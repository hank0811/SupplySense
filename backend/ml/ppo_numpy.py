"""
A from-scratch PPO implementation in plain NumPy (no PyTorch / Stable-Baselines3).

Why not Stable-Baselines3: SB3 requires PyTorch, and the only PyTorch wheel this
sandbox can reach from PyPI is the default CUDA build (500MB+ core wheel plus
several GB of mandatory nvidia-* companion packages just so `import torch`
doesn't crash on a machine with no GPU). That's a bad trade for a CPU-only
service that will eventually run on Render/Railway. This module implements the
same algorithm --- clipped surrogate objective, Generalized Advantage
Estimation, a Gaussian policy over a continuous action, Adam --- by hand.

Architecture: one hidden layer (tanh) for both the policy-mean network and the
value network, plus a single learned (state-independent) log_std parameter.
"""
import numpy as np


def _init_layer(rng, n_in, n_out, scale=None):
    scale = scale or np.sqrt(2.0 / n_in)
    W = rng.normal(0, scale, size=(n_in, n_out))
    b = np.zeros(n_out)
    return W, b


class Adam:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class PPOAgent:
    """Gaussian policy + value function, both single-hidden-layer MLPs, trained with PPO-clip."""

    def __init__(self, obs_dim, action_low=0.0, action_high=50.0, hidden=32, seed=0, lr=3e-4, policy_bias_init=0.0):
        rng = np.random.default_rng(seed)
        self.obs_dim = obs_dim
        self.hidden = hidden
        self.action_low = action_low
        self.action_high = action_high
        self.action_mid = (action_high + action_low) / 2
        self.action_range = (action_high - action_low) / 2

        W1, b1 = _init_layer(rng, obs_dim, hidden)
        W2, b2 = _init_layer(rng, hidden, 1, scale=0.01)
        b2 = b2 + policy_bias_init  # bias the initial mean toward a sane starting reorder qty
        vW1, vb1 = _init_layer(rng, obs_dim, hidden)
        vW2, vb2 = _init_layer(rng, hidden, 1, scale=0.1)

        self.params = {
            "pW1": W1, "pb1": b1, "pW2": W2, "pb2": b2,
            "log_std": np.array([-0.5]),
            "vW1": vW1, "vb1": vb1, "vW2": vW2, "vb2": vb2,
        }
        self.opt = Adam(self.params, lr=lr)
        self.rng = rng

    # ---- forward passes ----
    def _policy_forward(self, obs):
        z1 = obs @ self.params["pW1"] + self.params["pb1"]
        h1 = np.tanh(z1)
        z2 = h1 @ self.params["pW2"] + self.params["pb2"]
        mean_tanh = np.tanh(z2).squeeze(-1)
        mean = self.action_mid + self.action_range * mean_tanh
        return mean, {"obs": obs, "z1": z1, "h1": h1, "z2": z2, "mean_tanh": mean_tanh}

    def _value_forward(self, obs):
        z1 = obs @ self.params["vW1"] + self.params["vb1"]
        h1 = np.tanh(z1)
        z2 = h1 @ self.params["vW2"] + self.params["vb2"]
        value = z2.squeeze(-1)
        return value, {"obs": obs, "z1": z1, "h1": h1}

    def act(self, obs):
        obs = np.asarray(obs, dtype=np.float64).reshape(1, -1)
        mean, _ = self._policy_forward(obs)
        std = np.exp(self.params["log_std"])
        action = mean + std * self.rng.normal(size=mean.shape)
        logprob = self._logprob(action, mean, std)
        value, _ = self._value_forward(obs)
        clipped = np.clip(action, self.action_low, self.action_high)
        return clipped[0], float(logprob[0]), float(value[0])

    def predict(self, obs):
        """Deterministic action (the policy mean, no exploration noise) -- used for serving/demo."""
        obs = np.asarray(obs, dtype=np.float64).reshape(1, -1)
        mean, _ = self._policy_forward(obs)
        clipped = np.clip(mean, self.action_low, self.action_high)
        return float(clipped[0])

    def value_of(self, obs):
        obs = np.asarray(obs, dtype=np.float64).reshape(1, -1)
        value, _ = self._value_forward(obs)
        return float(value[0])

    @staticmethod
    def _logprob(a, mean, std):
        var = std ** 2
        return -0.5 * (((a - mean) ** 2) / var + 2 * np.log(std) + np.log(2 * np.pi))

    # ---- training ----
    def update(self, obs, actions, old_logprobs, advantages, returns, clip_eps=0.2,
               epochs=8, batch_size=64, vf_coef=0.5, ent_coef=0.005):
        n = len(obs)
        obs = np.asarray(obs, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64).reshape(-1)
        old_logprobs = np.asarray(old_logprobs, dtype=np.float64)
        advantages = np.asarray(advantages, dtype=np.float64)
        returns = np.asarray(returns, dtype=np.float64)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        idx_all = np.arange(n)
        last_stats = {}
        for _ in range(epochs):
            self.rng.shuffle(idx_all)
            for start in range(0, n, batch_size):
                idx = idx_all[start:start + batch_size]
                last_stats = self._update_batch(
                    obs[idx], actions[idx], old_logprobs[idx],
                    advantages[idx], returns[idx], clip_eps, vf_coef, ent_coef,
                )
        return last_stats

    def _update_batch(self, obs, a, old_logp, adv, ret, clip_eps, vf_coef, ent_coef):
        p = self.params
        mean, cache = self._policy_forward(obs)
        std = np.exp(p["log_std"])[0]

        logp = self._logprob(a, mean, std)
        ratio = np.exp(logp - old_logp)
        clipped_ratio = np.clip(ratio, 1 - clip_eps, 1 + clip_eps)
        surr1 = ratio * adv
        surr2 = clipped_ratio * adv
        use_surr1 = surr1 <= surr2  # min(): pick the smaller term

        entropy = 0.5 * np.log(2 * np.pi * np.e * std ** 2)

        # d(loss)/d(logp): loss = -mean(min(surr1,surr2)) - ent_coef*entropy
        # d(surr1)/d(logp) = ratio*adv (since d(ratio)/d(logp)=ratio)
        # d(surr2)/d(logp) = adv*ratio inside the unclipped band, 0 outside it
        in_band = (ratio > 1 - clip_eps) & (ratio < 1 + clip_eps)
        dsurr_dlogp = np.where(use_surr1, ratio * adv, np.where(in_band, ratio * adv, 0.0))
        n = len(a)
        dloss_dlogp = -dsurr_dlogp / n

        # d(logp)/d(mean) and d(logp)/d(log_std)
        var = std ** 2
        dlogp_dmean = (a - mean) / var
        dlogp_dlogstd = ((a - mean) / std) ** 2 - 1.0

        dloss_dmean = dloss_dlogp * dlogp_dmean
        # entropy = 0.5*log(2*pi*e*std^2) = const + log_std, so d(entropy)/d(log_std) = 1
        dloss_dlogstd = float(np.sum(dloss_dlogp * dlogp_dlogstd)) - ent_coef

        # backprop dloss_dmean through mean = mid + range*tanh(z2)
        dmean_dz2 = self.action_range * (1 - cache["mean_tanh"] ** 2)
        dz2 = (dloss_dmean * dmean_dz2)[:, None]  # (n,1)

        h1 = cache["h1"]
        dpW2 = h1.T @ dz2
        dpb2 = dz2.sum(axis=0)
        dh1 = dz2 @ p["pW2"].T
        dz1 = dh1 * (1 - h1 ** 2)
        dpW1 = cache["obs"].T @ dz1
        dpb1 = dz1.sum(axis=0)

        # ---- value function: MSE loss ----
        value, vcache = self._value_forward(obs)
        dvalue = (value - ret) * (2.0 / n) * vf_coef
        vh1 = vcache["h1"]
        dvz2 = dvalue[:, None]
        dvW2 = vh1.T @ dvz2
        dvb2 = dvz2.sum(axis=0)
        dvh1 = dvz2 @ p["vW2"].T
        dvz1 = dvh1 * (1 - vh1 ** 2)
        dvW1 = vcache["obs"].T @ dvz1
        dvb1 = dvz1.sum(axis=0)

        grads = {
            "pW1": dpW1, "pb1": dpb1, "pW2": dpW2, "pb2": dpb2,
            "log_std": np.array([dloss_dlogstd]),
            "vW1": dvW1, "vb1": dvb1, "vW2": dvW2, "vb2": dvb2,
        }
        self.opt.step(p, grads)
        p["log_std"] = np.clip(p["log_std"], -2.0, 0.7)  # keep exploration noise bounded/stable

        policy_loss = -np.mean(np.minimum(surr1, surr2))
        value_loss = np.mean((value - ret) ** 2)
        return {"policy_loss": float(policy_loss), "value_loss": float(value_loss),
                "entropy": float(entropy), "std": float(std)}

    def save(self, path):
        np.savez(path, **self.params, obs_dim=self.obs_dim, hidden=self.hidden,
                  action_low=self.action_low, action_high=self.action_high)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        agent = cls(int(data["obs_dim"]), float(data["action_low"]), float(data["action_high"]),
                    hidden=int(data["hidden"]))
        for k in agent.params:
            agent.params[k] = data[k]
        return agent


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    n = len(rewards)
    advantages = np.zeros(n)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_value = last_value if t == n - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns
