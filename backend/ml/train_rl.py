"""
Trains the PPO agent against InventoryEnv and saves:
  - backend/models/ppo_policy.npz         trained policy + value net weights
  - backend/artifacts/rl_reward_history.json   reward-per-episode + rolling mean,
                                                for the RL Performance dashboard page.
No retraining happens at request time.
"""
import os
import json
import time
import numpy as np

from ml.rl_env import InventoryEnv, EPISODE_LENGTH
from ml.ppo_numpy import PPOAgent, compute_gae

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

EPISODES_PER_UPDATE = 8
N_UPDATES = 600


def collect_rollout(env, agent, n_episodes):
    obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []
    episode_rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        for _ in range(EPISODE_LENGTH):
            action, logp, value = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(logp)
            val_buf.append(value)
            rew_buf.append(reward)
            done_buf.append(float(terminated or truncated))
            ep_reward += info["raw_reward"]
            obs = next_obs
            if terminated or truncated:
                break
        episode_rewards.append(ep_reward)

    last_value = agent.value_of(obs)
    advantages, returns = compute_gae(
        np.array(rew_buf), np.array(val_buf), np.array(done_buf), last_value
    )
    return (np.array(obs_buf), np.array(act_buf), np.array(logp_buf),
            advantages, returns, episode_rewards)


def main():
    t0 = time.time()
    env = InventoryEnv(n_rows=20000, seed=0)
    agent = PPOAgent(obs_dim=env.observation_space.shape[0], action_low=0.0, action_high=50.0,
                      hidden=32, seed=0, lr=3e-3, policy_bias_init=-1.0986)

    history = []
    for update in range(N_UPDATES):
        obs, acts, logps, adv, ret, ep_rewards = collect_rollout(env, agent, EPISODES_PER_UPDATE)
        stats = agent.update(obs, acts, logps, adv, ret)
        mean_ep_reward = float(np.mean(ep_rewards))
        history.append({
            "update": update,
            "mean_episode_reward": mean_ep_reward,
            "policy_loss": stats["policy_loss"],
            "value_loss": stats["value_loss"],
            "std": stats["std"],
        })
        if update % 10 == 0 or update == N_UPDATES - 1:
            print(f"update {update:4d}  mean_ep_reward={mean_ep_reward:8.2f}  "
                  f"policy_loss={stats['policy_loss']:.4f}  value_loss={stats['value_loss']:.4f}  "
                  f"std={stats['std']:.3f}  elapsed={time.time()-t0:.1f}s")

    agent.save(os.path.join(MODELS_DIR, "ppo_policy.npz"))

    rewards = [h["mean_episode_reward"] for h in history]
    window = 10
    rolling = [float(np.mean(rewards[max(0, i - window + 1):i + 1])) for i in range(len(rewards))]
    for h, r in zip(history, rolling):
        h["rolling_mean_reward"] = r

    with open(os.path.join(ARTIFACTS_DIR, "rl_reward_history.json"), "w") as f:
        json.dump({"history": history, "episodes_per_update": EPISODES_PER_UPDATE,
                   "episode_length": EPISODE_LENGTH}, f)

    print(f"Done in {time.time()-t0:.1f}s. Final mean episode reward: {rewards[-1]:.2f}")


if __name__ == "__main__":
    main()
