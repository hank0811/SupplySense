"""
A trivial fixed reorder-point baseline (no ML) for the "existing system" side
of the existing-vs-proposed comparison. Classic (s, Q) inventory policy:
if on-hand inventory falls below the reorder point s, order a fixed quantity Q;
otherwise order nothing. s and Q are set from the historical mean order
quantity in the dataset, not tuned against the RL reward.
"""
import numpy as np


class FixedReorderPointPolicy:
    def __init__(self, reorder_point=2.0, order_quantity=4.0):
        self.s = reorder_point
        self.Q = order_quantity

    def act(self, obs, inventory_raw):
        # obs[3] is normalized inventory; act on the raw inventory level directly.
        if inventory_raw < self.s:
            return self.Q
        return 0.0


def run_episodes(env, policy_fn, n_episodes, episode_length, seed_offset=0):
    """policy_fn(obs, inventory_raw) -> reorder_qty. Returns list of per-episode total raw reward."""
    episode_rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        total = 0.0
        for _ in range(episode_length):
            action = policy_fn(obs, env.inventory)
            obs, reward, terminated, truncated, info = env.step(np.array([action]))
            total += info["raw_reward"]
            if terminated or truncated:
                break
        episode_rewards.append(total)
    return episode_rewards
