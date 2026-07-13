"""
Runs the fixed-reorder-point baseline and the trained PPO agent over the same
number of held-out episodes on InventoryEnv, and saves a Table-7.1-style
existing-vs-proposed comparison (mean reward, variance, std dev) to
backend/artifacts/rl_compare.json.
"""
import os
import json
import numpy as np

from ml.rl_env import InventoryEnv, EPISODE_LENGTH
from ml.ppo_numpy import PPOAgent
from ml.baseline import FixedReorderPointPolicy, run_episodes

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

N_EPISODES = 200


def main():
    env = InventoryEnv(n_rows=20000, seed=123)
    baseline = FixedReorderPointPolicy(reorder_point=2.0, order_quantity=4.0)
    baseline_rewards = run_episodes(env, baseline.act, N_EPISODES, EPISODE_LENGTH, seed_offset=10_000)

    agent = PPOAgent.load(os.path.join(MODELS_DIR, "ppo_policy.npz"))

    def rl_policy(obs, inventory_raw):
        action, _, _ = agent.act(obs)
        return float(action)

    rl_rewards = run_episodes(env, rl_policy, N_EPISODES, EPISODE_LENGTH, seed_offset=20_000)

    def stats(name, rewards):
        arr = np.array(rewards)
        return {
            "name": name,
            "n_episodes": len(arr),
            "mean_reward": float(arr.mean()),
            "variance": float(arr.var()),
            "std_dev": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    comparison = {
        "baseline": stats("Existing system (fixed reorder-point)", baseline_rewards),
        "rl_agent": stats("Proposed system (PPO agent)", rl_rewards),
        "episode_length": EPISODE_LENGTH,
        "reward_series": {
            "baseline": [float(r) for r in baseline_rewards],
            "rl_agent": [float(r) for r in rl_rewards],
        },
    }

    with open(os.path.join(ARTIFACTS_DIR, "rl_compare.json"), "w") as f:
        json.dump(comparison, f)

    print("Baseline:", comparison["baseline"])
    print("RL agent:", comparison["rl_agent"])
    improvement = comparison["rl_agent"]["mean_reward"] - comparison["baseline"]["mean_reward"]
    print(f"Improvement in mean reward: {improvement:.2f}")


if __name__ == "__main__":
    main()
