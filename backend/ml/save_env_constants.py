"""One-off script: persist the InventoryEnv normalization constants and
representative dataset averages so the FastAPI /simulate endpoint can build
a state vector without reloading the full 20k-row env at request time."""
import os
import json
import numpy as np

from ml.rl_env import InventoryEnv, MAX_INVENTORY, MAX_DEMAND

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


def main():
    env = InventoryEnv(n_rows=20000, seed=0)
    constants = {
        "sales_scale": env._sales_scale,
        "profit_scale": env._profit_scale,
        "delivery_scale": env._delivery_scale,
        "max_inventory": MAX_INVENTORY,
        "max_demand": MAX_DEMAND,
        "mean_pred_sales": float(np.mean(env._pred_sales)),
        "mean_pred_profit": float(np.mean(env._pred_profit)),
        "mean_pred_delivery": float(np.mean(env._pred_delivery)),
        "mean_unit_price": float(np.mean(env._unit_price)),
        "holding_cost_rate": 0.5,
        "stockout_penalty_rate": 8.0,
        "reorder_unit_cost_rate": 0.6,
    }
    with open(os.path.join(ARTIFACTS_DIR, "env_constants.json"), "w") as f:
        json.dump(constants, f, indent=2)
    print(constants)


if __name__ == "__main__":
    main()
