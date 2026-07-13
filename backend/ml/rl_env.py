"""
Gymnasium environment for the inventory / reorder decision problem.

State  : [predicted_sales, predicted_profit, predicted_delivery_days,
          current_inventory, pending_demand]  (all min-max normalized)
Action : continuous reorder quantity, Box(low=0, high=50)
Reward : revenue_from_sold_units - holding_cost - stockout_penalty - reorder_cost

Each step draws a random historical order row, runs it through the three
trained CatBoost models to get realistic (sales, profit, delivery) predictions,
and uses the row's actual item quantity as the "true" demand signal the agent
has to plan against. This keeps the RL state grounded in the supervised
models rather than a fully synthetic distribution.
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from catboost import CatBoostRegressor

from ml.data_pipeline import load_and_prepare, get_feature_sets

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

HOLDING_COST_RATE = 0.5       # $ per unit of leftover inventory per step
STOCKOUT_PENALTY_RATE = 8.0   # $ per unit of unmet demand
REORDER_UNIT_COST_RATE = 0.6  # fraction of unit price paid to reorder a unit
MAX_INVENTORY = 200.0
MAX_DEMAND = 20.0
EPISODE_LENGTH = 30


def _load_models():
    models = {}
    for name in ["sales", "delivery", "profit"]:
        path = os.path.join(MODELS_DIR, f"{name}.cbm")
        m = CatBoostRegressor()
        m.load_model(path)
        models[name] = m
    return models


class InventoryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, n_rows=20000, seed=0):
        super().__init__()
        self._rng = np.random.default_rng(seed)

        df = load_and_prepare()
        if len(df) > n_rows:
            df = df.sample(n=n_rows, random_state=seed).reset_index(drop=True)
        sets = get_feature_sets(df)
        self._models = _load_models()

        X_sales, _, _ = sets["sales"]
        X_delivery, _, _ = sets["delivery"]
        X_profit, _, _ = sets["profit"]

        self._pred_sales = self._models["sales"].predict(X_sales)
        self._pred_delivery = self._models["delivery"].predict(X_delivery)
        self._pred_profit = self._models["profit"].predict(X_profit)
        self._quantity = df["Order Item Quantity"].values.astype(float)
        self._unit_price = df["Product Price"].values.astype(float)
        self._unit_price = np.clip(self._unit_price, 1.0, None)

        self._n = len(df)

        self.action_space = spaces.Box(low=0.0, high=50.0, shape=(1,), dtype=np.float32)
        # normalized: predicted_sales, predicted_profit, predicted_delivery, inventory, pending_demand
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(5,), dtype=np.float32)

        self._sales_scale = max(1.0, float(np.percentile(np.abs(self._pred_sales), 95)))
        self._profit_scale = max(1.0, float(np.percentile(np.abs(self._pred_profit), 95)))
        self._delivery_scale = max(1.0, float(np.percentile(np.abs(self._pred_delivery), 95)))

        self.inventory = 0.0
        self.t = 0
        self._idx = None

    def _draw_row(self):
        return int(self._rng.integers(0, self._n))

    def _obs(self, idx, pending_demand):
        return np.array([
            self._pred_sales[idx] / self._sales_scale,
            self._pred_profit[idx] / self._profit_scale,
            self._pred_delivery[idx] / self._delivery_scale,
            self.inventory / MAX_INVENTORY,
            pending_demand / MAX_DEMAND,
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.inventory = float(self._rng.uniform(0, 20))
        self.t = 0
        self._idx = self._draw_row()
        pending_demand = self._quantity[self._idx] + self._rng.normal(0, 0.5)
        pending_demand = max(0.0, pending_demand)
        self._pending_demand = pending_demand
        return self._obs(self._idx, pending_demand), {}

    def step(self, action):
        reorder_qty = float(np.clip(action[0] if hasattr(action, "__len__") else action, 0.0, 50.0))

        idx = self._idx
        demand = self._pending_demand
        unit_price = self._unit_price[idx]

        available = self.inventory + reorder_qty
        sold = min(available, demand)
        stockout = max(demand - available, 0.0)
        leftover = max(available - sold, 0.0)

        revenue = sold * unit_price
        holding_cost = HOLDING_COST_RATE * leftover
        stockout_penalty = STOCKOUT_PENALTY_RATE * stockout
        reorder_cost = REORDER_UNIT_COST_RATE * unit_price * reorder_qty

        reward = revenue - holding_cost - stockout_penalty - reorder_cost
        reward = reward / 100.0  # scale down for stable RL training

        self.inventory = min(leftover, MAX_INVENTORY)
        self.t += 1

        self._idx = self._draw_row()
        next_demand = self._quantity[self._idx] + self._rng.normal(0, 0.5)
        next_demand = max(0.0, next_demand)
        self._pending_demand = next_demand

        terminated = False
        truncated = self.t >= EPISODE_LENGTH
        obs = self._obs(self._idx, next_demand)
        info = {
            "sold": sold, "stockout": stockout, "leftover": leftover,
            "revenue": revenue, "holding_cost": holding_cost,
            "stockout_penalty": stockout_penalty, "reorder_cost": reorder_cost,
            "raw_reward": reward * 100.0,
        }
        return obs, reward, terminated, truncated, info
