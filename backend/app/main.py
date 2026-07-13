"""
SupplySense FastAPI backend.

Serves everything from pre-trained artifacts saved by the ml/ scripts --
no model training or retraining happens at request time:
  - backend/models/{sales,delivery,profit}.cbm   CatBoost regressors
  - backend/models/ppo_policy.npz                trained PPO policy
  - backend/artifacts/*.json                     metrics, predictions, RL history/comparison
"""
import os
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml.ppo_numpy import PPOAgent

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = FastAPI(title="SupplySense API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache = {}


def _load_json(name):
    if name not in _cache:
        path = os.path.join(ARTIFACTS_DIR, name)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"artifact not found: {name}")
        with open(path) as f:
            _cache[name] = json.load(f)
    return _cache[name]


def _load_agent():
    if "agent" not in _cache:
        _cache["agent"] = PPOAgent.load(os.path.join(MODELS_DIR, "ppo_policy.npz"))
    return _cache["agent"]


def _env_constants():
    return _load_json("env_constants.json")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predictions/{target}")
def predictions(target: str):
    if target not in ("sales", "delivery", "profit"):
        raise HTTPException(status_code=404, detail="target must be one of sales|delivery|profit")
    return _load_json(f"predictions_{target}.json")


@app.get("/predictions")
def all_predictions():
    return _load_json("metrics.json")


@app.get("/rl/performance")
def rl_performance():
    return _load_json("rl_reward_history.json")


@app.get("/rl/compare")
def rl_compare():
    return _load_json("rl_compare.json")


class SimulateRequest(BaseModel):
    inventory: float = Field(10.0, ge=0, le=200, description="Current on-hand inventory (units)")
    pending_demand: float = Field(3.0, ge=0, le=20, description="Expected demand this period (units)")
    lead_time_days: float = Field(3.0, ge=0, le=15, description="Scheduled/expected shipping lead time (days)")
    n_episodes: int = Field(1, ge=1, le=50, description="How many simulated steps/episodes to average over")


@app.post("/simulate")
def simulate(req: SimulateRequest):
    agent = _load_agent()
    c = _env_constants()

    obs = np.array([
        c["mean_pred_sales"] / c["sales_scale"],
        c["mean_pred_profit"] / c["profit_scale"],
        req.lead_time_days / c["delivery_scale"],
        req.inventory / c["max_inventory"],
        req.pending_demand / c["max_demand"],
    ], dtype=np.float64)

    reorder_qty = agent.predict(obs)

    unit_price = c["mean_unit_price"]
    available = req.inventory + reorder_qty
    sold = min(available, req.pending_demand)
    stockout = max(req.pending_demand - available, 0.0)
    leftover = max(available - sold, 0.0)

    revenue = sold * unit_price
    holding_cost = c["holding_cost_rate"] * leftover
    stockout_penalty = c["stockout_penalty_rate"] * stockout
    reorder_cost = c["reorder_unit_cost_rate"] * unit_price * reorder_qty
    net_reward = revenue - holding_cost - stockout_penalty - reorder_cost

    return {
        "reorder_qty": round(reorder_qty, 2),
        "projected_outcome": {
            "units_sold": round(sold, 2),
            "stockout_units": round(stockout, 2),
            "leftover_inventory": round(leftover, 2),
            "revenue": round(revenue, 2),
            "holding_cost": round(holding_cost, 2),
            "stockout_penalty": round(stockout_penalty, 2),
            "reorder_cost": round(reorder_cost, 2),
            "net_reward": round(net_reward, 2),
        },
        "assumed_unit_price": round(unit_price, 2),
    }
