# SupplySense

A hybrid AI system for predictive and adaptive e-commerce supply chain management. Three CatBoost
regressors forecast sales, delivery time, and profit from historical order data; those forecasts
feed a PPO reinforcement-learning agent that decides how much inventory to reorder, trained to
maximize revenue net of holding and stockout costs. A React dashboard visualizes both halves of
the system plus a live "try it" panel backed by the trained policy.

## Problem statement

E-commerce supply chains make two kinds of decisions badly when they're handled separately:
forecasting (how much will sell, how long will it take to ship, how profitable is this order) and
inventory control (how much to reorder, given forecasts and current stock). SupplySense couples
the two — CatBoost handles the forecasting, and its output becomes the state a PPO agent uses to
make reorder decisions — and benchmarks the learned policy against a naive fixed reorder-point
baseline on the same environment.

## Architecture

```
DataCo orders (180.5k rows)
        │
        ▼
Feature engineering  (lead_time_diff, order_urgency, date parts)
        │
        ▼
CatBoost x3  (sales · delivery time · profit)
        │
        ▼
RL state = [pred_sales, pred_profit, pred_delivery, inventory, pending_demand]
        │
        ▼
PPO agent  (clipped surrogate objective + GAE, from-scratch NumPy)
        │
        ▼
Reorder quantity  (continuous, 0-50 units)
```

## Tech stack

| Layer | Choice |
|---|---|
| Forecasting | CatBoost (`CatBoostRegressor` x3) |
| RL environment | Gymnasium (`InventoryEnv`, continuous action) |
| RL algorithm | PPO — **implemented from scratch in NumPy**, not Stable-Baselines3 (see below) |
| Backend | FastAPI, serving only pre-trained artifacts (no retraining at request time) |
| Frontend | React 19 + Vite + Tailwind v4 + Recharts |
| Data | [DataCo Smart Supply Chain for Big Data Analysis](https://data.mendeley.com/datasets/8gx2fvg2k6/5) (Constante, Silva, Pereira — Mendeley Data, 2019) |

### Why PPO from scratch instead of Stable-Baselines3

The build environment for this project had a locked-down network allowlist. Stable-Baselines3
requires PyTorch, and the only PyTorch wheel reachable from PyPI in that environment was the
default CUDA build — a 500MB+ core wheel plus several gigabytes of mandatory `nvidia-*` companion
packages just so `import torch` doesn't crash on a machine with no GPU. The CPU-only wheel lives on
`download.pytorch.org`, which wasn't reachable either. Rather than ship a CPU-only service with a
multi-gigabyte GPU dependency tree it will never use, `backend/ml/ppo_numpy.py` implements the same
algorithm by hand: a Gaussian policy over a continuous action, a single hidden layer (tanh) for
both the policy-mean and value networks, GAE for advantage estimation, the standard PPO clipped
surrogate objective, and Adam — all in plain NumPy, with manually-derived backprop. It trains
600 policy updates in about 13 seconds on 2 CPU cores.

## Key results

Trained on an 80/20 split of the 180,519-row dataset (see `backend/artifacts/metrics.json`
for the live numbers your run produces):

| Target | RMSE | MAE | R² |
|---|---|---|---|
| Sales per customer | 1.56 | 0.84 | 0.9998 |
| Delivery time (days) | 1.24 | 0.97 | 0.417 |
| Profit per order | 101.61 | 54.22 | -0.004 |

Sales is close to deterministic from price/quantity/discount, so the near-perfect R² is expected
rather than suspicious. Delivery time has real, moderate predictability from shipping mode and
region. Profit's R² near zero is an honest result, not a bug — `Benefit per order` in this dataset
carries enormous unexplained variance (std ≈ 104 against a mean of ≈ 22, with a small number of
extreme outliers), a characteristic noted in other public analyses of the same dataset.

**RL vs. baseline** (200 held-out episodes each, `backend/artifacts/rl_compare.json`):

| | Existing (fixed reorder-point) | Proposed (PPO agent) |
|---|---|---|
| Mean reward | 1342.7 | 1843.0 |
| Std dev | 1063.2 | 1567.4 |

The trained agent improves mean reward by roughly **+37%** over the naive (s, Q) baseline, at the
cost of higher variance — it takes bigger, more deliberate reorder swings rather than the
baseline's flat fixed-quantity response.

## Repository layout

```
backend/
  data/DataCoSupplyChainDataset.csv   the raw dataset (not committed — see Setup)
  ml/
    data_pipeline.py     load, clean, feature-engineer; per-target feature sets
    train_catboost.py    trains & saves the 3 CatBoost models + prediction artifacts
    rl_env.py            Gymnasium InventoryEnv (state = CatBoost preds + inventory + demand)
    ppo_numpy.py          from-scratch PPO (policy, value net, GAE, Adam)
    train_rl.py          trains the PPO agent, saves policy + reward history
    baseline.py          fixed reorder-point (s, Q) policy
    compare.py           runs both policies, saves the existing-vs-proposed comparison
    save_env_constants.py  persists env normalization constants for the API
  models/           trained artifacts: *.cbm, ppo_policy.npz
  artifacts/        metrics.json, predictions_*.json, rl_reward_history.json, rl_compare.json
  app/main.py       FastAPI app (reads artifacts only, never retrains)
  Dockerfile
frontend/
  src/pages/        Overview, Predictions, RLPerformance, Compare, TryIt
  src/components/   Sidebar, shared UI primitives
  vercel.json
render.yaml         Render deploy config for the backend
docker-compose.yml  one-command local run of both services
```

## Setup

### 1. Data

The raw CSV isn't committed (91MB). Download it from the source repo and place it at
`backend/data/DataCoSupplyChainDataset.csv`:

```
https://raw.githubusercontent.com/ashishpatel26/DataCo-SMART-SUPPLY-CHAIN-FOR-BIG-DATA-ANALYSIS/main/DataCoSupplyChainDataset.csv
```

### 2. Backend

```bash
cd backend
pip install -r requirements.txt

# Train everything (only needs to be done once; artifacts are committed under
# models/ and artifacts/ so this step is optional if you just want to run the API):
python -m ml.train_catboost
python -m ml.train_rl
python -m ml.compare
python -m ml.save_env_constants

# Serve
uvicorn app.main:app --reload --port 8000
```

API docs at `http://localhost:8000/docs` (FastAPI auto-generated OpenAPI).

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/api/*` to `http://localhost:8000` in dev (see `vite.config.js`). For a deployed
backend, set `VITE_API_BASE_URL` (see `.env.example`).

### 4. Or just Docker Compose

```bash
docker compose up --build
```

## Deployment (prepared, not yet deployed)

- **Frontend → Vercel**: `frontend/vercel.json` is set up for a static Vite build. Set
  `VITE_API_BASE_URL` as a Vercel environment variable pointing at the deployed backend.
- **Backend → Render**: `render.yaml` builds `backend/Dockerfile` directly. Railway works the same
  way pointed at the same Dockerfile. The image only needs the trained artifacts already committed
  under `backend/models/` and `backend/artifacts/` — it never trains at boot.

## API

| Endpoint | Description |
|---|---|
| `GET /health` | liveness check |
| `GET /predictions/{sales\|delivery\|profit}` | metrics, error histogram, actual-vs-predicted sample, feature importance |
| `GET /predictions` | RMSE/MAE/R² for all three targets |
| `GET /rl/performance` | reward-per-update training history |
| `GET /rl/compare` | baseline vs. RL agent comparison stats |
| `POST /simulate` | `{inventory, pending_demand, lead_time_days}` → the agent's reorder decision + projected outcome |

## Original report

The problem framing follows the DataCo Smart Supply Chain dataset's original academic context
(Constante, Silva, Pereira, 2019 — [Mendeley Data](https://data.mendeley.com/datasets/8gx2fvg2k6/5)).
This build regenerates all metrics and comparisons from a real training run rather than reusing
any report's placeholder numbers.
