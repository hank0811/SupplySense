"""
Trains three CatBoostRegressor models (sales, delivery time, profit) on the
DataCo dataset, reports RMSE/MAE/R2 on a held-out test split, and saves:
  - backend/models/{target}.cbm            trained model
  - backend/artifacts/metrics.json         RMSE/MAE/R2 for all three targets
  - backend/artifacts/predictions_{target}.json
        a sample of (actual, predicted) pairs + error histogram bins, for the
        dashboard's scatter/histogram charts.
No retraining happens at request time -- the FastAPI layer only reads these
saved artifacts.
"""
import os
import json
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ml.data_pipeline import load_and_prepare, get_feature_sets

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

SAMPLE_SIZE = 500
HIST_BINS = 30


def train_one(name, X, y, cat_features, iterations=180):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=0.18,
        depth=5,
        loss_function="RMSE",
        cat_features=cat_features,
        random_seed=42,
        thread_count=2,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    tmp_model_path = f"/tmp/{name}.cbm"
    model.save_model(tmp_model_path)
    with open(tmp_model_path, "rb") as src, open(os.path.join(MODELS_DIR, f"{name}.cbm"), "wb") as dst:
        dst.write(src.read())

    errors = (preds - y_test.values)
    hist_counts, hist_edges = np.histogram(errors, bins=HIST_BINS)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), size=min(SAMPLE_SIZE, len(X_test)), replace=False)
    sample = {
        "actual": [float(v) for v in y_test.values[idx]],
        "predicted": [float(v) for v in preds[idx]],
    }

    artifact = {
        "target": name,
        "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
        "error_histogram": {
            "counts": [int(c) for c in hist_counts],
            "bin_edges": [float(e) for e in hist_edges],
        },
        "sample": sample,
        "feature_importance": _feature_importance(model, X_train),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    with open(os.path.join(ARTIFACTS_DIR, f"predictions_{name}.json"), "w") as f:
        json.dump(artifact, f)

    print(f"[{name}] RMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.4f}")
    return artifact["metrics"]


def _feature_importance(model, X_train, top_k=12):
    importances = model.get_feature_importance()
    names = X_train.columns.tolist()
    pairs = sorted(zip(names, importances), key=lambda p: -p[1])[:top_k]
    return [{"feature": n, "importance": float(v)} for n, v in pairs]


def main(target=None):
    df = load_and_prepare()
    sets = get_feature_sets(df)

    if target:
        sets = {target: sets[target]}

    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    all_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            all_metrics = json.load(f)

    for name, (X, y, cat_features) in sets.items():
        all_metrics[name] = train_one(name, X, y, cat_features)

    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("Saved metrics.json:", all_metrics)


if __name__ == "__main__":
    import sys
    t = None
    if len(sys.argv) > 1:
        t = sys.argv[1]
    main(target=t)
