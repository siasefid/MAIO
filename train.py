from __future__ import annotations
import argparse, json, os, sys, time, platform, joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from src.data import load_split, FEATURES
from src.model import make_pipeline
from importlib.metadata import version as pkgver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v0.1", help="v0.1 | v0.2_ridge | v0.2_rf")
    parser.add_argument("--outdir", default="models")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    X_train, X_test, y_train, y_test = load_split(test_size=args.test_size, random_state=args.seed)

    np.random.seed(args.seed)
    model = make_pipeline(args.version)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, preds, squared=False)

    # save artifacts
    model_path = os.path.join(args.outdir, "model.joblib")
    metrics_path = os.path.join(args.outdir, "metrics.json")
    joblib.dump(model, model_path)

    metrics = {
        "model_version": args.version,
        "rmse": float(rmse),
        "test_size": args.test_size,
        "seed": args.seed,
        "timestamp": int(time.time()),
        "feature_names": FEATURES,
        "env": {
            "python": platform.python_version(),
            "numpy": pkgver("numpy"),
            "pandas": pkgver("pandas"),
            "scikit_learn": pkgver("scikit-learn")
        }
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({"rmse": rmse, "model_version": args.version, "model_path": model_path}, indent=2))

if __name__ == "__main__":
    sys.exit(main())
