from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from dataclasses import dataclass
import numpy as np
import joblib
import os

def load_model(path=None):
    """Load the most recent trained model (or a specific path)."""
    if path is None:
        artifacts = os.listdir("artifacts")
        pkl_files = [f for f in artifacts if f.endswith(".pkl")]
        if not pkl_files:
            raise FileNotFoundError("No model file found in artifacts/")
        # pick the latest saved model
        path = os.path.join("artifacts", sorted(pkl_files)[-1])
    model = joblib.load(path)
    return model


def build_model(model_type="linear"):
    if model_type == "linear":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ])
    elif model_type == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0))
        ])
    elif model_type == "rf":
        return Pipeline([
            ("regressor", RandomForestRegressor(
                n_estimators=200, random_state=42, n_jobs=-1
            ))
        ])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")



@dataclass
class TrainResult:
    rmse: float
    n_train: int
    n_test: int
    model_path: str

def train_and_eval_models(X, y, out_dir="artifacts", test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import numpy as np
    import joblib, os

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state)

    results = {}
    
    for name in ["linear", "ridge", "rf"]:
        model = build_model(name)
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        rmse = float(np.sqrt(mean_squared_error(yte, preds)))
        results[name] = rmse

    best_name = min(results, key=results.get)
    best_model = build_model(best_name)
    best_model.fit(X, y)

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"model_{best_name}.pkl")
    joblib.dump(best_model, model_path)

    # ✅ compute test metrics for compatibility
    best_rmse = results[best_name]

    return TrainResult(rmse=best_rmse, n_train=len(Xtr), n_test=len(Xte), model_path=model_path)