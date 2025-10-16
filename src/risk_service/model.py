from dataclasses import dataclass
from typing import Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import os

@dataclass
class TrainResult:
    rmse: float
    n_train: int
    n_test: int
    model_path: str

def build_baseline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])

def train_and_eval(X, y, random_state=42, test_size=0.2, out_dir="artifacts") -> TrainResult:
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    pipe = build_baseline()
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)
    rmse = float(np.sqrt(mean_squared_error(yte, preds)))
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model_baseline.pkl")
    joblib.dump({
        "pipeline": pipe,
        "feature_order": list(X.columns)  # reproducibility check in API
    }, model_path)
    return TrainResult(rmse=rmse, n_train=len(Xtr), n_test=len(Xte), model_path=model_path)

def load_model(path="artifacts/model_baseline.pkl"):
    bundle = joblib.load(path)
    return bundle["pipeline"], bundle["feature_order"]
