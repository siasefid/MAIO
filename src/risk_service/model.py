from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from dataclasses import dataclass
import numpy as np
import joblib
from pathlib import Path
from math import sqrt

def train_and_eval_models(X, y, out_dir=Path("artifacts")):
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, preds))

    model_path = out_dir / "model_baseline.pkl"
    joblib.dump(pipe, model_path)

    print(f"✅ Model trained. RMSE: {rmse:.2f}")
    return rmse, model_path
