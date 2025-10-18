from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pathlib import Path
from math import sqrt
import pandas as pd
import joblib


MODEL_PATH = Path("artifacts/model_baseline.pkl")


def train_and_eval_models(X, y, out_dir=Path("artifacts")):
    """
    Train baseline LinearRegression model on diabetes data and evaluate RMSE.
    Saves the trained pipeline to out_dir/model_baseline.pkl.
    Returns (rmse, model_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, preds))

    model_path = out_dir / "model_baseline.pkl"
    joblib.dump(pipe, model_path)

    print(f"✅ Model trained and saved to {model_path}. RMSE={rmse:.2f}")
    return rmse, model_path


def predict_single(features: dict) -> float:
    """
    Load the trained model and predict a single sample.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([features])
    pred = model.predict(X)[0]
    return float(pred)
