from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

def make_pipeline(version: str = "v0.1") -> Pipeline:
    if version == "v0.1":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression())
        ])
    elif version == "v0.2_ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=42))
        ])
    elif version == "v0.2_rf":
        return Pipeline([
            ("reg", RandomForestRegressor(
                n_estimators=300, max_depth=8, min_samples_leaf=3,
                random_state=42, n_jobs=-1))
        ])
    else:
        raise ValueError(f"Unknown version: {version}")
