from sklearn.datasets import load_diabetes
import pandas as pd


FEATURE_ORDER = None  # will be set at import time


def load_dataset(as_frame=True):
    Xy = load_diabetes(as_frame=as_frame)
    frame = Xy.frame  # includes "target"
    X = frame.drop(columns=["target"])
    y = frame["target"]
    global FEATURE_ORDER
    FEATURE_ORDER = list(X.columns)
    return X, y


def ensure_feature_order(df: pd.DataFrame) -> pd.DataFrame:
    # reorder and ensure no missing columns
    missing = set(FEATURE_ORDER) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature(s): {sorted(missing)}")
    return df[FEATURE_ORDER]
