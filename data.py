from __future__ import annotations
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd

FEATURES = ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]

def load_split(test_size: float = 0.2, random_state: int = 42):
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]
    # enforce column order just in case
    X = X[FEATURES]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=random_state
    )
    return (X_train, X_test, y_train, y_test)
