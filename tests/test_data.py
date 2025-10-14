from src.data import load_split, FEATURES

def test_load_split_shapes():
    X_train, X_test, y_train, y_test = load_split()
    assert list(X_train.columns) == FEATURES
    assert len(X_train) > 0 and len(X_test) > 0
    assert len(X_train) == len(y_train)
