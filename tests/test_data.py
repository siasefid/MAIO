from risk_service.data import load_dataset


def test_load_dataset_shapes():
    X, y = load_dataset(as_frame=True)
    assert len(X) == len(y)
    assert "bmi" in X.columns  # sanity check
