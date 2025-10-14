from src.model import make_pipeline
from src.data import load_split

def test_pipeline_predict_shape():
    X_train, X_test, y_train, y_test = load_split()
    model = make_pipeline("v0.1")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape[0] == y_test.shape[0]
