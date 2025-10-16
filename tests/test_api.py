from risk_service.data import load_dataset
from risk_service.model import train_and_eval_models, load_model
import pandas as pd

def test_pipeline_predict(tmp_path):
    X, y = load_dataset(as_frame=True)
    res = train_and_eval_models(X, y, out_dir=tmp_path)
    pipe, feats = load_model(res.model_path)
    df = X.iloc[[0]][feats]
    pred = pipe.predict(df)[0]
    assert isinstance(float(pred), float)
