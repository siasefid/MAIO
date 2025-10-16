from risk_service.data import load_dataset
from risk_service.model import train_and_eval_models

def test_train_and_eval_runs(tmp_path):
    X, y = load_dataset(as_frame=True)
    rmse, model_path = train_and_eval_models(X, y, out_dir=tmp_path)
    assert rmse > 0
    assert model_path.exists()
