from risk_service.data import load_dataset
from risk_service.model import train_and_eval
def test_train_and_eval_runs(tmp_path):
    X, y = load_dataset(as_frame=True)
    res = train_and_eval(X, y, out_dir=tmp_path)
    assert res.rmse > 0
