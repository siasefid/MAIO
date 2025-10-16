from risk_service.data import load_dataset
from risk_service.model import train_and_eval_models

def main():
    X, y = load_dataset(as_frame=True)
    res = train_and_eval_models(X, y)
print(f"RMSE: {res.rmse:.2f}")
print(f"Training samples: {res.n_train}, Test samples: {res.n_test}")
print(f"Saved model at: {res.model_path}")


if __name__ == "__main__":
    main()
