from risk_service.data import load_dataset
from risk_service.model import train_and_eval_models


def main():
    X, y = load_dataset(as_frame=True)
    rmse, model_path = train_and_eval_models(X, y)
    print(f"✅ Model trained successfully and saved to {model_path}")
    print(f"RMSE: {rmse:.2f}")


if __name__ == "__main__":
    main()
