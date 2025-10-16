from risk_service.data import load_dataset
from risk_service.model import train_and_eval_models

def main():
    X, y = load_dataset(as_frame=True)
    results, best_name, model_path = train_and_eval_models(X, y)
    print("Model performance (RMSE):", results)
    print(f"Best model: {best_name}")
    print(f"Saved model at: {model_path}")

if __name__ == "__main__":
    main()
