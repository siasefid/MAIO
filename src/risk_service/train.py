from risk_service.data import load_dataset
from risk_service.model import train_and_eval

def main():
    X, y = load_dataset(as_frame=True)
    result = train_and_eval(X, y)
    print({
        "rmse": result.rmse,
        "n_train": result.n_train,
        "n_test": result.n_test,
        "model_path": result.model_path
    })

if __name__ == "__main__":
    main()
