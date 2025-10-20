# Diabetes Risk Prediction Service

## Overview
Small FastAPI app that predicts short-term diabetes progression using the scikit-learn diabetes dataset. Higher scores mean higher follow-up priority.

## Setup
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the model (recreates `artifacts/model_baseline.pkl`)
```bash
python -m risk_service.train
```

## Run the API locally
```bash
uvicorn risk_service.api:app --host 0.0.0.0 --port 8000
```

### Health check
```
GET http://127.0.0.1:8000/health
→ {"status": "ok", "model_version": "v0.2"}
```

### Sample prediction request
```
POST http://127.0.0.1:8000/predict
Content-Type: application/json

{
	"age": 0.02,
	"sex": -0.044,
	"bmi": 0.06,
	"bp": -0.03,
	"s1": -0.02,
	"s2": 0.03,
	"s3": -0.02,
	"s4": 0.02,
	"s5": 0.02,
	"s6": -0.001
}

→ {"prediction": 142.31}
```

## Run tests
```bash
pytest -q
```

## Docker
```bash
docker build -t diabetes-risk-service .
docker run -p 8000:8000 diabetes-risk-service
```

## Collaborating
- Keep pull requests focused (one feature or fix).
- Write a short summary of what changed and why.
- Squash or rebase before merging so the history stays tidy.
