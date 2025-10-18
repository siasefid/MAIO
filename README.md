# Diabetes Risk Prediction Service

Baseline FastAPI microservice that wraps a StandardScaler + LinearRegression model trained on the scikit-learn diabetes dataset. The service powers iteration v0.1 of the project.

## Local Setup
1. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the baseline model and write `artifacts/model_baseline.pkl`:
   ```bash
   PYTHONPATH=src python -m risk_service.train
   ```
4. Start the API:
   ```bash
   uvicorn risk_service.api:app --host 0.0.0.0 --port 8000
   ```

## API
- `GET /health` → `{"status": "ok", "model_version": "v0.1"}`
- `POST /predict` expects a JSON body with all 10 normalized diabetes features. Example payload (the same values used in automated smoke tests):
  ```json
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
  ```
  Response shape:
  ```json
  {
    "prediction": <float>
  }
  ```

Sample curl invocations once the service is running locally:
```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":0.02,"sex":-0.044,"bmi":0.06,"bp":-0.03,"s1":-0.02,"s2":0.03,"s3":-0.02,"s4":0.02,"s5":0.02,"s6":-0.001}'
```

## Docker
Build and run the v0.1 image locally:
```bash
docker build -t diabetes-risk-service:v0.1 .
docker run --rm -p 8000:8000 diabetes-risk-service:v0.1
```
Official releases are published to GitHub Container Registry as `ghcr.io/<org>/<repo>:v0.1`. Use the same curl commands above to validate once the container is up.

## Metrics
- v0.1 (StandardScaler + LinearRegression): RMSE ≈ 53.85 on a 20% hold-out split (`random_state=42`).

## Automation
- `.github/workflows/Ci.yml` (push / PR): installs deps, runs lint + pytest, retrains the baseline, and uploads the artifact.
- `.github/workflows/release.yml` (tags `v*`): retrains, builds the Docker image, smoke-tests both endpoints, pushes to GHCR, and publishes a GitHub Release containing the RMSE.
