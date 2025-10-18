Diabetes Risk Prediction Service
Overview
This project builds a small ML service that predicts short-term diabetes progression risk using the scikit-learn diabetes dataset.

Run locally
python -m risk_service.train
uvicorn risk_service.api:app --port 8000


Version note

- This commit reverts the project from version 2 back to version 1.
- Reason: v1 is required to successfully build a Docker image for testing; v2 contains changes that break the current Docker build/test workflow.
- Plan: use this v1 state to create and validate Docker-based tests, then reintroduce v2 changes once the image and CI pipeline are updated.