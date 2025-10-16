# Diabetes Risk Prediction Service

## Overview
This project builds a small ML service that predicts short-term diabetes progression risk using the scikit-learn diabetes dataset.

## Run locally
```bash
python -m risk_service.train
uvicorn risk_service.api:app --port 8000
