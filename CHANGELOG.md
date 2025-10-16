# Changelog

## [v0.1] - 2025-10-15
- Baseline: StandardScaler + LinearRegression
- RMSE reported on 20% test split
- API endpoint `/predict` returns continuous risk score
- Docker image built and tested locally
- Added GitHub Actions CI pipeline


## [v0.2] - 2025-10-18
- Added Ridge and RandomForest models.
- Compared RMSE across all models.
- Automatically saves the best model.
- Improved performance and reproducibility.
