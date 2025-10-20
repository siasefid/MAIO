# Changelog

## [v0.3] - 2025-10-20

### Added
- Documented prediction parity between deployed models.

### Metrics
- `predict` response on the regression benchmark returns **226.9131468172938** with the v2 model.
- The previous v1 model returned **235.9496372217627** on the same input.
- RMSE continues to improve compared with v0.2.

### Changed
- Clarified model provenance and release notes.


## [v0.2] - 2025-10-18

### Added
- `/health` endpoint to report service status and model version.
- Ridge and RandomForest model training pipeline.

### Changed
- Improved preprocessing, data loading, and pipeline structure.
- Updated Docker image build process and CI configuration; pipeline now passes consistently.d
- Model selection now automatically persists the best-performing candidate.

### Metrics
- Compared RMSE across Ridge and RandomForest candidates, selecting the best-performing model.


## [v0.1] - 2025-10-15

### Added
- Baseline StandardScaler + LinearRegression pipeline with `/predict` endpoint.
- Initial Docker image build and local smoke tests.
- GitHub Actions CI workflow.
