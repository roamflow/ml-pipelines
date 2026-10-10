# ml-pipelines

Dagster User Code Location serving ML pipelines over gRPC for the `roamflow` MLOps platform.

## Architecture

This repository is isolated from the Dagster Control Plane (Webserver + Daemon) following the Dagster gRPC Code Location architecture:
- **Code Execution**: Runs as a separate gRPC server (`dagster api grpc -p 4000`).
- **Dependency Isolation**: Data science and ML packages (`scikit-learn`, `mlflow`, `pandas`, `numpy`) live exclusively here.
- **Model Tracking & Registry**: Logs runs, artifacts, and parameters to MLflow 3.17 (`http://mlflow.mlflow.svc.cluster.local:5000`) and promotes models with the `@Production` alias.

## Local Development

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run linting
uv run ruff check .

# Validate Dagster definitions
uv run dagster definitions validate -m ml_pipelines.definitions

# Run gRPC server locally
uv run dagster api grpc -h 127.0.0.1 -p 4000 -m ml_pipelines.definitions
```

## Docker Build

Native ARM64 build using `uv` cache mount:

```bash
docker build -t ghcr.io/roamflow/ml-pipelines:latest .
```
