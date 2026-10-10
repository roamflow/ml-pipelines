# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.12

# Pinned uv binary, copied in below. Multi-arch (amd64 + arm64).
FROM ghcr.io/astral-sh/uv:0.11.32 AS uv

# ---- builder: resolve and install dependencies into /app/.venv ----
FROM python:${PYTHON_VERSION}-slim AS builder
COPY --from=uv /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Layer 1: dependencies only. Re-runs only when pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Layer 2: the project itself, installed (not editable) into the venv.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# ---- runtime: clean slim container without build tools ----
FROM python:${PYTHON_VERSION}-slim AS runtime

# Non-root, fixed UID for Kubernetes securityContext (runAsNonRoot)
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --no-create-home app

COPY --from=builder --chown=app:app /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DAGSTER_GRPC_HOST=0.0.0.0 \
    DAGSTER_GRPC_PORT=4000 \
    DAGSTER_GRPC_MODULE=ml_pipelines.definitions

USER 10001:10001
WORKDIR /app
EXPOSE 4000

CMD ["dagster", "api", "grpc", "-h", "0.0.0.0", "-p", "4000", "-m", "ml_pipelines.definitions"]
