#!/usr/bin/env bash
# Runs the whole pipeline once, without Airflow.
# Useful for a first check that every stage works before scheduling it.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

echo "==> Stage 1: data engineering"
"$PYTHON" "$PROJECT_ROOT/code/datasets/prepare_data.py"

echo "==> Stage 2: model engineering"
"$PYTHON" "$PROJECT_ROOT/code/models/train_model.py"

echo "==> Stage 3: deployment"
cd "$PROJECT_ROOT/code/deployment"
docker compose up -d --build --remove-orphans

echo
echo "Done. API: http://localhost:8000/docs   App: http://localhost:8501"
