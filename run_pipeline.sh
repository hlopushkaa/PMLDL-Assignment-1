#!/usr/bin/env bash
# Runs the whole pipeline once, without Airflow.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

echo "==> Stage 1: data engineering"
"$PYTHON" "$PROJECT_ROOT/code/datasets/prepare_data.py"

echo "==> Stage 2: model engineering"
"$PYTHON" "$PROJECT_ROOT/code/models/train_model.py"

echo "==> Stage 3: deployment"
# The image must run the same Python the model was pickled with.
PYTHON_VERSION="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
export PYTHON_VERSION
echo "    building images on python:${PYTHON_VERSION}-slim"

cd "$PROJECT_ROOT/code/deployment"
docker compose up -d --build --remove-orphans

echo
echo "Done. API: http://localhost:8000/docs   App: http://localhost:8501"
