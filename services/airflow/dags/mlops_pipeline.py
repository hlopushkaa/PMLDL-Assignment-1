"""Automated MLOps pipeline: data engineering -> model engineering -> deployment.

Runs every 5 minutes. Each task is a thin wrapper around the stage scripts in
``code/``, so the very same commands can be run by hand (see README).
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow import DAG

# BashOperator moved into the "standard" provider in Airflow 3.
try:  # Airflow 3.x
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:  # Airflow 2.x
    from airflow.operators.bash import BashOperator

# services/airflow/dags/mlops_pipeline.py -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PYTHON = sys.executable  # the interpreter Airflow itself runs in

COMPOSE_DIR = PROJECT_ROOT / "code" / "deployment"

DEFAULT_ARGS = {
    "owner": "pmldl",
    "retries": 1,
    "retry_delay": timedelta(seconds=30),
}

with DAG(
    dag_id="mlops_pipeline",
    description="Data engineering -> model engineering -> deployment, every 5 minutes",
    default_args=DEFAULT_ARGS,
    start_date=pendulum.datetime(2026, 1, 1, tz="Europe/Moscow"),
    schedule="*/5 * * * *",
    catchup=False,
    max_active_runs=1,  # never let two runs rebuild the images at the same time
    dagrun_timeout=timedelta(minutes=10),
    tags=["pmldl", "mlops"],
) as dag:

    data_engineering = BashOperator(
        task_id="data_engineering",
        bash_command=f"{PYTHON} {PROJECT_ROOT}/code/datasets/prepare_data.py",
        doc_md="Stage 1: load the raw data, clean it, split it into train/test.",
    )

    model_engineering = BashOperator(
        task_id="model_engineering",
        bash_command=f"{PYTHON} {PROJECT_ROOT}/code/models/train_model.py",
        doc_md="Stage 2: build features, train, evaluate, log to MLflow, package the model.",
    )

    deployment = BashOperator(
        task_id="deployment",
        cwd=str(COMPOSE_DIR),
        bash_command="docker compose up -d --build --remove-orphans",
        doc_md=(
            "Stage 3: rebuild the API image around the freshly trained model and "
            "(re)start the API and app containers."
        ),
    )

    data_engineering >> model_engineering >> deployment
