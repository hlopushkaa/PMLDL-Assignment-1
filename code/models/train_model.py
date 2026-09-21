"""Stage 2: Model engineering.

Builds features, trains a model, evaluates it on the test split, logs
everything to MLflow and packages the trained model into a file.

Input  : data/processed/train.csv, data/processed/test.csv
Output : models/model.pkl, models/model_meta.json, models/metrics.json
         plus an MLflow run under mlruns/
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "models"))

from features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RAW_CATEGORICAL,
    RAW_NUMERIC,
    TARGET,
    build_features,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
# Local MLflow backend: a SQLite database plus an artifact folder. Point
# MLFLOW_TRACKING_URI at a tracking server to use one instead.
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACTS = PROJECT_ROOT / "mlruns"

EXPERIMENT_NAME = "california-housing"
MODEL_NAME = "HousingPriceRegressor"
RANDOM_SEED = 42

HYPERPARAMS = {
    "n_estimators": 150,
    "max_depth": 18,
    "min_samples_leaf": 5,
    "max_features": 0.5,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("model-engineering")

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing. Run stage 1 first.")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def build_pipeline() -> Pipeline:
    """Preprocessing + estimator in a single artifact.

    Packaging the preprocessing together with the estimator means the API only
    ever has to hand over raw-shaped rows.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(**HYPERPARAMS)),
        ]
    )


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    preds = model.predict(X)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "mae": float(mean_absolute_error(y, preds)),
        "r2": float(r2_score(y, preds)),
    }


def build_meta(train_raw: pd.DataFrame, metrics: dict[str, float]) -> dict:
    """Input schema for the web app: ranges and sensible defaults per field."""
    fields = []
    for col in RAW_NUMERIC:
        series = train_raw[col]
        fields.append(
            {
                "name": col,
                "type": "number",
                "min": float(series.min()),
                "max": float(series.max()),
                "default": float(series.median()),
                "label": col.replace("_", " ").capitalize(),
            }
        )
    for col in RAW_CATEGORICAL:
        options = sorted(train_raw[col].dropna().unique().tolist())
        fields.append(
            {
                "name": col,
                "type": "category",
                "options": options,
                "default": train_raw[col].mode()[0],
                "label": col.replace("_", " ").capitalize(),
            }
        )
    return {
        "model_name": MODEL_NAME,
        "target": TARGET,
        "target_label": "Median house value, USD",
        "fields": fields,
        "metrics": metrics,
    }


def setup_experiment(mlflow) -> None:
    """Point MLflow at this checkout's artifact folder.

    An existing tracking database carries the absolute artifact path of the
    machine that created it. If the project was copied from somewhere else,
    that path does not exist here and every run would fail on writing
    artifacts, so the stale experiment is archived and a fresh one is created.
    """
    expected = MLFLOW_ARTIFACTS.as_uri()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is not None and experiment.artifact_location != expected:
        log.warning(
            "Experiment %r points at %s, which does not belong to this checkout. "
            "Renaming it and starting a fresh experiment.",
            EXPERIMENT_NAME,
            experiment.artifact_location,
        )
        from mlflow.tracking import MlflowClient

        MlflowClient().rename_experiment(
            experiment.experiment_id, f"{EXPERIMENT_NAME}-stale-{experiment.experiment_id}"
        )
        experiment = None

    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=expected)

    mlflow.set_experiment(EXPERIMENT_NAME)


def main() -> None:
    train_raw, test_raw = load_splits()
    log.info("Train: %d rows | Test: %d rows", len(train_raw), len(test_raw))

    X_train = build_features(train_raw)
    y_train = train_raw[TARGET]
    X_test = build_features(test_raw)
    y_test = test_raw[TARGET]

    model = build_pipeline()

    import mlflow
    import mlflow.sklearn

    MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB}")
    mlflow.set_tracking_uri(tracking_uri)
    log.info("MLflow tracking URI: %s", tracking_uri)

    setup_experiment(mlflow)

    with mlflow.start_run() as run:
        mlflow.log_params(HYPERPARAMS)
        mlflow.log_param("n_train_rows", len(train_raw))
        mlflow.log_param("n_test_rows", len(test_raw))
        mlflow.log_param("n_features", len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES))

        model.fit(X_train, y_train)

        train_metrics = evaluate(model, X_train, y_train)
        test_metrics = evaluate(model, X_test, y_test)

        for name, value in train_metrics.items():
            mlflow.log_metric(f"train_{name}", value)
        for name, value in test_metrics.items():
            mlflow.log_metric(f"test_{name}", value)

        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_train.head(3),
            serialization_format="cloudpickle",
        )

        log.info("MLflow run id: %s", run.info.run_id)

    log.info(
        "Test metrics: RMSE=%.1f  MAE=%.1f  R2=%.4f",
        test_metrics["rmse"],
        test_metrics["mae"],
        test_metrics["r2"],
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "model.pkl", compress=3)
    (MODELS_DIR / "metrics.json").write_text(
        json.dumps({"train": train_metrics, "test": test_metrics}, indent=2)
    )
    (MODELS_DIR / "model_meta.json").write_text(
        json.dumps(build_meta(train_raw, test_metrics), indent=2)
    )

    log.info("Saved models/model.pkl, models/metrics.json, models/model_meta.json")
    log.info("Stage 2 (model engineering) finished")


if __name__ == "__main__":
    main()
