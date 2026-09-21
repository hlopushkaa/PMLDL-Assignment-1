# PMLDL Assignment 1 — Deployment

An automated MLOps pipeline that cleans data, trains a model and redeploys it
behind an API with a web front-end. Orchestrated by **Apache Airflow**, it runs
end to end **every 5 minutes**.

| Stage | What it does | Code | Artifacts |
|---|---|---|---|
| 1. Data engineering | load → clean (missing values, outliers) → split | `code/datasets/prepare_data.py` | `data/processed/train.csv`, `data/processed/test.csv` |
| 2. Model engineering | feature engineering → train → evaluate → log to MLflow → package | `code/models/train_model.py` | `models/model.pkl`, `models/metrics.json`, MLflow run |
| 3. Deployment | build & start the API and app images | `code/deployment/` | two running Docker containers |

**Task.** Predict the median house value of a California census district
(regression).
**Dataset.** [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
(StatLib, 1990 census) — 20 640 districts, 9 features, already in
`data/raw/housing.csv`. It is neither CelebFaces nor the smoking-status data
used in the labs.
**Model.** `RandomForestRegressor` inside a scikit-learn `Pipeline` —
**test R² ≈ 0.79, RMSE ≈ $45 000, MAE ≈ $31 000**.

---

## Repository structure

```
.
├── code
│   ├── datasets
│   │   └── prepare_data.py        # stage 1
│   ├── models
│   │   ├── features.py            # feature engineering (shared by training and API)
│   │   └── train_model.py         # stage 2
│   └── deployment
│       ├── api
│       │   ├── main.py            # FastAPI model API
│       │   ├── requirements.txt
│       │   └── Dockerfile
│       ├── app
│       │   ├── app.py             # Streamlit web application
│       │   ├── requirements.txt
│       │   └── Dockerfile
│       └── docker-compose.yml     # stage 3: two separate containers
├── data
│   ├── raw/housing.csv            # raw data
│   └── processed/                 # train.csv, test.csv (generated)
├── models/                        # model.pkl, model_meta.json, metrics.json (generated)
├── notebooks
│   └── eda.ipynb                  # why the cleaning/feature steps look the way they do
├── services
│   └── airflow
│       ├── dags/mlops_pipeline.py # the scheduled pipeline
│       └── logs/
├── run_pipeline.sh                # run all three stages once, without Airflow
└── requirements.txt
```

---

## Quick start

### 0. Prerequisites

* Python 3.11
* Docker with the Compose plugin (`docker compose version`)
* Linux or macOS — Airflow does not run on Windows natively, use WSL

### 1. Environment

```bash
git clone <this-repository>
cd pmldl-assignment1

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the pipeline once, by hand

Do this before scheduling anything — it proves each stage works and leaves a
trained model on disk.

```bash
./run_pipeline.sh
```

or stage by stage:

```bash
python code/datasets/prepare_data.py                 # stage 1
python code/models/train_model.py                    # stage 2

# stage 3 -- the image must run the same Python the model was pickled with
export PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
cd code/deployment && docker compose up -d --build
```

Then open:

* **Web application** — <http://localhost:8501>
* **API docs (Swagger)** — <http://localhost:8000/docs>

### 3. Schedule it every 5 minutes with Airflow

```bash
# from the repository root, with the virtualenv active
export AIRFLOW_HOME=$PWD/services/airflow
export AIRFLOW__CORE__DAGS_FOLDER=$AIRFLOW_HOME/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=False

airflow db migrate          # `airflow db init` on Airflow 2.x
airflow standalone
```

`airflow standalone` starts the scheduler, the API server and the UI on
<http://localhost:8080> (it prints the generated admin password on first run).
Unpause the **`mlops_pipeline`** DAG in the UI — from then on it runs on
`*/5 * * * *`.

Prefer the components separately (as in Lab 3):

```bash
airflow scheduler --daemon --log-file services/airflow/logs/scheduler.log
airflow api-server --daemon                 # `airflow webserver` on Airflow 2.x
```

Test a single run without waiting for the schedule:

```bash
airflow dags test mlops_pipeline
```

> `max_active_runs=1` and `catchup=False` are set on the DAG, so a slow run can
> never overlap with the next one and Airflow will not backfill history.

---

## How the stages work

### Stage 1 — data engineering

`code/datasets/prepare_data.py`

1. **Load** `data/raw/housing.csv`.
2. **Clean:**
   * drop duplicate rows and rows with no target;
   * impute missing values — median for numeric columns (207 gaps in
     `total_bedrooms`), mode for the categorical one;
   * drop outliers — the 992 rows where `median_house_value` sits at the
     500 001 cap (a recording artifact, not a real price), then an IQR rule
     (factor 3) on the heavy-tailed count columns.
3. **Split** 80/20 into `data/processed/train.csv` and `test.csv`.

20 640 raw rows → 18 915 clean rows (15 132 train / 3 783 test). A summary of
what was dropped is written to `data/processed/data_report.json`.

### Stage 2 — model engineering

`code/models/train_model.py`

1. **Feature engineering** (`code/models/features.py`): the raw block totals
   are turned into ratios — `rooms_per_household`, `bedrooms_per_room`,
   `population_per_household` — which correlate with the target far better than
   the counts themselves.
2. **Training**: one `Pipeline` holding a `ColumnTransformer`
   (one-hot for `ocean_proximity`, `handle_unknown="ignore"`) and a
   `RandomForestRegressor`. Preprocessing travels with the estimator, so the API
   only ever passes raw-shaped rows.
3. **Evaluation** on the test split: RMSE, MAE, R².
4. **Logging**: parameters, metrics and the model go to MLflow
   (SQLite backend `mlflow.db`, artifacts in `mlruns/`).
5. **Packaging**: `joblib.dump(..., compress=3)` → `models/model.pkl` (~10 MB),
   plus `models/metrics.json` and `models/model_meta.json` (the input schema the
   web app builds its form from).

Browse the experiment history:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://localhost:5000
```

To use a tracking server instead of the local backend, set
`MLFLOW_TRACKING_URI` before running the stage.

### Stage 3 — deployment

`code/deployment/docker-compose.yml` builds and starts **two separate
containers**:

* **`pmldl-api`** — FastAPI + Uvicorn on port **8000**. The build context is the
  repository root, so each build copies the freshly trained `models/model.pkl`
  into the image; the model layer is copied last, so only that layer is rebuilt.
  The base image follows the `PYTHON_VERSION` build argument, which the pipeline
  sets from the interpreter that trained the model — a pickle written by one
  Python version is not guaranteed to load under another.
  * `GET /health` — liveness
  * `GET /meta` — input schema + test metrics of the deployed model
  * `POST /predict` — prediction for one district
* **`pmldl-app`** — Streamlit on port **8501**. It reads `/meta`, builds the
  input fields from it, and on **Predict** sends the values to the API and shows
  the returned price. It waits for the API's healthcheck before starting.

The deployment task runs `docker compose up -d --build`, so every pipeline run
redeploys the model that was just trained.

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"longitude":-118.49,"latitude":34.26,"housing_median_age":29,
       "total_rooms":2127,"total_bedrooms":435,"population":1166,
       "households":409,"median_income":3.53,"ocean_proximity":"<1H OCEAN"}'
# {"prediction":213751.8,"target":"median_house_value","unit":"USD"}
```

---

## Useful commands

```bash
docker compose -f code/deployment/docker-compose.yml ps        # container status
docker compose -f code/deployment/docker-compose.yml logs -f   # follow logs
docker compose -f code/deployment/docker-compose.yml down      # stop everything
```

## Notes

* `data/processed/`, `models/*.pkl`, `mlruns/` and `mlflow.db` are generated by
  the pipeline and git-ignored — clone the repository and run
  `./run_pipeline.sh` to recreate them.
* Ports 8000, 8501 and 8080 must be free.
* Stage 1 and stage 2 together take about 15 seconds, so the 5-minute interval
  leaves plenty of room; the deployment stage is fast after the first build
  thanks to Docker layer caching.
