"""Stage 1: Data engineering.

Loads the raw dataset, cleans it (missing values + outliers) and splits it
into train/test files.

Input  : data/raw/housing.csv
Output : data/processed/train.csv, data/processed/test.csv,
         data/processed/data_report.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "housing.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET = "median_house_value"
CATEGORICAL = ["ocean_proximity"]
# The target in this dataset is capped at 500001 -- those rows are artifacts,
# not real observations, so they are treated as outliers.
TARGET_CAP = 500_000
IQR_COLUMNS = ["total_rooms", "total_bedrooms", "population", "households"]
IQR_FACTOR = 3.0  # conservative: drops only extreme tails
TEST_SIZE = 0.2
RANDOM_SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("data-engineering")


def load_data(path: Path) -> pd.DataFrame:
    """Read the raw data file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. See README for how to obtain it."
        )
    df = pd.read_csv(path)
    log.info("Loaded raw data: %d rows, %d columns", *df.shape)
    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove duplicates, impute missing values and drop outliers."""
    report: dict = {"rows_raw": int(len(df))}

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = before - len(df)

    before = len(df)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    report["rows_without_target_removed"] = before - len(df)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    missing_filled: dict[str, int] = {}
    for col in numeric_cols:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            df[col] = df[col].fillna(df[col].median())
            missing_filled[col] = n_missing
    for col in CATEGORICAL:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            df[col] = df[col].fillna(df[col].mode()[0])
            missing_filled[col] = n_missing
    report["missing_values_imputed"] = missing_filled

    before = len(df)
    df = df[df[TARGET] < TARGET_CAP].reset_index(drop=True)
    report["capped_target_rows_removed"] = before - len(df)

    iqr_removed: dict[str, int] = {}
    for col in IQR_COLUMNS:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - IQR_FACTOR * iqr, q3 + IQR_FACTOR * iqr
        mask = df[col].between(low, high)
        iqr_removed[col] = int((~mask).sum())
        df = df[mask].reset_index(drop=True)
    report["iqr_outliers_removed"] = iqr_removed

    report["rows_clean"] = int(len(df))
    log.info(
        "Cleaned data: %d -> %d rows (%s)",
        report["rows_raw"],
        report["rows_clean"],
        json.dumps({k: v for k, v in report.items() if k.endswith("removed")}),
    )
    return df, report


def split_and_save(df: pd.DataFrame, report: dict) -> None:
    """Split into train/test and write the processed files."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=True
    )
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    report["rows_train"] = int(len(train_df))
    report["rows_test"] = int(len(test_df))
    (PROCESSED_DIR / "data_report.json").write_text(json.dumps(report, indent=2))

    log.info("Saved %s (%d rows)", train_path.relative_to(PROJECT_ROOT), len(train_df))
    log.info("Saved %s (%d rows)", test_path.relative_to(PROJECT_ROOT), len(test_df))


def main() -> None:
    df = load_data(RAW_PATH)
    df, report = clean_data(df)
    split_and_save(df, report)
    log.info("Stage 1 (data engineering) finished")


if __name__ == "__main__":
    main()
