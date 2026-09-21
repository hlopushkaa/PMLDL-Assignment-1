"""Feature engineering shared by the training stage and the model API.

Keeping this in one module guarantees that the features built at training time
and the features built at serving time are identical.
"""

from __future__ import annotations

import pandas as pd

TARGET = "median_house_value"

RAW_NUMERIC = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
RAW_CATEGORICAL = ["ocean_proximity"]

DERIVED_NUMERIC = [
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
]

NUMERIC_FEATURES = RAW_NUMERIC + DERIVED_NUMERIC
CATEGORICAL_FEATURES = RAW_CATEGORICAL
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw columns into the feature matrix the model expects.

    Ratios carry much more signal than the raw block-level counts, because a
    district with 5000 rooms is only informative relative to how many
    households live there.
    """
    out = df.copy()

    households = out["households"].replace(0, pd.NA)
    total_rooms = out["total_rooms"].replace(0, pd.NA)

    out["rooms_per_household"] = out["total_rooms"] / households
    out["bedrooms_per_room"] = out["total_bedrooms"] / total_rooms
    out["population_per_household"] = out["population"] / households

    out[DERIVED_NUMERIC] = out[DERIVED_NUMERIC].astype("float64")
    out[DERIVED_NUMERIC] = out[DERIVED_NUMERIC].fillna(0.0)

    return out[FEATURE_COLUMNS]
