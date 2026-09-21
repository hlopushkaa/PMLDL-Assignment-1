"""Stage 3b: Web application (Streamlit).

Builds its input form from the schema the API exposes at /meta, sends the
values to the API and displays the prediction. Runs in its own container.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")
TIMEOUT = 10

st.set_page_config(page_title="California Housing Price", layout="centered")


@st.cache_data(ttl=30)
def fetch_meta() -> dict | None:
    try:
        response = requests.get(f"{API_URL}/meta", timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


st.markdown(
    "<h1 style='white-space:nowrap; font-size:clamp(1.1rem,3.6vw,2.4rem);"
    " margin:0 0 1.5rem'>California Housing Price Prediction</h1>",
    unsafe_allow_html=True,
)

meta = fetch_meta()

if meta is None:
    st.error(
        f"Model API is not reachable at {API_URL}. "
        "Make sure both containers are up (`docker compose up -d --build`)."
    )
    st.stop()

metrics = meta.get("metrics", {})
if metrics:
    col_rmse, col_mae, col_r2 = st.columns(3)
    col_rmse.metric("Test RMSE", f"${metrics['rmse']:,.0f}")
    col_mae.metric("Test MAE", f"${metrics['mae']:,.0f}")
    col_r2.metric("Test R²", f"{metrics['r2']:.3f}")

st.subheader("District parameters")

values: dict[str, float | str] = {}
left, right = st.columns(2)

for index, field in enumerate(meta["fields"]):
    container = left if index % 2 == 0 else right
    if field["type"] == "category":
        options = field["options"]
        default_index = options.index(field["default"]) if field["default"] in options else 0
        values[field["name"]] = container.selectbox(
            field["label"], options, index=default_index
        )
    else:
        values[field["name"]] = container.number_input(
            field["label"],
            min_value=float(field["min"]),
            max_value=float(field["max"]),
            value=float(field["default"]),
            step=(float(field["max"]) - float(field["min"])) / 100 or 0.01,
            format="%.4f",
        )

if st.button("Predict", type="primary", use_container_width=True):
    try:
        response = requests.post(f"{API_URL}/predict", json=values, timeout=TIMEOUT)
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        st.error(f"Request to the model API failed: {exc}")
    else:
        st.success(f"Predicted median house value: **${result['prediction']:,.0f}**")
        with st.expander("Request sent to the API"):
            st.json(values)
