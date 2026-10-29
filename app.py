"""Streamlit starter UI for StayWise.

This starter intentionally avoids pretending that a trained recommender or live
hotel inventory exists before data and model artifacts have been configured.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="StayWise | Hotel Recommendations", page_icon="🏨", layout="wide")

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "processed" / "hotels.csv"

st.title("🏨 StayWise")
st.subheader("Hotel Booking Recommendation System")
st.write(
    "Explore hotel data and, once a recommendation model is trained, discover "
    "properties that match your preferences."
)

with st.sidebar:
    st.header("Preferences")
    destination = st.text_input("Destination or city")
    min_rating = st.slider("Minimum hotel rating", 0.0, 10.0, 7.0, 0.5)
    top_k = st.selectbox("Number of recommendations", [5, 10, 15, 20], index=1)
    st.caption("Filters will become active when the selected dataset's schema is confirmed.")

st.info(
    "Project setup stage: no trained recommendation model is bundled yet. "
    "The dataset and evaluation protocol will be finalized before recommendations are shown."
)

if DEFAULT_DATA.exists():
    try:
        hotels = pd.read_csv(DEFAULT_DATA)
        st.success(f"Loaded {len(hotels):,} rows from {DEFAULT_DATA.relative_to(ROOT)}.")
        st.write("### Dataset preview")
        st.dataframe(hotels.head(20), use_container_width=True)
        st.write("### Dataset overview")
        left, middle, right = st.columns(3)
        left.metric("Rows", f"{len(hotels):,}")
        middle.metric("Columns", f"{len(hotels.columns):,}")
        right.metric("Missing cells", f"{int(hotels.isna().sum().sum()):,}")
        st.caption(
            f"Requested destination: {destination or 'not specified'} · "
            f"Minimum rating preference: {min_rating:.1f} · Top K: {top_k}"
        )
    except Exception as exc:
        st.error(f"Could not read the configured CSV: {exc}")
else:
    st.warning(
        "Dataset not configured yet. After the dataset is selected and cleaned, "
        "place the prepared hotel catalogue at data/processed/hotels.csv."
    )

st.divider()
st.markdown("**Planned modules:** dataset exploration · preprocessing comparison · five recommender approaches · ranking evaluation · model comparison.")
