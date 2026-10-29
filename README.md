# StayWise — Hotel Booking Recommendation System

A research-oriented hotel recommendation project built with Python, Google Colab, pandas, scikit-learn, and Streamlit.

## Project goals

This project compares recommendation approaches and investigates how preprocessing choices affect recommendation quality. Conclusions will be based on measured experiments, not assumed results.

## Selected dataset

The initial dataset is [ViHoRec](https://github.com/MinhNguyenDS/ViHoRec), a public research dataset containing pseudonymous user–hotel ratings, dates, and a hotel catalogue. It covers hotels in Vietnam and the basic catalogue provides hotel names and locations. The data are marked CC BY-NC 4.0; review the license before reuse or broader deployment. See [data/README.md](data/README.md).

## Planned recommendation approaches

1. Popularity-based baseline
2. Content-based recommendation using hotel name/location text
3. User-based collaborative filtering from actual user–hotel ratings
4. Matrix factorization
5. Hybrid recommendation combining compatible content and collaborative scores

## Planned preprocessing study

- Missing-value handling
- Duplicate detection and removal
- Domain-aware outlier inspection
- Categorical feature encoding
- Text normalization and TF-IDF

## Evaluation plan

Ranking metrics will include NDCG@10, Precision@10, Recall@10, and Hit Rate@10 when supported by a valid held-out evaluation. Catalog coverage and inference latency may be reported as additional measures. RMSE/MAE will only be used for rating prediction tasks.

## Repository structure

```text
app.py
notebooks/                 Google Colab notebook(s)
src/                       Reusable preprocessing and evaluation code
data/                      Dataset instructions; raw data is not committed
artifacts/                 Model artifact instructions
results/                   Measured experiment outputs
reports/                   Methodology and references
tests/                     Unit tests
requirements.txt
```

## Getting started

1. Create a Python environment (Python 3.10 or newer is recommended).
2. Install dependencies: `pip install -r requirements.txt`.
3. Open [notebooks/hotel_recommendation_system.ipynb](notebooks/hotel_recommendation_system.ipynb) in Google Colab.
4. Run the dataset-loading and schema-audit cells.
5. Run the Streamlit starter locally with `streamlit run app.py`.

The app currently displays a setup message and a configured CSV preview. It does not yet provide trained recommendations and does not claim live availability or booking capability.

## Research integrity

- Cite the dataset and follow its license.
- Treat user IDs as pseudonymous keys, not real-world identities.
- Keep train/validation/test splits consistent across comparable experiments.
- Fit learned preprocessing steps on training data only.
- Do not fabricate users, interactions, or results.
- State the limitations of the dataset's limited hotel attributes and geographic coverage.

## Status

**Initial scaffold and dataset selection complete.** The next milestone is running the dataset audit and implementing reproducible model comparisons.
