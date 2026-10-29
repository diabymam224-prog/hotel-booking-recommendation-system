# StayWise — Hotel Booking Recommendation System

A research-oriented hotel recommendation project built with Python, Google Colab, pandas, scikit-learn, and Streamlit.

## Project goals

This project will compare recommendation approaches and investigate how data preprocessing affects recommendation quality. Final conclusions will be based on measured experiments, not assumed results.

### Planned recommendation approaches
1. Popularity-based baseline
2. Content-based recommendation using hotel features and text
3. User-based collaborative filtering (only if the selected dataset has genuine user identifiers and interactions)
4. Matrix factorization (only if the interaction data supports it)
5. Hybrid recommendation combining compatible content and collaborative scores

### Planned preprocessing study
- Missing-value handling
- Duplicate detection and removal
- Outlier inspection with domain-aware rules
- Categorical feature encoding
- Review-text normalization and TF-IDF features

## Evaluation plan

Ranking metrics will include NDCG@10, Precision@10, Recall@10, and Hit Rate@10 where the data supports a valid held-out user-item evaluation. Catalog coverage and inference latency may be reported as additional measures. RMSE/MAE will only be used for rating prediction tasks.

## Repository structure

```text
app.py
notebooks/                 Google Colab notebook(s)
src/                       Reusable preprocessing and evaluation code
data/                      Dataset instructions; raw data is not committed
artifacts/                 Instructions for model artifacts; large files are not committed
results/                   Measured experiment outputs
reports/                   Methodology and final research report
tests/                     Unit tests
requirements.txt
```

## Getting started

1. Create a Python environment (Python 3.10 or newer is recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Read [data/README.md](data/README.md) and choose/download the dataset.
4. Keep raw data local or in Google Drive; do not commit large datasets or private data.
5. Start the Streamlit interface after placing a compatible CSV in the configured data location:

   ```bash
   streamlit run app.py
   ```

The starter app will explain when data is not yet configured. It does not claim live room availability or perform bookings.

## Research integrity

- Dataset fields and licensing will be verified before model selection.
- User-item interactions will never be fabricated to enable collaborative filtering.
- Train/validation/test splits will be fixed and shared across comparable experiments.
- Learned preprocessing steps will be fitted on training data only to reduce leakage.
- Results tables must contain actual measured outputs; no scores are pre-filled.
- Comparisons with published papers will discuss dataset and protocol differences.

## Status

**Initial scaffold created.** Dataset selection, schema validation, experiments, and deployment are still in progress.
