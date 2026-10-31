# StayWise — Hotel Booking Recommendation System

A research-oriented hotel recommendation project using the ViHoRec dataset, Python, scikit-learn, and Streamlit.

## Streamlit app
The application is launched from the repository root with `streamlit run app.py`.

### Required runtime files
Place the exported Colab outputs in these paths:
- `data/hotels_catalog.csv`
- `artifacts/metadata.joblib`
- `artifacts/collaborative_arrays.joblib`
- `artifacts/hotel_content_matrix.joblib`
- `artifacts/tfidf_vectorizer.joblib` (retained for reproducibility)
- `results/model_comparison.csv`
- `results/preprocessing_comparison.csv`
- `results/rating_ge_7_sensitivity.csv`
- `results/experiment_config.json`

Install dependencies with `pip install -r requirements.txt`, then run `streamlit run app.py`.

## Research notes
The app presents recommendations from historical interactions only; it does not provide live availability, room inventory, or prices. User identifiers are pseudonymous.

ViHoRec is licensed CC BY-NC 4.0. Keep use of the dataset and derived materials within the licence's non-commercial research permissions and retain attribution. The current preprocessing ablation compares two interaction-data cleaning variants; it does not isolate the effect of every preprocessing technique.
