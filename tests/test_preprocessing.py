import pandas as pd

from src.preprocessing import (
    combine_text_fields,
    normalize_column_names,
    normalize_review_text,
    remove_exact_duplicates,
    summarize_missing_values,
)


def test_normalize_column_names():
    df = pd.DataFrame([[1, 2]], columns=[" Hotel Name ", "Review-Score"])
    result = normalize_column_names(df)
    assert list(result.columns) == ["hotel_name", "review_score"]


def test_remove_exact_duplicates():
    df = pd.DataFrame({"hotel": ["A", "A", "B"]})
    result, removed = remove_exact_duplicates(df)
    assert len(result) == 2
    assert removed == 1


def test_missing_summary():
    df = pd.DataFrame({"x": [1, None], "y": ["a", "b"]})
    result = summarize_missing_values(df)
    assert result.loc["x", "missing_count"] == 1
    assert result.loc["x", "missing_percent"] == 50.0


def test_normalize_review_text_handles_nulls():
    assert normalize_review_text(None) == ""
    assert normalize_review_text("<b>Great</b> HOTEL! https://example.com") == "great hotel"


def test_combine_text_fields_ignores_missing_columns():
    df = pd.DataFrame({"positive": ["Lovely room"], "negative": [None]})
    result = combine_text_fields(df, ["positive", "negative", "not_a_column"])
    assert result.iloc[0] == "lovely room"
