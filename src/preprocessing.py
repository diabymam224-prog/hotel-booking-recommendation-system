"""Small, reusable preprocessing helpers for hotel recommendation data.

Keep dataset-specific column mappings explicit. These helpers intentionally avoid
fitting learned transformations globally; fit vectorizers/encoders on training
data only in the model pipeline to prevent leakage.
"""
from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with trimmed, lowercase, underscore-separated column names."""
    result = df.copy()
    result.columns = [
        re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip().lower())).strip("_")
        for col in result.columns
    ]
    return result


def summarize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing count and percentage for each column."""
    count = df.isna().sum()
    summary = pd.DataFrame(
        {
            "missing_count": count,
            "missing_percent": (count / max(len(df), 1) * 100).round(2),
        }
    )
    return summary.sort_values("missing_percent", ascending=False)


def remove_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicate rows and return (cleaned_copy, removed_count)."""
    before = len(df)
    cleaned = df.drop_duplicates().copy()
    return cleaned, before - len(cleaned)


def normalize_review_text(text: object) -> str:
    """Conservative text normalization that handles null/non-string values."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def combine_text_fields(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    """Combine existing text columns; absent columns are ignored."""
    available = [column for column in columns if column in df.columns]
    if not available:
        return pd.Series("", index=df.index, dtype="object")
    return (
        df[available]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .map(normalize_review_text)
    )
