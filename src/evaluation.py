"""Ranking metrics for recommendation lists.

The functions expect one list of recommended item IDs and one set/list of relevant
held-out item IDs per user. They do not train models or create relevance labels.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def precision_at_k(recommended: Sequence, relevant: Iterable, k: int = 10) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = set(relevant)
    top_k = list(recommended[:k])
    if not top_k:
        return 0.0
    return sum(item in relevant_set for item in top_k) / len(top_k)


def recall_at_k(recommended: Sequence, relevant: Iterable, k: int = 10) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = list(recommended[:k])
    return sum(item in relevant_set for item in top_k) / len(relevant_set)


def hit_rate_at_k(recommended: Sequence, relevant: Iterable, k: int = 10) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = set(relevant)
    return float(any(item in relevant_set for item in list(recommended[:k])))


def ndcg_at_k(recommended: Sequence, relevant: Iterable, k: int = 10) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = set(relevant)
    top_k = list(recommended[:k])
    dcg = sum(1.0 / math.log2(rank + 2) for rank, item in enumerate(top_k) if item in relevant_set)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg else 0.0
