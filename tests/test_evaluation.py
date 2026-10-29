import pytest

from src.evaluation import hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k


def test_ranking_metrics_perfect_hit():
    recommended = ["hotel_a", "hotel_b", "hotel_c"]
    relevant = {"hotel_a", "hotel_c"}
    assert precision_at_k(recommended, relevant, 3) == pytest.approx(2 / 3)
    assert recall_at_k(recommended, relevant, 3) == pytest.approx(1.0)
    assert hit_rate_at_k(recommended, relevant, 3) == 1.0
    assert ndcg_at_k(["hotel_a", "hotel_c"], relevant, 2) == pytest.approx(1.0)


def test_empty_relevant_set():
    assert recall_at_k(["hotel_a"], set(), 10) == 0.0
    assert ndcg_at_k(["hotel_a"], set(), 10) == 0.0


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        precision_at_k([], set(), 0)
