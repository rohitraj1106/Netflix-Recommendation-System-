"""Unit tests for evaluation.metrics module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from evaluation.metrics import (
    calculate_rmse,
    f1_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------

class TestCalculateRMSE:
    def test_perfect_predictions(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert calculate_rmse(y, y) == pytest.approx(0.0)

    def test_known_error(self) -> None:
        y_true = np.array([3.0, 3.0])
        y_pred = np.array([1.0, 5.0])
        assert calculate_rmse(y_true, y_pred) == pytest.approx(2.0)

    def test_handles_nan_predictions(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, np.nan, 3.0])
        assert calculate_rmse(y_true, y_pred) == pytest.approx(0.0)

    def test_all_nan_returns_nan(self) -> None:
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([np.nan, np.nan])
        assert math.isnan(calculate_rmse(y_true, y_pred))


# ---------------------------------------------------------------------------
# Precision@K
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_all_relevant(self) -> None:
        recommended = [1, 2, 3]
        relevant = [1, 2, 3]
        assert precision_at_k(recommended, relevant, k=3) == pytest.approx(1.0)

    def test_none_relevant(self) -> None:
        recommended = [4, 5, 6]
        relevant = [1, 2, 3]
        assert precision_at_k(recommended, relevant, k=3) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        recommended = [1, 4, 5, 2, 6]
        relevant = [1, 2, 3]
        assert precision_at_k(recommended, relevant, k=5) == pytest.approx(2 / 5)

    def test_empty_recommendations(self) -> None:
        assert precision_at_k([], [1, 2], k=5) == 0.0

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError):
            precision_at_k([1], [1], k=0)


# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_all_captured(self) -> None:
        recommended = [1, 2, 3]
        relevant = [1, 2, 3]
        assert recall_at_k(recommended, relevant, k=3) == pytest.approx(1.0)

    def test_partial(self) -> None:
        recommended = [1, 4, 5]
        relevant = [1, 2, 3]
        assert recall_at_k(recommended, relevant, k=3) == pytest.approx(1 / 3)

    def test_no_relevant_items(self) -> None:
        assert recall_at_k([1, 2], [], k=2) == 0.0


# ---------------------------------------------------------------------------
# F1@K
# ---------------------------------------------------------------------------

class TestF1AtK:
    def test_perfect(self) -> None:
        recs = [1, 2, 3]
        rels = [1, 2, 3]
        assert f1_at_k(recs, rels, k=3) == pytest.approx(1.0)

    def test_zero(self) -> None:
        assert f1_at_k([4, 5], [1, 2], k=2) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# NDCG@K
# ---------------------------------------------------------------------------

class TestNDCGAtK:
    def test_perfect_ranking(self) -> None:
        """All relevant items at the top positions → NDCG = 1."""
        recs = [1, 2, 3, 99, 98]
        rels = [1, 2, 3]
        assert ndcg_at_k(recs, rels, k=5) == pytest.approx(1.0)

    def test_worst_ranking(self) -> None:
        """No relevant items in top-K → NDCG = 0."""
        recs = [10, 11, 12]
        rels = [1, 2, 3]
        assert ndcg_at_k(recs, rels, k=3) == pytest.approx(0.0)

    def test_no_relevant_items(self) -> None:
        assert ndcg_at_k([1, 2], [], k=2) == pytest.approx(0.0)
