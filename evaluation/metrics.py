"""
Evaluation metrics for recommendation systems.

Includes both rating-prediction metrics (RMSE) and ranking metrics
(Precision@K, Recall@K, F1@K, NDCG@K).
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rating-prediction metrics
# ---------------------------------------------------------------------------

def calculate_rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute Root Mean Squared Error, ignoring NaN predictions.

    Args:
        y_true: Array of actual ratings.
        y_pred: Array of predicted ratings (may contain NaN).

    Returns:
        RMSE value, or ``np.nan`` if no valid predictions exist.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        logger.warning("No valid predictions to compute RMSE.")
        return np.nan

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def precision_at_k(
    recommended_items: Sequence,
    relevant_items: Sequence,
    k: int = 10,
) -> float:
    """Precision@K — fraction of top-K recommendations that are relevant.

    Args:
        recommended_items: Ordered list of recommended item IDs.
        relevant_items: Set/list of ground-truth relevant item IDs.
        k: Cut-off position.

    Returns:
        Precision value in [0, 1].
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if len(recommended_items) == 0:
        return 0.0

    recommended_k = list(recommended_items)[:k]
    num_relevant_in_k = len(set(recommended_k) & set(relevant_items))
    return num_relevant_in_k / k


def recall_at_k(
    recommended_items: Sequence,
    relevant_items: Sequence,
    k: int = 10,
) -> float:
    """Recall@K — fraction of relevant items captured in top-K.

    Args:
        recommended_items: Ordered list of recommended item IDs.
        relevant_items: Set/list of ground-truth relevant item IDs.
        k: Cut-off position.

    Returns:
        Recall value in [0, 1].
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if len(relevant_items) == 0:
        return 0.0  # undefined, but returning 0 is the common convention

    recommended_k = list(recommended_items)[:k]
    num_relevant_in_k = len(set(recommended_k) & set(relevant_items))
    return num_relevant_in_k / len(relevant_items)


def f1_at_k(
    recommended_items: Sequence,
    relevant_items: Sequence,
    k: int = 10,
) -> float:
    """F1@K — harmonic mean of Precision@K and Recall@K.

    Args:
        recommended_items: Ordered list of recommended item IDs.
        relevant_items: Set/list of ground-truth relevant item IDs.
        k: Cut-off position.

    Returns:
        F1 value in [0, 1].
    """
    p = precision_at_k(recommended_items, relevant_items, k)
    r = recall_at_k(recommended_items, relevant_items, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def ndcg_at_k(
    recommended_items: Sequence,
    relevant_items: Sequence,
    k: int = 10,
) -> float:
    """Normalized Discounted Cumulative Gain @ K.

    Measures ranking quality — items ranked higher contribute more.

    Args:
        recommended_items: Ordered list of recommended item IDs.
        relevant_items: Set/list of ground-truth relevant item IDs.
        k: Cut-off position.

    Returns:
        NDCG value in [0, 1].
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if len(relevant_items) == 0:
        return 0.0

    relevant_set = set(relevant_items)
    recommended_k = list(recommended_items)[:k]

    # DCG: binary relevance — 1 if item is relevant, else 0
    dcg = sum(
        1.0 / np.log2(i + 2)  # i+2 because positions are 1-indexed
        for i, item in enumerate(recommended_k)
        if item in relevant_set
    )

    # Ideal DCG: all relevant items ranked at the top
    ideal_length = min(len(relevant_items), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_length))

    if idcg == 0:
        return 0.0
    return dcg / idcg
