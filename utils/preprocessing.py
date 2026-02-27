"""
Preprocessing utilities for user-item interaction data.

Converts raw rating DataFrames into the matrices consumed by
collaborative-filtering models.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_user_item_matrix(
    df: pd.DataFrame,
    n_users: Optional[int] = None,
    n_items: Optional[int] = None,
) -> pd.DataFrame:
    """Pivot a ratings DataFrame into a dense User × Item matrix.

    Args:
        df: DataFrame with columns ``[user_id, movie_id, rating]``.
        n_users: Expected number of users (inferred from data if *None*).
        n_items: Expected number of items (inferred from data if *None*).

    Returns:
        A ``pd.DataFrame`` of shape (n_users, n_items) where missing
        entries are filled with ``0``.

    Note:
        For production systems a ``scipy.sparse.csr_matrix`` would be
        preferable; a dense DataFrame is used here for readability and
        compatibility with ``sklearn.metrics.pairwise.cosine_similarity``.
    """
    if n_users is None:
        n_users = int(df["user_id"].max())
    if n_items is None:
        n_items = int(df["movie_id"].max())

    logger.info("Creating interaction matrix: %d users × %d items", n_users, n_items)

    interaction_matrix: pd.DataFrame = (
        df.pivot(index="user_id", columns="movie_id", values="rating")
        .fillna(0)
    )

    return interaction_matrix


def normalize_user_ratings(
    interaction_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Mean-center each user's ratings to handle strict/lenient raters.

    Replaces unrated entries (``0``) with ``NaN`` before computing the
    per-user mean so that missing values do not deflate the average.

    Args:
        interaction_matrix: Dense User × Item matrix (0 = unrated).

    Returns:
        A tuple of:
        - **centered_matrix** — the mean-centered interaction matrix
          (``NaN`` filled back to ``0``).
        - **user_means** — ``pd.Series`` mapping each user to their mean
          rating.
    """
    matrix_nan = interaction_matrix.replace(0, np.nan)
    user_means: pd.Series = matrix_nan.mean(axis=1)

    centered_matrix = matrix_nan.sub(user_means, axis=0).fillna(0)

    logger.info("Normalized ratings for %d users.", len(user_means))
    return centered_matrix, user_means


def calculate_sparsity(df: pd.DataFrame, n_users: int, n_items: int) -> float:
    """Return the sparsity percentage of the interaction matrix.

    Args:
        df: Ratings DataFrame.
        n_users: Total number of users.
        n_items: Total number of items.

    Returns:
        Sparsity as a percentage (0–100).
    """
    n_total = n_users * n_items
    if n_total == 0:
        return 100.0
    sparsity = (1 - len(df) / n_total) * 100
    return sparsity
