"""
Collaborative Filtering models for the Netflix Recommendation System.

Provides two concrete implementations on top of a common base class:

* **UserBasedCF** — finds users with similar taste, then aggregates their
  ratings to predict what the active user would enjoy.
* **ItemBasedCF** — the approach Netflix made famous: builds an item-item
  similarity matrix and scores unseen items based on how similar they are
  to items the user already liked.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from config import DEFAULT_K_NEIGHBORS_ITEM, DEFAULT_K_NEIGHBORS_USER, DEFAULT_N_RECOMMENDATIONS

logger = logging.getLogger(__name__)


class BaseRecommender(ABC):
    """Abstract base class for recommendation models."""

    def __init__(self) -> None:
        self.interaction_matrix: Optional[pd.DataFrame] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        self.user_means: Optional[pd.Series] = None

    @abstractmethod
    def fit(self, interaction_matrix: pd.DataFrame, **kwargs) -> None:  # noqa: ANN003
        """Train the model on an interaction matrix."""

    @abstractmethod
    def predict(
        self,
        user_id: int,
        n_recommendations: int = DEFAULT_N_RECOMMENDATIONS,
        k_neighbors: int = 20,
    ) -> pd.Series:
        """Return predicted scores for the given user."""


class UserBasedCF(BaseRecommender):
    """User-Based Collaborative Filtering.

    Algorithm
    ---------
    1. Compute User–User cosine similarity on the mean-centered matrix.
    2. For a target user, find the *k* most similar neighbours.
    3. Predict each unseen item's rating as a weighted average of the
       neighbours' (normalized) ratings, then add the user's own mean
       back to recover the original scale.

    When to use
    -----------
    Good for *serendipity*; struggles at Netflix scale because the
    user–user similarity matrix grows as O(M²) with the number of
    users.
    """

    def fit(
        self,
        interaction_matrix: pd.DataFrame,
        user_means: Optional[pd.Series] = None,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Compute the User–User similarity matrix.

        Args:
            interaction_matrix: Mean-centered User × Item matrix.
            user_means: Per-user mean ratings (needed to un-center
                predictions).
        """
        logger.info("Training User-Based CF model …")
        self.interaction_matrix = interaction_matrix
        self.user_means = user_means

        # Cosine similarity between rows (users) → shape (n_users, n_users)
        self.similarity_matrix = cosine_similarity(self.interaction_matrix)

        # Zero the diagonal to prevent self-recommendation bias.
        np.fill_diagonal(self.similarity_matrix, 0)

        logger.info(
            "User similarity matrix shape: %s", self.similarity_matrix.shape
        )

    def predict(
        self,
        user_id: int,
        n_recommendations: int = DEFAULT_N_RECOMMENDATIONS,
        k_neighbors: int = DEFAULT_K_NEIGHBORS_USER,
    ) -> pd.Series:
        """Predict top-N items for *user_id*.

        Args:
            user_id: Must exist in the interaction matrix index.
            n_recommendations: Number of items to return.
            k_neighbors: Number of similar users to aggregate.

        Returns:
            A ``pd.Series`` of predicted ratings indexed by movie_id,
            sorted descending.  Returns an *empty* Series for unknown
            users (cold-start).
        """
        if user_id not in self.interaction_matrix.index:
            logger.warning("User %s not found — cold-start.", user_id)
            return pd.Series(dtype=np.float64)

        user_idx: int = self.interaction_matrix.index.get_loc(user_id)
        user_sims: np.ndarray = self.similarity_matrix[user_idx]

        # Top-k neighbours by descending similarity
        top_k_indices = user_sims.argsort()[-k_neighbors:][::-1]

        n_items = self.interaction_matrix.shape[1]
        weighted_sum = np.zeros(n_items)
        sim_sum = np.zeros(n_items)

        for neighbour_idx in top_k_indices:
            sim_score = user_sims[neighbour_idx]
            if sim_score <= 0:
                continue
            neighbour_ratings = self.interaction_matrix.iloc[neighbour_idx].values
            weighted_sum += sim_score * neighbour_ratings
            sim_sum += sim_score

        # Weighted average, guarding against division by zero
        pred_ratings = np.divide(
            weighted_sum, sim_sum,
            out=np.zeros_like(weighted_sum),
            where=sim_sum != 0,
        )

        # Un-center: add the user's mean back
        if self.user_means is not None:
            pred_ratings += self.user_means.loc[user_id]

        pred_series = pd.Series(
            pred_ratings, index=self.interaction_matrix.columns
        )
        return pred_series.sort_values(ascending=False).head(n_recommendations)


class ItemBasedCF(BaseRecommender):
    """Item-Based Collaborative Filtering (Netflix style).

    Algorithm
    ---------
    1. Transpose the User × Item matrix to Item × User.
    2. Compute cosine similarity between items (rows of the transposed
       matrix).
    3. For a target user, score every unseen item as the dot product of
       the user's rating vector and the item similarity row.

    Why Netflix prefers this
    ------------------------
    * **Stability** — item profiles change slowly; user tastes drift.
    * **Scalability** — N items ≪ M users → the N×N matrix is tractable.
    * **Pre-computation** — similarity can be computed offline and cached.
    """

    def __init__(self) -> None:
        super().__init__()
        self.similarity_df: Optional[pd.DataFrame] = None

    def fit(
        self,
        interaction_matrix: pd.DataFrame,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Compute the Item–Item similarity matrix.

        Args:
            interaction_matrix: User × Item matrix (may be mean-centered).
        """
        logger.info("Training Item-Based CF model …")
        self.interaction_matrix = interaction_matrix

        item_user_matrix = self.interaction_matrix.T
        self.similarity_matrix = cosine_similarity(item_user_matrix)

        np.fill_diagonal(self.similarity_matrix, 0)

        self.similarity_df = pd.DataFrame(
            self.similarity_matrix,
            index=self.interaction_matrix.columns,
            columns=self.interaction_matrix.columns,
        )
        logger.info(
            "Item similarity matrix shape: %s", self.similarity_matrix.shape
        )

    def predict(
        self,
        user_id: int,
        n_recommendations: int = DEFAULT_N_RECOMMENDATIONS,
        k_neighbors: int = DEFAULT_K_NEIGHBORS_ITEM,
    ) -> pd.Series:
        """Predict top-N items for *user_id*.

        Scores are computed as: ``user_ratings · similarity_matrix``.
        Items already seen by the user are excluded.

        Args:
            user_id: Must exist in the interaction matrix index.
            n_recommendations: Number of items to return.
            k_neighbors: (Unused — kept for API parity with
                ``UserBasedCF``.)

        Returns:
            A ``pd.Series`` of predicted scores indexed by movie_id,
            sorted descending.  Returns an *empty* Series for unknown
            users (cold-start).
        """
        if user_id not in self.interaction_matrix.index:
            logger.warning("User %s not found — cold-start.", user_id)
            return pd.Series(dtype=np.float64)

        user_ratings: pd.Series = self.interaction_matrix.loc[user_id]

        # (1 × Items) · (Items × Items) → (1 × Items) predicted scores
        scores: pd.Series = user_ratings.dot(self.similarity_df)

        scores = scores.sort_values(ascending=False)

        # Remove items the user has already interacted with
        seen_items = user_ratings[user_ratings != 0].index
        scores = scores.drop(seen_items, errors="ignore")

        return scores.head(n_recommendations)
