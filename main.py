"""
CLI entry-point for the Netflix Recommendation System.

Runs the full pipeline:
1. Load (or generate) data
2. Train/test split
3. Train User-Based and Item-Based CF models
4. Evaluate with Precision@K, Recall@K, NDCG@K, F1@K
5. Print sample recommendations + cold-start demo
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

from config import (
    DEFAULT_DENSITY,
    DEFAULT_K_NEIGHBORS_ITEM,
    DEFAULT_K_NEIGHBORS_USER,
    DEFAULT_N_ITEMS,
    DEFAULT_N_RECOMMENDATIONS,
    DEFAULT_N_USERS,
    EVAL_USER_SAMPLE_SIZE,
    RELEVANCE_THRESHOLD,
    TEST_RATIO,
)
from data.loader import generate_movielens_data, get_movie_metadata, load_movielens_data
from evaluation.metrics import f1_at_k, ndcg_at_k, precision_at_k, recall_at_k
from models.collaborative_filtering import ItemBasedCF, UserBasedCF
from utils.preprocessing import create_user_item_matrix, normalize_user_ratings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def train_test_split(
    df: pd.DataFrame,
    test_ratio: float = TEST_RATIO,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Randomly split a ratings DataFrame into train and test sets.

    Args:
        df: Full ratings DataFrame.
        test_ratio: Fraction of rows to hold out.
        random_seed: Seed for reproducibility.

    Returns:
        ``(train_df, test_df)``
    """
    test_df = df.sample(frac=test_ratio, random_state=random_seed)
    train_df = df.drop(test_df.index)
    return train_df, test_df


def evaluate_model(
    model: UserBasedCF | ItemBasedCF,
    test_df: pd.DataFrame,
    model_name: str = "Model",
    k: int = DEFAULT_N_RECOMMENDATIONS,
) -> dict[str, float]:
    """Compute ranking metrics for a trained model.

    Args:
        model: A fitted recommender.
        test_df: Held-out ratings DataFrame.
        model_name: Label for logging.
        k: Cut-off for ranking metrics.

    Returns:
        A dict with keys ``precision``, ``recall``, ``f1``, ``ndcg``.
    """
    logger.info("--- Evaluating %s ---", model_name)

    eval_users = test_df["user_id"].unique()[:EVAL_USER_SAMPLE_SIZE]

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    ndcgs: list[float] = []

    logger.info(
        "Computing Precision@%d, Recall@%d, F1@%d, NDCG@%d for %d users …",
        k, k, k, k, len(eval_users),
    )

    for user_id in eval_users:
        user_test = test_df[test_df["user_id"] == user_id]
        relevant_items = user_test[
            user_test["rating"] >= RELEVANCE_THRESHOLD
        ]["movie_id"].tolist()

        if not relevant_items:
            continue

        preds = model.predict(
            user_id,
            n_recommendations=k,
            k_neighbors=DEFAULT_K_NEIGHBORS_ITEM,
        )
        recommended_items = preds.index.tolist()

        precisions.append(precision_at_k(recommended_items, relevant_items, k))
        recalls.append(recall_at_k(recommended_items, relevant_items, k))
        f1s.append(f1_at_k(recommended_items, relevant_items, k))
        ndcgs.append(ndcg_at_k(recommended_items, relevant_items, k))

    results = {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }

    logger.info("%s results:", model_name)
    for metric, value in results.items():
        logger.info("  %s@%d: %.4f", metric.capitalize(), k, value)

    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full recommendation pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 50)
    logger.info("Netflix Recommendation System")
    logger.info("=" * 50)

    # 1. Data loading --------------------------------------------------------
    try:
        logger.info("Attempting real MovieLens data …")
        ratings_df, movies_df = load_movielens_data()
        logger.info("Success — using real data.")
    except Exception:
        logger.warning("Could not load real data — falling back to synthetic.", exc_info=True)
        movies_df = get_movie_metadata(n_items=DEFAULT_N_ITEMS)
        ratings_df = generate_movielens_data(
            n_users=DEFAULT_N_USERS,
            n_items=DEFAULT_N_ITEMS,
            density=DEFAULT_DENSITY,
        )

    # 2. Preprocessing -------------------------------------------------------
    train_df, test_df = train_test_split(ratings_df)
    logger.info("Train: %d rows | Test: %d rows", len(train_df), len(test_df))

    train_matrix = create_user_item_matrix(train_df)
    norm_matrix, user_means = normalize_user_ratings(train_matrix)

    # 3. Training & evaluation -----------------------------------------------
    ub_cf = UserBasedCF()
    ub_cf.fit(norm_matrix, user_means=user_means)
    evaluate_model(ub_cf, test_df, "User-Based CF")

    ib_cf = ItemBasedCF()
    ib_cf.fit(norm_matrix)
    evaluate_model(ib_cf, test_df, "Item-Based CF")

    # 4. Demo recommendations ------------------------------------------------
    valid_users = train_df["user_id"].value_counts()
    valid_users = valid_users[valid_users > 10].index.tolist()
    demo_user = valid_users[0] if valid_users else 1

    logger.info("--- Recommendations for User %s ---", demo_user)

    history = train_df[train_df["user_id"] == demo_user].join(
        movies_df.set_index("movie_id"), on="movie_id"
    )
    logger.info(
        "User history (top 3):\n%s",
        history.sort_values("rating", ascending=False)
        .head(3)[["title", "genres", "rating"]]
        .to_string(),
    )

    logger.info("User-Based picks:")
    ub_preds = ub_cf.predict(demo_user)
    if len(ub_preds) > 0:
        logger.info(
            "\n%s",
            movies_df[movies_df["movie_id"].isin(ub_preds.index)][
                ["title", "genres"]
            ].to_string(),
        )

    logger.info("Item-Based picks (Netflix style):")
    ib_preds = ib_cf.predict(demo_user)
    if len(ib_preds) > 0:
        logger.info(
            "\n%s",
            movies_df[movies_df["movie_id"].isin(ib_preds.index)][
                ["title", "genres"]
            ].to_string(),
        )

    # 5. Cold-start demo -----------------------------------------------------
    logger.info("--- Cold-start demo (unknown user 999999) ---")
    cold_preds = ub_cf.predict(999_999)
    if len(cold_preds) == 0:
        logger.info(
            "Model returned no results (expected). Falling back to popularity …"
        )
        trending = (
            train_df.groupby("movie_id")
            .size()
            .sort_values(ascending=False)
            .head(5)
        )
        logger.info(
            "\n%s",
            movies_df[movies_df["movie_id"].isin(trending.index)][
                ["title", "genres"]
            ].to_string(),
        )


if __name__ == "__main__":
    main()
