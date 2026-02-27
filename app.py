"""
Streamlit dashboard for the Netflix Recommendation System.

Provides an interactive UI to:
- Select a user and view their watch history.
- Toggle between User-Based and Item-Based CF.
- Generate and display personalised recommendations.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    DEFAULT_K_NEIGHBORS_ITEM,
    DEFAULT_K_NEIGHBORS_USER,
    DEFAULT_N_ITEMS,
    MIN_USER_RATINGS_FOR_DEMO,
    TOP_RATED_DISPLAY_COUNT,
)
from data.loader import generate_movielens_data, get_movie_metadata, load_movielens_data
from models.collaborative_filtering import ItemBasedCF, UserBasedCF
from utils.preprocessing import create_user_item_matrix, normalize_user_ratings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Netflix Recommender System", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ratings and movie metadata (cached across reruns)."""
    with st.spinner("Loading MovieLens data …"):
        try:
            ratings_df, movies_df = load_movielens_data()
            st.success("Loaded real MovieLens data!")
        except Exception as e:
            logger.warning("Could not load real data (%s). Using synthetic.", e)
            st.warning("Could not load real data — using synthetic fallback.")
            ratings_df = generate_movielens_data(
                n_users=1_000, n_items=DEFAULT_N_ITEMS, density=0.05,
            )
            movies_df = get_movie_metadata(n_items=DEFAULT_N_ITEMS, use_real_data=False)
    return ratings_df, movies_df


@st.cache_resource(show_spinner=False)
def train_models(
    _ratings_df: pd.DataFrame,
) -> tuple[UserBasedCF, ItemBasedCF, pd.DataFrame, pd.DataFrame]:
    """Build interaction matrix and train both CF models (cached)."""
    with st.spinner("Building interaction matrix …"):
        interaction_matrix = create_user_item_matrix(_ratings_df)

    norm_matrix, user_means = normalize_user_ratings(interaction_matrix)

    with st.spinner("Training User-Based CF …"):
        ub_cf = UserBasedCF()
        ub_cf.fit(norm_matrix, user_means=user_means)

    with st.spinner("Training Item-Based CF (Netflix) …"):
        ib_cf = ItemBasedCF()
        ib_cf.fit(norm_matrix)

    return ub_cf, ib_cf, interaction_matrix, norm_matrix


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    """Render the Streamlit dashboard."""
    st.title("🎬 Netflix Recommendation System")
    st.markdown("### User-Based & Item-Based Collaborative Filtering")

    # Load & train
    ratings_df, movies_df = load_data()
    ub_cf, ib_cf, raw_matrix, norm_matrix = train_models(ratings_df)

    # --- Sidebar: user selection -------------------------------------------
    st.sidebar.header("User Control")

    user_counts = ratings_df["user_id"].value_counts()
    active_users = (
        user_counts[user_counts >= MIN_USER_RATINGS_FOR_DEMO]
        .index.sort_values()
        .tolist()
    )

    selected_user = st.sidebar.selectbox("Select User ID", active_users, index=0)

    user_ratings = ratings_df[ratings_df["user_id"] == selected_user]
    st.sidebar.markdown(f"**Ratings count:** {len(user_ratings)}")
    st.sidebar.markdown(f"**Avg rating:** {user_ratings['rating'].mean():.2f}")

    # --- Main layout -------------------------------------------------------
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Watch History")
        st.markdown(f"*Top {TOP_RATED_DISPLAY_COUNT} highest-rated movies*")

        top_rated = (
            user_ratings.sort_values(by="rating", ascending=False)
            .head(TOP_RATED_DISPLAY_COUNT)
            .merge(movies_df, on="movie_id", how="left")
        )
        for _, row in top_rated.iterrows():
            st.info(f"**{row['title']}**\n\n⭐ {row['rating']} | {row['genres']}")

    with col2:
        st.subheader("Recommended for You 🍿")

        model_choice = st.radio(
            "Choose algorithm:",
            ("Item-Based CF (Netflix Standard)", "User-Based CF"),
            horizontal=True,
        )

        if st.button("Generate Recommendations"):
            start_time = time.time()

            if "Item-Based" in model_choice:
                model = ib_cf
                k_neighbors = DEFAULT_K_NEIGHBORS_ITEM
            else:
                model = ub_cf
                k_neighbors = DEFAULT_K_NEIGHBORS_USER

            preds = model.predict(
                selected_user,
                n_recommendations=10,
                k_neighbors=k_neighbors,
            )

            elapsed = time.time() - start_time
            st.success(f"Generated in {elapsed:.4f} s")

            if len(preds) == 0:
                st.warning("No recommendations found (cold-start or data gap).")
            else:
                recs_df = pd.DataFrame(
                    {"movie_id": preds.index, "score": preds.values}
                ).merge(movies_df, on="movie_id")

                for i, row in recs_df.iterrows():
                    with st.container():
                        st.markdown(f"#### {i + 1}. {row['title']}")
                        st.caption(
                            f"Genre: {row['genres']} | Score: {row['score']:.4f}"
                        )
                        st.markdown("---")

    # --- Explanation -------------------------------------------------------
    with st.expander("How does this work?"):
        st.markdown(
            """
            **Item-Based CF (Netflix Style):**
            - Finds movies similar to what you already liked using cosine
              similarity across the entire user population's ratings.
            - *Stable, scalable, pre-computable offline.*

            **User-Based CF:**
            - Finds users with taste profiles similar to yours and
              recommends what *they* enjoyed that you haven't seen.
            - *Good for serendipity, but hard to scale beyond ~10 M users.*
            """
        )


if __name__ == "__main__":
    main()
