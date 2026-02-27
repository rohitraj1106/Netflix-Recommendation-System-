"""
Data loading and synthetic generation for the MovieLens dataset.

Supports two modes:
1. **Real data** — automatically downloads MovieLens Small and injects
   modern (2020–2024) blockbuster titles so recommendations feel current.
2. **Synthetic data** — generates a configurable-density random rating
   matrix for rapid prototyping and unit testing.
"""

from __future__ import annotations

import io
import logging
import os
import urllib.request
import zipfile
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    DEFAULT_DENSITY,
    DEFAULT_N_ITEMS,
    DEFAULT_N_USERS,
    DEFAULT_RANDOM_SEED,
    GENRE_POOL,
    MODERN_MOVIE_RATING_PROBS,
    MODERN_MOVIE_RATING_VALUES,
    MODERN_MOVIE_USER_FRACTION,
    MODERN_MOVIE_WATCH_PROBABILITY,
    MOVIELENS_DIR,
    MOVIELENS_URL,
    RATING_PROBABILITIES,
    RATING_VALUES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Real data helpers
# ---------------------------------------------------------------------------

def download_and_extract_movielens() -> None:
    """Download and extract MovieLens Small if not already present."""
    ratings_path = os.path.join(MOVIELENS_DIR, "ratings.csv")
    movies_path = os.path.join(MOVIELENS_DIR, "movies.csv")

    if os.path.isfile(ratings_path) and os.path.isfile(movies_path):
        logger.info("MovieLens data already exists at %s", MOVIELENS_DIR)
        return

    logger.info("Downloading MovieLens Small from %s …", MOVIELENS_URL)
    try:
        response = urllib.request.urlopen(MOVIELENS_URL)  # noqa: S310
        zip_content = response.read()
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
            target_path = str(DATA_DIR)
            zf.extractall(target_path)  # noqa: S202
            logger.info("Dataset extracted to %s", target_path)
    except Exception:
        logger.exception("Failed to download MovieLens data.")
        raise


def load_movielens_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the real MovieLens Small dataset.

    Returns:
        A tuple of ``(ratings_df, movies_df)`` with standardised column
        names: ``[user_id, movie_id, rating, timestamp]`` and
        ``[movie_id, title, genres]``.
    """
    download_and_extract_movielens()

    ratings_path = os.path.join(MOVIELENS_DIR, "ratings.csv")
    movies_path = os.path.join(MOVIELENS_DIR, "movies.csv")

    logger.info("Loading real MovieLens CSVs …")
    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)

    # Standardise column names
    ratings_df = ratings_df.rename(
        columns={"userId": "user_id", "movieId": "movie_id"}
    )
    movies_df = movies_df.rename(columns={"movieId": "movie_id"})

    # Enrich with modern titles
    ratings_df, movies_df = inject_modern_movies(ratings_df, movies_df)

    return ratings_df, movies_df


# ---------------------------------------------------------------------------
# Synthetic / metadata helpers
# ---------------------------------------------------------------------------

def get_movie_metadata(
    n_items: int = DEFAULT_N_ITEMS,
    use_real_data: bool = False,
) -> pd.DataFrame:
    """Return movie metadata — real or synthetic.

    Args:
        n_items: Number of synthetic movies to generate.
        use_real_data: If *True*, load real MovieLens metadata instead.

    Returns:
        DataFrame with columns ``[movie_id, title, genres]``.
    """
    if use_real_data:
        _, movies_df = load_movielens_data()
        return movies_df

    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
    genres = [
        f"{rng.choice(GENRE_POOL)}|{rng.choice(GENRE_POOL)}"
        for _ in range(n_items)
    ]
    return pd.DataFrame(
        {
            "movie_id": np.arange(1, n_items + 1),
            "title": [f"Movie {i}" for i in range(1, n_items + 1)],
            "genres": genres,
        }
    )


def generate_movielens_data(
    n_users: int = DEFAULT_N_USERS,
    n_items: int = DEFAULT_N_ITEMS,
    density: float = DEFAULT_DENSITY,
    random_seed: int = DEFAULT_RANDOM_SEED,
    use_real_data: bool = False,
) -> pd.DataFrame:
    """Generate a synthetic ratings DataFrame or load real data.

    Args:
        n_users: Number of unique users.
        n_items: Number of unique items.
        density: Fraction of the matrix to fill with ratings.
        random_seed: Seed for reproducibility.
        use_real_data: If *True*, return the real MovieLens ratings.

    Returns:
        DataFrame with columns ``[user_id, movie_id, rating, timestamp]``.
    """
    if use_real_data:
        ratings_df, _ = load_movielens_data()
        return ratings_df

    rng = np.random.default_rng(random_seed)

    n_ratings = int(n_users * n_items * density)
    logger.info(
        "Generating synthetic data: %d users, %d movies, ~%d ratings …",
        n_users, n_items, n_ratings,
    )

    user_ids = rng.integers(1, n_users + 1, size=n_ratings)
    movie_ids = rng.integers(1, n_items + 1, size=n_ratings)
    ratings = rng.choice(RATING_VALUES, size=n_ratings, p=RATING_PROBABILITIES)
    timestamps = rng.integers(1_000_000_000, 1_700_000_000, size=n_ratings)

    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "movie_id": movie_ids,
            "rating": ratings,
            "timestamp": timestamps,
        }
    )

    before = len(df)
    df = df.drop_duplicates(subset=["user_id", "movie_id"])
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d duplicate (user, movie) pairs.", dropped)

    logger.info("Final synthetic data shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Modern-movie injection
# ---------------------------------------------------------------------------

_MODERN_MOVIES: list[dict] = [
    # --- 2026 ---
    {"movie_id": 200_001, "title": "The Odyssey (2026)", "genres": "Adventure|Drama"},
    {"movie_id": 200_002, "title": "Jurassic World: Rebirth (2026)", "genres": "Action|Sci-Fi|Adventure"},
    {"movie_id": 200_003, "title": "Zootopia 2 (2026)", "genres": "Animation|Comedy|Adventure"},
    # --- 2025 ---
    {"movie_id": 200_004, "title": "Mission: Impossible - The Final Reckoning (2025)", "genres": "Action|Thriller"},
    {"movie_id": 200_005, "title": "Captain America: Brave New World (2025)", "genres": "Action|Sci-Fi"},
    {"movie_id": 200_006, "title": "Thunderbolts* (2025)", "genres": "Action|Sci-Fi"},
    {"movie_id": 200_007, "title": "Superman (2025)", "genres": "Action|Sci-Fi|Adventure"},
    {"movie_id": 200_008, "title": "Sinners (2025)", "genres": "Horror|Thriller"},
    {"movie_id": 200_009, "title": "Snow White (2025)", "genres": "Fantasy|Adventure|Musical"},
    # --- 2024 ---
    {"movie_id": 200_010, "title": "Dune: Part Two (2024)", "genres": "Sci-Fi|Action|Adventure"},
    {"movie_id": 200_011, "title": "Inside Out 2 (2024)", "genres": "Animation|Comedy|Drama"},
    {"movie_id": 200_012, "title": "Deadpool & Wolverine (2024)", "genres": "Action|Comedy|Sci-Fi"},
    {"movie_id": 200_013, "title": "Gladiator II (2024)", "genres": "Action|Drama|History"},
    {"movie_id": 200_014, "title": "Wicked (2024)", "genres": "Fantasy|Musical|Drama"},
    {"movie_id": 200_015, "title": "Moana 2 (2024)", "genres": "Animation|Adventure|Comedy"},
    {"movie_id": 200_016, "title": "Alien: Romulus (2024)", "genres": "Horror|Sci-Fi|Thriller"},
    {"movie_id": 200_017, "title": "Furiosa: A Mad Max Saga (2024)", "genres": "Action|Sci-Fi|Adventure"},
    {"movie_id": 200_018, "title": "The Wild Robot (2024)", "genres": "Animation|Sci-Fi|Drama"},
    # --- 2023 ---
    {"movie_id": 200_019, "title": "Oppenheimer (2023)", "genres": "Drama|History|Thriller"},
    {"movie_id": 200_020, "title": "Barbie (2023)", "genres": "Comedy|Fantasy|Adventure"},
    {"movie_id": 200_021, "title": "Spider-Man: Across the Spider-Verse (2023)", "genres": "Animation|Action|Adventure"},
    {"movie_id": 200_022, "title": "Guardians of the Galaxy Vol. 3 (2023)", "genres": "Action|Sci-Fi|Comedy"},
    {"movie_id": 200_023, "title": "Killers of the Flower Moon (2023)", "genres": "Crime|Drama|History"},
    {"movie_id": 200_024, "title": "John Wick: Chapter 4 (2023)", "genres": "Action|Crime|Thriller"},
    {"movie_id": 200_025, "title": "The Super Mario Bros. Movie (2023)", "genres": "Animation|Adventure|Comedy"},
    # --- 2022 ---
    {"movie_id": 200_026, "title": "Top Gun: Maverick (2022)", "genres": "Action|Drama"},
    {"movie_id": 200_027, "title": "Everything Everywhere All At Once (2022)", "genres": "Sci-Fi|Adventure|Comedy"},
    {"movie_id": 200_028, "title": "The Batman (2022)", "genres": "Action|Crime|Drama"},
    {"movie_id": 200_029, "title": "Avatar: The Way of Water (2022)", "genres": "Sci-Fi|Action|Adventure"},
    {"movie_id": 200_030, "title": "Glass Onion: A Knives Out Mystery (2022)", "genres": "Comedy|Crime|Thriller"},
    # --- 2021 ---
    {"movie_id": 200_031, "title": "Spider-Man: No Way Home (2021)", "genres": "Action|Adventure|Sci-Fi"},
    {"movie_id": 200_032, "title": "Dune (2021)", "genres": "Sci-Fi|Action|Adventure"},
    {"movie_id": 200_033, "title": "No Time to Die (2021)", "genres": "Action|Thriller"},
    # --- 2020 ---
    {"movie_id": 200_034, "title": "Tenet (2020)", "genres": "Action|Sci-Fi|Thriller"},
    {"movie_id": 200_035, "title": "Soul (2020)", "genres": "Animation|Comedy|Drama"},
]


def inject_modern_movies(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject 2020–2024 blockbusters and simulated ratings.

    A fraction of existing users are randomly selected to "watch" the
    new titles so that collaborative filtering can surface them.

    Args:
        ratings_df: Existing ratings DataFrame.
        movies_df: Existing movie metadata DataFrame.

    Returns:
        Updated ``(ratings_df, movies_df)`` with injected rows.
    """
    logger.info("Injecting modern movies (2020–2024) …")

    new_movies_df = pd.DataFrame(_MODERN_MOVIES)
    updated_movies = pd.concat([movies_df, new_movies_df], ignore_index=True)

    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
    user_ids = ratings_df["user_id"].unique()
    n_active = int(len(user_ids) * MODERN_MOVIE_USER_FRACTION)
    active_users = rng.choice(user_ids, size=n_active, replace=False)

    new_ratings: list[dict] = []
    for movie in _MODERN_MOVIES:
        mid = movie["movie_id"]
        for uid in active_users:
            if rng.random() < MODERN_MOVIE_WATCH_PROBABILITY:
                new_ratings.append(
                    {
                        "user_id": uid,
                        "movie_id": mid,
                        "rating": rng.choice(
                            MODERN_MOVIE_RATING_VALUES,
                            p=MODERN_MOVIE_RATING_PROBS,
                        ),
                        "timestamp": 1_700_000_000,
                    }
                )

    updated_ratings = pd.concat(
        [ratings_df, pd.DataFrame(new_ratings)], ignore_index=True
    )
    logger.info("Added %d ratings for %d modern movies.", len(new_ratings), len(_MODERN_MOVIES))
    return updated_ratings, updated_movies


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        r, m = load_movielens_data()
        logger.info("Real data loaded successfully! Movies tail:\n%s", m.tail(10))
    except Exception:
        logger.exception("Real data load failed — falling back to synthetic.")
        df = generate_movielens_data()
        logger.info("Synthetic data head:\n%s", df.head())
