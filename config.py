"""
Centralized configuration constants for the Netflix Recommendation System.

All magic numbers, paths, and tunable hyperparameters are defined here
to avoid scattering them across the codebase.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
MOVIELENS_DIR: Path = DATA_DIR / "ml-latest-small"
MOVIELENS_URL: str = (
    "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
)

# ---------------------------------------------------------------------------
# Data Generation Defaults
# ---------------------------------------------------------------------------
DEFAULT_N_USERS: int = 1_000
DEFAULT_N_ITEMS: int = 500
DEFAULT_DENSITY: float = 0.05
DEFAULT_RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Model Hyperparameters
# ---------------------------------------------------------------------------
DEFAULT_K_NEIGHBORS_USER: int = 50
DEFAULT_K_NEIGHBORS_ITEM: int = 20
DEFAULT_N_RECOMMENDATIONS: int = 10
RELEVANCE_THRESHOLD: float = 4.0  # rating >= this is "relevant"

# ---------------------------------------------------------------------------
# Evaluation Defaults
# ---------------------------------------------------------------------------
EVAL_USER_SAMPLE_SIZE: int = 50
TEST_RATIO: float = 0.2

# ---------------------------------------------------------------------------
# Synthetic Rating Distribution
# ---------------------------------------------------------------------------
RATING_VALUES: list[int] = [1, 2, 3, 4, 5]
RATING_PROBABILITIES: list[float] = [0.05, 0.10, 0.25, 0.35, 0.25]

# ---------------------------------------------------------------------------
# Modern Movie Injection
# ---------------------------------------------------------------------------
MODERN_MOVIE_USER_FRACTION: float = 0.15  # % of users who rate new movies
MODERN_MOVIE_WATCH_PROBABILITY: float = 0.60
MODERN_MOVIE_RATING_VALUES: list[float] = [3.5, 4.0, 4.5, 5.0]
MODERN_MOVIE_RATING_PROBS: list[float] = [0.1, 0.2, 0.3, 0.4]

# ---------------------------------------------------------------------------
# Streamlit Settings
# ---------------------------------------------------------------------------
MIN_USER_RATINGS_FOR_DEMO: int = 10
TOP_RATED_DISPLAY_COUNT: int = 5

# ---------------------------------------------------------------------------
# Genre Pool (for synthetic data)
# ---------------------------------------------------------------------------
GENRE_POOL: list[str] = [
    "Action", "Comedy", "Drama", "Horror",
    "Sci-Fi", "Romance", "Thriller",
]
