"""Unit tests for utils.preprocessing module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.preprocessing import calculate_sparsity, create_user_item_matrix, normalize_user_ratings


@pytest.fixture()
def sample_ratings() -> pd.DataFrame:
    """Small ratings DataFrame for testing."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "movie_id": [10, 20, 10, 30, 20],
            "rating": [5.0, 3.0, 4.0, 2.0, 5.0],
        }
    )


class TestCreateUserItemMatrix:
    def test_shape(self, sample_ratings: pd.DataFrame) -> None:
        matrix = create_user_item_matrix(sample_ratings)
        # Users 1-3, Items up to 30 (inferred from max movie_id)
        assert matrix.shape[0] == 3  # 3 users
        assert 10 in matrix.columns
        assert 20 in matrix.columns
        assert 30 in matrix.columns

    def test_values_filled(self, sample_ratings: pd.DataFrame) -> None:
        matrix = create_user_item_matrix(sample_ratings)
        assert matrix.loc[1, 10] == 5.0
        assert matrix.loc[2, 30] == 2.0

    def test_missing_is_zero(self, sample_ratings: pd.DataFrame) -> None:
        matrix = create_user_item_matrix(sample_ratings)
        assert matrix.loc[1, 30] == 0.0  # user 1 did not rate movie 30


class TestNormalizeUserRatings:
    def test_centered_mean_is_zero(self, sample_ratings: pd.DataFrame) -> None:
        matrix = create_user_item_matrix(sample_ratings)
        centered, means = normalize_user_ratings(matrix)
        # For user 1: rated [5, 3] → mean=4 → centered [1, -1, 0...]
        assert means.loc[1] == pytest.approx(4.0)

    def test_returns_tuple(self, sample_ratings: pd.DataFrame) -> None:
        matrix = create_user_item_matrix(sample_ratings)
        result = normalize_user_ratings(matrix)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestCalculateSparsity:
    def test_full_matrix(self) -> None:
        df = pd.DataFrame({"x": range(100)})
        assert calculate_sparsity(df, 10, 10) == pytest.approx(0.0)

    def test_empty_matrix(self) -> None:
        df = pd.DataFrame({"x": []})
        assert calculate_sparsity(df, 10, 10) == pytest.approx(100.0)

    def test_half_sparse(self) -> None:
        df = pd.DataFrame({"x": range(50)})
        assert calculate_sparsity(df, 10, 10) == pytest.approx(50.0)

    def test_zero_dimensions(self) -> None:
        df = pd.DataFrame({"x": [1]})
        assert calculate_sparsity(df, 0, 10) == 100.0
