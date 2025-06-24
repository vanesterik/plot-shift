import pandas as pd
import pytest

from plot_shift.correlation_plot import get_correlations


def test_get_correlations_basic() -> None:
    df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [2, 4, 6, 8], "C": [4, 3, 2, 1]})
    result = get_correlations(df)
    # There are 3 pairs: A-B, A-C, B-C
    assert len(result) == 3
    # A and B are perfectly correlated
    assert pytest.approx(result["A - B"]) == 1.0
    # A and C are perfectly negatively correlated
    assert pytest.approx(result["A - C"]) == -1.0
    # B and C are perfectly negatively correlated
    assert pytest.approx(result["B - C"]) == -1.0
    # Sorted descending: A-B first, then A-C/B-C (both -1)
    assert list(result.index) == ["A - B", "A - C", "B - C"]


def test_get_correlations_non_numeric_columns() -> None:
    df = pd.DataFrame(
        {"A": [1, 2, 3], "B": [3, 2, 1], "C": ["x", "y", "z"]}  # Non-numeric
    )
    result = get_correlations(df)
    # Only numeric columns should be considered
    assert set(result.index) == {"A - B"}


def test_get_correlations_single_column() -> None:
    df = pd.DataFrame({"A": [1, 2, 3, 4]})
    result = get_correlations(df)
    # No pairs possible
    assert len(result) == 0


def test_get_correlations_empty_dataframe() -> None:
    df = pd.DataFrame()
    result = get_correlations(df)
    assert isinstance(result, pd.Series)
    assert len(result) == 0


def test_get_correlations_duplicate_values() -> None:
    df = pd.DataFrame({"A": [1, 1, 1, 1], "B": [2, 2, 2, 2]})
    result = get_correlations(df)
    # Correlation is undefined (nan) for constant columns
    assert result.isna().all()
