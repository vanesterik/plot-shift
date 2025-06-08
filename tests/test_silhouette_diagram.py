from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray
from plot_shift.silhouette_diagram import silhouette_diagram

SampleData = Tuple[int, NDArray[np.int32], NDArray[np.float64], float]


@pytest.fixture
def sample_data() -> SampleData:
    # 3 clusters, 15 samples
    k = 3
    y_pred = np.array([0] * 5 + [1] * 5 + [2] * 5, dtype=np.int32)
    silhouette_coefficients = np.linspace(0.1, 0.9, 15).astype(np.float64)
    silhouette_score = 0.5
    return k, y_pred, silhouette_coefficients, silhouette_score


def test_silhouette_diagram_runs_without_error(sample_data: SampleData) -> None:
    k, y_pred, silhouette_coefficients, silhouette_score = sample_data
    fig, ax = plt.subplots()
    silhouette_diagram(k, y_pred, silhouette_coefficients, silhouette_score, ax)
    plt.close(fig)


def test_silhouette_diagram_sets_title_and_labels(sample_data: SampleData) -> None:
    k, y_pred, silhouette_coefficients, silhouette_score = sample_data
    fig, ax = plt.subplots()
    silhouette_diagram(k, y_pred, silhouette_coefficients, silhouette_score, ax)
    assert ax.get_title() == f"$k={k}$"
    assert ax.get_xlabel() == "Silhouette Coefficient"
    assert ax.get_ylabel() == "Cluster"
    plt.close(fig)


def test_silhouette_diagram_xticks(sample_data: SampleData) -> None:
    k, y_pred, silhouette_coefficients, silhouette_score = sample_data
    fig, ax = plt.subplots()
    silhouette_diagram(k, y_pred, silhouette_coefficients, silhouette_score, ax)
    expected_xticks = [-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1]
    np.testing.assert_allclose(ax.get_xticks(), expected_xticks)
    plt.close(fig)


def test_silhouette_diagram_handles_single_cluster() -> None:
    k = 1
    y_pred = np.zeros(5, dtype=np.int32)
    silhouette_coefficients = np.linspace(0.2, 0.8, 5).astype(np.float64)
    silhouette_score = 0.5
    fig, ax = plt.subplots()
    silhouette_diagram(k, y_pred, silhouette_coefficients, silhouette_score, ax)
    assert ax.get_title() == "$k=1$"
    plt.close(fig)


def test_silhouette_diagram_handles_empty_cluster() -> None:
    k = 2
    y_pred = np.zeros(5, dtype=np.int32)
    silhouette_coefficients = np.linspace(0.2, 0.8, 5).astype(np.float64)
    silhouette_score = 0.5
    fig, ax = plt.subplots()
    silhouette_diagram(k, y_pred, silhouette_coefficients, silhouette_score, ax)
    assert ax.get_title() == "$k=2$"
    plt.close(fig)
