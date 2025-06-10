import numpy as np
import pytest
from matplotlib import pyplot as plt

from plot_shift.cost_threshold_plot import binary_classifier_curve, cost_threshold_plot


def test_binary_classifier_curve_basic() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int32)
    y_score = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float32)
    tps, fps, fns, thresholds = binary_classifier_curve(y_true, y_score)
    # There should be as many thresholds as unique scores
    assert len(thresholds) == len(np.unique(y_score))
    # tps, fps, fns should all have the same length as thresholds
    assert len(tps) == len(fps) == len(fns) == len(thresholds)
    # tps should be non-decreasing
    assert np.all(np.diff(tps) >= 0)
    # fps should be non-decreasing
    assert np.all(np.diff(fps) >= 0)
    # fns should be non-increasing
    assert np.all(np.diff(fns) <= 0)


def test_binary_classifier_curve_all_ones() -> None:
    y_true = np.ones(5, dtype=np.int32)
    y_score = np.linspace(0, 1, 5, dtype=np.float32)
    tps, fps, fns, _ = binary_classifier_curve(y_true, y_score)
    assert np.all(fps == 0)
    assert np.all(tps == np.arange(1, 6))
    assert np.all(fns == tps[-1] - tps)


def test_binary_classifier_curve_all_zeros() -> None:
    y_true = np.zeros(5, dtype=np.int32)
    y_score = np.linspace(0, 1, 5, dtype=np.float32)
    tps, fps, fns, _ = binary_classifier_curve(y_true, y_score)
    assert np.all(tps == 0)
    assert np.all(fns == 0)
    assert np.all(fps == np.arange(1, 6))


def test_cost_threshold_plot_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    total_costs = np.array([10, 5, 3, 7], dtype=np.int32)
    thresholds = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    fig, ax = plt.subplots()
    # Patch plt.show to avoid displaying the plot during tests
    monkeypatch.setattr(plt, "show", lambda: None)
    cost_threshold_plot(total_costs, thresholds, ax, model_name="TestModel")
    # Check that the title is set correctly
    assert ax.get_title() == "TestModel"
    # Check that xlabel and ylabel are set
    assert ax.get_xlabel() == "Threshold"
    assert ax.get_ylabel() == "Cost"
    plt.close(fig)


def test_cost_threshold_plot_no_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    total_costs = np.array([2, 1, 4], dtype=np.int32)
    thresholds = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    fig, ax = plt.subplots()
    monkeypatch.setattr(plt, "show", lambda: None)
    cost_threshold_plot(total_costs, thresholds, ax)
    assert ax.get_title() == "Costs vs. Thresholds"
    assert ax.get_xlabel() == "Threshold"
    assert ax.get_ylabel() == "Cost"
    plt.close(fig)
