import numpy as np
import pytest
from matplotlib import pyplot as plt

from plot_shift.profit_based_thresholds_plot import (
    calculate_binary_classifier_curve,
    calculate_profit_thresholds,
    plot_profit_thresholds,
)


def test_plot_profit_thresholds_sets_labels_and_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    total_profits = np.array([10, 20, 30], dtype=np.int32)
    thresholds = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    optimal_threshold = 0.5
    fig, ax = plt.subplots()
    monkeypatch.setattr(plt, "show", lambda: None)
    plot_profit_thresholds(
        total_profits, thresholds, optimal_threshold, ax, model_name="MyModel"
    )
    assert ax.get_title() == "MyModel"
    assert ax.get_xlabel() == "Threshold"
    assert ax.get_ylabel() == "Profit"
    legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert any("Total Expected Profit" in label for label in legend_labels)
    assert any("Optimal Threshold" in label for label in legend_labels)
    plt.close(fig)


def test_plot_profit_thresholds_default_title(monkeypatch: pytest.MonkeyPatch) -> None:
    total_profits = np.array([1, 2], dtype=np.int32)
    thresholds = np.array([0.2, 0.8], dtype=np.float32)
    optimal_threshold = 0.2
    fig, ax = plt.subplots()
    monkeypatch.setattr(plt, "show", lambda: None)
    plot_profit_thresholds(total_profits, thresholds, optimal_threshold, ax)
    assert ax.get_title() == "Expected Profit Across Classification Thresholds"
    plt.close(fig)


def test_calculate_binary_classifier_curve_typical() -> None:
    y_true = np.array([0, 1, 0, 1, 1], dtype=np.int32)
    y_score = np.array([0.2, 0.8, 0.4, 0.6, 0.9], dtype=np.float32)
    tps, fns, fps, tns, thresholds = calculate_binary_classifier_curve(y_true, y_score)
    assert len(tps) == len(fns) == len(fps) == len(tns) == len(thresholds)
    assert np.all(tps >= 0)
    assert np.all(fns >= 0)
    assert np.all(fps >= 0)
    assert np.all(tns >= 0)
    assert np.all(np.diff(thresholds) < 0)  # descending order


def test_calculate_binary_classifier_curve_all_zeros() -> None:
    y_true = np.zeros(4, dtype=np.int32)
    y_score = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    tps, fns, fps, tns, thresholds = calculate_binary_classifier_curve(y_true, y_score)
    assert np.all(tps == 0)
    assert np.all(fns == 0)
    assert np.all(fps >= 0)
    assert np.all(tns >= 0)


def test_calculate_binary_classifier_curve_all_ones() -> None:
    y_true = np.ones(4, dtype=np.int32)
    y_score = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    tps, fns, fps, tns, thresholds = calculate_binary_classifier_curve(y_true, y_score)
    assert np.all(fps == 0)
    assert np.all(tns == 0)
    assert np.all(tps >= 0)
    assert np.all(fns >= 0)


def test_calculate_profit_thresholds_basic() -> None:
    tps = np.array([1, 2, 3], dtype=np.int32)
    fns = np.array([2, 1, 0], dtype=np.int32)
    fps = np.array([0, 1, 2], dtype=np.int32)
    tns = np.array([3, 2, 1], dtype=np.int32)
    thresholds = np.array([0.9, 0.5, 0.1], dtype=np.float32)
    result = calculate_profit_thresholds(tps, fns, fps, tns, thresholds)
    (
        total_profit,
        maximum_profit,
        optimal_threshold,
        precision,
        recall,
        calibrated_optimal_threshold,
        calibrated_total_profit,
    ) = result
    assert isinstance(total_profit, np.ndarray)
    assert isinstance(maximum_profit, int)
    assert isinstance(optimal_threshold, float)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert isinstance(calibrated_optimal_threshold, float)
    assert isinstance(calibrated_total_profit, int)


def test_calculate_profit_thresholds_custom_params() -> None:
    tps = np.array([0, 1], dtype=np.int32)
    fns = np.array([1, 0], dtype=np.int32)
    fps = np.array([2, 1], dtype=np.int32)
    tns = np.array([1, 2], dtype=np.int32)
    thresholds = np.array([0.7, 0.3], dtype=np.float32)
    result = calculate_profit_thresholds(
        tps, fns, fps, tns, thresholds, revenue_tp=5, revenue_tn=2, cost_fp=3, cost_fn=4
    )
    (
        total_profit,
        maximum_profit,
        optimal_threshold,
        precision,
        recall,
        calibrated_optimal_threshold,
        calibrated_total_profit,
    ) = result
    assert total_profit.shape == tps.shape
    assert isinstance(maximum_profit, int)
    assert isinstance(optimal_threshold, float)
    assert isinstance(calibrated_total_profit, int)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1


def test_calculate_profit_thresholds_all_zeros() -> None:
    tps = np.zeros(3, dtype=np.int32)
    fns = np.zeros(3, dtype=np.int32)
    fps = np.zeros(3, dtype=np.int32)
    tns = np.zeros(3, dtype=np.int32)
    thresholds = np.array([0.8, 0.5, 0.2], dtype=np.float32)
    result = calculate_profit_thresholds(tps, fns, fps, tns, thresholds)
    (
        total_profit,
        maximum_profit,
        optimal_threshold,
        precision,
        recall,
        calibrated_optimal_threshold,
        calibrated_total_profit,
    ) = result
    assert np.all(total_profit == 0)
    assert maximum_profit == 0
    assert isinstance(optimal_threshold, float)
    assert precision == 0 or np.isnan(precision)
    assert recall == 0 or np.isnan(recall)
