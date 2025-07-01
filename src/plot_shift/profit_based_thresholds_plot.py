from typing import Optional, Tuple

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray


def plot_profit_thresholds(
    total_profits: NDArray[np.int32],
    thresholds: NDArray[np.float32],
    optimal_threshold: float,
    ax: Axes,
    model_name: Optional[str] = None,
) -> None:
    """
    Plot total expected profit across classification thresholds.

    Parameters
    ----------
    total_profits : NDArray[np.int32]
        Array of total expected profits for each threshold.

    thresholds : NDArray[np.float32]
        Array of threshold values.

    optimal_threshold : float
        Threshold value that yields the maximum profit.

    ax : Axes
        Matplotlib Axes object to plot on.

    model_name : Optional[str], default=None
        Name of the model for labeling the plot.

    Returns
    -------
    None
    """

    ax.plot(
        thresholds,
        total_profits,
        marker="o",
        color="blue",
        linestyle="-",
    )
    ax.plot(
        thresholds,
        total_profits,
        linestyle="-",
        color="blue",
        linewidth=4,
        label="Total Expected Profit",
    )
    ax.axvline(
        optimal_threshold,
        color="r",
        linestyle="--",
        label=f"Optimal Threshold: {optimal_threshold:.2f}",
    )

    if model_name:
        ax.set_title(model_name)
    else:
        ax.set_title("Expected Profit Across Classification Thresholds")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Profit")
    ax.legend()


def calculate_binary_classifier_curve(
    y_true: NDArray[np.int32],
    y_score: NDArray[np.float32],
) -> Tuple[
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.float32],
]:
    """
    Calculate true positives, false negatives, false positives, true negatives, and thresholds
    for a binary classifier across all possible thresholds.

    Parameters
    ----------
    y_true : NDArray[np.int32]
        1D array of shape (n_samples,) containing the true binary labels.

    y_score : NDArray[np.float32]
        1D array of shape (n_samples,) containing the scores or decision
        function values for the positive class. These can be probabilities or
        raw scores.

    Returns
    -------
    tps : NDArray[np.int32]
        1D array of shape (n_thresholds,) containing the number of true
        positives for each threshold.

    fns : NDArray[np.int32]
        1D array of shape (n_thresholds,) containing the number of false
        negatives for each threshold.

    fps : NDArray[np.int32]
        1D array of shape (n_thresholds,) containing the number of false
        positives for each threshold.

    tns : NDArray[np.int32]
        1D array of shape (n_thresholds,) containing the number of true
        negatives for each threshold.

    thresholds : NDArray[np.float32]
        1D array of shape (n_thresholds,) containing the score values of
        thresholds, sorted in descending order.
    """

    # Sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_indices = np.r_[distinct_value_indices, y_true.size - 1]

    # Accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_indices]

    fps = 1 + threshold_indices - tps
    fns = tps[-1] - tps
    thresholds = y_score[threshold_indices]

    # Calculate true negatives
    total_negatives = np.sum(y_true == 0)
    tns = total_negatives - fps

    return tps, fns, fps, tns, thresholds


def calculate_profit_thresholds(
    tps: NDArray[np.int32],
    fns: NDArray[np.int32],
    fps: NDArray[np.int32],
    tns: NDArray[np.int32],
    thresholds: NDArray[np.float32],
    revenue_tp: int = 10,
    revenue_tn: int = 0,
    cost_fp: int = 1,
    cost_fn: int = 1,
) -> Tuple[
    int,
    int,
    float,
    float,
    NDArray[np.int32],
    float,
    float,
]:
    """
    Compute the total profit for each threshold and find the maximum profit and
    corresponding threshold, as well as related metrics.

    Parameters
    ----------
    tps : NDArray[np.int32]
        Array containing the number of true positives for each threshold.

    fns : NDArray[np.int32]
        Array containing the number of false negatives for each threshold.

    fps : NDArray[np.int32]
        Array containing the number of false positives for each threshold.

    tns : NDArray[np.int32]
        Array containing the number of true negatives for each threshold.

    thresholds : NDArray[np.float32]
        Array of threshold values.

    revenue_tp : int, optional
        Revenue assigned to a true positive (default: 10).

    revenue_tn : int, optional
        Revenue assigned to a true negative (default: 0).

    cost_fp : int, optional
        Cost assigned to a false positive (default: 1).

    cost_fn : int, optional
        Cost assigned to a false negative (default: 1).

    Returns
    -------
    maximum_profit : int
        The maximum profit across all thresholds.

    calibrated_maximum_profit : int
        The profit at the calibrated optimal threshold.

    optimal_threshold : float
        The threshold corresponding to the maximum profit.

    calibrated_optimal_threshold : float
        The threshold that balances costs and revenues.

    profits : NDArray[np.int32]
        Array with total profit for each threshold.

    precision : float
        Precision at the optimal threshold.

    recall : float
        Recall at the optimal threshold.
    """
    # Calculate the total profit for each threshold
    profits = (revenue_tp * tps + revenue_tn * tns) - (cost_fp * fps - cost_fn * fns)
    # Calculate the total profit for each threshold and identify the threshold
    # with the maximum profit
    maximum_profit_index = np.argmax(profits)
    maximum_profit = profits[maximum_profit_index]
    optimal_threshold = thresholds[maximum_profit_index]
    # Calculate precision and recall at the optimal threshold
    tp = tps[maximum_profit_index]
    fp = fps[maximum_profit_index]
    fn = fns[maximum_profit_index]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # Calculate the calibrated optimal threshold This is the threshold that
    # balances the cost of false positives and false negatives with the revenue
    # from true positives and true negatives.
    calibrated_optimal_threshold = (cost_fp + revenue_tn) / (
        (cost_fp + revenue_tn) + (revenue_tp - cost_fn)
    )
    calibrated_maximum_profit = profits[
        np.searchsorted(
            -thresholds,
            -calibrated_optimal_threshold,
            side="right",
        )
        - 1
    ]

    return (
        int(maximum_profit),
        int(calibrated_maximum_profit),
        float(optimal_threshold),
        float(calibrated_optimal_threshold),
        profits,
        float(precision),
        float(recall),
    )
