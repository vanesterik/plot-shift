from typing import Optional, Tuple

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray


def cost_threshold_plot(
    total_costs: NDArray[np.int32],
    thresholds: NDArray[np.float32],
    ax: Axes,
    model_name: Optional[str] = None,
) -> None:
    """
    Plot classification costs against thresholds.

    Parameters
    ----------
    model_name : str
        Name of the model for which costs are being plotted.

    ax : Axes
        Matplotlib Axes object to plot on.

    Returns
    -------
    None

    """

    minimum_cost_index = np.argmin(total_costs)
    optimal_threshold = thresholds[minimum_cost_index]

    ax.plot(
        thresholds,
        total_costs,
        marker="o",
        color="blue",
        linestyle="-",
    )
    ax.plot(
        thresholds,
        total_costs,
        linestyle="-",
        color="blue",
        linewidth=4,
        label="Total Expected Cost",
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
        ax.set_title("Costs vs. Thresholds")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Cost")
    ax.legend()


def binary_classifier_curve(
    y_true: NDArray[np.int32],
    y_score: NDArray[np.float32],
) -> Tuple[
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.float32],
]:
    """
    Calculate thresholds, true positives, false positives and false negatives
    per threshold.

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

    fps : NDArray[np.int32]
        1D array of shape (n_thresholds,) containing the number of false
        positives for each threshold.

    fns : NDArray[np.int32]
        1D array of shape (n_thresholds,) containing the number of false
        negatives for each threshold.

    thresholds : NDArray[np.float32]
        1D array of shape (n_thresholds,) containing the score values of
        thresholds, start high with all negative classification

    """

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_idxs]

    fps = 1 + threshold_idxs - tps
    fns = tps[-1] - tps
    thresholds = y_score[threshold_idxs]

    return tps, fps, fns, thresholds


def calculate_threshold_costs(
    false_positives: NDArray[np.int32],
    false_negatives: NDArray[np.int32],
    C_FP: int = 1,
    C_FN: int = 1,
) -> Tuple[NDArray[np.int32], np.int32]:
    """
    Compute the total costs for each threshold and find the minimal cost.

    Parameters
    ----------
    false_positives : NDArray[np.int32]
        Array containing the number of false positives for each threshold.

    false_negatives : NDArray[np.int32]
        Array containing the number of false negatives for each threshold.

    C_FP : int, optional
        Cost assigned to a false positive (default: 1).

    C_FN : int, optional
        Cost assigned to a false negative (default: 1).

    Returns
    -------
    total_costs : NDArray[np.int32]
        Array with total costs for each threshold.

    minimal_costs : np.int32
        The minimal total cost across all thresholds.

    """
    # Calculate the total costs for each threshold
    total_costs = C_FP * false_positives + C_FN * false_negatives
    # Define the minimal expected costs as the minimum of the total costs
    # across all thresholds
    minimum_cost_index = np.argmin(total_costs)
    minimal_costs = total_costs[minimum_cost_index]

    return (total_costs, minimal_costs)
