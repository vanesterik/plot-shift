from typing import Optional, Tuple

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix


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


def calculate_profit_thresholds(
    y_true: NDArray[np.int32],
    y_probs: NDArray[np.float32],
    revenue: int = 10,
    cost: int = 1,
) -> Tuple[
    NDArray[np.float32],
    NDArray[np.int32],
    int,
    float,
    float,
    float,
]:
    """
    Compute the total profit for each threshold, identify the maximum profit and
    corresponding threshold, and calculate precision and recall at the optimal threshold.

    Parameters
    ----------
    y_true : NDArray[np.int32]
        Array of true binary labels.

    y_probs : NDArray[np.float32]
        Array of predicted probabilities for the positive class.

    revenue : int, optional
        Revenue assigned to a true positive (default: 10).

    cost : int, optional
        Cost assigned to each prediction (default: 1).

    Returns
    -------
    thresholds : NDArray[np.float32]
        Array of threshold values.

    profits : NDArray[np.int32]
        Array with total profit for each threshold.

    maximum_profit : int
        The maximum profit across all thresholds.

    optimal_threshold : float
        The threshold corresponding to the maximum profit.

    precision : float
        Precision at the optimal threshold.

    recall : float
        Recall at the optimal threshold.
    """

    # Define thresholds from 0.0 to 1.0 with a step of 0.01
    thresholds = np.arange(0.0, 1, 0.01)

    # Initialize arrays to store counts of true positives, false negatives,
    # false positives, and true negatives for each threshold
    tps = np.zeros(len(thresholds), dtype=np.int32)
    fns = np.zeros(len(thresholds), dtype=np.int32)
    fps = np.zeros(len(thresholds), dtype=np.int32)
    tns = np.zeros(len(thresholds), dtype=np.int32)

    for i, threshold in enumerate(thresholds):
        # Apply threshold to get predicted labels
        y_pred = (y_probs >= threshold).astype(int)

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # Store the counts for each threshold
        tps[i] = tp
        fns[i] = fn
        fps[i] = fp
        tns[i] = tn

    # Calculate the total profit for each threshold
    profits = revenue * tps - cost * (fps + tps)

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

    return (
        thresholds,
        profits,
        int(maximum_profit),
        float(optimal_threshold),
        float(precision),
        float(recall),
    )
