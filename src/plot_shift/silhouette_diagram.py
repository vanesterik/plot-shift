import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FixedFormatter, FixedLocator
from numpy.typing import NDArray


def silhouette_diagram(
    k: int,
    y_pred: NDArray[np.int32],
    silhouette_coefficients: NDArray[np.float32],
    silhouette_score: float,
    ax: Axes,
) -> None:
    """
    Plot a silhouette diagram for a given cluster count.

    Parameters
    ----------
    k : int
        Number of clusters.

    y_pred : NDArray[np.int32]
        Cluster labels for each sample.

    silhouette_coefficients : NDArray[np.float32]
        Silhouette coefficients for each sample.

    silhouette_score : float
        Average silhouette score for the clustering.

    ax : Axes
        Matplotlib Axes object to plot on.

    Returns
    -------
    None

    """

    # Set up color map
    cmap = plt.get_cmap("Spectral")

    # Define initial parameters for plotting
    padding = len(y_pred) // 30
    position = padding
    ticks = []

    # Loop through each cluster and plot its silhouette coefficients
    for i in range(k):
        coefficients = np.sort(silhouette_coefficients[y_pred == i])
        color = cmap(i / k)
        ax.fill_betweenx(
            np.arange(position, position + len(coefficients)),
            0,
            coefficients,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        ticks.append(position + len(coefficients) // 2)
        position += len(coefficients) + padding

    # Plot the average silhouette score line
    ax.axvline(x=silhouette_score, color="red", linestyle="--")

    # Set the y-axis ticks and labels
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([str(i) for i in range(k)]))

    # Set title and axis labels
    ax.set_title(f"$k={k}$")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_ylabel("Cluster")
