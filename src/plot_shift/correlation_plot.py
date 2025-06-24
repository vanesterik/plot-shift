from typing import Optional, Tuple

import pandas as pd
from matplotlib.axes import Axes


def correlation_plot(
    correlations: pd.Series,
    ax: Axes,
    colormap: Tuple[str, str] = (
        "#27A69A",  # Positive correlation color
        "#F0534F",  # Negative correlation color
    ),
    title: Optional[str] = None,
) -> None:
    """
    Create a horizontal bar plot of correlations.

    Parameters
    ----------
    correlations : pd.Series
        A Series containing the pairwise correlations between columns, sorted in
        ascending order. The index of the Series is a MultiIndex with pairs of
        column names.

    ax : Axes
        The matplotlib Axes object on which to plot the correlations.

    colormap : Tuple[str, str], optional
        A tuple containing two colors for positive and negative correlations,
        respectively. The default is ("#27A69A", "#F0534F").

    title : Optional[str], optional
        The title of the plot. If None, a default title "Correlations" will be
        used.

    Returns
    -------
    None

    """
    colors = [colormap[0] if v > 0 else colormap[1] for v in correlations.values]

    ax.barh(
        correlations.index,
        correlations.values,
        color=colors,
        edgecolor="white",
    )

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Correlations")


def get_correlations(df: pd.DataFrame) -> pd.Series:
    """
    Calculate pairwise correlations between columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing numerical data for which to calculate correlations.

    Returns
    -------
    pd.Series
        A Series containing the pairwise correlations between columns, sorted in
        ascending order. The index of the Series is a MultiIndex with pairs of
        column names.

    """

    correlations = df.select_dtypes(include="number")
    correlations = correlations.corr().unstack()
    correlations = correlations[
        correlations.index.get_level_values(0) != correlations.index.get_level_values(1)
    ]
    correlations = correlations[
        correlations.index.get_level_values(0) < correlations.index.get_level_values(1)
    ]
    correlations = correlations.sort_values(ascending=False)
    correlations.index = [f"{i[0]} - {i[1]}" for i in correlations.index]

    return correlations
