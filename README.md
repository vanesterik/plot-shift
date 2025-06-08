![PLOT-SHIFT](https://raw.githubusercontent.com/vanesterik/plot-shift/refs/heads/main/references/rick-and-morty-plot.jpg)

# PLOT-SHIFT

A growing collection of Python utilities for advanced data visualization, with a focus on exploratory data analysis. This project is a personal effort to gather and refine useful plotting functions developed during my data science studies and research.

## Purpose

The goal of this package is to provide reusable, well-documented visualization tools that support a variety of data science workflows. Over time, new plotting functions and utilities will be added as they are created or improved.

## Features

- Visualizations for cluster analysis, including silhouette diagram matrices
- Designed for integration with scikit-learn and other common data science libraries
- Usable in research, notebooks, and production code

## Installation

Install via [PDM](https://pdm.fming.dev/) (recommended):

```sh
pdm add plot-shift
```

Or with pip (after publishing to PyPI):

```sh
pip install plot-shift
```

> **Note:** If you are using the code locally (not from PyPI), make sure your `PYTHONPATH` includes the `src/` directory, or install the package in editable mode:
>
> ```sh
> pip install -e ./src
> ```

## Usage

Import and use the available plotting functions in your data science projects. Example for clustering visualization:

```python
from plot_shift.silhouette_diagram import silhouette_diagram

# Example usage:
silhouette_diagram(
    k,                          # Number of clusters
    y_pred,                     # Predicted cluster labels
    silhouette_coefficients,    # Silhouette coefficients for sample
    silhouette_score,           # Silhouette score for sample
    ax,                         # Matplotlib Axes object
)
```

More plotting functions will be added over time. See the `notebooks/` directory for practical examples and demonstrations.

## Contributing

Contributions and suggestions are welcome! If you have a useful plotting function or improvement, feel free to open an issue or pull request.

## License

[MIT License](https://github.com/vanesterik/plot-shift/blob/main/LICENSE)

## Author

[Koen van Esterik](https://github.com/vanesterik)
