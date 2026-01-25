"""
Visualization utilities for probe training results.

Provides plotting functions for:
- AUC vs layer comparisons
- Probe performance analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Sequence, Mapping, Any, Optional

import math


def plot_auc_by_layer(
    results: Union[list[dict], pd.DataFrame],
    target_field: str,
    title: str,
    metric: str = "test_auc",
    figsize: tuple = (6, 4)
) -> None:
    """
    Plot AUC (or other metric) vs layer for different positions.

    Based on plot_auc_by_layer() from semantic_entropy_probes.ipynb (lines 1654-1698).

    Args:
        results: List of result dicts or DataFrame with columns:
                 ['position', 'layer', 'target', metric]
        target_field: Target field to filter by (e.g., 'se_binary', 'is_correct')
        title: Plot title
        metric: Metric column name to plot (default: 'test_auc')
        figsize: Figure size tuple

    Example:
        >>> results = manager.train_every_combination(...)
        >>> plot_auc_by_layer(
        ...     results, target_field='se_binary',
        ...     title='Test AUROC vs layer (target = se_binary)'
        ... )
    """
    # Convert to DataFrame if needed
    if isinstance(results, list):
        df = pd.DataFrame(results)
    else:
        df = results.copy()

    # Filter by target
    df_t = df[df["target"] == target_field].copy()

    if len(df_t) == 0:
        print(f"Warning: No results found for target '{target_field}'")
        return

    # Position labels for legend
    pos_labels = {0: "TBG (pos=0)", -2: "SLT (pos=-2)"}

    plt.figure(figsize=figsize)

    # Plot each position
    for pos in sorted(df_t["position"].unique()):
        sub = df_t[df_t["position"] == pos].sort_values("layer")
        plt.plot(
            sub["layer"],
            sub[metric],
            marker="o",
            label=pos_labels.get(pos, f"pos={pos}"),
        )

    plt.xlabel("Layer index")
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_sep_vs_accuracy_auc(
    results: Union[list[dict], pd.DataFrame],
    position: int,
    title: str,
    metric: str = "test_auc_is_correct",
    figsize: tuple = (6, 4)
) -> None:
    """
    Compare SEP probes vs Accuracy probes for predicting correctness.

    Based on plot_sep_vs_accuracy_auc() from semantic_entropy_probes.ipynb (lines 1803-1855).

    Args:
        results: Results with 'test_auc_is_correct' metric
        position: Position to plot (0, -2, etc.)
        title: Plot title
        metric: Metric to plot (default: 'test_auc_is_correct')
        figsize: Figure size
    """
    # Convert to DataFrame if needed
    if isinstance(results, list):
        df = pd.DataFrame(results)
    else:
        df = results.copy()

    df_pos = df[df["position"] == position].sort_values("layer")

    if len(df_pos) == 0:
        print(f"Warning: No results found for position {position}")
        return

    plt.figure(figsize=figsize)

    # SEP probes
    sep_df = df_pos[df_pos["target"] == "se_binary"]
    if len(sep_df) > 0:
        plt.plot(
            sep_df["layer"], sep_df[metric],
            marker="o", linewidth=2.5, markersize=8,
            label="SEP probes (se_binary → inverted)", color="red"
        )

    # Accuracy probes
    acc_df = df_pos[df_pos["target"] == "is_correct"]
    if len(acc_df) > 0:
        plt.plot(
            acc_df["layer"], acc_df[metric],
            marker="s", linewidth=2.5, markersize=8,
            label="Accuracy probes", color="blue"
        )

    plt.xlabel("Layer index")
    plt.ylabel(f"Test AUROC (correlation with is_correct)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_history_grid(
    history: Sequence[Mapping[str, Any]],
    x_list: Sequence[str],
    y_list: Sequence[str],
    *,
    ncols: int = 3,
    figsize_per_ax: tuple[float, float] = (5.5, 3.6),
    marker: str = "o",
    linewidth: float = 1.5,
    sharey: bool = False,
    suptitle: Optional[str] = None,
):
    if not history:
        raise ValueError("history is empty")

    pairs = [(x, y) for y in y_list for x in x_list]
    n_plots = len(pairs)

    ncols = min(ncols, n_plots)
    nrows = math.ceil(n_plots / ncols)

    figsize = (figsize_per_ax[0] * ncols, figsize_per_ax[1] * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False, sharey=sharey)
    axes_flat = axes.ravel()

    for i, (x_key, y_key) in enumerate(pairs):
        ax = axes_flat[i]

        xs = [row.get(x_key) for row in history]
        ys = [row.get(y_key) for row in history]

        # drop None pairs
        xy = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        if not xy:
            ax.set_title(f"{y_key} vs {x_key} (no data)")
            ax.axis("off")
            continue

        xs, ys = zip(*xy)

        ax.plot(xs, ys, marker=marker, linewidth=linewidth)
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.set_title(f"{y_key} от {x_key}")
        ax.grid(True, alpha=0.3)

    # hide unused axes
    for j in range(n_plots, len(axes_flat)):
        axes_flat[j].axis("off")

    if suptitle:
        fig.suptitle(suptitle)

    fig.tight_layout()
    return fig, axes
