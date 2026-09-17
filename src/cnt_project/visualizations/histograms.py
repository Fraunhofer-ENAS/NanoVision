from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from .styles import apply_style


def _clean_numeric_array(data: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(list(data) if not isinstance(data, np.ndarray) else data, dtype=float)
    return arr[~np.isnan(arr)]


def _compute_bin_edges(
    data_a: np.ndarray,
    data_b: np.ndarray | None = None,
    bins: int = 60,
) -> np.ndarray:
    all_data = data_a if data_b is None else np.concatenate([data_a, data_b])

    if len(all_data) == 0:
        return np.linspace(0, 1, bins + 1)

    dmin = float(np.min(all_data))
    dmax = float(np.max(all_data))

    if abs(dmax - dmin) < 1e-12:
        dmin -= 0.5
        dmax += 0.5

    return np.linspace(dmin, dmax, bins + 1)


def _adaptive_bin_count(n: int) -> int:
    if n < 10:
        return max(3, n)
    if n < 30:
        return 8
    if n < 100:
        return 12
    return 15


def plot_histogram_comparison(
    data_true: Iterable[float] | np.ndarray | None,
    data_pred: Iterable[float] | np.ndarray,
    *,
    xlabel: str,
    save_path: str | Path | None = None,
    title: str | None = None,
    bins: int = 60,
    density: bool = True,
    show_kde: bool = True,
    color_true: str = "#f0e68c",
    color_pred: str = "#4169e1",
    legend_labels: tuple[str, str] = ("GT", "Prediction"),
    use_science_style: bool = True,
    use_ieee_style: bool = False,
    use_latex: bool = False,
    integer_x_ticks: bool = False,
    x_ticks: list[float] | None = None,
) -> dict[str, float]:
    if use_science_style:
        apply_style(style="science", ieee=use_ieee_style, no_latex=not use_latex)
    else:
        plt.rcdefaults()

    import seaborn as sns

    pred = _clean_numeric_array(data_pred)
    true = _clean_numeric_array(data_true) if data_true is not None else None

    if len(pred) == 0:
        raise ValueError("data_pred is empty after NaN removal.")

    bin_edges = _compute_bin_edges(true if true is not None else pred, pred, bins=bins)
    results: dict[str, float] = {}

    fig, ax = plt.subplots()

    if true is not None and len(true) > 0:
        sns.histplot(
            true,
            bins=bin_edges,
            color=color_true,
            stat="density" if density else "count",
            alpha=0.75,
            ax=ax,
            edgecolor="black",
            linewidth=0.5,
            label=legend_labels[0],
        )
        if show_kde and len(true) > 1:
            sns.kdeplot(true, color=color_true, ax=ax, linewidth=1.1, bw_adjust=0.8)

        mean_true = float(np.mean(true))
        std_true = float(np.std(true))
        ax.axvline(mean_true, color=color_true, linestyle="--", linewidth=1.2)
        results["mean_true"] = mean_true
        results["std_true"] = std_true

    sns.histplot(
        pred,
        bins=bin_edges,
        color=color_pred,
        stat="density" if density else "count",
        alpha=0.75,
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
        label=legend_labels[1],
    )
    if show_kde and len(pred) > 1:
        sns.kdeplot(pred, color=color_pred, ax=ax, linewidth=1.1, bw_adjust=0.8)

    mean_pred = float(np.mean(pred))
    std_pred = float(np.std(pred))
    ax.axvline(mean_pred, color=color_pred, linestyle="--", linewidth=1.2)
    results["mean_pred"] = mean_pred
    results["std_pred"] = std_pred

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density" if density else "Frequency")

    if integer_x_ticks:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if x_ticks is not None:
        ax.set_xticks(x_ticks)

    ax.legend(frameon=False)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = save_path.suffix.lstrip(".") or None
        fig.savefig(save_path, format=fmt)
        plt.close(fig)
    else:
        plt.show()

    return results


def plot_single_distribution_histogram(
    data: Iterable[float] | np.ndarray,
    *,
    xlabel: str,
    save_path: str | Path | None = None,
    title: str | None = None,
    bins: int | None = None,
    color: str = "#87ceeb",
    edgecolor: str = "black",
    density: bool = False,
    add_mean_line: bool = True,
    add_median_line: bool = True,
    integer_x_ticks: bool = False,
) -> dict[str, float]:
    apply_style(style="science", no_latex=True)

    values = _clean_numeric_array(data)
    if len(values) == 0:
        raise ValueError("Input data is empty after NaN removal.")

    if bins is None:
        bins = _adaptive_bin_count(len(values))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        values,
        bins=bins,
        alpha=0.75,
        color=color,
        edgecolor=edgecolor,
        linewidth=0.8,
        density=density,
    )

    mean_val = float(np.mean(values))
    std_val = float(np.std(values))
    median_val = float(np.median(values))

    if add_mean_line:
        ax.axvline(mean_val, color="red", linestyle="-", linewidth=2, label=f"Mean: {mean_val:.2f}")
    if add_median_line:
        ax.axvline(median_val, color="green", linestyle="--", linewidth=2, label=f"Median: {median_val:.2f}")

    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel("Density" if density else "Frequency", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=14)

    if add_mean_line or add_median_line:
        ax.legend(fontsize=10, frameon=False)

    if integer_x_ticks:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    stats_text = (
        f"Sample Size: {len(values)}\n"
        f"Mean: {mean_val:.2f}\n"
        f"Std Dev: {std_val:.2f}\n"
        f"Median: {median_val:.2f}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black", linewidth=1),
        fontsize=10,
    )

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
    else:
        plt.show()

    return {
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "n": float(len(values)),
    }
