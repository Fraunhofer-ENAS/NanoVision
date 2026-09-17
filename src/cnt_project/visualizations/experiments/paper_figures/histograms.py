# src/plotting/histograms.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from typing import Sequence

# try:
#     from ..utils.constants import COLOR_MAP, FIG_SIZES
# except ModuleNotFoundError:
#     COLOR_MAP = {"NanoVision": "#4169e1"}
#     FIG_SIZES = {"hist_fig": (8, 6)}

from .constants import COLOR_MAP, FIG_SIZES

# ---------------------------------------------------
# CNT count log-scale histogram
# ---------------------------------------------------
def plot_cnt_per_image_histogram_log(
    counts,
    q1=None,
    q2=None,
    bins=60,
    figsize=FIG_SIZES["hist_fig"],
    save_path=None,
):
    """
    EXACT reproduction of your original histogram:
    - plt.hist (not seaborn)
    - blue bars (#4169e1)
    - black edges
    - alpha=0.7
    - log-spaced bins
    - red threshold lines q1, q2
    - SciencePlots style
    """
    x = np.asarray(counts)
    x = x[x > 0]

    # Log-spaced bins 
    bin_edges = np.logspace(
        np.log10(x.min()),
        np.log10(x.max()),
        bins
    )

    plt.figure(figsize=figsize)
    print("Current font sizes:")
    print(f"Font size: {plt.rcParams['font.size']}")
    print(f"Axes labelsize: {plt.rcParams['axes.labelsize']}")
    print(f"XTick labelsize: {plt.rcParams['xtick.labelsize']}")
    print(f"YTick labelsize: {plt.rcParams['ytick.labelsize']}")
    plt.hist(
        x,
        bins=bin_edges,
        color="#4169e1",
        edgecolor="black",
        alpha=0.7
    )

    # Red dashed threshold lines
    if q1 is not None:
        plt.axvline(q1, color="red", linestyle="--", linewidth=2)
    if q2 is not None:
        plt.axvline(q2, color="red", linestyle="--", linewidth=2)

    plt.xscale("log")
    plt.xlabel("Number of CNTs per image (log scale)")
    plt.ylabel("Number of images")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.02)

    plt.show()


# ---------------------------------------------------
# Line density per micron (linear scale)
# ---------------------------------------------------
def plot_line_density_histogram(
    densities: Sequence[float],
    bins: int = 30,
    figsize: tuple = FIG_SIZES["hist_fig"],
    save_path: str | None = None,
    xlabel: str = "Mean Line Density (CNT/$\\mu$m)",
    ylabel: str = "Number of images"
):
    """Linear histogram of CNT/μm per image."""
    x = np.asarray(densities)
    q1 = np.percentile(x, 33)
    q2 = np.percentile(x, 66)
    plt.figure(figsize=figsize)
    print("Current font sizes:")
    print(f"Font size: {plt.rcParams['font.size']}")
    print(f"Axes labelsize: {plt.rcParams['axes.labelsize']}")
    print(f"XTick labelsize: {plt.rcParams['xtick.labelsize']}")
    print(f"YTick labelsize: {plt.rcParams['ytick.labelsize']}")
    sns.histplot(
        x,
        bins=bins,
        color=COLOR_MAP["NanoVision"],
        edgecolor="black",
        alpha=0.7,
    )
    # Add red threshold lines
    plt.axvline(q1, color='red', linestyle='--', linewidth=2, label=f'33rd percentile: {q1:.3f}')
    plt.axvline(q2, color='red', linestyle='--', linewidth=2, label=f'66th percentile: {q2:.3f}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.02)

    plt.show()


# ---------------------------------------------------
# Line density per micron (log scale)
# ---------------------------------------------------
def plot_line_density_histogram_log(
    densities: Sequence[float],
    bins: int = 42,
    figsize: tuple = FIG_SIZES["hist_fig"],
    save_path: str | None = None,
    xlabel: str = "Mean Line Density (CNT/$\\mu$m, log scale)",
    ylabel: str = "Number of images"
):
    """Log-scale histogram of CNT/μm per image."""
    x = np.asarray(densities)
    x = x[x > 0]  # log scale cannot show zero or negative

    # Compute tertiles
    q1 = np.percentile(x, 33)
    q2 = np.percentile(x, 66)

    # Log-spaced bins
    bin_edges = np.logspace(
        np.log10(x.min()),
        np.log10(x.max()),
        bins
    )

    plt.figure(figsize=figsize)

    plt.hist(
        x, bins=bin_edges,
        color=COLOR_MAP["NanoVision"],
        edgecolor="black",
        alpha=0.7
    )

    # Red tertile markers
    plt.axvline(q1, color='red', linestyle='--', linewidth=2)
    plt.axvline(q2, color='red', linestyle='--', linewidth=2)

    # ----- Improved log-scale ticks (5 ticks) -----
    xmin, xmax = x.min(), x.max()
    xticks = np.logspace(
        np.log10(xmin),
        np.log10(xmax),
        5   # <<— EXACTLY 5 TICKS
    )
    plt.xscale("log")
    plt.xticks(xticks, [f"{t:.1f}" for t in xticks])  # clean numeric labels

    # Labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.02)

    plt.show()



# ---------------------------------------------------
# Noise destribution
# ---------------------------------------------------
def plot_noise_histogram(
    noise_values,
    bins=42,
    threshold=14,
    figsize=(4.3, 2.5),
    save_path=None
):
    """
    EXACT reproduction of original noise histogram:
    - bins = 42
    - bar color = royal blue (#4169e1)
    - black edges, alpha = 0.7
    - red threshold line at 14
    - SciencePlots style
    """

    plt.figure(figsize=figsize)

    # Print current font sizes (same as your original debug)
    print("Current font sizes:")
    print("Font size:", plt.rcParams['font.size'])
    print("Axis labelsize:", plt.rcParams['axes.labelsize'])
    print("XTick labelsize:", plt.rcParams['xtick.labelsize'])
    print("YTick labelsize:", plt.rcParams['ytick.labelsize'])

    plt.hist(
        noise_values,
        bins=bins,
        color="#4169e1",
        edgecolor="black",
        alpha=0.7
    )

    # Fixed threshold line at 14
    plt.axvline(threshold, color="red", linestyle="--", linewidth=2)

    plt.xlabel(r"Noise ($\sigma$)")
    plt.ylabel("Number of images")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="svg", dpi=300,
                    bbox_inches="tight", pad_inches=0.02)

    plt.show()
