from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_mean_cnt_count_by_density(
    summary_df,
    *,
    save_path: str | Path | None = None,
    figsize=(4.3, 2.5),
    ylabel: str = "Mean CNT count per image",
    show: bool = True,
):
    plt.figure(figsize=figsize)
    plt.style.use(["science", "no-latex"])

    bar_color = "#4169e1"

    bars = plt.bar(
        summary_df["density_display"],
        summary_df["mean_cnt_count"],
        yerr=summary_df["std_cnt_count"],
        capsize=3,
        width=0.65,
        color=bar_color,
        edgecolor="black",
        alpha=0.7,
    )

    handles = [
        Patch(
            facecolor=bar_color,
            edgecolor="black",
            alpha=0.7,
            label=f"{row['density_display']}: n={int(row['n_images'])}",
        )
        for _, row in summary_df.iterrows()
    ]

    plt.legend(
        handles=handles,
        loc="best",
        frameon=True,
        title="Images",
    )

    plt.xlabel("")
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.2)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.02)

    if show:
        plt.show()
    else:
        plt.close()