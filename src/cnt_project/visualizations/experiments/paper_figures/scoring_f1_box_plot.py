from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from .constants import COLOR_MAP, FIG_SIZES
from .constants import SCORING_COLOR_MAP
from pathlib import Path
# enforce scienceplots if available
try:
    import scienceplots  
    plt.style.use(['science', 'no-latex'])
except Exception:
    pass


# Use the paper double-half figure size for compact side-by-side panels.
FIGSIZE_FULL = FIG_SIZES.get("double_half", (3.3, 2.5))
FIGSIZE_STAGE_COMPARISON = FIG_SIZES.get("single_full", (6.0, 4.0))

def save_svg_png(
    fig: plt.Figure,
    output_dir: Path,
    filename_stem: str,
    *,
    png_dpi: int = 600,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    svg_path = output_dir / f"{filename_stem}.svg"
    png_path = output_dir / f"{filename_stem}.png"

    fig.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    fig.savefig(
        png_path,
        format="png",
        dpi=png_dpi,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    print(f"Saved: {svg_path}")
    print(f"Saved: {png_path}")


def plot_horizontal_f1_comparison(
    df: pd.DataFrame,
    *,
    scoring_order: list[str],
    output_dir: Path,
    xlabel: str,
    filename_stem: str,
    figsize=FIGSIZE_FULL,
    xlim=(0, 1),
):
    summary = (
        df[df["scoring"].isin(scoring_order)]
        .groupby("scoring", as_index=False)["value"]
        .mean()
    )

    summary["scoring"] = pd.Categorical(
        summary["scoring"],
        categories=scoring_order,
        ordered=True,
    )
    summary = summary.sort_values("scoring")

    plt.figure(figsize=figsize)
    plt.style.use(["science", "no-latex"])

    colors = [SCORING_COLOR_MAP.get(str(s), "#4169E1") for s in summary["scoring"]]

    plt.barh(
        summary["scoring"].astype(str),
        summary["value"],
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )

    for i, value in enumerate(summary["value"]):
        plt.text(
            min(value + 0.015, xlim[1] - 0.02),
            i,
            f"{100 * value:.1f}%",
            va="center",
            # fontsize=8,
        )

    # plt.xlabel(xlabel)
    plt.ylabel("")
    plt.xlim(*xlim)
    # plt.grid(axis="x", linestyle="--", alpha=0.25)
    plt.tight_layout()

    fig = plt.gcf()
    save_svg_png(
        fig,
        output_dir=output_dir,
        filename_stem=filename_stem,
    )
    plt.show()


def plot_edt_sym_stage_comparison(
    *,
    stage1_pixel_mean: float,
    stage2_pixel_mean: float,
    stage2_object_mean: float,
    output_dir: Path,
    filename_stem: str = "barh_edt_sym_stage1_vs_stage2_f1",
    figsize=(5.8, 2.7),
    
):
    plt.style.use(["science", "no-latex"])

    y_positions = [3.2, 2.4, 1.1, 0.3]

    metric_labels = [
        "Pixel F1",
        "Object F1@0.05",
        "Pixel F1",
        "Object F1@0.05",
    ]

    values = [
        stage1_pixel_mean,
        0.99,  # visual placeholder only
        stage2_pixel_mean,
        stage2_object_mean,
    ]

    display_values = [
        stage1_pixel_mean,
        None,
        stage2_pixel_mean,
        stage2_object_mean,
    ]

    colors = [
        "#B88A63",   # Stage 1
        "#D8D8D8",   # N/A

        "#4F81BD",   # Stage 2
        "#4F81BD",   # Stage 2
    ]


    hatches = [
        "",
        "///",
        "",
        "",
    ]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(
        y_positions,
        values,
        height=0.52,
        color=colors,
        edgecolor="#333333",
        linewidth=0.6,
        alpha=0.95,
    )

    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(metric_labels)

    # Push stage labels further left
    ax.text(
        -0.42,
        np.mean(y_positions[:2]),
        "StarDist output\nbefore merging",
        ha="right",
        va="center",
        # fontsize=8.5,
        fontweight="bold",
        transform=ax.get_yaxis_transform(),
    )

    ax.text(
        -0.42,
        np.mean(y_positions[2:]),
        "Reconstructed output\nafter merging",
        ha="right",
        va="center",
        # fontsize=8.5,
        fontweight="bold",
        transform=ax.get_yaxis_transform(),
    )

    ax.axhline(
        y=1.75,
        color="#BDBDBD",
        linewidth=0.8,
        linestyle="-",
        alpha=0.8,
    )

    for bar, shown_val in zip(bars, display_values):
        y = bar.get_y() + bar.get_height() / 2.0
        width = bar.get_width()

        if shown_val is None:
            ax.text(
                0.495,
                y,
                "N/A fragments only",
                va="center",
                ha="center",
                # fontsize=8.2,
                color="#000000",
                fontstyle="italic",
                fontweight="bold",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.75,
                    boxstyle="round,pad=0.15",
                ),
            )
        else:
            ax.text(
                min(width + 0.018, 0.98),
                y,
                f"{100 * shown_val:.1f}%",
                va="center",
                ha="left",
                # fontsize=8,
                color="#222222",
            )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean F1 score")
    ax.set_ylabel("")

    # ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", direction="in")

    # Gives room for stage labels on the left
    plt.subplots_adjust(left=0.34, right=0.98, bottom=0.22, top=0.95)

    save_svg_png(fig, output_dir=output_dir, filename_stem=filename_stem)
    plt.show()