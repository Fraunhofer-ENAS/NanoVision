import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Literal

def plot_ap_iou_curves(
    df: pd.DataFrame,
    color_map: Dict[str, str],
    legend_order: list[str],
    figsize=(3.3, 2.5),
    save_path: str | None = None,
    average: Literal["macro", "mean_counts"] = "macro",
):
    """
    Plot AP (precision) vs IoU threshold.
    df must contain:
        experiment (pretty name: NanoVision, MaskRCNN, ...)
        IoU_thresh
        TP, FP, FN   (precision already in dataframe)

    average:
      - 'macro':
            mean per-image precision in each (experiment, IoU) group.
      - 'mean_counts':
            mean TP/FP per image in each group, then precision from mean counts.
    """

    plt.figure(figsize=figsize)

    required = {"experiment", "IoU_thresh", "TP", "FP", "FN"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    if average == "macro":
        if "precision" in df.columns:
            mean_df = (
                df.groupby(["experiment", "IoU_thresh"])[["precision"]]
                .mean()
                .reset_index()
            )
        else:
            tmp = df.copy()
            tmp["precision"] = tmp["TP"] / (tmp["TP"] + tmp["FP"]).replace(0, np.nan)
            mean_df = (
                tmp.groupby(["experiment", "IoU_thresh"])[["precision"]]
                .mean()
                .reset_index()
            )
    elif average == "mean_counts":
        mean_counts = (
            df.groupby(["experiment", "IoU_thresh"])[["TP", "FP"]]
            .mean()
            .reset_index()
        )
        mean_counts["precision"] = mean_counts["TP"] / (mean_counts["TP"] + mean_counts["FP"]).replace(0, np.nan)
        mean_df = mean_counts[["experiment", "IoU_thresh", "precision"]]
    else:
        raise ValueError("average must be one of: 'macro', 'mean_counts'")

    # Plot precision curves
    for exp in legend_order:
        sub = mean_df[mean_df["experiment"] == exp]
        if sub.empty:
            continue

        plt.plot(
            sub["IoU_thresh"],
            sub["precision"],
            label=exp,
            color=color_map.get(exp, "#000000"),
        )

    # Axis labels and formatting
    plt.xlabel("IoU Threshold")
    plt.ylabel("AP")
    plt.yticks(np.arange(0, 0.7, 0.1))
    plt.xticks(np.arange(0, 0.85, 0.2))
    plt.grid(False)

    plt.legend(title="Models", loc="upper right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.02)

    plt.show()
