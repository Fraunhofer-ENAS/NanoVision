import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Literal

def plot_f1_iou_curves(
    df: pd.DataFrame,
    name_map: Dict[str, str],
    color_map: Dict[str, str],
    legend_order: list[str],
    figsize=(3.3, 2.5),
    save_path: str | None = None,
    average: Literal['mean_counts', 'macro', 'micro'] = 'mean_counts',
):
    """
    Plot F1-score vs IoU threshold for multiple models.

    df columns expected:
      - experiment (already MAPPED to display names, e.g., 'NanoVision', 'MaskRCNN', ...)
      - IoU_thresh, TP, FP, FN

    average:
      - 'mean_counts' (matches your ORIGINAL notebook):
            mean TP/FP/FN per image → compute F1 from those means
      - 'macro':
            compute per-image F1 first → average F1 (macro-F1)
      - 'micro':
            sum TP/FP/FN across images → compute F1 from sums (true micro-F1)
    """
    plt.figure(figsize=figsize)

    # ensure required columns
    required = {'experiment', 'IoU_thresh', 'TP', 'FP', 'FN'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    # choose aggregation mode
    if average == 'mean_counts':
        # OLD behavior
        agg = (
            df.groupby(['experiment', 'IoU_thresh'])[['TP', 'FP', 'FN']]
            .mean()
            .reset_index()
        )
        agg['F1'] = 2 * agg['TP'] / (2 * agg['TP'] + agg['FP'] + agg['FN'])

    elif average == 'macro':
        tmp = df.copy()
        tmp['F1'] = 2 * tmp['TP'] / (2 * tmp['TP'] + tmp['FP'] + tmp['FN'])
        agg = (
            tmp.groupby(['experiment', 'IoU_thresh'])[['F1']]
            .mean()
            .reset_index()
        )

    elif average == 'micro':
        agg = (
            df.groupby(['experiment', 'IoU_thresh'])[['TP', 'FP', 'FN']]
            .sum()
            .reset_index()
        )
        agg['F1'] = 2 * agg['TP'] / (2 * agg['TP'] + agg['FP'] + agg['FN'])

    else:
        raise ValueError("average must be one of: 'mean_counts', 'macro', 'micro'")

    # plot in the specified legend order
    for label in legend_order:
        sub = agg[agg['experiment'] == label]
        if sub.empty:
            continue
        plt.plot(
            sub['IoU_thresh'],
            sub['F1'],
            label=label,
            color=color_map.get(label, '#000000'),
        )

    plt.xlabel("IoU Threshold")
    plt.ylabel("F1-score")
    ax = plt.gca()
    y_ticks = np.arange(0.0, 0.9, 0.1)
    ax.set_ylim(0.0, 0.9)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y:.1f}" for y in y_ticks])
    plt.grid(False)
    plt.legend(title="Models", loc="upper right", handlelength=1.5, handleheight=1.0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.02)

    plt.show()
