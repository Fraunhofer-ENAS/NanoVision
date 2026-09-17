from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def save_pixel_metrics_debug_figure(
    image_name: str,
    *,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    true_binary: np.ndarray,
    pred_binary: np.ndarray,
    debug_dir: str,
) -> None:
    """Save a compact debug panel for pixel-metrics inputs.

    Panels: GT labels, Pred labels, GT binary, Pred binary.
    """
    os.makedirs(debug_dir, exist_ok=True)
    stem = Path(str(image_name)).stem
    out_path = os.path.join(debug_dir, f"{stem}_pixel_metrics_debug.png")

    fig, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)

    axes[0].imshow(true_labels, cmap="nipy_spectral")
    axes[0].set_title("GT Labels")
    axes[0].axis("off")

    axes[1].imshow(pred_labels, cmap="nipy_spectral")
    axes[1].set_title("Pred Labels")
    axes[1].axis("off")

    axes[2].imshow(true_binary, cmap="gray")
    axes[2].set_title("GT Binary")
    axes[2].axis("off")

    axes[3].imshow(pred_binary, cmap="gray")
    axes[3].set_title("Pred Binary")
    axes[3].axis("off")

    fig.suptitle(stem, fontsize=10)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
