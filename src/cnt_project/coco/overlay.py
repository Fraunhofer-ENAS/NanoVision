from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mask_utils


def overlay_annotations_on_ax(
    ax: plt.Axes,
    anns: list[dict[str, object]],
    *,
    alpha: float = 0.5,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)

    for ann in anns:
        segs = ann.get("segmentation")
        if segs is None:
            continue

        # One random color per annotation/CNT.
        color = rng.random(3)

        if isinstance(segs, list):
            for seg in segs:
                poly = np.asarray(seg, dtype=float).reshape((-1, 2))

                patch = plt.Polygon(
                    poly,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=2,
                    alpha=alpha,
                )

                ax.add_patch(patch)

        elif isinstance(segs, dict):
            decoded = mask_utils.decode(segs)

            # Create an RGBA overlay using the annotation's color.
            rgba = np.zeros((*decoded.shape, 4), dtype=float)
            rgba[..., :3] = color
            rgba[..., 3] = decoded.astype(float) * alpha

            ax.imshow(rgba)


def save_image_with_annotations(
    *,
    image_rgb: np.ndarray,
    anns: list[dict[str, object]],
    out_path: str | Path,
    alpha: float = 0.5,
    dpi: int = 150,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.imshow(image_rgb)
    overlay_annotations_on_ax(ax, anns, alpha=alpha)
    ax.axis("off")
    plt.tight_layout()

    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)