from __future__ import annotations


import numpy as np
import matplotlib.pyplot as plt


def save_image_with_mask(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    save_path: str,
    alpha: float = 0.5,
) -> None:
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
    ax.imshow(np.ma.masked_where(mask == 0, mask), cmap="jet", alpha=alpha)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
