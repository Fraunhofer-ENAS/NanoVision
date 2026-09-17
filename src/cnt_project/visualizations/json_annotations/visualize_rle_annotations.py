from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mask_utils


def _decode_rle(segmentation: dict) -> np.ndarray:
    rle = dict(segmentation)

    # pycocotools can handle string counts in many cases,
    # but encoding to bytes is safer for compressed RLE.
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("utf-8")

    decoded = mask_utils.decode(rle)

    if decoded.ndim == 3:
        decoded = np.any(decoded > 0, axis=2)

    return decoded.astype(bool)


def rle_json_to_instance_mask(
    coco_json_path: str | Path,
    *,
    image_stem: str,
    sort_by_score: bool = True,
) -> np.ndarray:
    coco_json_path = Path(coco_json_path)

    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    image_info = next(
        img for img in coco["images"]
        if Path(img["file_name"]).stem == image_stem
    )

    image_id = image_info["id"]
    height = int(image_info["height"])
    width = int(image_info["width"])

    anns = [
        ann for ann in coco["annotations"]
        if ann["image_id"] == image_id
    ]

    if sort_by_score:
        anns = sorted(anns, key=lambda a: float(a.get("score", 1.0)))

    instance_mask = np.zeros((height, width), dtype=np.uint16)

    for inst_id, ann in enumerate(anns, start=1):
        seg = ann.get("segmentation")

        if not isinstance(seg, dict):
            continue

        decoded = _decode_rle(seg)
        instance_mask[decoded] = inst_id

    return instance_mask


def save_rle_overlay_for_image(
    *,
    coco_json_path: str | Path,
    image_dir: str | Path,
    image_stem: str,
    out_path: str | Path,
    image_ext: str = ".tif",
    alpha: float = 0.5,
    sort_by_score: bool = True,
) -> None:
    image_dir = Path(image_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / f"{image_stem}{image_ext}"

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    if image.ndim == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    instance_mask = rle_json_to_instance_mask(
        coco_json_path,
        image_stem=image_stem,
        sort_by_score=sort_by_score,
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_rgb, cmap="gray" if image_rgb.ndim == 2 else None)
    ax.imshow(
        np.ma.masked_where(instance_mask == 0, instance_mask),
        cmap="nipy_spectral",
        alpha=alpha,
    )
    ax.axis("off")
    plt.tight_layout()

    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_rle_overlays(
    *,
    coco_json_path: str | Path,
    image_dir: str | Path,
    out_dir: str | Path,
    image_ext: str = ".tif",
    alpha: float = 0.5,
    sort_by_score: bool = True,
) -> None:
    coco_json_path = Path(coco_json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    for image_info in coco["images"]:
        image_stem = Path(image_info["file_name"]).stem
        out_path = out_dir / f"{image_stem}_rle_overlay.png"

        try:
            save_rle_overlay_for_image(
                coco_json_path=coco_json_path,
                image_dir=image_dir,
                image_stem=image_stem,
                out_path=out_path,
                image_ext=image_ext,
                alpha=alpha,
                sort_by_score=sort_by_score,
            )
            print("Saved:", out_path)
        except FileNotFoundError as exc:
            print("Warning:", exc)