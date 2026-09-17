from __future__ import annotations

from typing import Any

import numpy as np
from pycocotools import mask as maskUtils
from cnt_project.features.core.cnt_feature_extraction import (
    MICRONS_PER_PIXEL,
)


def _normalize_angle_minus90_90(angle: float) -> float:
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    return float(angle)

def _polygon_pca_orientation_from_annotation(ann: dict[str, Any]) -> float:
    """
    Orientation from original COCO polygon coordinates.

    Convention:
    - 0° = horizontal, left-to-right
    - positive = downward in image coordinates
    - range = [-90°, 90°]
    """
    seg = ann.get("segmentation", [])

    if not isinstance(seg, list):
        return np.nan

    coords_all = []

    for poly in seg:
        if not isinstance(poly, list) or len(poly) < 4:
            continue

        coords = np.asarray(poly, dtype=float).reshape(-1, 2)  # x, y
        if coords.shape[0] >= 2:
            coords_all.append(coords)

    if not coords_all:
        return np.nan

    coords = np.vstack(coords_all)

    if coords.shape[0] < 2:
        return np.nan

    coords_centered = coords - coords.mean(axis=0, keepdims=True)

    cov = np.cov(coords_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    major_axis = eigvecs[:, np.argmax(eigvals)]

    dx, dy = major_axis[0], major_axis[1]
    angle = np.degrees(np.arctan2(dy, dx))

    return _normalize_angle_minus90_90(float(angle))


def _decode_annotation_mask(
    ann: dict[str, Any],
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """Turns a COCO annotation into a binary mask using pycocotools."""
    rles = maskUtils.frPyObjects(ann["segmentation"], image_height, image_width)
    rle = maskUtils.merge(rles)
    return maskUtils.decode(rle).astype(bool)


def _annotation_coords_xy(ann: dict[str, Any]) -> np.ndarray:
    """
    Return all segmentation coordinates as an (N, 2) array in x,y order.

    For normal COCO polygons this gives polygon boundary coordinates.
    For Nano1D this gives centerline/polyline coordinates.
    """
    seg = ann.get("segmentation", [])

    if not isinstance(seg, list):
        return np.empty((0, 2), dtype=float)

    coords_all = []

    for poly in seg:
        if not isinstance(poly, list) or len(poly) < 4:
            continue

        coords = np.asarray(poly, dtype=float).reshape(-1, 2)

        if coords.shape[0] >= 2:
            coords_all.append(coords)

    if not coords_all:
        return np.empty((0, 2), dtype=float)

    return np.vstack(coords_all)


def _centerline_polyline_length_from_annotation(
    ann: dict[str, Any],
    microns_per_pixel: float = MICRONS_PER_PIXEL,
) -> tuple[float, float]:
    """
    Compute Nano1D centerline length from ordered segments without retracing.

    Uses the same local segment-length formula as before:
        sum(||p_{i+1} - p_i||)
    but counts each undirected traced edge at most once. This removes A->B->A
    inflation while preserving geometric segment lengths.
    """
    seg = ann.get("segmentation", [])

    if not isinstance(seg, list):
        return np.nan, np.nan

    # key: undirected edge between rounded pixel nodes, value: edge length in px
    unique_edges: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}

    def _to_pixel_node(xy: np.ndarray) -> tuple[int, int]:
        return (int(round(float(xy[0]))), int(round(float(xy[1]))))

    for poly in seg:
        if not isinstance(poly, list) or len(poly) < 4:
            continue

        coords = np.asarray(poly, dtype=float).reshape(-1, 2)
        if coords.shape[0] < 2:
            continue

        prev_node = _to_pixel_node(coords[0])
        for i in range(1, coords.shape[0]):
            cur_node = _to_pixel_node(coords[i])
            if cur_node == prev_node:
                continue

            # Compute length in rounded pixel space to match node identity.
            dx = float(cur_node[0] - prev_node[0])
            dy = float(cur_node[1] - prev_node[1])
            w = float(np.hypot(dx, dy))
            if w <= 0:
                prev_node = cur_node
                continue

            a, b = (prev_node, cur_node) if prev_node <= cur_node else (cur_node, prev_node)
            old = unique_edges.get((a, b))
            if old is None or w < old:
                unique_edges[(a, b)] = w
            prev_node = cur_node

    if not unique_edges:
        return np.nan, np.nan

    length_px = float(sum(unique_edges.values()))
    if length_px <= 0:
        return np.nan, np.nan

    return length_px, length_px * microns_per_pixel

#Centerline-based morphology extractor for Nano1D-style predictions 
def calculate_centerline_properties_from_json(
    annotations: list[dict[str, Any]],
    image_height: int,
    image_width: int,
    visualize: bool = False,
) -> dict[str, Any]:
    """
    Centerline-specific morphology extractor for Nano1D-style predictions.

    One annotation gives one feature entry.

    Orientation:
        PCA on segmentation coordinates.

    Length:
        Polyline length through ordered segmentation coordinates.

    Notes:
        This function does not rely on rasterized masks for length/orientation.
        This is appropriate when COCO segmentation stores line/centerline points
        rather than closed filled object contours.
    """
    import matplotlib.pyplot as plt

    all_props: list[dict[str, float]] = []

    for ann in annotations:
        if "segmentation" not in ann:
            continue

        coords = _annotation_coords_xy(ann)

        if coords.shape[0] < 2:
            continue

        orientation_angle = _polygon_pca_orientation_from_annotation(ann)

        length_px, length_um = _centerline_polyline_length_from_annotation(
            ann,
            microns_per_pixel=MICRONS_PER_PIXEL,
        )

        if np.isnan(length_px) or length_px <= 0:
            continue

        # Optional mask-derived area only for compatibility.
        try:
            mask = _decode_annotation_mask(ann, image_height, image_width)
            area = float(mask.sum())
        except Exception:
            mask = None
            area = np.nan

        all_props.append(
            {
                "area": area,
                "perimeter": np.nan,
                "length": float(length_px),
                "length_um": float(length_um),
                "width": np.nan,
                "width_um": np.nan,
                "aspect_ratio": np.nan,
                "orientation_angle": float(orientation_angle),
            }
        )

        if visualize:
            fig, ax = plt.subplots(figsize=(4, 4))

            if mask is not None:
                ax.imshow(mask, cmap="gray", interpolation="nearest")
            else:
                ax.set_xlim(0, image_width)
                ax.set_ylim(image_height, 0)

            ax.plot(coords[:, 0], coords[:, 1], linewidth=1.5, label="centerline coords")

            cx = float(coords[:, 0].mean())
            cy = float(coords[:, 1].mean())

            line_len = max(5.0, float(length_px) / 2.0)
            dx = np.cos(np.deg2rad(orientation_angle)) * line_len
            dy = np.sin(np.deg2rad(orientation_angle)) * line_len

            ax.plot(
                [cx - dx, cx + dx],
                [cy - dy, cy + dy],
                linewidth=2,
                label="PCA orientation",
            )

            ax.set_title(
                f"Centerline length={length_px:.2f}px, "
                f"{length_um:.2f}µm, "
                f"Angle={orientation_angle:.1f}°"
            )
            ax.set_aspect("equal")
            ax.legend(fontsize=8)
            plt.show()

    if not all_props:
        return {"num_polygons": 0}

    keys = all_props[0].keys()
    output = {k: [d[k] for d in all_props] for k in keys}
    output["num_polygons"] = len(all_props)

    for k in keys:
        output[f"average_{k}"] = float(np.nanmean(output[k])) if output[k] else 0.0

    return output