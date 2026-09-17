from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import (
    find_boundaries,
)
from cnt_project.features.core.cnt_feature_extraction import (
    calculate_cnt_features_from_annotations,
    decode_annotation_mask,
)
from cnt_project.features.plotting.utils import (
    slugify_plot_name,
)



def _image_shape(
    image_record: dict[str, Any],
    default_h: int,
    default_w: int,
) -> tuple[int, int]:
    h = int(image_record.get("height", default_h))
    w = int(image_record.get("width", default_w))
    return h, w


def _load_prediction_coco(
    pred_json_path: Path,
) -> dict[str, Any]:
    with pred_json_path.open("r", encoding="utf-8") as f:
        return json.load(f)



def _load_image_or_blank(image_path: Path, image_height: int, image_width: int) -> tuple[np.ndarray, bool]:
    if image_path.exists():
        img = plt.imread(image_path)
        return img, True

    return np.zeros((image_height, image_width), dtype=np.float32), False


def _resolve_image_path(image_name: str, image_roots: list[Path]) -> Path | None:
    image_rel = Path(image_name)

    # 1) Try as-is if prediction JSON stores an absolute path.
    if image_rel.is_absolute() and image_rel.exists():
        return image_rel

    # 2) Try each known image root with the stored relative path.
    for root in image_roots:
        candidate = root / image_rel
        if candidate.exists():
            return candidate

    # 2b) If JSON stores names without extension, try common image extensions.
    if image_rel.suffix == "":
        ext_candidates = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]
        for root in image_roots:
            for ext in ext_candidates:
                candidate = root / f"{image_rel.name}{ext}"
                if candidate.exists():
                    return candidate

    # 3) Fallback: search by basename under each image root.
    basename = image_rel.name
    for root in image_roots:
        if not root.exists():
            continue
        matches = list(root.rglob(basename))
        if matches:
            return matches[0]

    # 4) Last resort: search by stem regardless of extension.
    stem = image_rel.stem if image_rel.suffix else image_rel.name
    for root in image_roots:
        if not root.exists():
            continue
        matches = list(root.rglob(f"{stem}.*"))
        if matches:
            # Prefer typical image formats first.
            preferred = sorted(
                matches,
                key=lambda p: p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"},
            )
            return preferred[0]

    return None



def _compute_owner_row(
    anns: list[dict[str, Any]],
    image_height: int,
    image_width: int,
    row_index: int,
) -> tuple[np.ndarray, int]:
    owner = np.full(image_width, fill_value=-1, dtype=int)
    overlap_count = 0

    row_clamped = int(np.clip(row_index, 0, image_height - 1))

    for obj_idx, ann in enumerate(anns):
        mask = decode_annotation_mask(ann, image_height, image_width)
        cols = np.flatnonzero(mask[row_clamped])

        if cols.size == 0:
            continue

        for col in cols.tolist():
            if owner[col] == -1:
                owner[col] = obj_idx
            elif owner[col] != obj_idx:
                owner[col] = -2
                overlap_count += 1

    return owner, overlap_count


def _plot_object_overlay_debug(
    *,
    image_array: np.ndarray,
    image_found: bool,
    image_name: str,
    anns: list[dict[str, Any]],
    metrics: dict[str, Any],
    image_height: int,
    image_width: int,
    max_objects: int,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    ax.imshow(image_array, cmap="gray")

    if not image_found:
        ax.text(
            0.01,
            0.99,
            "Image file not found in configured dataset image roots. Showing blank canvas with predictions.",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="yellow",
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 3},
        )

    area_values = metrics.get("area_pixels2", [])
    orientation_values = metrics.get("orientation_angle", [])
    length_values = metrics.get("length", [])
    width_values = metrics.get("width", [])

    n_objects = int(metrics.get("num_polygons", 0))

    sort_indices = list(range(min(n_objects, len(anns))))

    if isinstance(area_values, list) and len(area_values) >= len(sort_indices):
        sort_indices = sorted(
            sort_indices,
            key=lambda i: float(area_values[i]) if np.isfinite(area_values[i]) else -1.0,
            reverse=True,
        )

    shown_indices = sort_indices[:max_objects]
    cmap = plt.get_cmap("tab20")

    for color_idx, obj_idx in enumerate(shown_indices):
        ann = anns[obj_idx]

        mask = decode_annotation_mask( ann, image_height, image_width, ).astype(bool)

        if not mask.any():
            continue

        color = cmap( color_idx % 20 )

        # Translucent object fill.
        fill_overlay = np.zeros( (*mask.shape, 4), dtype=float, )

        fill_overlay[mask] = [
            color[0],
            color[1],
            color[2],
            0.25,
        ]

        ax.imshow( fill_overlay, interpolation="nearest", )

        # Solid contour extracted from the decoded mask.
        ax.contour( mask.astype(float), levels=[0.5], colors=[color], linewidths=1.6, )

        ys, xs = np.where(mask)

        cx = float(np.mean(xs))
        cy = float(np.mean(ys))

        angle_deg = float(orientation_values[obj_idx]) if isinstance(orientation_values, list) and obj_idx < len(orientation_values) else np.nan
        length_px = float(length_values[obj_idx]) if isinstance(length_values, list) and obj_idx < len(length_values) else np.nan
        width_px = float(width_values[obj_idx]) if isinstance(width_values, list) and obj_idx < len(width_values) else np.nan
        area_px2 = float(area_values[obj_idx]) if isinstance(area_values, list) and obj_idx < len(area_values) else np.nan

        if np.isfinite(angle_deg) and np.isfinite(length_px) and length_px > 0:
            half_len = max(4.0, 0.5 * length_px)
            dx = np.cos(np.deg2rad(angle_deg)) * half_len
            dy = -np.sin(np.deg2rad(angle_deg)) * half_len
            ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color="cyan", linewidth=1.3)

        label = (
            f"#{obj_idx} | A={area_px2:.1f}px² | L={length_px:.1f}px | "
            f"W={width_px:.2f}px | θ={angle_deg:.1f}°"
        )
        ax.text(
            cx,
            cy,
            label,
            fontsize=7,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.5},
        )

    title = (
        f"CNT debug overlay - {image_name}\n"
        f"Showing top {len(shown_indices)} objects by area"
    )
    ax.set_title(title)
    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _plot_line_density_row_debug(
    *,
    image_array: np.ndarray,
    image_name: str,
    anns: list[dict[str, Any]],
    image_height: int,
    image_width: int,
    line_density: list[int],
    owner_row: np.ndarray,
    row_index: int,
    overlap_count: int,
    out_path: Path,
) -> None:
    """
    Plot predicted objects and ownership along one image row.
    """
    row_clamped = int(
        np.clip(
            row_index,
            0,
            image_height - 1,
        )
    )

    density_on_row = (
        int(line_density[row_clamped])
        if row_clamped < len(line_density)
        else 0
    )

    unique_obj_ids = sorted(
        set(
            owner_row[
                owner_row >= 0
            ].tolist()
        )
    )

    number_of_objects = len(anns)

    color_positions = (
        np.arange(
            max(number_of_objects, 1),
            dtype=float,
        )
        * 0.618033988749895
    ) % 1.0

    object_colors = plt.cm.hsv(
        color_positions
    )[:, :3]

    figure, (image_axis, row_axis) = plt.subplots(
        2,
        1,
        figsize=(10.0, 6.5),
        gridspec_kw={
            "height_ratios": [4.0, 1.2],
        },
        sharex=True,
    )

    image_axis.imshow(
        image_array,
        cmap=(
            "gray"
            if image_array.ndim == 2
            else None
        ),
    )

    prediction_overlay = np.zeros(
        (
            image_height,
            image_width,
            4,
        ),
        dtype=float,
    )

    for object_index, annotation in enumerate(
        anns
    ):
        mask = decode_annotation_mask(
            annotation,
            image_height,
            image_width,
        ).astype(bool)

        if not mask.any():
            continue

        color = object_colors[
            object_index
        ]

        # Translucent object fill.
        prediction_overlay[mask] = [
            color[0],
            color[1],
            color[2],
            0.25,
        ]

        # Solid object contour.
        boundary = find_boundaries(
            mask,
            mode="outer",
        )

        prediction_overlay[boundary] = [
            color[0],
            color[1],
            color[2],
            1.0,
        ]

    image_axis.imshow(
        prediction_overlay,
        interpolation="nearest",
    )

    image_axis.axhline(
        y=row_clamped,
        color="white",
        linestyle="--",
        linewidth=2.0,
    )

    image_axis.set_title(
        f"Line-density debug — {image_name}\n"
        f"row={row_clamped}; "
        f"objects crossing row={density_on_row}"
    )

    image_axis.set_ylabel("y (pixels)")

    # Use the same colors in the row-ownership panel.
    row_display = np.zeros(
        (
            30,
            image_width,
            3,
        ),
        dtype=float,
    )

    row_display[:] = [
        0.90,
        0.90,
        0.90,
    ]

    for column, owner_index in enumerate(
        owner_row
    ):
        if owner_index >= 0:
            row_display[
                :,
                column,
            ] = object_colors[
                owner_index
            ]

        elif owner_index == -2:
            # A black column represents overlapping objects.
            row_display[
                :,
                column,
            ] = [
                0.0,
                0.0,
                0.0,
            ]

    row_axis.imshow(
        row_display,
        aspect="auto",
        interpolation="nearest",
    )

    row_axis.set_yticks([])
    row_axis.set_xlabel("x (pixels)")

    row_axis.set_title(
        "Object ownership along selected row "
        f"(unique objects={len(unique_obj_ids)}, "
        f"overlap pixels={overlap_count})"
    )

    figure.tight_layout()

    figure.savefig(
        out_path,
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(figure)

def build_prediction_feature_debug_visualizations(
    *,
    pred_json_path: Path,
    output_dir: Path,
    image_roots: list[Path],
    default_image_height: int,
    default_image_width: int,
    max_images: int,
    max_objects_per_image: int,
    row_index: int | None,
) -> list[str]:
    pred_coco = _load_prediction_coco(pred_json_path)
    images = pred_coco.get("images", [])
    annotations = pred_coco.get("annotations", [])

    image_id_to_record = {
        int(im["id"]): im
        for im in images
        if "id" in im
    }

    anns_by_image: dict[int, list[dict[str, Any]]] = {}

    for ann in annotations:
        image_id = int(ann.get("image_id", -1))

        if image_id < 0:
            continue

        anns_by_image.setdefault(image_id, []).append(ann)

    ranked_items = sorted(
        anns_by_image.items(),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )

    selected_items = ranked_items[: max(0, max_images)]

    overlay_dir = output_dir / "debug_visualizations" / "object_overlays"
    row_dir = output_dir / "debug_visualizations" / "line_density_rows"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    row_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []

    for image_id, anns in selected_items:
        image_record = image_id_to_record.get(image_id, {})
        h, w = _image_shape(image_record, default_image_height, default_image_width)
        image_name = str(image_record.get("file_name", image_id))
        image_path = _resolve_image_path(image_name, image_roots)

        if image_path is not None:
            image_array, found_image = _load_image_or_blank(image_path, h, w)
        else:
            image_array, found_image = np.zeros((h, w), dtype=np.float32), False

        metrics = calculate_cnt_features_from_annotations(
            anns,
            image_height=h,
            image_width=w,
        )

        stem = slugify_plot_name(Path(image_name).stem)

        overlay_path = overlay_dir / f"{stem}_overlay.png"
        _plot_object_overlay_debug(
            image_array=image_array,
            image_found=found_image,
            image_name=image_name,
            anns=anns,
            metrics=metrics,
            image_height=h,
            image_width=w,
            max_objects=max_objects_per_image,
            out_path=overlay_path,
        )
        saved_paths.append(str(overlay_path))

        line_density = metrics.get("line_density", [])
        if row_index is None:
            if isinstance(line_density, list) and len(line_density) > 0:
                chosen_row = int(np.argmax(np.asarray(line_density, dtype=float)))
            else:
                chosen_row = h // 2
        else:
            chosen_row = int(row_index)

        owner_row, overlap_count = _compute_owner_row(
            anns,
            image_height=h,
            image_width=w,
            row_index=chosen_row,
        )

        row_path = row_dir / f"{stem}_row_{int(np.clip(chosen_row, 0, h - 1))}.png"
        _plot_line_density_row_debug(
            image_array=image_array,
            image_name=image_name,
            anns=anns,
            image_height=h,
            image_width=w,
            line_density=(
                line_density
                if isinstance(line_density, list)
                else []
            ),
            owner_row=owner_row,
            row_index=chosen_row,
            overlap_count=overlap_count,
            out_path=row_path,
        )
        saved_paths.append(str(row_path))

    return saved_paths
