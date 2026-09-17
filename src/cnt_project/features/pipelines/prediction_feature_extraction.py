from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cnt_project.features.core.cnt_feature_extraction import (
    calculate_cnt_features_from_annotations,
)


OBJECT_FEATURE_KEYS = [
    "area_pixels2",
    "area_um2",
    "perimeter",
    "perimeter_um",
    "length",
    "length_um",
    "tree_length",
    "tree_length_um",
    "width",
    "width_um",
    "aspect_ratio",
    "orientation_angle",
    "n_connected_components",
]


IMAGE_FEATURE_KEYS = [
    "cnt_density_per_um2",
    "line_density_mean",
    "line_density_std",
    "line_density_max",
    "nematic_order_parameter",
    "nematic_director_angle_deg",
    "von_mises_mean_orientation_deg",
    "von_mises_concentration_kappa",
    "von_mises_circular_variance",
    "von_mises_confidence_95_deg",
    "num_objects_for_fit",
]


def _image_shape(
    image_record: dict[str, Any],
    default_h: int,
    default_w: int,
) -> tuple[int, int]:
    h = int(image_record.get("height", default_h))
    w = int(image_record.get("width", default_w))
    return h, w


def _safe_series_value(
    values: Any,
    index: int,
) -> float:
    if not isinstance(values, list) or index >= len(values):
        return np.nan

    val = values[index]

    try:
        return float(val)
    except Exception:
        return np.nan


def _load_prediction_coco(
    pred_json_path: Path,
) -> dict[str, Any]:
    with pred_json_path.open( "r", encoding="utf-8-sig", ) as file:
        return json.load(file)


def extract_features_from_prediction_json(
    pred_json_path: Path,
    *,
    default_image_height: int = 256,
    default_image_width: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract canonical CNT features from a prediction COCO JSON.

    Returns
    -------
    object_df
        One row per predicted CNT object.

    image_df
        One row per image containing image-level metrics and mean
        object-level metrics.
    """
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

    object_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []

    # Iterate over every declared image, including images without
    # any predicted annotations.
    for image_id, image_record in ( image_id_to_record.items() ):
        anns = anns_by_image.get( image_id, [], )

        h, w = _image_shape(
            image_record,
            default_image_height,
            default_image_width,
        )

        image_name = str(
            image_record.get(
                "file_name",
                image_id,
            )
        )

        image_key = Path(image_name).stem.lower()

        metrics = calculate_cnt_features_from_annotations(
            anns,
            image_height=h,
            image_width=w,
        )

        n_obj = int(
            metrics.get(
                "num_polygons",
                0,
            )
        )

        image_row: dict[str, Any] = {
            "image_id": image_key,
            "image_file": image_name,
            "n_objects": n_obj,
        }

        for key in IMAGE_FEATURE_KEYS:
            image_row[key] = metrics.get(
                key,
                np.nan,
            )

        for key in OBJECT_FEATURE_KEYS:
            image_row[f"mean_{key}"] = float(
                metrics.get(
                    f"average_{key}",
                    np.nan,
                )
            )

        image_rows.append(image_row)

        if n_obj <= 0:
            continue

        feature_annotation_ids = metrics.get( "annotation_ids", [], )

        for idx in range(n_obj):
            ann_id = (
                feature_annotation_ids[idx]
                if idx < len(
                    feature_annotation_ids
                )
                else None
            )

            row: dict[str, Any] = {
                "image_id": image_key,
                "image_file": image_name,
                "annotation_id": ann_id,
                "object_index": idx,
            }

            for key in OBJECT_FEATURE_KEYS:
                row[key] = _safe_series_value(
                    metrics.get(key),
                    idx,
                )

            object_rows.append(row)

    object_df = pd.DataFrame(object_rows)
    image_df = pd.DataFrame(image_rows)

    return object_df, image_df