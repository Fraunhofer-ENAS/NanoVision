from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, List, Literal, Sequence

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from pycocotools import mask as mask_utils
from skimage.draw import polygon as sk_polygon



def _bbox_from_coords(bbox_coords: list[tuple[int | float, int | float, int | float, int | float]]) -> list[float] | None:
    if not bbox_coords:
        return None

    x_min = min(x for x, _, _, _ in bbox_coords)
    y_min = min(y for _, y, _, _ in bbox_coords)
    x_max = max(x2 for _, _, x2, _ in bbox_coords)
    y_max = max(y2 for _, _, _, y2 in bbox_coords)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def _normalize_selected_filenames(selected_filenames: Sequence[str] | None) -> list[str] | None:
    if selected_filenames is None:
        return None

    normalized = [str(name) for name in selected_filenames]
    duplicates = sorted({name for name in normalized if normalized.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate selected filenames are not allowed: {duplicates[:10]}")

    invalid = [name for name in normalized if not name.lower().endswith(".tif")]
    if invalid:
        raise ValueError(f"Selected filenames must all end with '.tif': {invalid[:10]}")

    return normalized


def _resolve_contour_approximation(mode: Literal["simple", "none"]) -> int:
    if mode == "simple":
        return cv2.CHAIN_APPROX_SIMPLE
    if mode == "none":
        return cv2.CHAIN_APPROX_NONE
    raise ValueError(f"Unsupported contour approximation mode: {mode}")


def _atomic_write_json(output_json_path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with NamedTemporaryFile("w", delete=False, dir=str(output_path.parent), encoding="utf-8", newline="") as tf:
        json.dump(payload, tf, indent=4)
        temp_path = Path(tf.name)

    temp_path.replace(output_path)


def build_coco_from_instance_masks(
    mask_folder: str | os.PathLike[str],
    *,
    selected_filenames: Sequence[str] | None = None,
    preserve_tif_filenames: bool = False,
    category_id: int = 1,
    contour_approximation: Literal["simple", "none"] = "simple",
    # added for cellpose inference conversion
    image_filename_suffix_to_remove: str | None = None,
) -> dict[str, Any]:
    """
    Build a COCO dictionary from instance-indexed TIFF masks.

    Each non-zero mask value is treated as one object instance.
    When selected_filenames is provided, only those masks are converted and each
    selected filename must exist exactly once in mask_folder.
    """
    mask_root = Path(mask_folder)
    if not mask_root.exists() or not mask_root.is_dir():
        raise ValueError(f"Mask folder does not exist or is not a directory: {mask_root}")

    normalized_filenames = _normalize_selected_filenames(selected_filenames)

    if normalized_filenames is None:
        mask_filenames = sorted(p.name for p in mask_root.iterdir() if p.is_file() and p.suffix.lower() == ".tif")
    else:
        missing = [name for name in normalized_filenames if not (mask_root / name).is_file()]
        if missing:
            raise ValueError(f"Selected mask files do not exist: {missing[:10]}")
        mask_filenames = list(normalized_filenames)

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": category_id, "name": "CNT"}],
    }

    contour_mode = _resolve_contour_approximation(contour_approximation)

    image_id = 1
    annotation_id = 1

    for mask_filename in mask_filenames:
        mask_path = mask_root / mask_filename
        mask_image = np.array(Image.open(mask_path))

        if mask_image.ndim != 2:
            raise ValueError(
                f"Mask '{mask_filename}' must be two-dimensional, got shape {mask_image.shape}"
            )

        height, width = mask_image.shape
        image_stem = mask_path.stem

        if image_filename_suffix_to_remove is not None:
            if not image_stem.endswith(
                image_filename_suffix_to_remove
            ):
                raise ValueError(
                    f"Mask filename '{mask_filename}' does not end "
                    f"with the expected suffix "
                    f"'{image_filename_suffix_to_remove}'."
                )

            image_stem = image_stem[
                :-len(image_filename_suffix_to_remove)
            ]

        file_name = (
            f"{image_stem}.tif"
            if preserve_tif_filenames
            else f"{image_stem}.jpg"
        )

        coco_data["images"].append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )

        unique_labels = np.unique(mask_image)
        unique_labels = unique_labels[unique_labels != 0]

        for label in unique_labels:
            obj_mask = (mask_image == label).astype(np.uint8)
            contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, contour_mode)

            segmentation = []
            bbox_coords = []
            contour_areas: list[float] = []

            for contour in contours:
                if len(contour) < 3:
                    continue
                segmentation.append(contour.flatten().tolist())
                x, y, w, h = cv2.boundingRect(contour)
                bbox_coords.append((x, y, x + w, y + h))
                contour_areas.append(float(cv2.contourArea(contour)))

            bbox = _bbox_from_coords(bbox_coords)
            if bbox is None or not segmentation:
                continue

            coco_data["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": segmentation,
                    "bbox": bbox,
                    "iscrowd": 0,
                    "area": float(sum(contour_areas)),
                }
            )
            annotation_id += 1

        image_id += 1

    return coco_data


def convert_tif_annotations_to_json(
    mask_folder,
    output_json_path,
    selected_filenames: Sequence[str] | None = None,
    preserve_tif_filenames: bool = False,
    category_id: int = 1,
    contour_approximation: Literal["simple", "none"] = "simple",
    # added for cellpose inference conversion
    image_filename_suffix_to_remove: str | None = None,
):
    coco_data = build_coco_from_instance_masks(
        mask_folder,
        selected_filenames=selected_filenames,
        preserve_tif_filenames=preserve_tif_filenames,
        category_id=category_id,
        contour_approximation=contour_approximation,
        image_filename_suffix_to_remove=(
            image_filename_suffix_to_remove
        ),
    )

    _atomic_write_json(output_json_path, coco_data)

    print(f"COCO annotations saved to {output_json_path}")
    return coco_data


def build_coco_from_stardist_label_predictions(
    *,
    y_preds: list[np.ndarray],
    filenames: list[str],
    scores: list,
    category_id: int = 1,
    score_thresh: float | None = None,
) -> dict:
    """
    Build COCO polygon annotations from StarDist instance-label predictions.

    Parameters
    ----------
    y_preds:
        One instance-label mask per image. Background must be label 0.
    filenames:
        Filenames corresponding to ``y_preds``.
    scores:
        Per-image object scores aligned with the non-zero instance labels.
    category_id:
        COCO category ID. Default: 1.
    score_thresh:
        Optional minimum score required for an annotation.

    Returns
    -------
    dict
        COCO-style dictionary containing images, annotations, and categories.

    Raises
    ------
    ValueError
        If the number of masks and filenames differs.
    """
    if len(y_preds) != len(filenames):
        raise ValueError(
            "y_preds and filenames must contain the same number of entries: "
            f"y_preds={len(y_preds)}, filenames={len(filenames)}."
        )

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": category_id,
                "name": "CNT",
            }
        ],
    }

    annotation_id = 1

    for image_id, (object_attr, filename) in enumerate(
        zip(y_preds, filenames),
        start=1,
    ):
        height, width = object_attr.shape

        image_scores = (
            scores[image_id - 1]
            if image_id - 1 < len(scores)
            else []
        )

        coco_data["images"].append(
            {
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height,
            }
        )

        unique_labels = np.unique(object_attr)
        unique_labels = unique_labels[unique_labels != 0]

        for object_index, label in enumerate(unique_labels):
            score = (
                float(image_scores[object_index])
                if object_index < len(image_scores)
                else None
            )

            if (
                score_thresh is not None
                and score is not None
                and score < score_thresh
            ):
                continue

            obj_mask = (
                object_attr == label
            ).astype(np.uint8)

            contours, _ = cv2.findContours(
                obj_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            segmentation = []
            bbox_coords = []
            valid_contours = []

            for contour in contours:
                if len(contour) < 3:
                    continue

                valid_contours.append(contour)
                segmentation.append(
                    contour.flatten().tolist()
                )

                x, y, w, h = cv2.boundingRect(contour)

                bbox_coords.append(
                    (
                        x,
                        y,
                        x + w,
                        y + h,
                    )
                )

            bbox = _bbox_from_coords(
                bbox_coords
            )

            if bbox is None or not valid_contours:
                continue

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": segmentation,
                "bbox": bbox,
                "iscrowd": 0,
                "area": cv2.contourArea(
                    np.vstack(valid_contours)
                ),
            }

            if score is not None:
                annotation["score"] = score

            coco_data["annotations"].append(
                annotation
            )

            annotation_id += 1

    return coco_data

def convert_stardist_predictions_to_json(
    loader,
    all_scores,
    output_json_path,
    target_attr="Y_pred",
    score_thresh: float | None = None,
):
    """
    Backward-compatible loader-based wrapper around
    ``build_coco_from_stardist_label_predictions``.
    """
    samples = loader.get_test_data()

    y_preds = [
        getattr(sample, target_attr)
        for sample in samples
    ]

    filenames = [
        sample.X_fn
        for sample in samples
    ]

    coco_data = build_coco_from_stardist_label_predictions(
        y_preds=y_preds,
        filenames=filenames,
        scores=all_scores,
        category_id=1,
        score_thresh=score_thresh,
    )

    _atomic_write_json(
        output_json_path,
        coco_data,
    )

    print(
        f"COCO annotations saved to {output_json_path}"
    )

    return coco_data

def _flatten_xy(coords) -> list[float]:
    return [float(v) for xy in coords for v in xy]


def _polygon_to_binary_mask(poly: Polygon, height: int, width: int) -> np.ndarray:
    mask_out = np.zeros((height, width), dtype=np.uint8)

    if poly.is_empty:
        return mask_out

    polygons = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]

    for p in polygons:
        if p.is_empty or len(p.exterior.coords) < 3:
            continue

        exterior = np.asarray(p.exterior.coords, dtype=float)
        rr, cc = sk_polygon(exterior[:, 1], exterior[:, 0], shape=(height, width))
        mask_out[rr, cc] = 1

        for interior in p.interiors:
            interior_coords = np.asarray(interior.coords, dtype=float)
            if len(interior_coords) < 3:
                continue

            rr, cc = sk_polygon(
                interior_coords[:, 1],
                interior_coords[:, 0],
                shape=(height, width),
            )
            mask_out[rr, cc] = 0

    return mask_out


def _binary_mask_to_coco_rle(binary_mask: np.ndarray) -> dict:
    binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))

    rle = mask_utils.encode(binary_mask)

    # pycocotools returns bytes, JSON needs string
    rle["counts"] = rle["counts"].decode("utf-8")

    return rle

def build_coco_rle_from_instance_masks(
    mask_folder: str | os.PathLike[str],
    *,
    selected_filenames: Sequence[str] | None = None,
    preserve_tif_filenames: bool = False,
    category_id: int = 1,
) -> dict[str, Any]:
    """
    Build COCO RLE annotations directly from instance-labelled TIFF masks.

    Every nonzero TIFF label is encoded independently as one RLE object.
    No polygon extraction or polygon rasterization is performed.
    """
    mask_root = Path(mask_folder)

    if not mask_root.exists() or not mask_root.is_dir():
        raise ValueError(
            "Mask folder does not exist or is not a directory: "
            f"{mask_root}"
        )

    normalized_filenames = _normalize_selected_filenames(
        selected_filenames
    )

    if normalized_filenames is None:
        mask_filenames = sorted(
            path.name
            for path in mask_root.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".tif"
        )
    else:
        missing = [
            filename
            for filename in normalized_filenames
            if not (mask_root / filename).is_file()
        ]

        if missing:
            raise ValueError(
                "Selected mask files do not exist: "
                f"{missing[:10]}"
            )

        mask_filenames = list(
            normalized_filenames
        )

    coco_data: dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": category_id,
                "name": "CNT",
            }
        ],
    }

    annotation_id = 1

    for image_id, mask_filename in enumerate(
        mask_filenames,
        start=1,
    ):
        mask_path = mask_root / mask_filename

        instance_mask = np.asarray(
            Image.open(mask_path)
        )

        if instance_mask.ndim != 2:
            raise ValueError(
                f"Mask '{mask_filename}' must be two-dimensional; "
                f"received shape {instance_mask.shape}."
            )

        height, width = instance_mask.shape

        file_name = (
            mask_filename
            if preserve_tif_filenames
            else f"{mask_path.stem}.jpg"
        )

        coco_data["images"].append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": int(width),
                "height": int(height),
            }
        )

        labels = np.unique(
            instance_mask
        )

        labels = labels[
            labels != 0
        ]

        for label in labels:
            binary_mask = (
                instance_mask == label
            ).astype(np.uint8)

            if not binary_mask.any():
                raise RuntimeError(
                    "Unexpected empty mask for "
                    f"'{mask_filename}', label {label}."
                )

            rle = _binary_mask_to_coco_rle(
                binary_mask
            )

            bbox = (
                mask_utils
                .toBbox(rle)
                .astype(float)
                .tolist()
            )

            area = float(
                mask_utils.area(rle)
            )

            coco_data["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": rle,
                    "bbox": bbox,
                    "iscrowd": 0,
                    "area": area,
                    "source_instance_label": int(label),
                }
            )

            annotation_id += 1

    return coco_data

def convert_tif_annotations_to_rle_json(
    mask_folder: str | os.PathLike[str],
    output_json_path: str | os.PathLike[str],
    *,
    selected_filenames: Sequence[str] | None = None,
    preserve_tif_filenames: bool = False,
    category_id: int = 1,
) -> dict[str, Any]:
    """
    Export instance-labelled TIFF masks directly to COCO RLE JSON.
    """
    coco_data = build_coco_rle_from_instance_masks(
        mask_folder,
        selected_filenames=selected_filenames,
        preserve_tif_filenames=(
            preserve_tif_filenames
        ),
        category_id=category_id,
    )

    _atomic_write_json(
        output_json_path,
        coco_data,
    )

    print(
        "COCO RLE annotations saved to "
        f"{output_json_path}"
    )

    return coco_data


def build_coco_rle_from_polygon_predictions(
    *,
    images: Sequence[np.ndarray],
    filenames: Sequence[str],
    polygons: Sequence[Sequence[Polygon]],
    scores: Sequence[Sequence[float]],
    category_id: int = 1,
) -> dict[str, Any]:
    """
    Build an RLE-segmentation COCO dictionary from polygon prediction data.
    """
    n_images = len(images)

    if len(filenames) != n_images:
        raise ValueError(
            "images and filenames must contain the same number of entries: "
            f"images={n_images}, filenames={len(filenames)}."
        )

    if len(polygons) != n_images:
        raise ValueError(
            "images and polygons must contain the same number of entries: "
            f"images={n_images}, polygons={len(polygons)}."
        )

    if len(scores) != n_images:
        raise ValueError(
            "images and scores must contain the same number of entries: "
            f"images={n_images}, scores={len(scores)}."
        )

    coco: dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": category_id,
                "name": "CNT",
            }
        ],
    }

    annotation_id = 1

    for image_id, (
        image,
        filename,
        image_polygons,
        image_scores,
    ) in enumerate(
        zip(
            images,
            filenames,
            polygons,
            scores,
        ),
        start=1,
    ):
        height, width = image.shape[:2]

        coco["images"].append(
            {
                "id": image_id,
                "file_name": str(filename),
                "width": int(width),
                "height": int(height),
            }
        )

        for object_index, poly in enumerate(
            image_polygons
        ):
            if poly.is_empty:
                continue

            mask = _polygon_to_binary_mask(
                poly,
                height,
                width,
            )

            if mask.sum() == 0:
                continue

            rle = _binary_mask_to_coco_rle(
                mask
            )

            bbox = (
                mask_utils
                .toBbox(rle)
                .astype(float)
                .tolist()
            )

            area = float(
                mask_utils.area(rle)
            )

            annotation: dict[str, Any] = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": rle,
                "bbox": bbox,
                "iscrowd": 0,
                "area": area,
            }

            if object_index < len(image_scores):
                annotation["score"] = float(
                    image_scores[object_index]
                )

            coco["annotations"].append(
                annotation
            )

            annotation_id += 1

    return coco

def export_polygon_predictions_to_coco_rle_json(
    *,
    images: Sequence[np.ndarray],
    filenames: Sequence[str],
    polygons: Sequence[Sequence[Polygon]],
    scores: Sequence[Sequence[float]],
    output_json_path: str | os.PathLike[str],
    category_id: int = 1,
) -> Path:
    """
    Build and save RLE-segmentation COCO predictions.
    """
    coco = build_coco_rle_from_polygon_predictions(
        images=images,
        filenames=filenames,
        polygons=polygons,
        scores=scores,
        category_id=category_id,
    )

    _atomic_write_json(
        output_json_path,
        coco,
    )

    output_path = Path(
        output_json_path
    )

    print(
        f"[+] Saved COCO RLE JSON to {output_path}"
    )

    return output_path

def convert_polygons_to_coco_rle_json(
    loader,
    all_scores: List[List[float]],
    output_json_path: str,
    poly_attr: str = "polygons",
    category_id: int = 1,
):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": category_id, "name": "CNT"}],
    }

    ann_id = 1

    for img_idx, sample in enumerate(loader.get_test_data(), start=1):
        height, width = sample.X.shape[:2]

        coco["images"].append(
            {
                "id": img_idx,
                "file_name": sample.X_fn,
                "width": width,
                "height": height,
            }
        )

        polys: List[Polygon] = getattr(sample, poly_attr, [])
        scores = all_scores[img_idx - 1] if img_idx - 1 < len(all_scores) else []

        for obj_idx, poly in enumerate(polys):
            if poly.is_empty:
                continue

            mask = _polygon_to_binary_mask(poly, height, width)

            if mask.sum() == 0:
                continue

            rle = _binary_mask_to_coco_rle(mask)

            bbox = mask_utils.toBbox(rle).astype(float).tolist()
            area = float(mask_utils.area(rle))

            annotation = {
                "id": ann_id,
                "image_id": img_idx,
                "category_id": category_id,
                "segmentation": rle,
                "bbox": bbox,
                "iscrowd": 0,
                "area": area,
            }

            if obj_idx < len(scores):
                annotation["score"] = float(scores[obj_idx])

            coco["annotations"].append(annotation)
            ann_id += 1

    _atomic_write_json(output_json_path, coco)

    print(f"[+] Saved COCO RLE JSON to {output_json_path}")

def build_coco_from_polygon_predictions(
    *,
    images: Sequence[np.ndarray],
    filenames: Sequence[str],
    polygons: Sequence[Sequence[Polygon]],
    scores: Sequence[Sequence[float]],
    category_id: int = 1,
) -> dict[str, Any]:
    """
    Build a polygon-segmentation COCO dictionary from prediction data.

    Parameters
    ----------
    images:
        Source images used for inference.
    filenames:
        Filename stems or names corresponding to ``images``.
    polygons:
        Predicted polygons for each image.
    scores:
        Prediction scores corresponding to the polygons for each image.
    category_id:
        COCO category identifier.

    Returns
    -------
    dict[str, Any]
        COCO-formatted dictionary containing images, annotations, and category.
    """
    n_images = len(images)

    if len(filenames) != n_images:
        raise ValueError(
            "images and filenames must contain the same number of entries: "
            f"images={n_images}, filenames={len(filenames)}."
        )

    if len(polygons) != n_images:
        raise ValueError(
            "images and polygons must contain the same number of entries: "
            f"images={n_images}, polygons={len(polygons)}."
        )

    if len(scores) != n_images:
        raise ValueError(
            "images and scores must contain the same number of entries: "
            f"images={n_images}, scores={len(scores)}."
        )

    coco: dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": category_id,
                "name": "CNT",
            }
        ],
    }

    annotation_id = 1

    for image_id, (
        image,
        filename,
        image_polygons,
        image_scores,
    ) in enumerate(
        zip(
            images,
            filenames,
            polygons,
            scores,
        ),
        start=1,
    ):
        height, width = image.shape[:2]

        coco["images"].append(
            {
                "id": image_id,
                "file_name": str(filename),
                "width": int(width),
                "height": int(height),
            }
        )

        for object_index, poly in enumerate(
            image_polygons
        ):
            if (
                not poly.is_valid
                or poly.is_empty
                or len(poly.exterior.coords) < 3
            ):
                continue

            segmentation = [
                _flatten_xy(
                    poly.exterior.coords
                )
            ]

            minx, miny, maxx, maxy = poly.bounds

            annotation: dict[str, Any] = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": segmentation,
                "bbox": [
                    float(minx),
                    float(miny),
                    float(maxx - minx),
                    float(maxy - miny),
                ],
                "iscrowd": 0,
                "area": float(poly.area),
            }

            if object_index < len(image_scores):
                annotation["score"] = float(
                    image_scores[object_index]
                )

            coco["annotations"].append(
                annotation
            )

            annotation_id += 1

    return coco

def export_polygon_predictions_to_coco_json(
    *,
    images: Sequence[np.ndarray],
    filenames: Sequence[str],
    polygons: Sequence[Sequence[Polygon]],
    scores: Sequence[Sequence[float]],
    output_json_path: str | os.PathLike[str],
    category_id: int = 1,
) -> Path:
    """
    Build and save polygon-segmentation COCO predictions.
    """
    coco = build_coco_from_polygon_predictions(
        images=images,
        filenames=filenames,
        polygons=polygons,
        scores=scores,
        category_id=category_id,
    )

    _atomic_write_json(
        output_json_path,
        coco,
    )

    output_path = Path(
        output_json_path
    )

    print(
        f"[+] Saved COCO JSON to {output_path}"
    )

    return output_path

def convert_polygons_to_coco_json(
    loader,
    all_scores: List[List[float]],
    output_json_path: str,
    poly_attr: str = "polygons",
    category_id: int = 1,
):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": category_id, "name": "CNT"}],
    }
    ann_id = 1

    for img_idx, sample in enumerate(loader.get_test_data(), start=1):
        height, width = sample.X.shape[:2]
        coco["images"].append(
            {
                "id": img_idx,
                "file_name": sample.X_fn,
                "width": width,
                "height": height,
            }
        )

        polys: List[Polygon] = getattr(sample, poly_attr, [])
        scores = all_scores[img_idx - 1] if img_idx - 1 < len(all_scores) else []

        for obj_idx, poly in enumerate(polys):
            if not poly.is_valid or poly.is_empty or len(poly.exterior.coords) < 3:
                continue

            seg = [list(coord) for coord in poly.exterior.coords]
            segmentation = [sum(seg, [])]
            minx, miny, maxx, maxy = poly.bounds

            annotation = {
                "id": ann_id,
                "image_id": img_idx,
                "category_id": category_id,
                "segmentation": segmentation,
                "bbox": [minx, miny, maxx - minx, maxy - miny],
                "iscrowd": 0,
                "area": poly.area,
            }

            if obj_idx < len(scores):
                annotation["score"] = float(scores[obj_idx])

            coco["annotations"].append(annotation)
            ann_id += 1

    _atomic_write_json(output_json_path, coco)

    print(f"[+] Saved COCO JSON to {output_json_path}")