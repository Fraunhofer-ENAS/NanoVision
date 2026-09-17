"""Decode and validate one non-empty COCO object."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from cnt_project.coco.masks import (
    rle_segmentation_to_mask,
)


@dataclass(frozen=True)
class CocoImageInstances:
    """Ground-truth and predicted instances for one image."""

    filename: str
    height: int
    width: int

    ground_truth_masks: tuple[np.ndarray, ...]
    prediction_masks: tuple[np.ndarray, ...]

    ground_truth_annotation_ids: tuple[int, ...]
    prediction_annotation_ids: tuple[int, ...]

    ground_truth_source_labels: tuple[int | None, ...]
    prediction_scores: tuple[float, ...]


def _normalize_filename(
    filename: str,
) -> str:
    """Return a case-insensitive, extension-independent image key."""
    return Path(
        str(filename).strip()
    ).stem.lower()


def _load_json(
    path: str | Path,
    *,
    name: str,
) -> dict:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"{name} does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            f"{name} must contain a JSON object."
        )

    if not isinstance(data.get("images"), list):
        raise ValueError(
            f"{name} must contain an 'images' list."
        )

    if not isinstance(
        data.get("annotations"),
        list,
    ):
        raise ValueError(
            f"{name} must contain an 'annotations' list."
        )

    return data


def _index_images_by_filename(
    images: list[dict],
    *,
    name: str,
) -> dict[str, dict]:
    """Index COCO images by normalized filename."""
    images_by_filename: dict[str, dict] = {}
    image_ids = set()

    for image in images:
        for required_key in (
            "id",
            "file_name",
            "height",
            "width",
        ):
            if required_key not in image:
                raise ValueError(
                    f"{name} image is missing "
                    f"{required_key!r}: {image}"
                )

        image_id = int(image["id"])
        image_key = _normalize_filename(
            image["file_name"]
        )

        if image_id in image_ids:
            raise ValueError(
                f"{name} contains duplicate image ID "
                f"{image_id}."
            )

        if image_key in images_by_filename:
            raise ValueError(
                f"{name} contains duplicate normalized "
                f"filename {image_key!r}."
            )

        height = int(image["height"])
        width = int(image["width"])

        if height <= 0 or width <= 0:
            raise ValueError(
                f"{name} image {image_key!r} has invalid "
                f"dimensions {(height, width)}."
            )

        image_ids.add(image_id)
        images_by_filename[image_key] = image

    return images_by_filename


def _group_annotations_by_image(
    annotations: list[dict],
    *,
    valid_image_ids: set[int],
    name: str,
) -> dict[int, list[dict]]:
    """Group COCO annotations using their JSON-local image IDs."""
    annotations_by_image: dict[
        int,
        list[dict],
    ] = defaultdict(list)

    annotation_ids = set()

    for annotation in annotations:
        for required_key in (
            "id",
            "image_id",
            "segmentation",
        ):
            if required_key not in annotation:
                raise ValueError(
                    f"{name} annotation is missing "
                    f"{required_key!r}: {annotation}"
                )

        annotation_id = int(
            annotation["id"]
        )

        image_id = int(
            annotation["image_id"]
        )

        if annotation_id in annotation_ids:
            raise ValueError(
                f"{name} contains duplicate annotation ID "
                f"{annotation_id}."
            )

        if image_id not in valid_image_ids:
            raise ValueError(
                f"{name} annotation {annotation_id} refers "
                f"to unknown image ID {image_id}."
            )

        if not isinstance(
            annotation["segmentation"],
            (dict, list),
        ):
            raise ValueError(
                f"{name} annotation {annotation_id} must contain "
                "RLE or polygon segmentation."
            )

        annotation_ids.add(annotation_id)

        annotations_by_image[
            image_id
        ].append(annotation)

    return dict(annotations_by_image)


def _decode_annotation(
    annotation: dict,
    *,
    height: int,
    width: int,
    name: str,
) -> np.ndarray:
    """Decode and validate one non-empty COCO RLE object."""
    annotation_id = int(
        annotation["id"]
    )

    segmentation = annotation["segmentation"]

    if isinstance(segmentation, dict):
        decoded = rle_segmentation_to_mask(
            segmentation,
            height,
            width,
        )
    else:
        rle = mask_utils.frPyObjects(
            segmentation,
            height,
            width,
        )

        decoded = mask_utils.decode(rle)

        if decoded.ndim == 3:
            decoded = np.any(
                decoded,
                axis=2,
            )

        decoded = decoded.astype(np.uint8)

    decoded = np.asarray(
        decoded,
        dtype=np.uint8,
    )

    if decoded.shape != (height, width):
        raise ValueError(
            f"{name} annotation {annotation_id} decoded "
            f"to shape {decoded.shape}; expected "
            f"{(height, width)}."
        )

    decoded = (
        decoded > 0
    ).astype(np.uint8)

    if not decoded.any():
        raise ValueError(
            f"{name} annotation {annotation_id} decoded "
            "to an empty mask."
        )

    return decoded


def load_coco_rle_instance_pairs(
    *,
    ground_truth_json_path: str | Path,
    prediction_json_path: str | Path,
    prediction_score_threshold: float = 0.0,
) -> tuple[CocoImageInstances, ...]:
    """
    Load corresponding GT and prediction instances from COCO RLE.

    Images are associated by normalized filename rather than image ID.
    Both JSON files must describe exactly the same image set.

    Predictions are retained when:

        score >= prediction_score_threshold

    A missing prediction score defaults to 1.0.
    """
    prediction_score_threshold = float(
        prediction_score_threshold
    )

    if (
        not math.isfinite(
            prediction_score_threshold
        )
        or not 0.0
        <= prediction_score_threshold
        <= 1.0
    ):
        raise ValueError(
            "prediction_score_threshold must be finite "
            "and between 0 and 1 inclusive."
        )

    ground_truth_data = _load_json(
        ground_truth_json_path,
        name="ground-truth JSON",
    )

    prediction_data = _load_json(
        prediction_json_path,
        name="prediction JSON",
    )

    ground_truth_images = (
        _index_images_by_filename(
            ground_truth_data["images"],
            name="ground-truth JSON",
        )
    )

    prediction_images = (
        _index_images_by_filename(
            prediction_data["images"],
            name="prediction JSON",
        )
    )

    ground_truth_keys = set(
        ground_truth_images
    )

    prediction_keys = set(
        prediction_images
    )

    missing_prediction_images = (
        ground_truth_keys
        - prediction_keys
    )

    extra_prediction_images = (
        prediction_keys
        - ground_truth_keys
    )

    if (
        missing_prediction_images
        or extra_prediction_images
    ):
        raise ValueError(
            "Ground-truth and prediction image sets do not "
            "match by normalized filename. "
            f"Missing prediction images: "
            f"{sorted(missing_prediction_images)[:10]}; "
            f"extra prediction images: "
            f"{sorted(extra_prediction_images)[:10]}."
        )

    ground_truth_image_ids = {
        int(image["id"])
        for image in ground_truth_data["images"]
    }

    prediction_image_ids = {
        int(image["id"])
        for image in prediction_data["images"]
    }

    ground_truth_annotations = (
        _group_annotations_by_image(
            ground_truth_data["annotations"],
            valid_image_ids=(
                ground_truth_image_ids
            ),
            name="ground-truth JSON",
        )
    )

    prediction_annotations = (
        _group_annotations_by_image(
            prediction_data["annotations"],
            valid_image_ids=(
                prediction_image_ids
            ),
            name="prediction JSON",
        )
    )

    image_results = []

    # Preserve GT JSON image order.
    for ground_truth_image in (
        ground_truth_data["images"]
    ):
        image_key = _normalize_filename(
            ground_truth_image["file_name"]
        )

        prediction_image = (
            prediction_images[image_key]
        )

        height = int(
            ground_truth_image["height"]
        )

        width = int(
            ground_truth_image["width"]
        )

        prediction_height = int(
            prediction_image["height"]
        )

        prediction_width = int(
            prediction_image["width"]
        )

        if (
            prediction_height != height
            or prediction_width != width
        ):
            raise ValueError(
                f"Image dimensions do not match for "
                f"{image_key!r}: GT={(height, width)}, "
                f"prediction="
                f"{(prediction_height, prediction_width)}."
            )

        gt_annotations_for_image = (
            ground_truth_annotations.get(
                int(ground_truth_image["id"]),
                [],
            )
        )

        prediction_annotations_for_image = (
            prediction_annotations.get(
                int(prediction_image["id"]),
                [],
            )
        )

        ground_truth_masks = []
        ground_truth_annotation_ids = []
        ground_truth_source_labels = []

        for annotation in (
            gt_annotations_for_image
        ):
            ground_truth_masks.append(
                _decode_annotation(
                    annotation,
                    height=height,
                    width=width,
                    name="ground-truth JSON",
                )
            )

            ground_truth_annotation_ids.append(
                int(annotation["id"])
            )

            source_label = annotation.get(
                "source_instance_label"
            )

            ground_truth_source_labels.append(
                int(source_label)
                if source_label is not None
                else None
            )

        prediction_masks = []
        prediction_annotation_ids = []
        prediction_scores = []

        for annotation in (
            prediction_annotations_for_image
        ):
            score = float(
                annotation.get("score", 1.0)
            )

            if not math.isfinite(score):
                raise ValueError(
                    "Prediction annotation "
                    f"{annotation['id']} has a non-finite "
                    f"score: {score}."
                )

            if score < prediction_score_threshold:
                continue

            prediction_masks.append(
                _decode_annotation(
                    annotation,
                    height=height,
                    width=width,
                    name="prediction JSON",
                )
            )

            prediction_annotation_ids.append(
                int(annotation["id"])
            )

            prediction_scores.append(score)

        image_results.append(
            CocoImageInstances(
                filename=str(
                    ground_truth_image[
                        "file_name"
                    ]
                ),
                height=height,
                width=width,
                ground_truth_masks=tuple(
                    ground_truth_masks
                ),
                prediction_masks=tuple(
                    prediction_masks
                ),
                ground_truth_annotation_ids=tuple(
                    ground_truth_annotation_ids
                ),
                prediction_annotation_ids=tuple(
                    prediction_annotation_ids
                ),
                ground_truth_source_labels=tuple(
                    ground_truth_source_labels
                ),
                prediction_scores=tuple(
                    prediction_scores
                ),
            )
        )

    return tuple(image_results)