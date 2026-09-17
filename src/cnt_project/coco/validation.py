from __future__ import annotations

from pathlib import Path
from typing import Any

from cnt_project.preprocessing.dataset_splitting.split_manifest import SUBSET_COL
from cnt_project.preprocessing.dataset_splitting.validation import SUPPORTED_SUBSETS
from cnt_project.preprocessing.metadata.schemas import FILENAME_COL
from cnt_project.preprocessing.metadata.validation import DatasetValidationError, validate_image_mask_pairs


class CocoValidationError(ValueError):
    """Raised when generated COCO content is structurally invalid."""


def validate_coco_dataset_structure(dataset_root: Path) -> list[str]:
    """Validate unified dataset image/mask structure and return sample filenames."""
    return validate_image_mask_pairs(dataset_root / "images", dataset_root / "masks")


def validate_coco_json_dict(coco: dict[str, Any]) -> None:
    required_top = {"images", "annotations", "categories"}
    missing_top = sorted(required_top - set(coco.keys()))
    if missing_top:
        raise CocoValidationError(f"COCO dict is missing top-level keys: {missing_top}")

    if not isinstance(coco["images"], list) or not isinstance(coco["annotations"], list) or not isinstance(coco["categories"], list):
        raise CocoValidationError("COCO dict fields images/annotations/categories must all be lists.")

    image_ids: set[int] = set()
    for image in coco["images"]:
        for key in ("id", "file_name", "width", "height"):
            if key not in image:
                raise CocoValidationError(f"COCO image entry is missing key: {key}")
        image_id = int(image["id"])
        if image_id in image_ids:
            raise CocoValidationError(f"Duplicate COCO image id detected: {image_id}")
        image_ids.add(image_id)

    annotation_ids: set[int] = set()
    for annotation in coco["annotations"]:
        for key in ("id", "image_id", "category_id", "segmentation", "bbox", "iscrowd", "area"):
            if key not in annotation:
                raise CocoValidationError(f"COCO annotation entry is missing key: {key}")
        ann_id = int(annotation["id"])
        if ann_id in annotation_ids:
            raise CocoValidationError(f"Duplicate COCO annotation id detected: {ann_id}")
        annotation_ids.add(ann_id)

        if int(annotation["image_id"]) not in image_ids:
            raise CocoValidationError(
                f"Annotation {ann_id} references missing image_id {annotation['image_id']}"
            )

        bbox = annotation["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise CocoValidationError(f"Annotation {ann_id} has invalid bbox shape: {bbox}")

    category_ids = {int(cat["id"]) for cat in coco["categories"] if "id" in cat}
    if not category_ids:
        raise CocoValidationError("COCO categories must contain at least one category id.")

    for annotation in coco["annotations"]:
        if int(annotation["category_id"]) not in category_ids:
            raise CocoValidationError(
                f"Annotation {annotation['id']} references missing category_id {annotation['category_id']}"
            )


def validate_split_manifest_for_coco(manifest_df, *, dataset_filenames: list[str] | None = None) -> None:
    required = {FILENAME_COL, SUBSET_COL}
    missing = sorted(required - set(manifest_df.columns))
    if missing:
        raise CocoValidationError(f"Split manifest is missing required columns for COCO generation: {missing}")

    subset_labels = set(manifest_df[SUBSET_COL].astype(str).unique().tolist())
    allowed_subsets = set(SUPPORTED_SUBSETS)
    if not subset_labels.issubset(allowed_subsets):
        invalid = sorted(subset_labels - allowed_subsets)
        raise CocoValidationError(f"Split manifest has unsupported subset labels: {invalid}")

    if dataset_filenames is not None:
        manifest_filenames = set(manifest_df[FILENAME_COL].astype(str).tolist())
        expected_filenames = set(dataset_filenames)

        missing_assignments = sorted(expected_filenames - manifest_filenames)
        unexpected_assignments = sorted(manifest_filenames - expected_filenames)
        if missing_assignments or unexpected_assignments:
            details: list[str] = []
            if missing_assignments:
                details.append(f"missing assignments: {missing_assignments[:10]}")
            if unexpected_assignments:
                details.append(f"unexpected assignments: {unexpected_assignments[:10]}")
            raise CocoValidationError("Split manifest coverage mismatch for dataset samples, " + "; ".join(details))
