from __future__ import annotations

from collections import Counter
from pathlib import Path


class DatasetValidationError(ValueError):
    """Raised when the canonical dataset or its metadata are invalid."""


def _assert_directory_exists(path: Path, label: str) -> None:
    """Ensure that a required directory exists."""
    if not path.exists() or not path.is_dir():
        raise DatasetValidationError(
            f"Missing required {label} directory: {path}"
        )


def list_tif_filenames(images_root: Path) -> list[str]:
    """Return sorted .tif filenames from the canonical images directory."""
    _assert_directory_exists(images_root, "images")

    filenames = sorted(
        path.name
        for path in images_root.glob("*.tif")
        if path.is_file()
    )

    if not filenames:
        raise DatasetValidationError(
            f"No .tif images found in {images_root}"
        )

    return filenames


def validate_image_mask_pairs(
    images_root: Path,
    masks_root: Path,
) -> list[str]:
    """
    Validate exact one-to-one image/mask pairing by .tif filename.

    Only .tif images are considered dataset samples. Every sample must have
    a mask with exactly the same filename.

    Returns
    -------
    list[str]
        Sorted dataset sample filenames.
    """
    _assert_directory_exists(images_root, "images")
    _assert_directory_exists(masks_root, "masks")

    image_names = {
        path.name
        for path in images_root.glob("*.tif")
        if path.is_file()
    }

    mask_names = {
        path.name
        for path in masks_root.glob("*.tif")
        if path.is_file()
    }

    if not image_names:
        raise DatasetValidationError(
            f"No .tif images found in {images_root}"
        )

    images_without_masks = sorted(image_names - mask_names)
    masks_without_images = sorted(mask_names - image_names)

    if images_without_masks or masks_without_images:
        problems: list[str] = []

        if images_without_masks:
            problems.append(
                "images without masks: "
                f"{images_without_masks[:10]}"
            )

        if masks_without_images:
            problems.append(
                "masks without images: "
                f"{masks_without_images[:10]}"
            )

        raise DatasetValidationError(
            "Invalid image/mask pairing: " + "; ".join(problems)
        )

    return sorted(image_names)


def ensure_metadata_coverage(
    sample_filenames: list[str],
    metadata_filenames: list[str],
    *,
    metadata_name: str,
) -> None:
    """
    Validate that metadata contains exactly one row per dataset sample.

    The validation rejects:

    - duplicate metadata rows;
    - missing dataset samples;
    - metadata rows for unknown samples.
    """
    sample_duplicates = sorted(
        filename
        for filename, count in Counter(sample_filenames).items()
        if count > 1
    )

    if sample_duplicates:
        raise DatasetValidationError(
            "Duplicate dataset filenames detected: "
            f"{sample_duplicates[:10]}"
        )

    metadata_duplicates = sorted(
        filename
        for filename, count in Counter(metadata_filenames).items()
        if count > 1
    )

    if metadata_duplicates:
        raise DatasetValidationError(
            f"Duplicate filename rows in {metadata_name}: "
            f"{metadata_duplicates[:10]}"
        )

    sample_set = set(sample_filenames)
    metadata_set = set(metadata_filenames)

    missing = sorted(sample_set - metadata_set)
    extra = sorted(metadata_set - sample_set)

    if missing or extra:
        problems: list[str] = []

        if missing:
            problems.append(
                f"missing from {metadata_name}: {missing[:10]}"
            )

        if extra:
            problems.append(
                f"unknown files in {metadata_name}: {extra[:10]}"
            )

        raise DatasetValidationError(
            "Metadata coverage mismatch: " + "; ".join(problems)
        )