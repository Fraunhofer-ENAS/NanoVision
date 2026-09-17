from __future__ import annotations

from collections import Counter
from pathlib import Path


class DatasetValidationError(ValueError):
    """Raised when the canonical dataset layout is invalid."""


def _assert_directory_exists(path: Path, label: str) -> None:
    if not path.exists() or not path.is_dir():
        raise DatasetValidationError(f"Missing required {label} directory: {path}")


def list_tif_filenames(images_root: Path) -> list[str]:
    """Return sorted .tif filenames from images_root."""
    _assert_directory_exists(images_root, "images")
    return sorted(p.name for p in images_root.glob("*.tif") if p.is_file())


def validate_image_mask_pairs(images_root: Path, masks_root: Path) -> list[str]:
    """
    Validate exact 1:1 image/mask pairing by .tif filename.

    Returns sorted sample filenames when valid.
    """
    _assert_directory_exists(images_root, "images")
    _assert_directory_exists(masks_root, "masks")

    image_names = {p.name for p in images_root.glob("*.tif") if p.is_file()}
    mask_names = {p.name for p in masks_root.glob("*.tif") if p.is_file()}

    if not image_names:
        raise DatasetValidationError(f"No .tif images found in {images_root}")

    only_images = sorted(image_names - mask_names)
    only_masks = sorted(mask_names - image_names)

    if only_images or only_masks:
        parts: list[str] = []
        if only_images:
            parts.append(f"images without masks: {only_images[:10]}")
        if only_masks:
            parts.append(f"masks without images: {only_masks[:10]}")
        raise DatasetValidationError("Invalid image/mask pairing, " + "; ".join(parts))

    return sorted(image_names)


def ensure_metadata_coverage(
    sample_filenames: list[str],
    metadata_filenames: list[str],
    *,
    metadata_name: str,
) -> None:
    """Validate metadata has exactly one row per dataset sample."""
    sample_counter = Counter(sample_filenames)
    metadata_counter = Counter(metadata_filenames)

    duplicate_samples = sorted([name for name, count in sample_counter.items() if count > 1])
    duplicate_metadata = sorted([name for name, count in metadata_counter.items() if count > 1])

    if duplicate_samples:
        raise DatasetValidationError(
            f"Duplicate sample identifiers detected before validating {metadata_name}: "
            f"{duplicate_samples[:10]}"
        )

    if duplicate_metadata:
        raise DatasetValidationError(
            f"Duplicate rows detected in {metadata_name}: {duplicate_metadata[:10]}"
        )

    sample_set = set(sample_filenames)
    meta_set = set(metadata_filenames)

    missing = sorted(sample_set - meta_set)
    extra = sorted(meta_set - sample_set)

    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append(f"missing in {metadata_name}: {missing[:10]}")
        if extra:
            parts.append(f"extra in {metadata_name}: {extra[:10]}")
        raise DatasetValidationError("Metadata coverage mismatch, " + "; ".join(parts))
