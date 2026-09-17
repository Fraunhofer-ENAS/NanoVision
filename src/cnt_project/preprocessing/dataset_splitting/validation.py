from __future__ import annotations

from pathlib import Path

from cnt_project.preprocessing.metadata.validation import DatasetValidationError, validate_image_mask_pairs

SUPPORTED_SUBSETS = ("train", "val", "test")


def validate_hierarchical_split_fractions(
    train: float,
    val: float,
    test: float,
    *,
    tol: float = 1e-9,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Validate hierarchical split fractions.

    Semantics:
      - `test` is the fraction of the complete dataset assigned to test.
      - `train` and `val` divide the remaining development pool.

    Returns:
      - development/test fractions for stage 1;
      - train/val fractions for stage 2.
    """
    train = float(train)
    val = float(val)
    test = float(test)

    if not 0.0 < test < 1.0:
        raise DatasetValidationError(
            "test_fraction must be greater than 0 and less than 1, "
            f"got {test}"
        )

    if train <= 0.0:
        raise DatasetValidationError(
            f"train_fraction must be greater than 0, got {train}"
        )

    if val <= 0.0:
        raise DatasetValidationError(
            f"val_fraction must be greater than 0, got {val}"
        )

    train_val_total = train + val
    if abs(train_val_total - 1.0) > tol:
        raise DatasetValidationError(
            "train_fraction and val_fraction must sum to 1.0 because "
            "they divide the non-test development pool. "
            f"Got train={train}, val={val}, total={train_val_total:.12f}"
        )

    development_test_fractions = {
        "development": 1.0 - test,
        "test": test,
    }

    train_val_fractions = {
        "train": train,
        "val": val,
    }

    return development_test_fractions, train_val_fractions


def validate_dataset_root(dataset_root: Path) -> list[str]:
    """Validate canonical image-mask pairing and return sorted .tif sample filenames."""
    images_root = dataset_root / "images"
    masks_root = dataset_root / "masks"
    return validate_image_mask_pairs(images_root, masks_root)
