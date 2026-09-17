from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from cnt_project.preprocessing.metadata.schemas import (
    DENSITY_CLASS_COL,
    DENSITY_CLASS_KMEANS_COL,
    DENSITY_CLASS_TERTILE_COL,
    FILENAME_COL,
    GT_OBJECT_COUNT_COL,
)
from cnt_project.preprocessing.metadata.validation import (
    DatasetValidationError,
    ensure_metadata_coverage,
    validate_image_mask_pairs,
)


def _count_instance_ids(mask: np.ndarray, min_area_px: int = 1) -> int:
    """
    Count GT objects using instance IDs encoded in mask values.

    Background is assumed to be 0.
    Each non-zero unique value is treated as one object instance.
    """
    if mask.ndim == 3:
        # Instance-indexed masks are expected to be single-channel; keep the first channel
        # for compatibility with image loaders that return HxWxC arrays.
        mask = mask[..., 0]

    values = mask.astype(np.int64)
    ids, counts = np.unique(values, return_counts=True)

    # Exclude background id 0, then apply optional area filtering per instance id.
    valid = (ids != 0) & (counts >= int(min_area_px))
    return int(np.count_nonzero(valid))


def compute_object_counts_from_masks(dataset_root: Path, min_area_px: int = 1) -> pd.DataFrame:
    """Build per-sample GT object counts from instance-indexed .tif masks."""
    images_root = dataset_root / "images"
    masks_root = dataset_root / "masks"

    sample_filenames = validate_image_mask_pairs(images_root, masks_root)

    rows: list[dict[str, int | str]] = []
    for filename in sample_filenames:
        mask_path = masks_root / filename
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise DatasetValidationError(f"Failed to read mask: {mask_path}")

        count = _count_instance_ids(mask, min_area_px=min_area_px)
        rows.append({FILENAME_COL: filename, GT_OBJECT_COUNT_COL: count})

    return pd.DataFrame(rows)


def classify_density_tertiles(counts: pd.Series) -> pd.Series:
    """Classify counts into Low/Mid/High using dataset tertile boundaries."""
    sorted_counts = counts.sort_values().reset_index(drop=True)
    n = len(sorted_counts)

    t1 = int(sorted_counts.iloc[n // 3])
    t2 = int(sorted_counts.iloc[(2 * n) // 3])

    def _label(x: float) -> str:
        if x <= t1:
            return "Low"
        if x <= t2:
            return "Mid"
        return "High"

    return counts.apply(_label)


def classify_density_kmeans(counts: pd.Series, random_seed: int = 42) -> pd.Series:
    """Classify counts into Low/Mid/High using 3-cluster KMeans."""
    X = counts.to_numpy(dtype=float).reshape(-1, 1)

    kmeans = KMeans(n_clusters=3, random_state=random_seed, init="k-means++", max_iter=500)
    labels = kmeans.fit_predict(X)

    tmp = pd.DataFrame({"label": labels, "count": counts.to_numpy(dtype=float)})
    means = tmp.groupby("label")["count"].mean().sort_values()
    mapping = {old: new for old, new in zip(means.index.tolist(), ["Low", "Mid", "High"])}

    return pd.Series(labels, index=counts.index).map(mapping)


def generate_density_metadata(
    dataset_root: Path,
    output_csv: Path,
    *,
    method: str = "kmeans",
    random_seed: int = 42,
    min_area_px: int = 1,
    include_comparison_columns: bool = True,
) -> pd.DataFrame:
    """
    Generate canonical density metadata from the unsplit dataset.

    Output always includes:
      - filename
      - gt_object_count
      - density_class
    """
    if method not in {"tertile", "kmeans"}:
        raise ValueError("method must be one of: 'tertile', 'kmeans'")

    dataset_sample_filenames = validate_image_mask_pairs(
        dataset_root / "images",
        dataset_root / "masks",
    )

    df = compute_object_counts_from_masks(dataset_root=dataset_root, min_area_px=min_area_px)

    tertile = classify_density_tertiles(df[GT_OBJECT_COUNT_COL])
    kmeans = classify_density_kmeans(df[GT_OBJECT_COUNT_COL], random_seed=random_seed)

    df[DENSITY_CLASS_TERTILE_COL] = tertile
    df[DENSITY_CLASS_KMEANS_COL] = kmeans
    df[DENSITY_CLASS_COL] = df[DENSITY_CLASS_KMEANS_COL] if method == "kmeans" else df[DENSITY_CLASS_TERTILE_COL]

    metadata_filenames = df[FILENAME_COL].astype(str).tolist()
    ensure_metadata_coverage(
        sample_filenames=dataset_sample_filenames,
        metadata_filenames=metadata_filenames,
        metadata_name="density metadata",
    )

    if not include_comparison_columns:
        df = df[[FILENAME_COL, GT_OBJECT_COUNT_COL, DENSITY_CLASS_COL]]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df
