from __future__ import annotations

from pathlib import Path

import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.ndimage import label as connected_component_label
from skimage.morphology import skeletonize

from cnt_project.features.core.geodesic_length import (
    polygon_mask_geodesic_length,
)
from cnt_project.preprocessing.metadata.schemas import (
    FILENAME_COL,
)
from cnt_project.preprocessing.metadata.validation import (
    DatasetValidationError,
    validate_image_mask_pairs,
)


INSTANCE_ID_COL = "instance_id"
AREA_PX_COL = "area_px"
CONNECTED_COMPONENT_COUNT_COL = "connected_component_count"

SKELETON_PIXEL_COUNT_COL = "skeleton_pixel_count"

GEODESIC_LENGTH_PX_COL = "geodesic_length_px"
GEODESIC_LENGTH_UM_COL = "geodesic_length_um"


OBJECT_LENGTH_COLUMNS = [
    FILENAME_COL,
    INSTANCE_ID_COL,
    AREA_PX_COL,
    CONNECTED_COMPONENT_COUNT_COL,
    SKELETON_PIXEL_COUNT_COL,
    GEODESIC_LENGTH_PX_COL,
    GEODESIC_LENGTH_UM_COL,
]
GT_OBJECT_COUNT_COL = "gt_object_count"

TOTAL_AREA_PX_COL = "total_area_px"
MEAN_AREA_PX_COL = "mean_area_px"
MEDIAN_AREA_PX_COL = "median_area_px"
MAX_AREA_PX_COL = "max_area_px"

MEAN_SKELETON_PIXEL_COUNT_COL = (
    "mean_skeleton_pixel_count"
)
MEDIAN_SKELETON_PIXEL_COUNT_COL = (
    "median_skeleton_pixel_count"
)
MAX_SKELETON_PIXEL_COUNT_COL = (
    "max_skeleton_pixel_count"
)

MEAN_GEODESIC_LENGTH_PX_COL = (
    "mean_geodesic_length_px"
)
MEDIAN_GEODESIC_LENGTH_PX_COL = (
    "median_geodesic_length_px"
)
MAX_GEODESIC_LENGTH_PX_COL = (
    "max_geodesic_length_px"
)
P95_GEODESIC_LENGTH_PX_COL = (
    "p95_geodesic_length_px"
)

MEAN_GEODESIC_LENGTH_UM_COL = (
    "mean_geodesic_length_um"
)
MEDIAN_GEODESIC_LENGTH_UM_COL = (
    "median_geodesic_length_um"
)
MAX_GEODESIC_LENGTH_UM_COL = (
    "max_geodesic_length_um"
)
P95_GEODESIC_LENGTH_UM_COL = (
    "p95_geodesic_length_um"
)

MULTI_COMPONENT_OBJECT_COUNT_COL = (
    "multi_component_object_count"
)
MULTI_COMPONENT_OBJECT_FRACTION_COL = (
    "multi_component_object_fraction"
)
MAX_CONNECTED_COMPONENT_COUNT_COL = (
    "max_connected_component_count"
)

# define the image-level schema
IMAGE_LENGTH_COLUMNS = [
    FILENAME_COL,
    GT_OBJECT_COUNT_COL,
    TOTAL_AREA_PX_COL,
    MEAN_AREA_PX_COL,
    MEDIAN_AREA_PX_COL,
    MAX_AREA_PX_COL,
    MEAN_SKELETON_PIXEL_COUNT_COL,
    MEDIAN_SKELETON_PIXEL_COUNT_COL,
    MAX_SKELETON_PIXEL_COUNT_COL,
    MEAN_GEODESIC_LENGTH_PX_COL,
    MEDIAN_GEODESIC_LENGTH_PX_COL,
    MAX_GEODESIC_LENGTH_PX_COL,
    P95_GEODESIC_LENGTH_PX_COL,
    MEAN_GEODESIC_LENGTH_UM_COL,
    MEDIAN_GEODESIC_LENGTH_UM_COL,
    MAX_GEODESIC_LENGTH_UM_COL,
    P95_GEODESIC_LENGTH_UM_COL,
    MULTI_COMPONENT_OBJECT_COUNT_COL,
    MULTI_COMPONENT_OBJECT_FRACTION_COL,
    MAX_CONNECTED_COMPONENT_COUNT_COL,
]

def _normalize_instance_mask(mask: np.ndarray) -> np.ndarray:
    """
    Normalize an instance-indexed mask to a two-dimensional array.

    Background is expected to use instance ID 0. Every positive unique value
    is treated as a separate object instance.
    """
    mask = np.asarray(mask)

    if mask.ndim == 3:
        # Instance masks should normally be single-channel. Keep channel zero
        # for compatibility with readers that return HxWxC arrays.
        mask = mask[..., 0]

    if mask.ndim != 2:
        raise DatasetValidationError(
            "Instance mask must be two-dimensional after normalization, "
            f"got shape {mask.shape}."
        )

    return mask


def extract_instance_ids(mask: np.ndarray) -> np.ndarray:
    """
    Return sorted nonzero instance IDs from an instance-indexed mask.
    """
    normalized_mask = _normalize_instance_mask(mask)

    instance_ids = np.unique(normalized_mask)
    instance_ids = instance_ids[instance_ids != 0]

    return np.sort(instance_ids)


def extract_instance_mask(
    mask: np.ndarray,
    instance_id: int,
) -> np.ndarray:
    """
    Extract one object instance as a boolean mask.
    """
    normalized_mask = _normalize_instance_mask(mask)

    if int(instance_id) == 0:
        raise ValueError(
            "instance_id 0 represents the background and cannot be "
            "extracted as an object."
        )

    instance_mask = normalized_mask == instance_id

    if not np.any(instance_mask):
        raise DatasetValidationError(
            f"Instance ID {instance_id} is not present in the mask."
        )

    return instance_mask


def calculate_skeleton_pixel_count(
    instance_mask: np.ndarray,
) -> int:
    """
    Count all foreground pixels in an object's skeleton.

    This measurement includes skeleton pixels from every disconnected component
    and every branch. It is therefore a skeleton-occupancy measurement, not a
    geometrically weighted end-to-end path length.
    """
    binary_mask = np.asarray(
        instance_mask,
        dtype=bool,
    )

    if binary_mask.ndim != 2:
        raise ValueError(
            "instance_mask must be two-dimensional, "
            f"got shape {binary_mask.shape}."
        )

    if not np.any(binary_mask):
        return 0

    skeleton = skeletonize(binary_mask)

    return int(np.count_nonzero(skeleton))


def calculate_componentwise_geodesic_length(
    instance_mask: np.ndarray,
    *,
    microns_per_pixel: float,
    geodesic_alpha: float = 2.0,
    geodesic_eps: float = 1e-3,
) -> dict[str, int | float]:
    """
    Calculate total weighted geodesic length across disconnected components.

    Each 8-connected component is measured independently with
    ``polygon_mask_geodesic_length``. The component lengths are summed because
    all components carrying the same instance ID represent the same annotated
    CNT object.

    Notes
    -----
    Summing components resolves the previous behavior where the geodesic
    algorithm effectively returned a path from only one connected component.

    Within each connected component, the geodesic implementation still reports
    one selected path. It does not sum separate branches inside the same
    connected component.
    """
    binary_mask = np.asarray(
        instance_mask,
        dtype=bool,
    )

    if binary_mask.ndim != 2:
        raise ValueError(
            "instance_mask must be two-dimensional, "
            f"got shape {binary_mask.shape}."
        )

    if microns_per_pixel <= 0:
        raise ValueError(
            "microns_per_pixel must be greater than zero, "
            f"got {microns_per_pixel}."
        )

    if geodesic_alpha < 0:
        raise ValueError(
            "geodesic_alpha must be greater than or equal to zero, "
            f"got {geodesic_alpha}."
        )

    if geodesic_eps <= 0:
        raise ValueError(
            "geodesic_eps must be greater than zero, "
            f"got {geodesic_eps}."
        )

    if not np.any(binary_mask):
        return {
            CONNECTED_COMPONENT_COUNT_COL: 0,
            GEODESIC_LENGTH_PX_COL: 0.0,
            GEODESIC_LENGTH_UM_COL: 0.0,
        }

    # Use 8-connectivity to match the neighborhood used by the geodesic graph.
    labeled_components, component_count = connected_component_label(
        binary_mask,
        structure=np.ones(
            (3, 3),
            dtype=np.uint8,
        ),
    )

    total_length_px = 0.0
    total_length_um = 0.0

    for component_id in range(
        1,
        int(component_count) + 1,
    ):
        component_mask = (
            labeled_components == component_id
        )

        component_length_px, component_length_um = (
            polygon_mask_geodesic_length(
                component_mask,
                microns_per_pixel=float(
                    microns_per_pixel
                ),
                alpha=float(geodesic_alpha),
                eps=float(geodesic_eps),
            )
        )

        total_length_px += float(
            component_length_px
        )
        total_length_um += float(
            component_length_um
        )

    return {
        CONNECTED_COMPONENT_COUNT_COL: int(
            component_count
        ),
        GEODESIC_LENGTH_PX_COL: float(
            total_length_px
        ),
        GEODESIC_LENGTH_UM_COL: float(
            total_length_um
        ),
    }


def measure_instance_lengths(
    instance_mask: np.ndarray,
    *,
    microns_per_pixel: float,
    geodesic_alpha: float = 2.0,
    geodesic_eps: float = 1e-3,
) -> dict[str, int | float]:
    """
    Calculate object-level area, skeleton occupancy, and geodesic length.

    The skeleton-pixel count includes all disconnected components. Geodesic
    length is calculated per connected component and summed across components.
    """
    binary_mask = np.asarray(
        instance_mask,
        dtype=bool,
    )

    if binary_mask.ndim != 2:
        raise ValueError(
            "instance_mask must be two-dimensional, "
            f"got shape {binary_mask.shape}."
        )

    if microns_per_pixel <= 0:
        raise ValueError(
            "microns_per_pixel must be greater than zero, "
            f"got {microns_per_pixel}."
        )

    area_px = int(
        np.count_nonzero(binary_mask)
    )

    if area_px == 0:
        return {
            AREA_PX_COL: 0,
            CONNECTED_COMPONENT_COUNT_COL: 0,
            SKELETON_PIXEL_COUNT_COL: 0,
            GEODESIC_LENGTH_PX_COL: 0.0,
            GEODESIC_LENGTH_UM_COL: 0.0,
        }

    skeleton_pixel_count = (
        calculate_skeleton_pixel_count(
            binary_mask
        )
    )

    geodesic_measurements = (
        calculate_componentwise_geodesic_length(
            binary_mask,
            microns_per_pixel=microns_per_pixel,
            geodesic_alpha=geodesic_alpha,
            geodesic_eps=geodesic_eps,
        )
    )

    return {
        AREA_PX_COL: area_px,
        SKELETON_PIXEL_COUNT_COL: (
            skeleton_pixel_count
        ),
        **geodesic_measurements,
    }


def measure_mask_instances(
    mask: np.ndarray,
    *,
    filename: str,
    microns_per_pixel: float,
    min_area_px: int = 1,
    geodesic_alpha: float = 2.0,
    geodesic_eps: float = 1e-3,
) -> pd.DataFrame:
    """
    Calculate object-level length measurements for one instance mask.

    One output row is generated for every nonzero instance ID whose total area
    across all of its components is at least ``min_area_px``.
    """
    if min_area_px <= 0:
        raise ValueError(
            "min_area_px must be greater than zero, "
            f"got {min_area_px}."
        )

    normalized_mask = _normalize_instance_mask(
        mask
    )

    rows: list[
        dict[str, int | float | str]
    ] = []

    for instance_id in extract_instance_ids(
        normalized_mask
    ):
        instance_mask = extract_instance_mask(
            normalized_mask,
            int(instance_id),
        )

        area_px = int(
            np.count_nonzero(instance_mask)
        )

        if area_px < min_area_px:
            continue

        measurements = measure_instance_lengths(
            instance_mask,
            microns_per_pixel=microns_per_pixel,
            geodesic_alpha=geodesic_alpha,
            geodesic_eps=geodesic_eps,
        )

        rows.append(
            {
                FILENAME_COL: str(filename),
                INSTANCE_ID_COL: int(instance_id),
                **measurements,
            }
        )

    return pd.DataFrame(
        rows,
        columns=OBJECT_LENGTH_COLUMNS,
    )

def _validate_object_length_metadata(
    object_lengths_df: pd.DataFrame,
) -> None:
    """
    Validate object-level length metadata before image aggregation.
    """
    required_columns = set(
        OBJECT_LENGTH_COLUMNS
    )

    missing_columns = (
        required_columns
        - set(object_lengths_df.columns)
    )

    if missing_columns:
        raise DatasetValidationError(
            "Object-length metadata is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if object_lengths_df.empty:
        raise DatasetValidationError(
            "Object-length metadata is empty."
        )

    duplicate_rows = object_lengths_df.duplicated(
        subset=[
            FILENAME_COL,
            INSTANCE_ID_COL,
        ]
    )

    if duplicate_rows.any():
        duplicates = (
            object_lengths_df.loc[
                duplicate_rows,
                [
                    FILENAME_COL,
                    INSTANCE_ID_COL,
                ],
            ]
            .head(10)
            .to_dict(
                orient="records"
            )
        )

        raise DatasetValidationError(
            "Object-length metadata contains duplicate "
            "(filename, instance_id) rows: "
            f"{duplicates}"
        )

    if object_lengths_df[FILENAME_COL].isna().any():
        raise DatasetValidationError(
            "Object-length metadata contains missing filenames."
        )

    if (
        pd.to_numeric(
            object_lengths_df[
                INSTANCE_ID_COL
            ],
            errors="coerce",
        ).isna().any()
    ):
        raise DatasetValidationError(
            "Object-length metadata contains invalid instance IDs."
        )

def aggregate_object_lengths_by_image(
    object_lengths_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate object-level length metadata into one row per image.

    Parameters
    ----------
    object_lengths_df
        Object-level DataFrame produced by
        ``generate_object_length_metadata``.

    Returns
    -------
    pandas.DataFrame
        One row per image containing object-count, area, length, and
        disconnected-component summary statistics.
    """
    _validate_object_length_metadata(
        object_lengths_df
    )

    rows: list[
        dict[str, int | float | str]
    ] = []

    grouped = object_lengths_df.groupby(
        FILENAME_COL,
        sort=True,
    )

    for filename, image_objects in grouped:
        area_values = image_objects[
            AREA_PX_COL
        ].to_numpy(
            dtype=float
        )

        skeleton_values = image_objects[
            SKELETON_PIXEL_COUNT_COL
        ].to_numpy(
            dtype=float
        )

        geodesic_px_values = image_objects[
            GEODESIC_LENGTH_PX_COL
        ].to_numpy(
            dtype=float
        )

        geodesic_um_values = image_objects[
            GEODESIC_LENGTH_UM_COL
        ].to_numpy(
            dtype=float
        )

        component_counts = image_objects[
            CONNECTED_COMPONENT_COUNT_COL
        ].to_numpy(
            dtype=int
        )

        object_count = int(
            len(image_objects)
        )

        multi_component_count = int(
            np.count_nonzero(
                component_counts > 1
            )
        )

        multi_component_fraction = (
            float(
                multi_component_count
                / object_count
            )
            if object_count > 0
            else 0.0
        )

        rows.append(
            {
                FILENAME_COL: str(filename),
                GT_OBJECT_COUNT_COL: object_count,
                TOTAL_AREA_PX_COL: int(
                    np.sum(area_values)
                ),
                MEAN_AREA_PX_COL: float(
                    np.mean(area_values)
                ),
                MEDIAN_AREA_PX_COL: float(
                    np.median(area_values)
                ),
                MAX_AREA_PX_COL: int(
                    np.max(area_values)
                ),
                MEAN_SKELETON_PIXEL_COUNT_COL: float(
                    np.mean(
                        skeleton_values
                    )
                ),
                MEDIAN_SKELETON_PIXEL_COUNT_COL: float(
                    np.median(
                        skeleton_values
                    )
                ),
                MAX_SKELETON_PIXEL_COUNT_COL: int(
                    np.max(
                        skeleton_values
                    )
                ),
                MEAN_GEODESIC_LENGTH_PX_COL: float(
                    np.mean(
                        geodesic_px_values
                    )
                ),
                MEDIAN_GEODESIC_LENGTH_PX_COL: float(
                    np.median(
                        geodesic_px_values
                    )
                ),
                MAX_GEODESIC_LENGTH_PX_COL: float(
                    np.max(
                        geodesic_px_values
                    )
                ),
                P95_GEODESIC_LENGTH_PX_COL: float(
                    np.percentile(
                        geodesic_px_values,
                        95,
                    )
                ),
                MEAN_GEODESIC_LENGTH_UM_COL: float(
                    np.mean(
                        geodesic_um_values
                    )
                ),
                MEDIAN_GEODESIC_LENGTH_UM_COL: float(
                    np.median(
                        geodesic_um_values
                    )
                ),
                MAX_GEODESIC_LENGTH_UM_COL: float(
                    np.max(
                        geodesic_um_values
                    )
                ),
                P95_GEODESIC_LENGTH_UM_COL: float(
                    np.percentile(
                        geodesic_um_values,
                        95,
                    )
                ),
                MULTI_COMPONENT_OBJECT_COUNT_COL: (
                    multi_component_count
                ),
                MULTI_COMPONENT_OBJECT_FRACTION_COL: (
                    multi_component_fraction
                ),
                MAX_CONNECTED_COMPONENT_COUNT_COL: int(
                    np.max(
                        component_counts
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=IMAGE_LENGTH_COLUMNS,
    )

def generate_object_length_metadata(
    dataset_root: str | Path,
    output_csv: str | Path | None = None,
    *,
    microns_per_pixel: float,
    min_area_px: int = 1,
    geodesic_alpha: float = 2.0,
    geodesic_eps: float = 1e-3,
) -> pd.DataFrame:
    """
    Generate object-level length metadata from canonical TIFF masks.

    The canonical dataset must contain exactly paired ``images/`` and
    ``masks/`` directories. Each nonzero mask value is treated as one object
    instance, even when that instance consists of several disconnected
    components.

    Parameters
    ----------
    dataset_root
        Canonical dataset root containing ``images/`` and ``masks/``.

    output_csv
        Optional CSV output path. When omitted, no file is written.

    microns_per_pixel
        Physical pixel size used for converting geodesic length to micrometres.

    min_area_px
        Minimum total area of an instance ID required for inclusion.

    geodesic_alpha
        Boundary-penalty strength used by the weighted geodesic algorithm.

    geodesic_eps
        Positive numerical stabilizer used by the weighted geodesic algorithm.

    Returns
    -------
    pandas.DataFrame
        One row per included object instance.
    """
    dataset_root = Path(
        dataset_root
    ).resolve()

    images_root = (
        dataset_root / "images"
    )
    masks_root = (
        dataset_root / "masks"
    )

    sample_filenames = (
        validate_image_mask_pairs(
            images_root,
            masks_root,
        )
    )

    per_mask_results: list[
        pd.DataFrame
    ] = []

    for filename in tqdm( sample_filenames, desc="Generating object length metadata", ):
        mask_path = (
            masks_root / filename
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_UNCHANGED,
        )

        if mask is None:
            raise DatasetValidationError(
                f"Failed to read mask: {mask_path}"
            )

        image_result = measure_mask_instances(
            mask,
            filename=filename,
            microns_per_pixel=microns_per_pixel,
            min_area_px=min_area_px,
            geodesic_alpha=geodesic_alpha,
            geodesic_eps=geodesic_eps,
        )

        per_mask_results.append(
            image_result
        )

    if per_mask_results:
        result = pd.concat(
            per_mask_results,
            ignore_index=True,
        )
    else:
        result = pd.DataFrame(
            columns=OBJECT_LENGTH_COLUMNS,
        )

    result = result.sort_values(
        [
            FILENAME_COL,
            INSTANCE_ID_COL,
        ]
    ).reset_index(
        drop=True
    )

    if output_csv is not None:
        output_csv = Path(
            output_csv
        )

        output_csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        result.to_csv(
            output_csv,
            index=False,
        )

    return result


def generate_image_length_metadata(
        dataset_root: str | Path,
        output_csv: str | Path | None = None,
        *,
        object_lengths_csv: str | Path | None = None,
        microns_per_pixel: float | None = None,
        min_area_px: int = 1,
        geodesic_alpha: float = 2.0,
        geodesic_eps: float = 1e-3,
    ) -> pd.DataFrame:
        """
        Generate image-level length metadata.

        Object-level metadata can either be loaded from an existing CSV or
        generated directly from the canonical dataset.

        Parameters
        ----------
        dataset_root
            Canonical dataset root.

        output_csv
            Optional output path for the image-level CSV.

        object_lengths_csv
            Optional path to an existing object-level metadata CSV.

            When supplied, object-level measurements are loaded from this file and
            no TIFF masks are processed again.

        microns_per_pixel
            Required only when ``object_lengths_csv`` is not supplied.

        min_area_px
            Minimum object area used when generating object-level measurements.

        geodesic_alpha
            Boundary penalty used when generating object-level measurements.

        geodesic_eps
            Numerical stabilizer used when generating object-level measurements.

        Returns
        -------
        pandas.DataFrame
            One row per image.
        """
        dataset_root = Path(
            dataset_root
        ).resolve()

        if object_lengths_csv is not None:
            object_lengths_csv = Path(
                object_lengths_csv
            )

            if not object_lengths_csv.exists():
                raise FileNotFoundError(
                    "Object-length metadata CSV not found: "
                    f"{object_lengths_csv}"
                )

            object_lengths_df = pd.read_csv(
                object_lengths_csv
            )

        else:
            if microns_per_pixel is None:
                raise ValueError(
                    "microns_per_pixel is required when object-level "
                    "metadata must be generated from TIFF masks."
                )

            object_lengths_df = (
                generate_object_length_metadata(
                    dataset_root=dataset_root,
                    output_csv=None,
                    microns_per_pixel=microns_per_pixel,
                    min_area_px=min_area_px,
                    geodesic_alpha=geodesic_alpha,
                    geodesic_eps=geodesic_eps,
                )
            )

        image_lengths_df = (
            aggregate_object_lengths_by_image(
                object_lengths_df
            )
        )

        expected_filenames = (
            validate_image_mask_pairs(
                dataset_root / "images",
                dataset_root / "masks",
            )
        )

        actual_filenames = (
            image_lengths_df[
                FILENAME_COL
            ]
            .astype(str)
            .tolist()
        )

        expected_set = set(
            expected_filenames
        )
        actual_set = set(
            actual_filenames
        )

        missing_filenames = sorted(
            expected_set
            - actual_set
        )

        unexpected_filenames = sorted(
            actual_set
            - expected_set
        )

        if missing_filenames or unexpected_filenames:
            problems: list[str] = []

            if missing_filenames:
                problems.append(
                    "missing images: "
                    f"{missing_filenames[:10]}"
                )

            if unexpected_filenames:
                problems.append(
                    "unexpected images: "
                    f"{unexpected_filenames[:10]}"
                )

            raise DatasetValidationError(
                "Image-level length metadata coverage mismatch: "
                + "; ".join(problems)
            )

        image_lengths_df = (
            image_lengths_df
            .sort_values(
                FILENAME_COL
            )
            .reset_index(
                drop=True
            )
        )

        if output_csv is not None:
            output_csv = Path(
                output_csv
            )

            output_csv.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            image_lengths_df.to_csv(
                output_csv,
                index=False,
            )

        return image_lengths_df