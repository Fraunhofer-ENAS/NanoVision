"""Feature differences between matched segmentation objects."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence
from skimage.morphology import skeletonize
import cv2
import numpy as np

from cnt_project.evaluation.core.object_matching import (
    ObjectMatch,
)
# from cnt_project.preprocessing.metadata.length import (
#     GEODESIC_LENGTH_UM_COL,
#     calculate_componentwise_geodesic_length,
# )
from cnt_project.features.core.tree_length import (
    branching_skeleton_length,
    prune_branches,
    skeleton_graph,
)

from cnt_project.features.core.cnt_feature_extraction import (
    MICRONS_PER_PIXEL,
    polygon_pca_orientation_professional_from_coordinates,
)

@dataclass(frozen=True)
class ObjectFeatures:
    """Length and polygon-PCA orientation of one object."""

    length_um: float
    orientation_deg: float


@dataclass(frozen=True)
class MatchedFeatureErrors:
    """Mean pairwise feature errors over accepted matches."""

    matched_pair_count: int
    irregular_matched_prediction_count: int
    mean_absolute_length_error_um: float
    mean_absolute_relative_length_error: float
    mean_absolute_orientation_error_deg: float


def _validate_mask(
    mask: np.ndarray,
) -> np.ndarray:
    """Validate and convert a nonempty 2D mask to Boolean."""
    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError(
            "mask must be two-dimensional; "
            f"received shape {mask.shape}."
        )

    mask = mask.astype(
        bool,
        copy=False,
    )

    if not mask.any():
        raise ValueError(
            "mask must contain foreground pixels."
        )

    return mask


# def calculate_mask_length_um(
#     mask: np.ndarray,
# ) -> float:
#     """Calculate canonical component-wise geodesic length. return the um version"""
#     mask = _validate_mask(mask)
# 
#     result = (
#         calculate_componentwise_geodesic_length(
#             mask,
#             microns_per_pixel=MICRONS_PER_PIXEL,
#         )
#     )
# 
#     return float(
#         result[GEODESIC_LENGTH_UM_COL]
#     )
def calculate_mask_length_um(
    mask: np.ndarray,
) -> float:
    """Calculate branch-sum skeleton length in micrometres."""
    mask = _validate_mask(
        mask
    )

    result = branching_skeleton_length(
        mask,
        min_branch_length=None,
    )

    return float(
        result["tree_length"]
        * MICRONS_PER_PIXEL
    )

def calculate_polygon_pca_orientation_deg(
    mask: np.ndarray,
) -> float:
    """
    Calculate polygon-PCA orientation from complete mask contours.

    Contours are extracted with CHAIN_APPROX_NONE and passed to
    the canonical coordinate-based polygon-PCA implementation.
    """
    mask = _validate_mask(mask)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    contour_coordinates = [
        contour.reshape(-1, 2)
        for contour in contours
        if contour.shape[0] >= 2
    ]

    if not contour_coordinates:
        return float("nan")

    coordinates_xy = np.concatenate(
        contour_coordinates,
        axis=0,
    )

    return (
        polygon_pca_orientation_professional_from_coordinates(
            coordinates_xy
        )
    )

def calculate_object_features(
    masks: Sequence[np.ndarray],
) -> tuple[ObjectFeatures, ...]:
    """Calculate reusable features for every object mask."""
    return tuple(
        ObjectFeatures(
            length_um=calculate_mask_length_um(
                mask
            ),
            orientation_deg=(
                calculate_polygon_pca_orientation_deg(
                    mask
                )
            ),
        )
        for mask in masks
    )


def axial_orientation_error_deg(
    first_angle_deg: float,
    second_angle_deg: float,
) -> float:
    """Calculate absolute axial orientation error."""
    if not (
        math.isfinite(first_angle_deg)
        and math.isfinite(second_angle_deg)
    ):
        return float("nan")

    difference = (
        abs(
            first_angle_deg
            - second_angle_deg
        )
        % 180.0
    )

    return float(
        min(
            difference,
            180.0 - difference,
        )
    )


def _mean_or_nan(
    values: Sequence[float],
) -> float:
    """Calculate the mean of finite values or return NaN."""
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return float("nan")

    return float(np.mean(values))


def is_irregular_prediction_mask(
    mask: np.ndarray,
    *,
    min_branch_length_px: float = 5,
) -> bool:
    """
    Determine whether a prediction remains branched after pruning.

    Processing:
    1. Skeletonize the prediction mask.
    2. Convert the skeleton into a graph.
    3. Remove terminal branches shorter than the selected length.
    4. Classify the object as irregular when the pruned graph
       contains more than two endpoints.
    """
    mask = _validate_mask(
        mask
    )

    raw_skeleton = skeletonize(
        mask
    )

    graph = skeleton_graph(
        raw_skeleton
    )

    pruned_graph = prune_branches(
        graph,
        min_length=min_branch_length_px,
    )

    pruned_endpoint_count = sum(
        pruned_graph.degree[node] == 1
        for node in pruned_graph.nodes
    )

    return bool(
        pruned_endpoint_count > 2
    )


def classify_irregular_prediction_masks(
    masks: Sequence[np.ndarray],
    *,
    min_branch_length_px: float = 5,
) -> tuple[bool, ...]:
    """
    Classify all prediction masks once for reuse across matching
    metrics and thresholds.
    """
    return tuple(
        is_irregular_prediction_mask(
            mask,
            min_branch_length_px=(
                min_branch_length_px
            ),
        )
        for mask in masks
    )


def evaluate_matched_feature_errors(
    *,
    ground_truth_features: Sequence[ObjectFeatures],
    prediction_features: Sequence[ObjectFeatures],
    prediction_irregular_flags: Sequence[bool],
    matches: Sequence[ObjectMatch],
) -> MatchedFeatureErrors:
    """
    Calculate mean feature error across accepted matched pairs.

    Errors are calculated independently for every matched pair
    and then averaged.
    """
    if (
        len(prediction_irregular_flags)
        != len(prediction_features)
    ):
        raise ValueError(
            "prediction_irregular_flags and "
            "prediction_features must have the same length. "
            f"Received {len(prediction_irregular_flags)} flags "
            f"and {len(prediction_features)} features."
        )
    
    absolute_length_errors = []
    relative_length_errors = []
    orientation_errors = []
    irregular_matched_prediction_count = 0

    for match in matches:
        ground_truth = ground_truth_features[
            match.ground_truth_index
        ]

        prediction = prediction_features[
            match.prediction_index
        ]

        if prediction_irregular_flags[ match.prediction_index ]:
            irregular_matched_prediction_count += 1

        absolute_length_error = abs(
            prediction.length_um
            - ground_truth.length_um
        )

        absolute_length_errors.append(
            absolute_length_error
        )

        if ground_truth.length_um > 0.0:
            relative_length_errors.append(
                absolute_length_error
                / ground_truth.length_um
            )

        orientation_errors.append(
            axial_orientation_error_deg(
                ground_truth.orientation_deg,
                prediction.orientation_deg,
            )
        )

    return MatchedFeatureErrors(
        matched_pair_count=len(matches),
        irregular_matched_prediction_count=(
            irregular_matched_prediction_count
        ),
        mean_absolute_length_error_um=(
            _mean_or_nan(
                absolute_length_errors
            )
        ),
        mean_absolute_relative_length_error=(
            _mean_or_nan(
                relative_length_errors
            )
        ),
        mean_absolute_orientation_error_deg=(
            _mean_or_nan(
                orientation_errors
            )
        ),
    )