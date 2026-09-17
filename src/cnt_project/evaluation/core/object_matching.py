"""Hungarian one-to-one matching of segmentation objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class ObjectMatch:
    """One accepted ground-truth/prediction object pair."""

    ground_truth_index: int
    prediction_index: int
    value: float


def _as_binary_2d(
    mask: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Validate and convert a two-dimensional mask to Boolean."""
    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError(
            f"{name} must be two-dimensional; "
            f"received shape {mask.shape}."
        )

    if mask.shape[0] == 0 or mask.shape[1] == 0:
        raise ValueError(
            f"{name} must have non-empty spatial dimensions."
        )

    if not (
        np.issubdtype(mask.dtype, np.number)
        or mask.dtype == np.bool_
    ):
        raise TypeError(
            f"{name} must contain numeric or Boolean values."
        )

    if np.issubdtype(mask.dtype, np.number):
        if not np.isfinite(mask).all():
            raise ValueError(
                f"{name} must contain only finite values."
            )

    return mask.astype(bool, copy=False)


def calculate_iou(
    ground_truth_mask: np.ndarray,
    prediction_mask: np.ndarray,
) -> float:
    """Calculate IoU between two binary object masks."""
    ground_truth_mask = _as_binary_2d(
        ground_truth_mask,
        name="ground_truth_mask",
    )

    prediction_mask = _as_binary_2d(
        prediction_mask,
        name="prediction_mask",
    )

    if ground_truth_mask.shape != prediction_mask.shape:
        raise ValueError(
            "ground_truth_mask and prediction_mask must have "
            "the same shape; "
            f"received {ground_truth_mask.shape} and "
            f"{prediction_mask.shape}."
        )

    intersection = np.count_nonzero(
        ground_truth_mask & prediction_mask
    )

    union = np.count_nonzero(
        ground_truth_mask | prediction_mask
    )

    return float(
        intersection / union
        if union > 0
        else 1.0
    )


def pairwise_iou_matrix(
    ground_truth_masks: Sequence[np.ndarray],
    prediction_masks: Sequence[np.ndarray],
) -> np.ndarray:
    """
    Calculate IoU for every GT/prediction object pair.

    Returns a matrix with shape:
        (number_of_ground_truth_objects,
         number_of_predicted_objects)
    """
    binary_ground_truth_masks = [
        _as_binary_2d(
            mask,
            name=f"ground_truth_masks[{index}]",
        )
        for index, mask in enumerate(
            ground_truth_masks
        )
    ]

    binary_prediction_masks = [
        _as_binary_2d(
            mask,
            name=f"prediction_masks[{index}]",
        )
        for index, mask in enumerate(
            prediction_masks
        )
    ]

    for index, mask in enumerate(
        binary_ground_truth_masks
    ):
        if not mask.any():
            raise ValueError(
                f"ground_truth_masks[{index}] is empty."
            )

    for index, mask in enumerate(
        binary_prediction_masks
    ):
        if not mask.any():
            raise ValueError(
                f"prediction_masks[{index}] is empty."
            )

    number_of_ground_truths = len(
        binary_ground_truth_masks
    )

    number_of_predictions = len(
        binary_prediction_masks
    )

    iou_matrix = np.zeros(
        (
            number_of_ground_truths,
            number_of_predictions,
        ),
        dtype=np.float64,
    )

    if (
        number_of_ground_truths == 0
        or number_of_predictions == 0
    ):
        return iou_matrix

    expected_shape = binary_ground_truth_masks[0].shape

    for index, mask in enumerate(
        binary_ground_truth_masks
    ):
        if mask.shape != expected_shape:
            raise ValueError(
                "All ground-truth masks must have the same shape; "
                f"ground_truth_masks[0] has shape {expected_shape}, "
                f"but ground_truth_masks[{index}] has "
                f"shape {mask.shape}."
            )

    for index, mask in enumerate(
        binary_prediction_masks
    ):
        if mask.shape != expected_shape:
            raise ValueError(
                "All prediction masks must have the same shape as "
                "the ground-truth masks; "
                f"expected {expected_shape}, but "
                f"prediction_masks[{index}] has shape {mask.shape}."
            )

    for ground_truth_index, ground_truth_mask in enumerate(
        binary_ground_truth_masks
    ):
        for prediction_index, prediction_mask in enumerate(
            binary_prediction_masks
        ):
            iou_matrix[
                ground_truth_index,
                prediction_index,
            ] = calculate_iou(
                ground_truth_mask,
                prediction_mask,
            )

    return iou_matrix


def _validate_matrix(
    matrix: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Validate a rectangular pairwise metric matrix."""
    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            f"{name} must be two-dimensional; "
            f"received shape {matrix.shape}."
        )

    if not np.isfinite(matrix).all():
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return matrix


def match_similarity_matrix_matching_bonus(
    score_matrix: np.ndarray,
    *,
    threshold: float,
) -> list[ObjectMatch]:
    """
    Perform threshold-aware Hungarian similarity matching.

    Matrix rows represent GT objects and columns represent predictions.

    Priorities:
        1. Maximize the number of pairs with score >= threshold.
        2. Among those assignments, maximize total accepted score.
    """
    score_matrix = _validate_matrix(
        score_matrix,
        name="score_matrix",
    )

    if np.any(
        (score_matrix < 0.0)
        | (score_matrix > 1.0)
    ):
        raise ValueError(
            "score_matrix values must be between 0 and 1."
        )

    threshold = float(threshold)

    if (
        not math.isfinite(threshold)
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError(
            "threshold must be finite and between 0 and 1."
        )

    if score_matrix.size == 0:
        return []

    valid_pairs = score_matrix >= threshold

    maximum_match_count = min(
        score_matrix.shape
    )

    matching_bonus = (
        maximum_match_count + 1.0
    )

    objective = (
        valid_pairs.astype(np.float64)
        * (
            matching_bonus
            + score_matrix
        )
    )

    ground_truth_indices, prediction_indices = (
        linear_sum_assignment(
            -objective
        )
    )

    accepted = valid_pairs[
        ground_truth_indices,
        prediction_indices,
    ]

    return [
        ObjectMatch(
            ground_truth_index=int(ground_truth_index),
            prediction_index=int(prediction_index),
            value=float(
                score_matrix[
                    ground_truth_index,
                    prediction_index,
                ]
            ),
        )
        for ground_truth_index, prediction_index in zip(
            ground_truth_indices[accepted],
            prediction_indices[accepted],
        )
    ]


def match_similarity_matrix(
    score_matrix: np.ndarray,
    *,
    threshold: float,
) -> list[ObjectMatch]:
    """
    Perform threshold-first maximum-cardinality similarity matching.

    Matrix rows represent GT objects and columns represent predictions.

    Pairs with score greater than the threshold qualify before
    Hungarian assignment. The assignment maximizes the number of
    valid one-to-one pairs.

    A pair is valid when:

        score >= threshold
    """
    score_matrix = _validate_matrix(
        score_matrix,
        name="score_matrix",
    )

    if np.any(
        (score_matrix < 0.0)
        | (score_matrix > 1.0)
    ):
        raise ValueError(
            "score_matrix values must be between 0 and 1."
        )

    threshold = float(threshold)

    if (
        not math.isfinite(threshold)
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError(
            "threshold must be finite and between "
            "0 and 1 inclusive."
        )

    if score_matrix.size == 0:
        return []

    valid_pairs = (
        score_matrix >= threshold
    )

    cost_matrix = (
        -valid_pairs.astype(np.float64)
    )

    ground_truth_indices, prediction_indices = (
        linear_sum_assignment(
            cost_matrix
        )
    )

    accepted = valid_pairs[
        ground_truth_indices,
        prediction_indices,
    ]

    return [
        ObjectMatch(
            ground_truth_index=int(
                ground_truth_index
            ),
            prediction_index=int(
                prediction_index
            ),
            value=float(
                score_matrix[
                    ground_truth_index,
                    prediction_index,
                ]
            ),
        )
        for ground_truth_index, prediction_index in zip(
            ground_truth_indices[accepted],
            prediction_indices[accepted],
        )
    ]

def match_distance_matrix(
    distance_matrix: np.ndarray,
    *,
    threshold: float,
) -> list[ObjectMatch]:
    """
    Perform threshold-first maximum-cardinality distance matching.

    Matrix rows represent GT objects and columns represent predictions.

    Pairs with distance less than or equal to the threshold qualify
    before Hungarian assignment. The assignment maximizes the number
    of valid one-to-one pairs.

    A pair is valid when:

        distance <= threshold
    """
    distance_matrix = _validate_matrix(
        distance_matrix,
        name="distance_matrix",
    )

    if np.any(distance_matrix < 0.0):
        raise ValueError(
            "distance_matrix values must be non-negative."
        )

    threshold = float(threshold)

    if (
        not math.isfinite(threshold)
        or threshold < 0.0
    ):
        raise ValueError(
            "threshold must be finite and non-negative."
        )

    if distance_matrix.size == 0:
        return []

    valid_pairs = (
        distance_matrix <= threshold
    )

    # Valid pairs have cost -1 and invalid pairs have cost 0.
    # Minimizing this matrix maximizes the number of valid
    # one-to-one pairs.
    cost_matrix = (
        -valid_pairs.astype(np.float64)
    )

    ground_truth_indices, prediction_indices = (
        linear_sum_assignment(
            cost_matrix
        )
    )

    assigned_distances = distance_matrix[
        ground_truth_indices,
        prediction_indices,
    ]

    accepted = (
        assigned_distances <= threshold
    )

    return [
        ObjectMatch(
            ground_truth_index=int(
                ground_truth_index
            ),
            prediction_index=int(
                prediction_index
            ),
            value=float(distance),
        )
        for ground_truth_index, prediction_index, distance
        in zip(
            ground_truth_indices[accepted],
            prediction_indices[accepted],
            assigned_distances[accepted],
        )
    ]


def match_distance_matrix_matching_bonus(
    distance_matrix: np.ndarray,
    *,
    threshold: float,
) -> list[ObjectMatch]:
    """
    Perform threshold-aware Hungarian distance matching.

    Matrix rows represent GT objects and columns represent predictions.

    Priorities:
        1. Maximize the number of pairs with distance <= threshold.
        2. Among those assignments, minimize total accepted distance.
    """
    distance_matrix = _validate_matrix(
        distance_matrix,
        name="distance_matrix",
    )

    if np.any(distance_matrix < 0.0):
        raise ValueError(
            "distance_matrix values must be non-negative."
        )

    threshold = float(threshold)

    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError(
            "threshold must be finite and non-negative."
        )

    if distance_matrix.size == 0:
        return []

    valid_pairs = distance_matrix <= threshold

    maximum_match_count = min(
        distance_matrix.shape
    )

    matching_bonus = (
        maximum_match_count + 1.0
    )

    # quality becomes 1 - distance-matrix / threshold 
    if threshold == 0.0:
        quality = np.where(
            valid_pairs,
            1.0,
            0.0,
        )
    else:
        quality = np.where(
            valid_pairs,
            distance_matrix ,
            0.0,
        )

    # objective = (
    #     valid_pairs.astype(np.float64)
    #     * (
    #         matching_bonus
    #         + quality
    #     )
    # )

    ground_truth_indices, prediction_indices = (
        linear_sum_assignment(
            quality
        )
    )

    accepted = valid_pairs[
        ground_truth_indices,
        prediction_indices,
    ]

    return [
        ObjectMatch(
            ground_truth_index=int(ground_truth_index),
            prediction_index=int(prediction_index),
            value=float(
                distance_matrix[
                    ground_truth_index,
                    prediction_index,
                ]
            ),
        )
        for ground_truth_index, prediction_index in zip(
            ground_truth_indices[accepted],
            prediction_indices[accepted],
        )
    ]