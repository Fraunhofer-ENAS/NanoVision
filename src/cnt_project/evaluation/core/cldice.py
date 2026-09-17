from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from skimage.morphology import skeletonize

from cnt_project.evaluation.core.object_matching import (
    match_similarity_matrix,
)
from cnt_project.evaluation.core.object_metrics import (
    calculate_object_metrics,
)


def _as_binary_2d(
    mask: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """
    Convert an input mask to a two-dimensional Boolean mask.
    """
    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D mask, "
            f"but received shape {mask.shape}."
        )

    return mask.astype(bool, copy=False)


def centerline_coverage(
    mask: np.ndarray,
    skeleton: np.ndarray,
) -> float:
    """
    Return the fraction of skeleton pixels contained in mask.

    Both inputs must be two-dimensional binary masks.
    """
    mask = _as_binary_2d(
        mask,
        name="mask",
    )

    skeleton = _as_binary_2d(
        skeleton,
        name="skeleton",
    )

    if mask.shape != skeleton.shape:
        raise ValueError(
            "mask and skeleton must have the same shape: "
            f"{mask.shape} and {skeleton.shape}."
        )

    skeleton_size = np.count_nonzero(
        skeleton
    )

    if skeleton_size == 0:
        return 0.0

    overlap_size = np.count_nonzero(
        mask & skeleton
    )

    return float(
        overlap_size / skeleton_size
    )


def cldice(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """
    Calculate centerline Dice between two 2-D binary masks.

    Topology precision measures how much of the predicted
    skeleton lies inside the ground-truth mask.

    Topology sensitivity measures how much of the ground-truth
    skeleton lies inside the predicted mask.
    """
    prediction = _as_binary_2d(
        prediction,
        name="prediction",
    )

    ground_truth = _as_binary_2d(
        ground_truth,
        name="ground_truth",
    )

    if prediction.shape != ground_truth.shape:
        raise ValueError(
            "prediction and ground_truth must have the same shape: "
            f"{prediction.shape} and {ground_truth.shape}."
        )

    prediction_is_empty = not prediction.any()
    ground_truth_is_empty = not ground_truth.any()

    if prediction_is_empty and ground_truth_is_empty:
        return 1.0

    if prediction_is_empty or ground_truth_is_empty:
        return 0.0

    prediction_skeleton = skeletonize(
        prediction
    )

    ground_truth_skeleton = skeletonize(
        ground_truth
    )

    topology_precision = centerline_coverage(
        ground_truth,
        prediction_skeleton,
    )

    topology_sensitivity = centerline_coverage(
        prediction,
        ground_truth_skeleton,
    )

    denominator = (
        topology_precision
        + topology_sensitivity
    )

    if denominator == 0.0:
        return 0.0

    return float(
        2.0
        * topology_precision
        * topology_sensitivity
        / denominator
    )


@dataclass(frozen=True)
class ClDiceObjectF1Result:
    """
    Result of clDice-based one-to-one instance matching.
    """

    threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    object_f1: float
    matched_cldice_scores: tuple[float, ...]


def pairwise_cldice_matrix(
    ground_truth_masks: Sequence[np.ndarray],
    prediction_masks: Sequence[np.ndarray],
) -> np.ndarray:
    """
    Calculate clDice for every GT/prediction object pair.

    Each object is skeletonized only once and its skeleton is
    reused across all pairwise comparisons.

    Returns an array with shape:
        (number_of_ground_truth_objects,
         number_of_predicted_objects)
    """
    binary_gt_masks = [
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
        binary_gt_masks
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


    number_of_gt_objects = len(
        binary_gt_masks
    )

    number_of_predictions = len(
        binary_prediction_masks
    )

    score_matrix = np.zeros(
        (
            number_of_gt_objects,
            number_of_predictions,
        ),
        dtype=float,
    )

    if (
        number_of_gt_objects == 0
        or number_of_predictions == 0
    ):
        return score_matrix

    all_masks = (
        binary_gt_masks
        + binary_prediction_masks
    )

    expected_shape = all_masks[0].shape

    for index, mask in enumerate(
        binary_gt_masks
    ):
        if mask.shape != expected_shape:
            raise ValueError(
                "All ground-truth object masks must have "
                "the same shape; "
                f"expected {expected_shape}, but "
                f"ground_truth_masks[{index}] has "
                f"shape {mask.shape}."
            )

    for index, mask in enumerate(
        binary_prediction_masks
    ):
        if mask.shape != expected_shape:
            raise ValueError(
                "All prediction object masks must have the "
                "same shape as the ground-truth masks; "
                f"expected {expected_shape}, but "
                f"prediction_masks[{index}] has "
                f"shape {mask.shape}."
            )


    gt_skeletons = [
        skeletonize(mask)
        for mask in binary_gt_masks
    ]

    prediction_skeletons = [
        skeletonize(mask)
        for mask in binary_prediction_masks
    ]

    for gt_index, (
        ground_truth_mask,
        ground_truth_skeleton,
    ) in enumerate(
        zip(
            binary_gt_masks,
            gt_skeletons,
        )
    ):
        for prediction_index, (
            prediction_mask,
            prediction_skeleton,
        ) in enumerate(
            zip(
                binary_prediction_masks,
                prediction_skeletons,
            )
        ):
            topology_precision = (
                centerline_coverage(
                    ground_truth_mask,
                    prediction_skeleton,
                )
            )

            topology_sensitivity = (
                centerline_coverage(
                    prediction_mask,
                    ground_truth_skeleton,
                )
            )

            denominator = (
                topology_precision
                + topology_sensitivity
            )

            if denominator > 0.0:
                score_matrix[
                    gt_index,
                    prediction_index,
                ] = (
                    2.0
                    * topology_precision
                    * topology_sensitivity
                    / denominator
                )

    return score_matrix

def cldice_object_f1(
    ground_truth_masks: Sequence[np.ndarray],
    prediction_masks: Sequence[np.ndarray],
    *,
    threshold: float,
) -> ClDiceObjectF1Result:
    """
    Calculate object F1 using clDice-based one-to-one matching.

    This compatibility function is retained for existing callers.
    Matching and metric calculation delegate to the shared
    object-level implementations.
    """
    threshold = float(threshold)

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1 inclusive; "
            f"received {threshold}."
        )

    score_matrix = pairwise_cldice_matrix(
        ground_truth_masks,
        prediction_masks,
    )

    matches = match_similarity_matrix(
        score_matrix,
        threshold=threshold,
    )

    metrics = calculate_object_metrics(
        number_of_ground_truths=len(
            ground_truth_masks
        ),
        number_of_predictions=len(
            prediction_masks
        ),
        number_of_matches=len(matches),
    )

    return ClDiceObjectF1Result(
        threshold=threshold,
        true_positives=metrics.true_positives,
        false_positives=metrics.false_positives,
        false_negatives=metrics.false_negatives,
        object_f1=metrics.f1_score,
        matched_cldice_scores=tuple(
            match.value
            for match in matches
        ),
    )