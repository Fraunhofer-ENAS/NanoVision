"""Pairwise and GT-width-normalized centerline distance matrices."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


from cnt_project.evaluation.core.centerline_distances import (
    DistanceMeasure,
    centerline_distance,
)


def _validate_instance_masks(
    masks: torch.Tensor,
    *,
    name: str,
) -> None:
    """
    Validate a collection of binary instance-centerline masks.

    Expected shape:
        (number_of_objects, 1, height, width)
    """
    if not isinstance(masks, torch.Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor."
        )

    if masks.ndim != 4 or masks.shape[1] != 1:
        raise ValueError(
            f"{name} must have shape (L, 1, H, W); "
            f"received {tuple(masks.shape)}."
        )

    if masks.shape[2] == 0 or masks.shape[3] == 0:
        raise ValueError(
            f"{name} must have non-empty spatial dimensions."
        )

    if masks.shape[0] > 0:
        non_empty_instances = (
            masks
            .flatten(start_dim=1)
            .bool()
            .any(dim=1)
        )

        if not bool(non_empty_instances.all()):
            raise ValueError(
                f"Every instance in {name} must contain "
                "at least one centerline pixel."
            )


def pairwise_centerline_distances(
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    *,
    measure: DistanceMeasure = "chamfer",
) -> np.ndarray:
    """
    Calculate the raw Chamfer/hausdorf distance for every GT/prediction pair.

    Args:
        ground_truth:
            Tensor with shape (L_gt, 1, H, W).

        prediction:
            Tensor with shape (L_pred, 1, H, W).

    Returns:
        Matrix with shape (L_gt, L_pred).

        Rows represent ground-truth objects.
        Columns represent predicted objects.
        Values are symmetric mean distances in pixels.
    """
    _validate_instance_masks(
        ground_truth,
        name="ground_truth",
    )

    _validate_instance_masks(
        prediction,
        name="prediction",
    )

    if ground_truth.shape[2:] != prediction.shape[2:]:
        raise ValueError(
            "Ground-truth and prediction spatial shapes must match; "
            f"received {tuple(ground_truth.shape[2:])} and "
            f"{tuple(prediction.shape[2:])}."
        )

    number_of_ground_truths = ground_truth.shape[0]
    number_of_predictions = prediction.shape[0]

    distances = np.empty(
        (
            number_of_ground_truths,
            number_of_predictions,
        ),
        dtype=np.float64,
    )

    for ground_truth_index, ground_truth_centerline in enumerate(
        ground_truth
    ):
        for prediction_index, prediction_centerline in enumerate(
            prediction
        ):
            distances[
                ground_truth_index,
                prediction_index,
            ] = centerline_distance(
                ground_truth_centerline,
                prediction_centerline,
                measure=measure,
            )

    return distances


def normalize_distances_by_ground_truth_width(
    distance_matrix: np.ndarray,
    *,
    ground_truth_widths: Sequence[float],
) -> np.ndarray:
    """
    Express every Chamfer distance in units of its GT object's width.

    Matrix rows must represent GT objects and columns must represent
    predicted objects.

    normalized_distance[i, j] =
        distance_matrix[i, j] / ground_truth_widths[i]
    """
    distance_matrix = np.asarray(
        distance_matrix,
        dtype=np.float64,
    )

    if distance_matrix.ndim != 2:
        raise ValueError(
            "distance_matrix must be two-dimensional; "
            f"received shape {distance_matrix.shape}."
        )

    if not np.isfinite(distance_matrix).all():
        raise ValueError(
            "distance_matrix must contain only finite values."
        )

    if np.any(distance_matrix < 0.0):
        raise ValueError(
            "distance_matrix values must be non-negative."
        )

    widths = np.asarray(
        ground_truth_widths,
        dtype=np.float64,
    )

    if widths.ndim != 1:
        raise ValueError(
            "ground_truth_widths must be one-dimensional; "
            f"received shape {widths.shape}."
        )

    if len(widths) != distance_matrix.shape[0]:
        raise ValueError(
            "There must be exactly one width for every "
            "ground-truth object; "
            f"received {len(widths)} widths for "
            f"{distance_matrix.shape[0]} GT objects."
        )

    if not np.isfinite(widths).all():
        raise ValueError(
            "ground_truth_widths must contain only finite values."
        )

    if np.any(widths <= 0.0):
        raise ValueError(
            "Every ground-truth width must be greater than zero."
        )

    return distance_matrix / widths[:, np.newaxis]

