"""Pixel-level metrics for CNT centerline predictions.

Buffered-DICE:
    - similar to this, buffered IoU is used by "Learning and Aggregating Lane Graphs for Urban Automated Driving".
        https://arxiv.org/abs/2302.06175
    

"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch


@dataclass(frozen=True)
class BufferedDiceResult:
    """Buffered Dice scores over distance thresholds and their mean."""

    thresholds: tuple[float, ...]
    dice_scores: tuple[float, ...]
    mean_dice: float


def _validate_masks(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
) -> None:
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            "Masks must have the same shape; received "
            f"{tuple(prediction.shape)} and {tuple(ground_truth.shape)}."
        )
    if prediction.ndim != 3 or prediction.shape[0] != 1:
        raise ValueError(
            "A mask must have shape (1, H, W); "
            f"received {tuple(prediction.shape)}."
        )
    if not isinstance(prediction, torch.Tensor):
        raise TypeError("prediction must be a torch.Tensor.")

    if not isinstance(ground_truth, torch.Tensor):
        raise TypeError("ground_truth must be a torch.Tensor.")

    if prediction.device != ground_truth.device:
        raise ValueError(
            "prediction and ground_truth must be on the same device."
        )


def _buffered_fraction(
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
    threshold: float,
) -> float:
    source_points = torch.nonzero(source_mask[0].bool(), as_tuple=False)
    target_points = torch.nonzero(target_mask[0].bool(), as_tuple=False)

    if source_points.shape[0] == 0:
        return 1.0 if target_points.shape[0] == 0 else 0.0
    if target_points.shape[0] == 0:
        return 0.0

    distances = torch.cdist(source_points.float(), target_points.float())
    minimum_distances = distances.amin(dim=1)
    # changed < to <= because in the case of perfectly overlaped centerlined 0<0 would be false 
    return (minimum_distances <= threshold).float().mean().item()


def buffered_dice_index(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    threshold: float,
) -> float:
    """Return the buffered Dice index for two binary centerline masks.

    Precision is the fraction of prediction pixels less than or equal ``threshold``
    pixels from a ground-truth pixel. Sensitivity is calculated in the
    opposite direction. Two empty centerlines have score one; when only one
    centerline is empty, the score is zero.
    """
    _validate_masks(prediction, ground_truth)
    threshold = float(threshold)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative.")

    precision = _buffered_fraction(prediction, ground_truth, threshold)
    sensitivity = _buffered_fraction(ground_truth, prediction, threshold)
    denominator = precision + sensitivity
    return 0.0 if denominator == 0 else 2 * precision * sensitivity / denominator


def evaluate_buffered_dice(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    thresholds: Iterable[float],
) -> BufferedDiceResult:
    """Calculate buffered Dice scores and their mean over thresholds.
    Args:
        prediction: Binary mask of all predicted centerlines, shape (1, H, W).
        ground_truth: Binary mask of all ground-truth centerlines, shape (1, H, W).
        thresholds: Iterable of distance thresholds in pixels.

    Returns:
        BufferedDiceResult containing Dice scores for each threshold and their mean.
        DICE score = 2 * (precision * sensitivity) / (precision + sensitivity), 
        it ranges from 0 to 1.
    """
    thresholds = tuple(float(threshold) for threshold in thresholds)
    if not thresholds:
        raise ValueError("thresholds must contain at least one value.")

    dice_scores = tuple(
        buffered_dice_index(prediction, ground_truth, threshold)
        for threshold in thresholds
    )
    return BufferedDiceResult(
        thresholds=thresholds,
        dice_scores=dice_scores,
        mean_dice=sum(dice_scores) / len(dice_scores),
    )

def mean_buffered_dice(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    thresholds: Iterable[float],
) -> float:
    """Return only the mean buffered Dice index over thresholds."""
    return evaluate_buffered_dice(
        prediction,
        ground_truth,
        thresholds,
    ).mean_dice
