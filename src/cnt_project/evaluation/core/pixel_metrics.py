"""Image-level pixel metrics for binary segmentation masks."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PixelMetricsResult:
    """Pixel-level confusion counts and segmentation metrics."""

    true_positive_pixels: int
    false_positive_pixels: int
    false_negative_pixels: int
    true_negative_pixels: int
    precision: float
    recall: float
    f1_score: float
    fdr: float
    iou: float
    dice: float


def binarize_mask(
    mask: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert a two-dimensional numeric mask into a Boolean mask.

    For an instance-label mask, the default threshold converts every
    positive integer label to foreground. For a probability map, the
    threshold controls foreground classification.
    """
    mask = np.asarray(mask)
    threshold = float(threshold)

    if mask.ndim != 2:
        raise ValueError(
            "mask must be two-dimensional; "
            f"received shape {mask.shape}."
        )

    if not math.isfinite(threshold):
        raise ValueError(
            "threshold must be finite."
        )

    if not np.issubdtype(mask.dtype, np.number):
        raise TypeError(
            "mask must contain numeric values."
        )

    if not np.isfinite(mask).all():
        raise ValueError(
            "mask must contain only finite values."
        )

    return mask > threshold


def _as_binary_2d(
    mask: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """
    Validate and convert an already-binary 2-D mask to Boolean.
    """
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

        if not np.all(
            (mask == 0) | (mask == 1)
        ):
            raise ValueError(
                f"{name} must be binary and contain only 0 and 1."
            )

    return mask.astype(bool, copy=False)


def _validate_binary_pair(
    true_binary: np.ndarray,
    pred_binary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    true_binary = _as_binary_2d(
        true_binary,
        name="true_binary",
    )

    pred_binary = _as_binary_2d(
        pred_binary,
        name="pred_binary",
    )

    if true_binary.shape != pred_binary.shape:
        raise ValueError(
            "true_binary and pred_binary must have the same shape; "
            f"received {true_binary.shape} and {pred_binary.shape}."
        )

    return true_binary, pred_binary


def calculate_pixel_confusion(
    true_binary: np.ndarray,
    pred_binary: np.ndarray,
) -> tuple[int, int, int, int]:
    """
    Return pixel counts in the order TP, FP, FN, TN.
    """
    true_binary, pred_binary = _validate_binary_pair(
        true_binary,
        pred_binary,
    )

    true_positives = int( np.count_nonzero( pred_binary & true_binary ) )

    false_positives = int( np.count_nonzero( pred_binary & ~true_binary ) )

    false_negatives = int( np.count_nonzero( ~pred_binary & true_binary ) )

    true_negatives = int( np.count_nonzero( ~pred_binary & ~true_binary ) )

    return (
        true_positives,
        false_positives,
        false_negatives,
        true_negatives,
    )


def calculate_pixel_precision(
    true_binary: np.ndarray,
    pred_binary: np.ndarray,
) -> float:
    """Calculate pixel-wise precision."""
    tp, fp, fn, _ = calculate_pixel_confusion(
        true_binary,
        pred_binary,
    )

    if tp == fp == fn == 0:
        return 1.0

    denominator = tp + fp
    return float(
        tp / denominator
        if denominator > 0
        else 0.0
    )


def calculate_pixel_recall(
    true_binary: np.ndarray,
    pred_binary: np.ndarray,
) -> float:
    """Calculate pixel-wise recall."""
    tp, fp, fn, _ = calculate_pixel_confusion(
        true_binary,
        pred_binary,
    )

    if tp == fp == fn == 0:
        return 1.0

    denominator = tp + fn
    return float(
        tp / denominator
        if denominator > 0
        else 0.0
    )


def calculate_pixel_f1(
    true_binary: np.ndarray,
    pred_binary: np.ndarray,
) -> float:
    """Calculate pixel-wise F1."""
    tp, fp, fn, _ = calculate_pixel_confusion(
        true_binary,
        pred_binary,
    )

    denominator = 2 * tp + fp + fn

    return float(
        2 * tp / denominator
        if denominator > 0
        else 1.0
    )


def calculate_pixel_iou(
    true_binary: np.ndarray,
    pred_binary: np.ndarray,
) -> float:
    """Calculate pixel-wise intersection over union."""
    tp, fp, fn, _ = calculate_pixel_confusion(
        true_binary,
        pred_binary,
    )

    denominator = tp + fp + fn

    return float(
        tp / denominator
        if denominator > 0
        else 1.0
    )


def calculate_pixel_fdr(
    true_binary: np.ndarray,
    pred_binary: np.ndarray,
) -> float:
    """Calculate pixel-wise false discovery rate."""
    tp, fp, _, _ = calculate_pixel_confusion(
        true_binary,
        pred_binary,
    )

    denominator = tp + fp

    return float(
        fp / denominator
        if denominator > 0
        else 0.0
    )


def calculate_pixel_dice(
    true_binary: np.ndarray,
    pred_binary: np.ndarray,
) -> float:
    """Calculate the pixel-wise Dice coefficient."""
    tp, fp, fn, _ = calculate_pixel_confusion(
        true_binary,
        pred_binary,
    )

    denominator = 2 * tp + fp + fn

    return float(
        2 * tp / denominator
        if denominator > 0
        else 1.0
    )


def evaluate_pixel_metrics(
    true_binary: np.ndarray,
    pred_binary: np.ndarray,
) -> PixelMetricsResult:
    """Calculate all supported image-level pixel metrics."""
    tp, fp, fn, tn = calculate_pixel_confusion(
        true_binary,
        pred_binary,
    )

    empty_agreement = (
        tp == 0
        and fp == 0
        and fn == 0
    )

    precision_denominator = tp + fp
    recall_denominator = tp + fn
    overlap_denominator = tp + fp + fn
    dice_denominator = 2 * tp + fp + fn

    precision = (
        1.0
        if empty_agreement
        else (
            tp / precision_denominator
            if precision_denominator > 0
            else 0.0
        )
    )

    recall = (
        1.0
        if empty_agreement
        else (
            tp / recall_denominator
            if recall_denominator > 0
            else 0.0
        )
    )

    f1_score = (
        2 * tp / dice_denominator
        if dice_denominator > 0
        else 1.0
    )

    iou = (
        tp / overlap_denominator
        if overlap_denominator > 0
        else 1.0
    )

    dice = (
        2 * tp / dice_denominator
        if dice_denominator > 0
        else 1.0
    )

    fdr = (
        fp / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    return PixelMetricsResult(
        true_positive_pixels=tp,
        false_positive_pixels=fp,
        false_negative_pixels=fn,
        true_negative_pixels=tn,
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1_score),
        fdr=float(fdr),
        iou=float(iou),
        dice=float(dice),
    )