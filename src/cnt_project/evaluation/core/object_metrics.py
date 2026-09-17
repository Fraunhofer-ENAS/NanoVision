"""Generic object-level metrics for thresholded one-to-one matching."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from cnt_project.evaluation.core.object_matching import (
    ObjectMatch,
    match_distance_matrix,
    match_similarity_matrix,
)


@dataclass(frozen=True)
class ObjectMetrics:
    """Object-level detection metrics at one threshold."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    dsb_precision: float


@dataclass(frozen=True)
class ObjectThresholdResult:
    """Matching and object metrics at one threshold."""

    threshold: float
    metrics: ObjectMetrics
    matches: tuple[ObjectMatch, ...]


@dataclass(frozen=True)
class ObjectThresholdSweepResult:
    """Object-level results over multiple matching thresholds."""

    thresholds: tuple[float, ...]
    results: tuple[ObjectThresholdResult, ...]
    mean_f1: float
    mean_dsb_precision: float


def calculate_object_metrics(
    *,
    number_of_ground_truths: int,
    number_of_predictions: int,
    number_of_matches: int,
) -> ObjectMetrics:
    """
    Calculate object-level metrics from object counts.

    Accepted one-to-one matches are true positives.

    Unmatched predictions are false positives.
    Unmatched ground-truth objects are false negatives.
    """
    for name, value in (
        (
            "number_of_ground_truths",
            number_of_ground_truths,
        ),
        (
            "number_of_predictions",
            number_of_predictions,
        ),
        (
            "number_of_matches",
            number_of_matches,
        ),
    ):
        if not isinstance(value, (int, np.integer)):
            raise TypeError(
                f"{name} must be an integer."
            )

        if value < 0:
            raise ValueError(
                f"{name} must be non-negative."
            )

    maximum_possible_matches = min(
        number_of_ground_truths,
        number_of_predictions,
    )

    if number_of_matches > maximum_possible_matches:
        raise ValueError(
            "number_of_matches cannot exceed the smaller of "
            "number_of_ground_truths and number_of_predictions; "
            f"received {number_of_matches} matches, "
            f"{number_of_ground_truths} ground truths, and "
            f"{number_of_predictions} predictions."
        )

    true_positives = int(number_of_matches)

    false_positives = int(
        number_of_predictions - true_positives
    )

    false_negatives = int(
        number_of_ground_truths - true_positives
    )

    empty_agreement = (
        number_of_ground_truths == 0
        and number_of_predictions == 0
    )

    precision_denominator = (
        true_positives + false_positives
    )

    recall_denominator = (
        true_positives + false_negatives
    )

    f1_denominator = (
        2 * true_positives
        + false_positives
        + false_negatives
    )

    dsb_denominator = (
        true_positives
        + false_positives
        + false_negatives
    )

    precision = (
        1.0
        if empty_agreement
        else (
            true_positives / precision_denominator
            if precision_denominator > 0
            else 0.0
        )
    )

    recall = (
        1.0
        if empty_agreement
        else (
            true_positives / recall_denominator
            if recall_denominator > 0
            else 0.0
        )
    )

    f1_score = (
        2 * true_positives / f1_denominator
        if f1_denominator > 0
        else 1.0
    )

    dsb_precision = (
        true_positives / dsb_denominator
        if dsb_denominator > 0
        else 1.0
    )

    return ObjectMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1_score),
        dsb_precision=float(dsb_precision),
    )


def _prepare_thresholds(
    thresholds: Iterable[float],
    *,
    kind: str,
) -> tuple[float, ...]:
    """Validate and return matching thresholds as a tuple."""
    thresholds = tuple(
        float(threshold)
        for threshold in thresholds
    )

    if not thresholds:
        raise ValueError(
            "thresholds must contain at least one value."
        )

    for threshold in thresholds:
        if not math.isfinite(threshold):
            raise ValueError(
                "Thresholds must contain only finite values."
            )

        if kind == "similarity":
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    "Similarity thresholds must be between "
                    "0 and 1 inclusive; "
                    f"received {threshold}."
                )

        elif kind == "distance":
            if threshold < 0.0:
                raise ValueError(
                    "Distance thresholds must be non-negative; "
                    f"received {threshold}."
                )

        else:
            raise ValueError(
                f"Unsupported threshold kind: {kind}."
            )

    return thresholds


def _build_sweep_result(
    threshold_results: list[ObjectThresholdResult],
) -> ObjectThresholdSweepResult:
    """Construct the final result and calculate threshold means."""
    results = tuple(threshold_results)

    thresholds = tuple(
        result.threshold
        for result in results
    )

    mean_f1 = float(
        np.mean([
            result.metrics.f1_score
            for result in results
        ])
    )

    mean_dsb_precision = float(
        np.mean([
            result.metrics.dsb_precision
            for result in results
        ])
    )

    return ObjectThresholdSweepResult(
        thresholds=thresholds,
        results=results,
        mean_f1=mean_f1,
        mean_dsb_precision=mean_dsb_precision,
    )


def evaluate_similarity_thresholds(
    score_matrix: np.ndarray,
    *,
    thresholds: Iterable[float],
) -> ObjectThresholdSweepResult:
    """
    Evaluate an object-similarity matrix over multiple thresholds.

    Matrix rows represent ground-truth objects.
    Matrix columns represent predicted objects.

    A pair is eligible when:

        similarity >= threshold

    Hungarian matching is rerun independently at every threshold.
    This function can be used for IoU and clDice matrices.
    """
    score_matrix = np.asarray(
        score_matrix,
        dtype=np.float64,
    )

    if score_matrix.ndim != 2:
        raise ValueError(
            "score_matrix must be two-dimensional; "
            f"received shape {score_matrix.shape}."
        )

    thresholds = _prepare_thresholds(
        thresholds,
        kind="similarity",
    )

    number_of_ground_truths = score_matrix.shape[0]
    number_of_predictions = score_matrix.shape[1]

    threshold_results = []

    for threshold in thresholds:
        matches = tuple(
            match_similarity_matrix(
                score_matrix,
                threshold=threshold,
            )
        )

        metrics = calculate_object_metrics(
            number_of_ground_truths=(
                number_of_ground_truths
            ),
            number_of_predictions=(
                number_of_predictions
            ),
            number_of_matches=len(matches),
        )

        threshold_results.append(
            ObjectThresholdResult(
                threshold=threshold,
                metrics=metrics,
                matches=matches,
            )
        )

    return _build_sweep_result(
        threshold_results
    )


def evaluate_distance_thresholds(
    distance_matrix: np.ndarray,
    *,
    thresholds: Iterable[float],
) -> ObjectThresholdSweepResult:
    """
    Evaluate an object-distance matrix over multiple thresholds.

    Matrix rows represent ground-truth objects.
    Matrix columns represent predicted objects.

    A pair is eligible when:

        distance <= threshold

    Hungarian matching is rerun independently at every threshold.
    This function can be used for GT-width-normalized Chamfer
    distance matrices.
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

    thresholds = _prepare_thresholds(
        thresholds,
        kind="distance",
    )

    number_of_ground_truths = distance_matrix.shape[0]
    number_of_predictions = distance_matrix.shape[1]

    threshold_results = []

    for threshold in thresholds:
        matches = tuple(
            match_distance_matrix(
                distance_matrix,
                threshold=threshold,
            )
        )

        metrics = calculate_object_metrics(
            number_of_ground_truths=(
                number_of_ground_truths
            ),
            number_of_predictions=(
                number_of_predictions
            ),
            number_of_matches=len(matches),
        )

        threshold_results.append(
            ObjectThresholdResult(
                threshold=threshold,
                metrics=metrics,
                matches=matches,
            )
        )

    return _build_sweep_result(
        threshold_results
    )