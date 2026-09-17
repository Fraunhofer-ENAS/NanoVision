"""Evaluate object- and pixel-level metrics from paired COCO RLE files."""

from __future__ import annotations

import argparse
import csv
import json
import tifffile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import torch
from skimage.morphology import skeletonize

from cnt_project.evaluation.core.centerline_distance_matrix import (
    normalize_distances_by_ground_truth_width,
    pairwise_centerline_distances,
)
from cnt_project.evaluation.core.object_metrics import (
    evaluate_distance_thresholds,
    evaluate_similarity_thresholds,
)
from cnt_project.features.core.mask_width import (
    calculate_mask_width_px,
)

from cnt_project.evaluation.core.cldice import (
    cldice,
    pairwise_cldice_matrix,
)
from cnt_project.evaluation.core.object_matching import (
    pairwise_iou_matrix,
)
from cnt_project.evaluation.core.pixel_metrics import (
    evaluate_pixel_metrics,
)
from cnt_project.evaluation.inputs.coco_instances import (
    load_coco_rle_instance_pairs,
)
from cnt_project.evaluation.core.matched_object_features import (
    calculate_object_features,
    classify_irregular_prediction_masks,
    evaluate_matched_feature_errors,
)
from cnt_project.model_development.postprocessing.filtering.shape_heuristics import (
    count_endpoints,
    skeletonize_mask_with_polygon_smoothing,
)
from cnt_project.evaluation.plotting.matched_objects import (
    save_cldice_pair_visualizations,
    save_matched_feature_visualization,
    save_object_matching_visualization,
    save_rejected_prediction_visualization,
)


def _mean_finite(
    values: Iterable[float],
) -> float:
    """Calculate the mean of finite values, or NaN if none exist."""
    values_array = np.asarray(
        list(values),
        dtype=np.float64,
    )

    finite_values = values_array[
        np.isfinite(values_array)
    ]

    if finite_values.size == 0:
        return float("nan")

    return float(
        np.mean(finite_values)
    )

def _parse_distance_thresholds(
    value: str,
) -> tuple[float, ...]:
    """Parse comma-separated nonnegative distance thresholds."""
    try:
        thresholds = tuple(
            float(item.strip())
            for item in value.split(",")
            if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Distance thresholds must be "
            "comma-separated numbers."
        ) from error

    if not thresholds:
        raise argparse.ArgumentTypeError(
            "At least one distance threshold is required."
        )

    for threshold in thresholds:
        if (
            not np.isfinite(threshold)
            or threshold < 0.0
        ):
            raise argparse.ArgumentTypeError(
                "Distance thresholds must be finite "
                "and non-negative."
            )

    return thresholds



def _parse_thresholds(value: str) -> tuple[float, ...]:
    """Parse comma-separated similarity thresholds."""
    try:
        thresholds = tuple(
            float(item.strip())
            for item in value.split(",")
            if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Thresholds must be comma-separated numbers."
        ) from error

    if not thresholds:
        raise argparse.ArgumentTypeError(
            "At least one threshold is required."
        )

    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise argparse.ArgumentTypeError(
                "Similarity thresholds must be between "
                "0 and 1 inclusive."
            )

    return thresholds


def _partition_prediction_masks_by_endpoints(
    masks: Sequence[np.ndarray],
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
]:
    """
    Reject predictions whose skeleton has more than two endpoints.

    This follows check_y_or_t_shape() from shape_heuristics.py.
    """
    retained_masks = []
    rejected_masks = []

    for index, mask in enumerate(masks):
        mask = np.asarray(mask)

        if mask.ndim != 2:
            raise ValueError(
                f"masks[{index}] must be two-dimensional; "
                f"received shape {mask.shape}."
            )

        binary_mask = mask.astype(
            bool,
            copy=False,
        )

        if not binary_mask.any():
            raise ValueError(
                f"masks[{index}] is empty."
            )

        skeleton = (
            skeletonize_mask_with_polygon_smoothing(
                binary_mask,
                smoothing_iterations=2,
            )
        )

        number_of_endpoints = count_endpoints(
            skeleton
        )

        if number_of_endpoints > 2:
            rejected_masks.append(
                binary_mask
            )
        else:
            retained_masks.append(
                binary_mask
            )

    return (
        tuple(retained_masks),
        tuple(rejected_masks),
    )

def _instance_masks_to_centerline_tensor(
    masks: Sequence[np.ndarray],
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Skeletonize instance masks and return shape (L, 1, H, W).
    """
    if not masks:
        return torch.empty(
            (0, 1, height, width),
            dtype=torch.bool,
        )

    centerlines = []

    for index, mask in enumerate(masks):
        mask = np.asarray(mask)

        if mask.shape != (height, width):
            raise ValueError(
                f"masks[{index}] has shape {mask.shape}; "
                f"expected {(height, width)}."
            )

        binary_mask = mask.astype(
            bool,
            copy=False,
        )

        if not binary_mask.any():
            raise ValueError(
                f"masks[{index}] is empty."
            )

        centerline = skeletonize(
            binary_mask
        )

        if not centerline.any():
            raise RuntimeError(
                "Skeletonization produced an empty "
                f"centerline for masks[{index}]."
            )

        centerlines.append(centerline)

    centerline_array = np.stack(
        centerlines,
        axis=0,
    )

    return torch.from_numpy(
        centerline_array
    ).unsqueeze(1)



def _union_instance_masks(
    masks: Sequence[np.ndarray],
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Combine separate instances into one binary foreground mask."""
    if not masks:
        return np.zeros(
            (height, width),
            dtype=bool,
        )

    union = np.zeros(
        (height, width),
        dtype=bool,
    )

    for index, mask in enumerate(masks):
        mask = np.asarray(mask)

        if mask.shape != (height, width):
            raise ValueError(
                f"masks[{index}] has shape {mask.shape}; "
                f"expected {(height, width)}."
            )

        union |= mask.astype(bool)

    return union


def _write_csv(
    path: Path,
    *,
    rows: list[dict],
    fieldnames: Iterable[str],
) -> None:
    """Write dictionaries to a CSV file."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
        )

        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float]) -> float:
    values = list(values)

    return float(
        np.mean(values)
        if values
        else 0.0
    )


def run_evaluation(
    *,
    ground_truth_rle_json: str | Path,
    prediction_rle_json: str | Path,
    output_directory: str | Path,
    prediction_score_threshold: float,
    iou_thresholds: Sequence[float],
    matching_metrics: Sequence[str] = ( "iou", "cldice", "centerline", ),
    cldice_thresholds: Sequence[float],
    centerline_distance_measure: str,
    centerline_distance_thresholds: Sequence[float],
    image_directory: str | Path | None = None,
    filter_predictions_by_endpoints: bool = False,
) -> None:
    """Run object- and pixel-level evaluation."""
    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    matching_metrics = tuple(
        matching_metrics
    )

    supported_matching_metrics = {
        "iou",
        "cldice",
        "centerline",
    }

    unknown_matching_metrics = (
        set(matching_metrics)
        - supported_matching_metrics
    )

    if unknown_matching_metrics:
        raise ValueError(
            "Unsupported matching metrics: "
            f"{sorted(unknown_matching_metrics)}"
        )

    if centerline_distance_measure == "both":
        centerline_distance_measures = (
            "chamfer",
            "hausdorff",
        )
    else:
        centerline_distance_measures = (
            centerline_distance_measure,
        )

    if "centerline" not in matching_metrics:
        centerline_distance_measures = ()


    if image_directory is not None:
        image_directory = Path(
            image_directory
        )

        if not image_directory.is_dir():
            raise FileNotFoundError(
                f"Image directory does not exist: "
                f"{image_directory}"
            )

    images = load_coco_rle_instance_pairs(
        ground_truth_json_path=(
            ground_truth_rle_json
        ),
        prediction_json_path=(
            prediction_rle_json
        ),
        prediction_score_threshold=(
            prediction_score_threshold
        ),
    )

    object_rows: list[dict] = []
    pixel_rows: list[dict] = []

    number_of_images = len(images)

    # Disable evaluation visualizations for large batch runs.
    matching_visualization_image_count = 10
    feature_visualization_pair_count = 10 # float( "inf" )

    saved_feature_visualizations: set[
        tuple[str, float]
    ] = set()

    for image_index, image in enumerate(
        images,
        start=1,
    ):
        number_of_ground_truths = len(
            image.ground_truth_masks
        )

        original_number_of_predictions = len(
            image.prediction_masks
        )

        if filter_predictions_by_endpoints:
            (
                prediction_masks,
                rejected_prediction_masks,
            ) = _partition_prediction_masks_by_endpoints(
                image.prediction_masks
            )
        else:
            prediction_masks = tuple(
                image.prediction_masks
            )

            rejected_prediction_masks = ()

        rejected_prediction_count = len(
            rejected_prediction_masks
        )

        number_of_predictions = len(
            prediction_masks
        )

        if filter_predictions_by_endpoints:
            prediction_status = (
                f"predictions before filter="
                f"{original_number_of_predictions}, "
                f"retained={number_of_predictions}, "
                f"rejected={rejected_prediction_count}"
            )
        else:
            prediction_status = (
                f"predictions={number_of_predictions}; "
                "endpoint filter disabled"
            )

        print(
            f"[{image_index}/{number_of_images}] "
            f"{image.filename}: "
            f"GT={number_of_ground_truths}, "
            f"{prediction_status}"
        )

        if (rejected_prediction_masks and image_directory is not None):
            image_path = (
                image_directory
                / image.filename
            )

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Original image not found: "
                    f"{image_path}"
                )

            original_image = tifffile.imread(
                image_path
            )

            save_rejected_prediction_visualization(
                filename=image.filename,
                image=original_image,
                rejected_masks=(
                    rejected_prediction_masks
                ),
                output_path=(
                    output_directory
                    / "rejected_prediction_visualizations"
                    / (
                        f"{Path(image.filename).stem}"
                        "_rejected_predictions.png"
                    )
                ),
            )

        # --------------------------------------------------------
        # Reusable object morphology features
        # --------------------------------------------------------

        ground_truth_features = (
            calculate_object_features(
                image.ground_truth_masks
            )
        )

        prediction_features = (
            calculate_object_features(
                prediction_masks    # image.prediction_masks
            )
        )

        prediction_irregular_flags = (
            classify_irregular_prediction_masks(
                prediction_masks,
                min_branch_length_px=5,
            )
        )

        # --------------------------------------------------------
        # Object-level IoU
        # --------------------------------------------------------
        if "iou" in matching_metrics:
            iou_matrix = pairwise_iou_matrix(
                image.ground_truth_masks,
                prediction_masks,
            )

            iou_sweep = evaluate_similarity_thresholds(
                iou_matrix,
                thresholds=iou_thresholds,
            )

            iou_results = iou_sweep.results
        else:
            iou_results = ()

        for threshold_result in iou_results:
            metrics = threshold_result.metrics
            feature_errors = (
                evaluate_matched_feature_errors(
                    ground_truth_features=(
                        ground_truth_features
                    ),
                    prediction_features=(
                        prediction_features
                    ),
                    prediction_irregular_flags=(
                        prediction_irregular_flags
                    ),
                    matches=threshold_result.matches,
                )
            )
            # visualize pairs
            matching_metric = "iou"
            threshold = float(
                threshold_result.threshold
            )

            threshold_name = (
                f"{threshold:.2f}"
                .replace(".", "p")
            )

            if (
                image_index
                <= matching_visualization_image_count
            ):
                save_object_matching_visualization(
                    matching_metric=matching_metric,
                    filename=image.filename,
                    height=image.height,
                    width=image.width,
                    ground_truth_masks=(
                        image.ground_truth_masks
                    ),
                    prediction_masks=(
                        prediction_masks    # image.prediction_masks
                    ),
                    matches=threshold_result.matches,
                    threshold=threshold,
                    output_path=(
                        output_directory
                        / "matching_visualizations"
                        / matching_metric
                        / f"threshold_{threshold_name}"
                        / (
                            f"{Path(image.filename).stem}"
                            "_matches.png"
                        )
                    ),
                )

            feature_key = (
                matching_metric,
                threshold,
            )

            if (
                feature_key
                not in saved_feature_visualizations
                and len(threshold_result.matches)
                >= feature_visualization_pair_count
            ):
                save_matched_feature_visualization(
                    matching_metric=matching_metric,
                    threshold=threshold,
                    filename=image.filename,
                    ground_truth_masks=(
                        image.ground_truth_masks
                    ),
                    prediction_masks=(
                        prediction_masks    # image.prediction_masks
                    ),
                    ground_truth_features=(
                        ground_truth_features
                    ),
                    prediction_features=(
                        prediction_features
                    ),
                    matches=threshold_result.matches,
                    maximum_pairs=(
                        feature_visualization_pair_count
                    ),
                    output_path=(
                        output_directory
                        / "matched_feature_visualizations"
                        / matching_metric
                        / (
                            f"threshold_{threshold_name}"
                            "_features.png"
                        )
                    ),
                )

                saved_feature_visualizations.add(
                    feature_key
                )

            object_rows.append(
                {
                    "filename": image.filename,
                    "matching_metric": "iou",
                    "threshold": (
                        threshold_result.threshold
                    ),
                    "num_gt_objects": (
                        number_of_ground_truths
                    ),
                    "num_pred_objects": (
                        number_of_predictions
                    ),
                    "true_positives": (
                        metrics.true_positives
                    ),
                    "false_positives": (
                        metrics.false_positives
                    ),
                    "false_negatives": (
                        metrics.false_negatives
                    ),
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1_score": metrics.f1_score,
                    "dsb_precision": (
                        metrics.dsb_precision
                    ),
                    "matched_feature_pair_count": (
                        feature_errors.matched_pair_count
                    ),
                    "irregular_matched_prediction_count": (
                        feature_errors
                        .irregular_matched_prediction_count
                    ),
                    "mean_absolute_length_error_um": (
                        feature_errors
                        .mean_absolute_length_error_um
                    ),
                    "mean_absolute_relative_length_error": (
                        feature_errors
                        .mean_absolute_relative_length_error
                    ),
                    "mean_absolute_orientation_error_deg": (
                        feature_errors
                        .mean_absolute_orientation_error_deg
                    ),
                    "num_pred_objects_before_endpoint_filter": (
                        original_number_of_predictions
                    ),
                    "num_pred_objects_rejected_by_endpoint_filter": (
                        rejected_prediction_count
                    ),
                }
            )

        # --------------------------------------------------------
        # Object-level clDice
        # --------------------------------------------------------

        if "cldice" in matching_metrics:
            cldice_matrix = pairwise_cldice_matrix(
                image.ground_truth_masks,
                prediction_masks,
            )

            cldice_sweep = evaluate_similarity_thresholds(
                cldice_matrix,
                thresholds=cldice_thresholds,
            )

            cldice_results = cldice_sweep.results
        else:
            cldice_results = ()

        for threshold_result in cldice_results:
            metrics = threshold_result.metrics
            feature_errors = (
                evaluate_matched_feature_errors(
                    ground_truth_features=(
                        ground_truth_features
                    ),
                    prediction_features=(
                        prediction_features
                    ),
                    prediction_irregular_flags=(
                        prediction_irregular_flags
                    ),
                    matches=threshold_result.matches,
                )
            )
            # visualize pairs
            matching_metric = "cldice"
            threshold = float(
                threshold_result.threshold
            )
            
            threshold_name = (
                f"{threshold:.2f}"
                .replace(".", "p")
            )
            
            if (
                image_index
                <= matching_visualization_image_count
            ):
                save_object_matching_visualization(
                    matching_metric=matching_metric,
                    filename=image.filename,
                    height=image.height,
                    width=image.width,
                    ground_truth_masks=(
                        image.ground_truth_masks
                    ),
                    prediction_masks=(
                        prediction_masks    # image.prediction_masks
                    ),
                    matches=threshold_result.matches,
                    threshold=threshold,
                    output_path=(
                        output_directory
                        / "matching_visualizations"
                        / matching_metric
                        / f"threshold_{threshold_name}"
                        / (
                            f"{Path(image.filename).stem}"
                            "_matches.png"
                        )
                    ),
                )
                save_cldice_pair_visualizations(
                    filename=image.filename,
                    ground_truth_masks=(
                        image.ground_truth_masks
                    ),
                    prediction_masks=(
                        prediction_masks
                    ),
                    matches=threshold_result.matches,
                    threshold=threshold,
                    output_directory=(
                        output_directory
                        / "cldice_pair_visualizations"
                        / f"threshold_{threshold_name}"
                        / Path(image.filename).stem
                    ),
                    pairs_per_page=12,
                )
            
            feature_key = (
                matching_metric,
                threshold,
            )
            
            if (
                feature_key
                not in saved_feature_visualizations
                and len(threshold_result.matches)
                >= feature_visualization_pair_count
            ):
                save_matched_feature_visualization(
                    matching_metric=matching_metric,
                    threshold=threshold,
                    filename=image.filename,
                    ground_truth_masks=(
                        image.ground_truth_masks
                    ),
                    prediction_masks=(
                        prediction_masks    # image.prediction_masks
                    ),
                    ground_truth_features=(
                        ground_truth_features
                    ),
                    prediction_features=(
                        prediction_features
                    ),
                    matches=threshold_result.matches,
                    maximum_pairs=(
                        feature_visualization_pair_count
                    ),
                    output_path=(
                        output_directory
                        / "matched_feature_visualizations"
                        / matching_metric
                        / (
                            f"threshold_{threshold_name}"
                            "_features.png"
                        )
                    ),
                )
            
                saved_feature_visualizations.add(
                    feature_key
                )



            object_rows.append(
                {
                    "filename": image.filename,
                    "matching_metric": "cldice",
                    "threshold": (
                        threshold_result.threshold
                    ),
                    "num_gt_objects": (
                        number_of_ground_truths
                    ),
                    "num_pred_objects": (
                        number_of_predictions
                    ),
                    "true_positives": (
                        metrics.true_positives
                    ),
                    "false_positives": (
                        metrics.false_positives
                    ),
                    "false_negatives": (
                        metrics.false_negatives
                    ),
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1_score": metrics.f1_score,
                    "dsb_precision": (
                        metrics.dsb_precision
                    ),
                    "matched_feature_pair_count": (
                        feature_errors.matched_pair_count
                    ),
                    "irregular_matched_prediction_count": (
                        feature_errors
                        .irregular_matched_prediction_count
                    ),
                    "mean_absolute_length_error_um": (
                        feature_errors
                        .mean_absolute_length_error_um
                    ),
                    "mean_absolute_relative_length_error": (
                        feature_errors
                        .mean_absolute_relative_length_error
                    ),
                    "mean_absolute_orientation_error_deg": (
                        feature_errors
                        .mean_absolute_orientation_error_deg
                    ),
                    "num_pred_objects_before_endpoint_filter": (
                        original_number_of_predictions
                    ),
                    "num_pred_objects_rejected_by_endpoint_filter": (
                        rejected_prediction_count
                    ),
                }
            )


        # --------------------------------------------------------
        # Object-level normalized centerline Chamfer distance
        # --------------------------------------------------------

        if "centerline" in matching_metrics:
            ground_truth_centerlines = (
                _instance_masks_to_centerline_tensor(
                    image.ground_truth_masks,
                    height=image.height,
                    width=image.width,
                )
            )

            prediction_centerlines = (
                _instance_masks_to_centerline_tensor(
                    prediction_masks,
                    height=image.height,
                    width=image.width,
                )
            )

            ground_truth_widths_px = [
                calculate_mask_width_px(
                    mask
                )
                for mask
                in image.ground_truth_masks
            ]
        else:
            ground_truth_centerlines = None
            prediction_centerlines = None
            ground_truth_widths_px = []

        for centerline_measure in (
            centerline_distance_measures
        ):
            centerline_matching_metric = (
                f"centerline_{centerline_measure}"
                "_gt_width"
            )

            centerline_distance_matrix_px = (
                pairwise_centerline_distances(
                    ground_truth=(
                        ground_truth_centerlines
                    ),
                    prediction=(
                        prediction_centerlines
                    ),
                    measure=centerline_measure,
                )
            )

            normalized_centerline_distance_matrix = (
                normalize_distances_by_ground_truth_width(
                    centerline_distance_matrix_px,
                    ground_truth_widths=(
                        ground_truth_widths_px
                    ),
                )
            )

            centerline_distance_sweep = (
                evaluate_distance_thresholds(
                    normalized_centerline_distance_matrix,
                    thresholds=(
                        centerline_distance_thresholds
                    ),
                )
            )

            for threshold_result in (
                centerline_distance_sweep.results
            ):
                metrics = threshold_result.metrics
                feature_errors = (
                    evaluate_matched_feature_errors(
                        ground_truth_features=(
                            ground_truth_features
                        ),
                        prediction_features=(
                            prediction_features
                        ),
                        prediction_irregular_flags=(
                            prediction_irregular_flags
                        ),
                        matches=threshold_result.matches,
                    )
                )
                # visualize pairs
                matching_metric = centerline_matching_metric
                threshold = float(
                    threshold_result.threshold
                )
                
                threshold_name = (
                    f"{threshold:.2f}"
                    .replace(".", "p")
                )
                
                if (
                    image_index
                    <= matching_visualization_image_count
                ):
                    save_object_matching_visualization(
                        matching_metric=matching_metric,
                        filename=image.filename,
                        height=image.height,
                        width=image.width,
                        ground_truth_masks=(
                            image.ground_truth_masks
                        ),
                        prediction_masks=(
                            prediction_masks    # image.prediction_masks
                        ),
                        matches=threshold_result.matches,
                        threshold=threshold,
                        output_path=(
                            output_directory
                            / "matching_visualizations"
                            / matching_metric
                            / f"threshold_{threshold_name}"
                            / (
                                f"{Path(image.filename).stem}"
                                "_matches.png"
                            )
                        ),
                    )
                
                feature_key = (
                    matching_metric,
                    threshold,
                )
                
                if (
                    feature_key
                    not in saved_feature_visualizations
                    and len(threshold_result.matches)
                    >= feature_visualization_pair_count
                ):
                    save_matched_feature_visualization(
                        matching_metric=matching_metric,
                        threshold=threshold,
                        filename=image.filename,
                        ground_truth_masks=(
                            image.ground_truth_masks
                        ),
                        prediction_masks=(
                            prediction_masks    # image.prediction_masks
                        ),
                        ground_truth_features=(
                            ground_truth_features
                        ),
                        prediction_features=(
                            prediction_features
                        ),
                        matches=threshold_result.matches,
                        maximum_pairs=(
                            feature_visualization_pair_count
                        ),
                        output_path=(
                            output_directory
                            / "matched_feature_visualizations"
                            / matching_metric
                            / (
                                f"threshold_{threshold_name}"
                                "_features.png"
                            )
                        ),
                    )
                
                    saved_feature_visualizations.add(
                        feature_key
                    )

                object_rows.append(
                    {
                        "filename": image.filename,
                        "matching_metric": (
                            centerline_matching_metric
                        ),
                        "threshold": (
                            threshold_result.threshold
                        ),
                        "num_gt_objects": (
                            number_of_ground_truths
                        ),
                        "num_pred_objects": (
                            number_of_predictions
                        ),
                        "true_positives": (
                            metrics.true_positives
                        ),
                        "false_positives": (
                            metrics.false_positives
                        ),
                        "false_negatives": (
                            metrics.false_negatives
                        ),
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "f1_score": metrics.f1_score,
                        "dsb_precision": (
                            metrics.dsb_precision
                        ),
                        "matched_feature_pair_count": (
                            feature_errors.matched_pair_count
                        ),
                        "irregular_matched_prediction_count": (
                            feature_errors
                            .irregular_matched_prediction_count
                        ),
                        "mean_absolute_length_error_um": (
                            feature_errors
                            .mean_absolute_length_error_um
                        ),
                        "mean_absolute_relative_length_error": (
                            feature_errors
                            .mean_absolute_relative_length_error
                        ),
                        "mean_absolute_orientation_error_deg": (
                            feature_errors
                            .mean_absolute_orientation_error_deg
                        ),
                        "num_pred_objects_before_endpoint_filter": (
                            original_number_of_predictions
                        ),
                        "num_pred_objects_rejected_by_endpoint_filter": (
                            rejected_prediction_count
                        ),
                    }
                )        

        # --------------------------------------------------------
        # Image-level pixel metrics
        # --------------------------------------------------------

        ground_truth_union = (
            _union_instance_masks(
                image.ground_truth_masks,
                height=image.height,
                width=image.width,
            )
        )

        prediction_union = (
            _union_instance_masks(
                prediction_masks,    # image.prediction_masks,
                height=image.height,
                width=image.width,
            )
        )

        pixel_metrics = evaluate_pixel_metrics(
            ground_truth_union,
            prediction_union,
        )

        pixel_cldice = cldice(
            prediction_union,
            ground_truth_union,
        )

        pixel_rows.append(
            {
                "filename": image.filename,
                "num_gt_objects": (
                    number_of_ground_truths
                ),
                "num_pred_objects": (
                    number_of_predictions
                ),
                "true_positive_pixels": (
                    pixel_metrics
                    .true_positive_pixels
                ),
                "false_positive_pixels": (
                    pixel_metrics
                    .false_positive_pixels
                ),
                "false_negative_pixels": (
                    pixel_metrics
                    .false_negative_pixels
                ),
                "true_negative_pixels": (
                    pixel_metrics
                    .true_negative_pixels
                ),
                "pixel_precision": (
                    pixel_metrics.precision
                ),
                "pixel_recall": (
                    pixel_metrics.recall
                ),
                "pixel_f1": (
                    pixel_metrics.f1_score
                ),
                "pixel_fdr": pixel_metrics.fdr,
                "pixel_iou": pixel_metrics.iou,
                "pixel_dice": pixel_metrics.dice,
                "pixel_cldice": pixel_cldice,
                "num_pred_objects_before_endpoint_filter": (
                    original_number_of_predictions
                ),
                "num_pred_objects_rejected_by_endpoint_filter": (
                    rejected_prediction_count
                ),
            }
        )

    # ------------------------------------------------------------
    # Per-image output
    # ------------------------------------------------------------

    object_columns = (
        "filename",
        "matching_metric",
        "threshold",
        "num_gt_objects",
        "num_pred_objects",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1_score",
        "dsb_precision",
        "matched_feature_pair_count",
        "irregular_matched_prediction_count",
        "mean_absolute_length_error_um",
        "mean_absolute_relative_length_error",
        "mean_absolute_orientation_error_deg",
        "num_pred_objects_before_endpoint_filter",
        "num_pred_objects_rejected_by_endpoint_filter",  
    )

    pixel_columns = (
        "filename",
        "num_gt_objects",
        "num_pred_objects",
        "true_positive_pixels",
        "false_positive_pixels",
        "false_negative_pixels",
        "true_negative_pixels",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "pixel_fdr",
        "pixel_iou",
        "pixel_dice",
        "pixel_cldice",
        "num_pred_objects_before_endpoint_filter",
        "num_pred_objects_rejected_by_endpoint_filter",
    )

    _write_csv(
        output_directory
        / "object_metrics_per_image_threshold.csv",
        rows=object_rows,
        fieldnames=object_columns,
    )

    _write_csv(
        output_directory
        / "pixel_metrics_per_image.csv",
        rows=pixel_rows,
        fieldnames=pixel_columns,
    )

    # ------------------------------------------------------------
    # Dataset-level macro averages
    # ------------------------------------------------------------

    object_summary_rows = []

    centerline_summary_groups = tuple(
        (
            f"centerline_{measure}_gt_width",
            centerline_distance_thresholds,
        )
        for measure in centerline_distance_measures
    )

    summary_groups = []

    if "iou" in matching_metrics:
        summary_groups.append(
            ("iou", iou_thresholds)
        )

    if "cldice" in matching_metrics:
        summary_groups.append(
            ("cldice", cldice_thresholds)
        )

    if "centerline" in matching_metrics:
        summary_groups.extend(
            centerline_summary_groups
        )

    for matching_metric, thresholds in summary_groups:
        metric_rows = [
            row
            for row in object_rows
            if row["matching_metric"]
            == matching_metric
        ]

        threshold_dsb_values = []
        threshold_f1_values = []

        for threshold in thresholds:
            rows_at_threshold = [
                row
                for row in metric_rows
                if np.isclose(
                    row["threshold"],
                    threshold,
                )
            ]

            mean_precision = _mean(
                row["precision"]
                for row in rows_at_threshold
            )

            mean_recall = _mean(
                row["recall"]
                for row in rows_at_threshold
            )

            mean_f1 = _mean(
                row["f1_score"]
                for row in rows_at_threshold
            )

            mean_dsb = _mean(
                row["dsb_precision"]
                for row in rows_at_threshold
            )

            mean_absolute_length_error_um = (
                _mean_finite(
                    row[
                        "mean_absolute_length_error_um"
                    ]
                    for row in rows_at_threshold
                )
            )

            mean_absolute_relative_length_error = (
                _mean_finite(
                    row[
                        "mean_absolute_relative_length_error"
                    ]
                    for row in rows_at_threshold
                )
            )

            mean_absolute_orientation_error_deg = (
                _mean_finite(
                    row[
                        "mean_absolute_orientation_error_deg"
                    ]
                    for row in rows_at_threshold
                )
            )

            threshold_f1_values.append(
                mean_f1
            )

            threshold_dsb_values.append(
                mean_dsb
            )

            object_summary_rows.append(
                {
                    "matching_metric": (
                        matching_metric
                    ),
                    "threshold": float(
                        threshold
                    ),
                    "number_of_images": (
                        number_of_images
                    ),
                    "mean_precision": (
                        mean_precision
                    ),
                    "mean_recall": mean_recall,
                    "mean_f1": mean_f1,
                    "mean_dsb_precision": (
                        mean_dsb
                    ),
                    "mean_absolute_length_error_um": (
                        mean_absolute_length_error_um
                    ),
                    "mean_absolute_relative_length_error": (
                        mean_absolute_relative_length_error
                    ),
                    "mean_absolute_orientation_error_deg": (
                        mean_absolute_orientation_error_deg
                    ),
                }
            )

        print()
        print(
            f"{matching_metric} mean F1 "
            "over thresholds:",
            _mean(threshold_f1_values),
        )

        print(
            f"{matching_metric} DSB mAP:",
            _mean(threshold_dsb_values),
        )

    _write_csv(
        output_directory
        / "object_metrics_dataset_summary.csv",
        rows=object_summary_rows,
        fieldnames=(
            "matching_metric",
            "threshold",
            "number_of_images",
            "mean_precision",
            "mean_recall",
            "mean_f1",
            "mean_dsb_precision",
            "mean_absolute_length_error_um",
            "mean_absolute_relative_length_error",
            "mean_absolute_orientation_error_deg",
        ),
    )

    pixel_summary = {
        "number_of_images": number_of_images,
        "mean_pixel_precision": _mean(
            row["pixel_precision"]
            for row in pixel_rows
        ),
        "mean_pixel_recall": _mean(
            row["pixel_recall"]
            for row in pixel_rows
        ),
        "mean_pixel_f1": _mean(
            row["pixel_f1"]
            for row in pixel_rows
        ),
        "mean_pixel_fdr": _mean(
            row["pixel_fdr"]
            for row in pixel_rows
        ),
        "mean_pixel_iou": _mean(
            row["pixel_iou"]
            for row in pixel_rows
        ),
        "mean_pixel_dice": _mean(
            row["pixel_dice"]
            for row in pixel_rows
        ),
        "mean_pixel_cldice": _mean(
            row["pixel_cldice"]
            for row in pixel_rows
        ),
    }

    with (
        output_directory
        / "pixel_metrics_dataset_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            pixel_summary,
            handle,
            indent=2,
        )

    metadata = {
        "ground_truth_rle_json": str(
            Path(ground_truth_rle_json).resolve()
        ),
        "prediction_rle_json": str(
            Path(prediction_rle_json).resolve()
        ),
        "prediction_score_threshold": float(
            prediction_score_threshold
        ),
        "number_of_images": number_of_images,
        "iou_thresholds": [
            float(value)
            for value in iou_thresholds
        ],
        "cldice_thresholds": [
            float(value)
            for value in cldice_thresholds
        ],
        "centerline_distance_thresholds": [
            float(value)
            for value in centerline_distance_thresholds
        ],
        "centerline_distance_measure_requested": (
            centerline_distance_measure
        ),
        "centerline_distance_measures_evaluated": list(
            centerline_distance_measures
        ),
        "centerline_distance_definitions": {
            "chamfer": (
                "symmetric mean Chamfer distance between "
                "skeletonized instance masks, normalized by "
                "canonical GT mask width"
            ),
            "hausdorff": (
                "symmetric Hausdorff distance between "
                "skeletonized instance masks, normalized by "
                "canonical GT mask width"
            ),
        },
        "ground_truth_width_definition": (
            "calculate_mask_width_px"
        ),
        "filter_predictions_by_endpoints": bool(
            filter_predictions_by_endpoints
        ),
        "endpoint_filter_rule": (
            "reject smoothed skeleton endpoint count > 2"
            if filter_predictions_by_endpoints
            else None
        ),
        "endpoint_filter_smoothing_iterations": (
            2
            if filter_predictions_by_endpoints
            else None
        ),
        "rejected_prediction_visualizations_enabled": (
            filter_predictions_by_endpoints
            and image_directory is not None
        ),
        "original_image_directory": (
            str(image_directory.resolve())
            if image_directory is not None
            else None
        ),
    }

    with (
        output_directory
        / "evaluation_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )

    print()
    print(
        "Evaluation outputs:",
        output_directory.resolve(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate object IoU, object clDice, normalized "
            "centerline Chamfer and/or Hausdorff distances, "
            "and image-level pixel metrics from COCO RLE."
        )
    )
    parser.add_argument(
        "--matching-metrics",
        nargs="+",
        choices=(
            "iou",
            "cldice",
            "centerline",
        ),
        default=[
            "iou",
            "cldice",
            "centerline",
        ],
        help=(
            "Object-matching metrics to evaluate. "
            "Example: --matching-metrics cldice"
        ),
    )

    parser.add_argument(
        "--gt-rle-json",
        required=True,
        help="Ground-truth COCO RLE JSON.",
    )

    parser.add_argument(
        "--pred-rle-json",
        required=True,
        help="Prediction COCO RLE JSON.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Evaluation output directory.",
    )

    parser.add_argument(
        "--prediction-score-threshold",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--iou-thresholds",
        type=_parse_thresholds,
        default=_parse_thresholds(
            "0.1,0.2,0.3,0.4,0.5,"
            "0.6,0.7,0.8,0.9,1.0"
        ),
    )

    parser.add_argument(
        "--cldice-thresholds",
        type=_parse_thresholds,
        default=_parse_thresholds(
            "0.1,0.2,0.3,0.4,0.5,"
            "0.6,0.7,0.8,0.9,1.0"
        ),
    )

    parser.add_argument(
        "--centerline-distance-measure",
        choices=(
            "chamfer",
            "hausdorff",
            "both",
        ),
        default="chamfer",
        help=(
            "Centerline distance used for object matching. "
            "Use 'both' to evaluate Chamfer and Hausdorff."
        ),
    )

    parser.add_argument(
        "--centerline-distance-thresholds",
        type=_parse_distance_thresholds,
        default=_parse_distance_thresholds(
            "0.1,0.2,0.3,0.4,0.5,"
            "0.6,0.7,0.8,0.9,1.0"
        ),
        help=(
            "Comma-separated centerline-distance thresholds "
            "expressed in units of GT object width."
        ),
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help=(
            "Optional directory containing original images. "
            "When endpoint filtering is enabled, rejected-mask "
            "overlays are generated from these images."
        ),
    )
    parser.add_argument(
        "--filter-predictions-by-endpoints",
        action="store_true",
        help=(
            "Reject predicted instances whose smoothed "
            "skeleton has more than two endpoints."
        ),
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    run_evaluation(
        matching_metrics=(
            args.matching_metrics
        ),
        ground_truth_rle_json=(
            args.gt_rle_json
        ),
        prediction_rle_json=(
            args.pred_rle_json
        ),
        output_directory=args.output_dir,
        prediction_score_threshold=(
            args.prediction_score_threshold
        ),
        iou_thresholds=args.iou_thresholds,
        cldice_thresholds=(
            args.cldice_thresholds
        ),

        centerline_distance_measure=(
            args.centerline_distance_measure
        ),
        centerline_distance_thresholds=(
            args.centerline_distance_thresholds
        ),
        image_directory=args.image_dir,
        filter_predictions_by_endpoints=(
            args.filter_predictions_by_endpoints
        ),
    )


if __name__ == "__main__":
    main()