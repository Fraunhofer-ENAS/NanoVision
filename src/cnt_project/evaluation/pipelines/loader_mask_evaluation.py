from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

from cnt_project.features.legacy.mask_morphology_legacy import (
    calculate_polygon_properties,
    compute_line_densities,
    compute_density_vs_angle,
)
from cnt_project.evaluation.outputs.csv_io import write_csv_header_if_needed, append_dict_rows_csv, append_rows_csv
from cnt_project.evaluation.plotting.loader_mask_debug import (
    save_pixel_metrics_debug_figure,
)
from cnt_project.evaluation.core.object_matching import pr_curve_over_thresholds
from cnt_project.evaluation.core.pixel_metrics import (
    binarize_mask,
    calculate_pixel_dice,
    calculate_pixel_precision,
    calculate_pixel_iou,
    calculate_pixel_recall,
    calculate_pixel_fdr,
)
from cnt_project.visualizations.evaluation.data_analysis.histograms import plot_combined_histogram
from cnt_project.visualizations.evaluation.reports import (
    plot_density_vs_angle,
    plot_performance_by_density,
)
from cnt_project.features.comparison.feature_distribution_metrics import (
    compute_distribution_stats,
)



def process_with_ground_truth(
    sample,
    dst_dir: str,
    *,
    feature_results: list,
    iou_metrics_results_by_algo: dict[str, list],
    image_metrics_results_by_algo: dict[str, list],
    mean_errors: list,
    std_errors: list,
    relative_mean_errors: list,
    relative_std_errors: list,
    num_labels: list,
    distribution_distance_results: list,
    object_matching_algorithms: Sequence[str] = ("greedy",),
    include_iou_metrics: bool = True,
    include_properties: bool = True,
    include_distributions: bool = True,
    include_distribution_plots: bool = True,
    debug_pixel_metrics: bool = False,
    debug_pixel_metrics_dir: str | None = None,
    fig: bool = True,
) -> None:
    """
    Process a sample that contains GT (sample.Y) + prediction (sample.Y_pred).
    """

    if fig:
        sample.plot_img_label(lbl_title="GT_LABEL", save_folder=dst_dir)
        sample.plot_img_label(lbl_title="PRED_LABEL", save_folder=dst_dir)

    pred, true = sample.Y_pred, sample.Y

    properties_true = None
    properties_pred = None
    if include_properties or include_distributions:
        properties_true = calculate_polygon_properties(true)
        properties_pred = calculate_polygon_properties(pred)

    # number of objects in GT (exclude background 0)
    num_labels.append(len(np.unique(true)) - 1)

    # Line density stats
    gt_line_densities = compute_line_densities(true, min_object_size=2)
    pred_line_densities = compute_line_densities(pred, min_object_size=2)

    gt_mean, gt_std = float(np.mean(gt_line_densities)), float(np.std(gt_line_densities))
    pr_mean, pr_std = float(np.mean(pred_line_densities)), float(np.std(pred_line_densities))

    mean_error = abs(gt_mean - pr_mean)
    std_error = abs(gt_std - pr_std)
    rel_mean_error = (mean_error / gt_mean) * 100 if gt_mean != 0 else 0.0
    rel_std_error = (std_error / gt_std) * 100 if gt_std != 0 else 0.0

    mean_errors.append(mean_error)
    std_errors.append(std_error)
    relative_mean_errors.append(rel_mean_error)
    relative_std_errors.append(rel_std_error)

    # Object-level PR/AP/F1 (your custom definition)
    pred_i = np.round(pred).astype(int)
    true_i = true.astype(int)
    iou_thresholds = np.arange(0.1, 1.0, 0.1)

    # Pixel-wise metrics
    true_bin = binarize_mask(true_i)
    pred_bin = binarize_mask(pred_i)
    if debug_pixel_metrics:
        save_pixel_metrics_debug_figure(
            sample.X_fn,
            true_labels=true_i,
            pred_labels=pred_i,
            true_binary=true_bin,
            pred_binary=pred_bin,
            debug_dir=debug_pixel_metrics_dir or os.path.join(dst_dir, "debug_pixel_metrics"),
        )

    px_prec = float(calculate_pixel_precision(true_bin, pred_bin))
    px_rec = float(calculate_pixel_recall(true_bin, pred_bin))
    px_f1 = float(2 * (px_prec * px_rec) / (px_prec + px_rec + np.finfo(float).eps))
    px_iou = float(calculate_pixel_iou(true_bin, pred_bin))
    px_dice = float(calculate_pixel_dice(true_bin, pred_bin))
    # FDR = FP / (TP + FP)
    # Precision = TP / (TP + FP)
    # Therefore FDR = 1 - Precision
    px_fdr = float(1.0 - px_prec)

    # Object-level metrics for each selected matching algorithm.
    for algo in object_matching_algorithms:
        precision_values, recall_values, ap_values, f1_values = pr_curve_over_thresholds(
            true_i,
            pred_i,
            iou_thresholds=iou_thresholds,
            matching_algorithm=algo,
        )

        # your mAP proxy = mean(p*r) (keep as-is)
        mAP = float(np.mean([p * r for p, r in zip(precision_values, recall_values)]))

        image_metrics_results_by_algo[algo].append(
            [sample.X_fn, mAP, px_prec, px_rec, px_f1, px_iou, gt_mean, gt_std, pr_mean, pr_std, px_dice]
        )

        if include_iou_metrics:
            for j in range(len(ap_values)):
                iou_metrics_results_by_algo[algo].append(
                    [
                        sample.X_fn,
                        round(float(iou_thresholds[j]), 2),
                        round(float(precision_values[j]), 4),
                        round(float(recall_values[j]), 4),
                        round(float(ap_values[j]), 4),
                        round(float(f1_values[j]), 4),
                    ]
                )

    # Properties row (GT + Pred)
    image_height_um = 5.0
    gt_cnts_per_um = len(np.unique(true_i)) / image_height_um
    pr_cnts_per_um = len(np.unique(pred_i)) / image_height_um

    if include_properties:
        assert properties_true is not None and properties_pred is not None
        mean_area_um2 = float(np.mean(properties_pred["areas_um2"])) if properties_pred["areas_um2"] else 0.0

        properties_row = [
            sample.X_fn,
            # GT
            len(np.unique(true_i)) - 1,
            gt_mean,
            gt_std,
            float(np.mean(properties_true["areas_um2"])) if properties_true["areas_um2"] else 0.0,
            float(np.std(properties_true["areas_um2"])) if properties_true["areas_um2"] else 0.0,
            float(np.mean(properties_true["lengths_um"])) if properties_true["lengths_um"] else 0.0,
            float(np.std(properties_true["lengths_um"])) if properties_true["lengths_um"] else 0.0,
            float(np.mean(properties_true["widths_um"])) if properties_true["widths_um"] else 0.0,
            float(np.std(properties_true["widths_um"])) if properties_true["widths_um"] else 0.0,
            float(np.mean(properties_true["orientation_angles"])) if properties_true["orientation_angles"] else 0.0,
            float(np.std(properties_true["orientation_angles"])) if properties_true["orientation_angles"] else 0.0,
            gt_cnts_per_um,
            mean_area_um2,
            # Pred
            len(np.unique(pred_i)) - 1,
            pr_mean,
            pr_std,
            float(np.mean(properties_pred["areas_um2"])) if properties_pred["areas_um2"] else 0.0,
            float(np.std(properties_pred["areas_um2"])) if properties_pred["areas_um2"] else 0.0,
            float(np.mean(properties_pred["lengths_um"])) if properties_pred["lengths_um"] else 0.0,
            float(np.std(properties_pred["lengths_um"])) if properties_pred["lengths_um"] else 0.0,
            float(np.mean(properties_pred["widths_um"])) if properties_pred["widths_um"] else 0.0,
            float(np.std(properties_pred["widths_um"])) if properties_pred["widths_um"] else 0.0,
            float(np.mean(properties_pred["orientation_angles"])) if properties_pred["orientation_angles"] else 0.0,
            float(np.std(properties_pred["orientation_angles"])) if properties_pred["orientation_angles"] else 0.0,
            pr_cnts_per_um,
            mean_area_um2,
        ]
        feature_results.append(properties_row)

    # Histogram comparisons + distribution distances
    if include_distributions:
        assert properties_true is not None and properties_pred is not None

        def _record_dist(feature_name: str, data_true, data_pred, filename_base: str) -> None:
            if include_distribution_plots:
                plot_combined_histogram(
                    data_true,
                    data_pred,
                    title=f"{feature_name} Distribution",
                    xlabel=feature_name,
                    filename_base=filename_base,
                    save_folder=dst_dir,
                )
            stats_df = compute_distribution_stats(data_true, data_pred, bins=60)
            stats_values = stats_df.iloc[0].tolist()
            distribution_distance_results.append([sample.X_fn, feature_name] + stats_values)

        _record_dist("Area (µm²)", properties_true["areas_um2"], properties_pred["areas_um2"], f"hist_area_um2_{sample.X_fn}")
        _record_dist("Length (µm)", properties_true["lengths_um"], properties_pred["lengths_um"], f"hist_length_um_{sample.X_fn}")
        _record_dist("Width (µm)", properties_true["widths_um"], properties_pred["widths_um"], f"hist_width_um_{sample.X_fn}")

        # Orientation mapped to [-90, 90]
        orientation_true = np.asarray(properties_true["orientation_angles"], dtype=float)
        orientation_pred = np.asarray(properties_pred["orientation_angles"], dtype=float)

        def restrict_to_90_range(angles: np.ndarray) -> np.ndarray:
            angles = np.where(angles < -90, angles + 180, angles)
            angles = np.where(angles > 90, angles - 180, angles)
            return angles

        orientation_true = restrict_to_90_range(orientation_true)
        orientation_pred = restrict_to_90_range(orientation_pred)
        _record_dist("Orientation (deg)", orientation_true, orientation_pred, f"hist_orientation_{sample.X_fn}")

        _record_dist("Line Density", gt_line_densities, pred_line_densities, f"hist_linedensity_{sample.X_fn}")


def process_without_ground_truth(
    sample,
    dst_dir: str,
    *,
    no_gt_results: list,
    no_gt_results_properties: list,
    fig: bool = True,
) -> None:
    """
    Process a sample that has only predictions (sample.Y_pred) and no GT.
    """
    if fig:
        sample.plot_img_label(lbl_title="PRED_LABEL", save_folder=dst_dir)

    pred = sample.Y_pred
    properties_pred = calculate_polygon_properties(pred)

    pred_line_densities = compute_line_densities(pred, min_object_size=3)
    angles = np.arange(0, 180, 10)
    density_vs_angle = compute_density_vs_angle(pred, angles)

    file_stem = Path(sample.X_fn).stem
    plot_density_vs_angle(density_vs_angle, dst_dir=dst_dir, file_stem=file_stem)

    pred_mean = float(np.mean(pred_line_densities))
    pred_std = float(np.std(pred_line_densities))

    cnts_per_um = pred_mean / 5.0  # preserve your scaling assumption
    mean_area_um2 = float(np.mean(properties_pred["areas"])) * ((5 / 256) ** 2) if properties_pred["areas"] else 0.0

    no_gt_results.append(
        [
            file_stem,
            len(np.unique(pred)) - 1,
            pred_mean,
            pred_std,
            float(np.mean(properties_pred["areas_um2"])) if properties_pred["areas_um2"] else 0.0,
            float(np.std(properties_pred["areas_um2"])) if properties_pred["areas_um2"] else 0.0,
            float(np.mean(properties_pred["lengths_um"])) if properties_pred["lengths_um"] else 0.0,
            float(np.std(properties_pred["lengths_um"])) if properties_pred["lengths_um"] else 0.0,
            float(np.mean(properties_pred["widths_um"])) if properties_pred["widths_um"] else 0.0,
            float(np.std(properties_pred["widths_um"])) if properties_pred["widths_um"] else 0.0,
            float(np.mean(properties_pred["orientation_angles"])) if properties_pred["orientation_angles"] else 0.0,
            float(np.std(properties_pred["orientation_angles"])) if properties_pred["orientation_angles"] else 0.0,
            cnts_per_um,
            mean_area_um2,
        ]
    )

    # per-object rows
    n_polys = len(properties_pred["areas"])
    for poly_idx in range(n_polys):
        no_gt_results_properties.append(
            {
                "Image": file_stem,
                "Polygon Index": poly_idx,
                "Area": properties_pred["areas_um2"][poly_idx],
                "Width": properties_pred["widths_um"][poly_idx],
                "Length": properties_pred["lengths_um"][poly_idx],
                "Orientation Angle": properties_pred["orientation_angles"][poly_idx],
            }
        )

    # histograms
    plot_combined_histogram(data_pred=properties_pred["areas_um2"], title="Area (µm²)", xlabel="Area (µm²)",
                            filename_base=f"{file_stem}_hist_area_um2", save_folder=dst_dir)
    plot_combined_histogram(data_pred=properties_pred["widths_um"], title="Width (µm)", xlabel="Width (µm)",
                            filename_base=f"{file_stem}_hist_width_um", save_folder=dst_dir)
    plot_combined_histogram(data_pred=properties_pred["lengths_um"], title="Length (µm)", xlabel="Length (µm)",
                            filename_base=f"{file_stem}_hist_length_um", save_folder=dst_dir)
    plot_combined_histogram(data_pred=properties_pred["orientation_angles"], title="Orientation (deg)", xlabel="Orientation (deg)",
                            filename_base=f"{file_stem}_hist_orientation", save_folder=dst_dir)
    plot_combined_histogram(data_pred=pred_line_densities, title="Line Density", xlabel="Line Density",
                            filename_base=f"{file_stem}_hist_linedensity", save_folder=dst_dir)


def run_pixel_metrics_only_from_loader(
    loader,
    dst_dir: str,
    *,
    rewrite: bool = False,
    pixel_metrics_csv: str = "pixel_metrics.csv",
    fig: bool = False,
) -> None:
    """
    Run pixel-level evaluation only.

    Expected sample fields:
      - sample.Y      : GT instance mask from TIFF labels
      - sample.Y_pred : prediction instance mask reconstructed from JSON or model output
      - sample.X_fn   : image filename

    Writes:
      pixel_metrics.csv

    Metrics:
      - Pixel Precision
      - Pixel Recall
      - Pixel F1
      - Pixel IoU
      - Pixel Dice
      - Pixel FDR
    """
    os.makedirs(dst_dir, exist_ok=True)

    pixel_metrics_path = os.path.join(dst_dir, pixel_metrics_csv)

    pixel_metrics_header = [
        "Image",
        "Pixel Precision",
        "Pixel Recall",
        "Pixel F1",
        "Pixel IoU",
        "Pixel Dice",
        "Pixel FDR",
    ]

    write_csv_header_if_needed(
        pixel_metrics_path,
        pixel_metrics_header,
        rewrite=rewrite,
    )

    rows = []

    test_samples = loader.get_test_data()

    for i, sample in enumerate(test_samples):
        print(f"Pixel metrics {i + 1}/{len(test_samples)} - {sample.X_fn}")

        if getattr(sample, "Y", None) is None:
            print(f"Skipping {sample.X_fn}: no GT mask found.")
            continue

        if getattr(sample, "Y_pred", None) is None:
            print(f"Skipping {sample.X_fn}: no prediction mask found.")
            continue
        if fig:
            sample.plot_img_label(lbl_title="GT_LABEL", save_folder=dst_dir)
            sample.plot_img_label(lbl_title="PRED_LABEL", save_folder=dst_dir)

        true_i = sample.Y.astype(int)
        pred_i = np.round(sample.Y_pred).astype(int)

        true_bin = binarize_mask(true_i)
        pred_bin = binarize_mask(pred_i)

        px_prec = float(calculate_pixel_precision(true_bin, pred_bin))
        px_rec = float(calculate_pixel_recall(true_bin, pred_bin))
        px_f1 = float(
            2 * (px_prec * px_rec) / (px_prec + px_rec + np.finfo(float).eps)
        )
        px_iou = float(calculate_pixel_iou(true_bin, pred_bin))
        px_dice = float(calculate_pixel_dice(true_bin, pred_bin))
        px_fdr = float(calculate_pixel_fdr(true_bin, pred_bin))

        rows.append(
            [
                sample.X_fn,
                px_prec,
                px_rec,
                px_f1,
                px_iou,
                px_dice,
                px_fdr,
            ]
        )

    append_rows_csv(pixel_metrics_path, rows)

    print(f"Pixel metrics saved to: {pixel_metrics_path}")


# -----------------------------
# main loop
# -----------------------------
def main_evaluation_loop(
    loader,
    dst_dir: str,
    *,
    rewrite: bool = False,
    fig: bool = True,
    stop_after_image: str | None = None,
    features_csv: str = "features_statistics.csv",
    iou_metrics_csv: str = "IoU_metrics.csv",
    image_metrics_csv: str = "image_metrics.csv",
    no_gt_csv: str = "output_no_gt.csv",
    no_gt_properties_csv: str = "output_no_gt_properties.csv",
    properties_comparison_csv: str = "properties_with_gt.csv",
    distribution_distance_csv: str = "histogram_comparison.csv",
    object_matching_algorithm: str = "both",
    include_outputs: Sequence[str] | None = None,
    debug_pixel_metrics: bool = False,
    debug_pixel_metrics_dirname: str = "debug_pixel_metrics",
    debug_pixel_metrics_max_images: int | None = None,
) -> None:
    """
    Evaluate loader samples where predictions are in sample.Y_pred.
    If GT exists (sample.Y is not None for at least one sample), it writes GT-based CSVs.
    Otherwise, it writes no-GT CSVs.

    This function is meant to be imported by other runners.
    """

    valid_outputs = {
        "iou_metrics",
        "image_metrics",
        "properties",
        "distribution",
        "distribution_csv",
        "distribution_plots",
        "performance_plot",
        "no_gt_summary",
        "no_gt_properties",
    }
    selected_outputs = set(include_outputs or valid_outputs)
    unknown_outputs = selected_outputs - valid_outputs
    if unknown_outputs:
        raise ValueError(f"Unsupported include_outputs values: {sorted(unknown_outputs)}")

    include_iou_metrics = "iou_metrics" in selected_outputs
    include_image_metrics = "image_metrics" in selected_outputs
    include_properties = "properties" in selected_outputs
    include_distributions = (
        "distribution" in selected_outputs or "distribution_csv" in selected_outputs
    )
    include_distribution_plots = (
        "distribution" in selected_outputs or "distribution_plots" in selected_outputs
    )
    include_performance_plot = "performance_plot" in selected_outputs
    include_no_gt_summary = "no_gt_summary" in selected_outputs
    include_no_gt_properties = "no_gt_properties" in selected_outputs

    os.makedirs(dst_dir, exist_ok=True)
    object_matching_algorithm = str(object_matching_algorithm).strip().lower()
    if object_matching_algorithm not in {"greedy", "hungarian", "both"}:
        raise ValueError(
            f"Unsupported object_matching_algorithm='{object_matching_algorithm}'. "
            "Use one of: 'greedy', 'hungarian', 'both'."
        )
    if object_matching_algorithm == "both":
        object_matching_algorithms = ("greedy", "hungarian")
    else:
        object_matching_algorithms = (object_matching_algorithm,)
    print(f"Object-level matching algorithm(s): {', '.join(object_matching_algorithms)}")

    # Determine whether we have GT in the dataset
    test_samples = loader.get_test_data()
    has_gt = any(getattr(s, "Y", None) is not None for s in test_samples)

    # Output paths
    features_csv_path = os.path.join(dst_dir, features_csv)
    def _algo_csv_path(filename: str, algo: str) -> str:
        stem, ext = os.path.splitext(filename)
        ext = ext or ".csv"
        return os.path.join(dst_dir, f"{stem}_{algo}{ext}")

    iou_metrics_csv_paths = {
        algo: _algo_csv_path(iou_metrics_csv, algo) for algo in object_matching_algorithms
    }
    image_metrics_csv_paths = {
        algo: _algo_csv_path(image_metrics_csv, algo) for algo in object_matching_algorithms
    }
    properties_comp_path = os.path.join(dst_dir, properties_comparison_csv)
    dist_path = os.path.join(dst_dir, distribution_distance_csv)

    no_gt_csv_path = os.path.join(dst_dir, no_gt_csv)
    no_gt_props_path = os.path.join(dst_dir, no_gt_properties_csv)

    # Headers
    distribution_header = [
        "mean_true", "std_true", "mean_pred", "std_pred", "wasserstein_distance", "jensen_shannon_distance"
    ]
    dist_csv_header = ["Image", "Feature"] + distribution_header

    iou_header = ["Image", "IoU_thresh", "precision", "recall", "ap", "f1"]
    image_metrics_header = [
        "Image", "Object mAP", "Pixel Precision", "Pixel Recall", "Pixel F1", "Pixel IoU",
        "GT Line Density Mean", "GT Line Density Std", "Pred Line Density Mean", "Pred Line Density Std", "Pixel Dice"

    ]
    # Keep your existing “properties” header (long but explicit)
    properties_header = [
        "Image",
        "GT Number of Cnts", "GT Mean Line Density", "GT Std Line Density",
        "GT Mean Area (µm²)", "GT Std Area (µm²)", "GT Mean Length (µm)", "GT Std Length (µm)",
        "GT Mean Width (µm)", "GT Std Width (µm)", "GT Mean Orientation", "GT Std Orientation",
        "GT CNTs per Micrometer", "GT Mean Area (µm²) (dup)",
        "Pred Number of Cnts", "Pred Mean Line Density", "Pred Std Line Density",
        "Pred Mean Area (µm²)", "Pred Std Area (µm²)", "Pred Mean Length (µm)", "Pred Std Length (µm)",
        "Pred Mean Width (µm)", "Pred Std Width (µm)", "Pred Mean Orientation", "Pred Std Orientation",
        "Pred CNTs per Micrometer", "Pred Mean Area (µm²) (dup)",
    ]

    no_gt_header = [
        "Image",
        "Number of Cnts", "Mean Line Density", "Std Line Density",
        "Mean Area (µm²)", "Std Area (µm²)",
        "Mean Length (µm)", "Std Length (µm)",
        "Mean Width (µm)", "Std Width (µm)",
        "Mean Orientation", "Std Orientation",
        "CNTs per Micrometer", "Mean Area (µm²) (alt)",
    ]
    no_gt_props_header = ["Image", "Polygon Index", "Area", "Width", "Length", "Orientation Angle"]

    # Initialize CSVs
    if has_gt:
        for algo in object_matching_algorithms:
            if include_iou_metrics:
                write_csv_header_if_needed(iou_metrics_csv_paths[algo], iou_header, rewrite=rewrite)
            if include_image_metrics:
                write_csv_header_if_needed(image_metrics_csv_paths[algo], image_metrics_header, rewrite=rewrite)
        if include_properties:
            write_csv_header_if_needed(properties_comp_path, properties_header, rewrite=rewrite)
        if include_distributions:
            write_csv_header_if_needed(dist_path, dist_csv_header, rewrite=rewrite)
        # features_csv is redundant in your current code; you were not writing it correctly anyway.
        # Keep it optional: if you still want it, add it explicitly later.
    else:
        if include_no_gt_summary:
            write_csv_header_if_needed(no_gt_csv_path, no_gt_header, rewrite=rewrite)
        # dict-based CSV for per-object rows
        if include_no_gt_properties and rewrite and os.path.exists(no_gt_props_path):
            os.remove(no_gt_props_path)
        if include_no_gt_properties:
            append_dict_rows_csv(no_gt_props_path, [], no_gt_props_header, rewrite=True)

    # Accumulators
    feature_results: list = []
    iou_metrics_results_by_algo: dict[str, list] = {algo: [] for algo in object_matching_algorithms}
    image_metrics_results_by_algo: dict[str, list] = {algo: [] for algo in object_matching_algorithms}
    distribution_distance_results: list = []

    no_gt_results: list = []
    no_gt_results_properties: list[dict] = []

    mean_errors: list[float] = []
    std_errors: list[float] = []
    rel_mean_errors: list[float] = []
    rel_std_errors: list[float] = []
    num_labels: list[int] = []
    debug_pixel_metrics_saved = 0

    # Main loop
    stop_processing = False
    for i, sample in enumerate(test_samples):
        if stop_processing:
            break

        print(f"Image {i + 1}/{len(test_samples)} - {sample.X_fn}")

        if stop_after_image and sample.X_fn == stop_after_image:
            stop_processing = True
            print(f"Stopping after processing image: {sample.X_fn}")

        if getattr(sample, "Y", None) is not None:
            do_debug_pixel = debug_pixel_metrics and (
                debug_pixel_metrics_max_images is None
                or debug_pixel_metrics_saved < int(debug_pixel_metrics_max_images)
            )

            process_with_ground_truth(
                sample,
                dst_dir,
                feature_results=feature_results,
                iou_metrics_results_by_algo=iou_metrics_results_by_algo,
                image_metrics_results_by_algo=image_metrics_results_by_algo,
                mean_errors=mean_errors,
                std_errors=std_errors,
                relative_mean_errors=rel_mean_errors,
                relative_std_errors=rel_std_errors,
                num_labels=num_labels,
                distribution_distance_results=distribution_distance_results,
                object_matching_algorithms=object_matching_algorithms,
                include_iou_metrics=include_iou_metrics,
                include_properties=include_properties,
                include_distributions=include_distributions,
                include_distribution_plots=include_distribution_plots,
                debug_pixel_metrics=do_debug_pixel,
                debug_pixel_metrics_dir=os.path.join(dst_dir, debug_pixel_metrics_dirname),
                fig=fig,
            )
            if do_debug_pixel:
                debug_pixel_metrics_saved += 1
        else:
            process_without_ground_truth(
                sample,
                dst_dir,
                no_gt_results=no_gt_results,
                no_gt_results_properties=no_gt_results_properties,
                fig=fig,
            )

    # Write outputs
    if has_gt:
        for algo in object_matching_algorithms:
            if include_iou_metrics:
                append_rows_csv(iou_metrics_csv_paths[algo], iou_metrics_results_by_algo[algo])
            if include_image_metrics:
                append_rows_csv(image_metrics_csv_paths[algo], image_metrics_results_by_algo[algo])
        if include_properties:
            append_rows_csv(properties_comp_path, feature_results)
        if include_distributions:
            append_rows_csv(dist_path, distribution_distance_results)

        # Optional: performance-by-density plot (keep behavior)
        if include_performance_plot and mean_errors and std_errors:
            plot_performance_by_density(
                num_labels=num_labels,
                mean_errors=mean_errors,
                rel_mean_errors=rel_mean_errors,
                std_errors=std_errors,
                rel_std_errors=rel_std_errors,
            )
        elif include_performance_plot:
            print("No error metrics computed; skipping performance-by-density plot.")
    else:
        if include_no_gt_summary:
            append_rows_csv(no_gt_csv_path, no_gt_results)
        if include_no_gt_properties:
            append_dict_rows_csv(no_gt_props_path, no_gt_results_properties, no_gt_props_header, rewrite=False)

        print("Ground truth not available: wrote no-GT outputs.")