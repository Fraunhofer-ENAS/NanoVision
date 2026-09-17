from __future__ import annotations

import os
from typing import Any

import pandas as pd

from cnt_project.coco.io import get_annotations_for_image_id
from cnt_project.features.comparison.feature_distribution_metrics import (
    MAX_JENSEN_SHANNON,
    MAX_WASSERSTEIN,
    build_penalty_result,
    compare_distributions,
    compare_distributions_enhanced,
)

from cnt_project.features.core.cnt_feature_extraction import (
    calculate_cnt_features_from_annotations,
    compute_line_densities_from_polygons,
)
from cnt_project.io.file_io import read_json, write_text


DEFAULT_METRICS_TO_COMPARE = [
    "line_density",
    "length_um",
    "width_um",
    "aspect_ratio",
    "orientation_angle",
]


def build_image_result(
    annotations: list[dict[str, Any]],
    *,
    image_height: int = 256,
    image_width: int = 256,
) -> dict[str, Any]:
    """
    Build GT/prediction result structure for one image.
    """
    return {
        "line_density": compute_line_densities_from_polygons(
            annotations,
            image_height,
            image_width,
        ),
        "metrics": calculate_cnt_features_from_annotations(
            annotations,
            image_height,
            image_width,
        ),
    }


def load_model_prediction_jsons(model_dirs: dict[str, str]) -> dict[str, dict[str, Any] | None]:
    """
    Load prediction JSONs from model output directories.

    Expects predicted_annotations_poly.json inside each model directory.
    """
    model_jsons: dict[str, dict[str, Any] | None] = {}

    for model_name, model_dir in model_dirs.items():
        pred_json_path = os.path.join(model_dir, "predicted_annotations_poly.json")
        if os.path.exists(pred_json_path):
            model_jsons[model_name] = read_json(pred_json_path)
        else:
            model_jsons[model_name] = None

    return model_jsons


def compare_one_image_against_models(
    *,
    image_id: int,
    filename: str,
    gt_json: dict[str, Any],
    model_jsons: dict[str, dict[str, Any] | None],
    image_height: int = 256,
    image_width: int = 256,
    metrics_to_compare: list[str] | None = None,
    enhanced: bool = False,
) -> dict[str, Any]:
    """
    Compare one GT image against all model predictions.
    """
    if metrics_to_compare is None:
        metrics_to_compare = list(DEFAULT_METRICS_TO_COMPARE)

    image_annotations_gt = get_annotations_for_image_id(gt_json, image_id)
    gt_results = build_image_result(
        image_annotations_gt,
        image_height=image_height,
        image_width=image_width,
    )

    image_results: dict[str, Any] = {
        "image_id": image_id,
        "filename": filename,
        "ground_truth": gt_results,
    }

    for model_name, pred_json in model_jsons.items():
        if pred_json is None:
            continue

        image_annotations_pred = get_annotations_for_image_id(pred_json, image_id)
        has_predictions = len(image_annotations_pred) > 0

        comparison_results: list[dict[str, Any]] = []

        if not has_predictions:
            for metric in metrics_to_compare:
                comparison_results.append(
                    build_penalty_result(metric, model_name, no_predictions=True)
                )
            image_results[model_name] = {
                "raw": None,
                "comparisons": comparison_results,
                "has_predictions": False,
            }
            continue

        model_results = build_image_result(
            image_annotations_pred,
            image_height=image_height,
            image_width=image_width,
        )

        for metric in metrics_to_compare:
            if metric == "line_density":
                gt_values = gt_results["line_density"]
                pred_values = model_results["line_density"]
            else:
                gt_values = gt_results["metrics"].get(metric, [])
                pred_values = model_results["metrics"].get(metric, [])

            if enhanced:
                result = compare_distributions_enhanced(
                    gt_values,
                    pred_values,
                    metric,
                    bins=60,
                )
            else:
                result = compare_distributions(
                    gt_values,
                    pred_values,
                    metric,
                    bins=60,
                )

            if result is None:
                comparison_results.append(
                    build_penalty_result(metric, model_name, no_predictions=False)
                )
            else:
                result["model"] = model_name
                result["no_predictions"] = False
                comparison_results.append(result)

        image_results[model_name] = {
            "raw": model_results,
            "comparisons": comparison_results,
            "has_predictions": True,
        }

    return image_results


def flatten_comparison_results(
    all_comparison_results: list[dict[str, Any]],
    model_names: list[str],
) -> pd.DataFrame:
    """
    Flatten nested image comparison results into a DataFrame.
    """
    flat_rows: list[dict[str, Any]] = []

    for image_result in all_comparison_results:
        image_id = image_result["image_id"]
        filename = image_result["filename"]

        for model_name in model_names:
            if model_name not in image_result:
                continue

            model_block = image_result[model_name]
            if not model_block or "comparisons" not in model_block:
                continue

            for comparison in model_block["comparisons"]:
                row = dict(comparison)
                row["image_id"] = image_id
                row["filename"] = filename
                flat_rows.append(row)

    return pd.DataFrame(flat_rows)


def ensure_complete_comparison_table(
    comparison_df: pd.DataFrame,
    *,
    gt_json: dict[str, Any],
    model_names: list[str],
    metrics_to_compare: list[str],
) -> pd.DataFrame:
    """
    Ensure all (image, model, metric) combinations exist in the comparison table.
    Missing combinations are filled with penalty rows.
    """
    expected_images = gt_json.get("images", [])
    all_image_ids = [img["id"] for img in expected_images]

    if len(comparison_df) == 0:
        extra_rows = []
        for img in expected_images:
            for model_name in model_names:
                for metric in metrics_to_compare:
                    extra_rows.append(
                        {
                            "image_id": img["id"],
                            "filename": img["file_name"],
                            "model": model_name,
                            "metric": metric,
                            "wasserstein": MAX_WASSERSTEIN,
                            "jensen_shannon": MAX_JENSEN_SHANNON,
                            "no_predictions": True,
                        }
                    )
        return pd.DataFrame(extra_rows)

    all_combinations = [
        (img_id, model_name, metric)
        for img_id in all_image_ids
        for model_name in model_names
        for metric in metrics_to_compare
    ]

    existing_combinations = set(
        zip(
            comparison_df["image_id"],
            comparison_df["model"],
            comparison_df["metric"],
        )
    )

    missing = [comb for comb in all_combinations if comb not in existing_combinations]

    if not missing:
        return comparison_df

    extra_rows = []
    for img_id, model_name, metric in missing:
        filename = next(
            (img["file_name"] for img in expected_images if img["id"] == img_id),
            "unknown",
        )
        extra_rows.append(
            {
                "image_id": img_id,
                "filename": filename,
                "model": model_name,
                "metric": metric,
                "wasserstein": MAX_WASSERSTEIN,
                "jensen_shannon": MAX_JENSEN_SHANNON,
                "no_predictions": True,
            }
        )

    return pd.concat([comparison_df, pd.DataFrame(extra_rows)], ignore_index=True)


def compute_performance_statistics(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate comparison statistics by model and metric.
    """
    performance_stats = (
        comparison_df.groupby(["model", "metric"])
        .agg(
            {
                "wasserstein": ["mean", "std"],
                "jensen_shannon": ["mean", "std"],
                "no_predictions": "mean",
            }
        )
        .reset_index()
    )

    performance_stats.columns = [
        "_".join(col).strip("_") for col in performance_stats.columns.values
    ]

    performance_stats = performance_stats.rename(
        columns={
            "wasserstein_mean": "mean_wasserstein",
            "wasserstein_std": "std_wasserstein",
            "jensen_shannon_mean": "mean_jensen_shannon",
            "jensen_shannon_std": "std_jensen_shannon",
            "no_predictions_mean": "percent_no_predictions",
        }
    )

    return performance_stats


def compute_pivoted_performance_statistics(performance_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot aggregated model/metric performance stats to a wide table.
    """
    pivoted = performance_stats.pivot_table(
        index="model",
        columns="metric",
        values=[
            "mean_wasserstein",
            "std_wasserstein",
            "mean_jensen_shannon",
            "std_jensen_shannon",
            "percent_no_predictions",
        ],
    ).reset_index()

    pivoted.columns = ["_".join(col).strip("_") for col in pivoted.columns.values]
    return pivoted


def compute_model_summary(performance_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Compute model-level mean-of-means summary table.
    """
    model_summary = (
        performance_stats.groupby("model")
        .agg(
            {
                "mean_wasserstein": "mean",
                "std_wasserstein": "mean",
                "mean_jensen_shannon": "mean",
                "std_jensen_shannon": "mean",
                "percent_no_predictions": "mean",
            }
        )
        .reset_index()
    )

    model_summary = model_summary.rename(
        columns={
            "mean_wasserstein": "mean_of_wasserstein_means",
            "std_wasserstein": "mean_of_wasserstein_stds",
            "mean_jensen_shannon": "mean_of_jensen_shannon_means",
            "std_jensen_shannon": "mean_of_jensen_shannon_stds",
            "percent_no_predictions": "mean_percent_no_predictions",
        }
    )

    return model_summary


def compute_model_coverage(
    all_comparison_results: list[dict[str, Any]],
    model_names: list[str],
) -> dict[str, float]:
    """
    Compute percentage of images with predictions for each model.
    """
    coverage: dict[str, float] = {}

    for model_name in model_names:
        flags: list[bool] = []
        for image_result in all_comparison_results:
            if model_name not in image_result:
                continue
            model_block = image_result[model_name]
            flags.append(bool(model_block.get("has_predictions", False)))

        coverage[model_name] = (sum(flags) / len(flags) * 100.0) if flags else 0.0

    return coverage


def create_comparison_summary_text(
    *,
    num_images: int,
    num_rows: int,
    model_names: list[str],
    model_coverage: dict[str, float],
    no_prediction_percent: pd.Series | None = None,
) -> str:
    """
    Create a human-readable summary for comparison outputs.
    """
    lines = [
        "Model Comparison Summary",
        "========================",
        "",
        f"Processed images: {num_images}",
        f"Generated comparison rows: {num_rows}",
        f"Models evaluated: {', '.join(model_names)}",
        "",
        "Model coverage (percentage of images with predictions):",
    ]

    for model_name, pct in model_coverage.items():
        lines.append(f"- {model_name}: {pct:.2f}%")

    if no_prediction_percent is not None and len(no_prediction_percent) > 0:
        lines.extend(
            [
                "",
                "Percentage of images with no predictions:",
            ]
        )
        for model_name, pct in no_prediction_percent.items():
            lines.append(f"- {model_name}: {pct:.2f}%")

    return "\n".join(lines)


def run_model_comparison_pipeline(
    *,
    gt_json_path: str,
    model_dirs: dict[str, str],
    dst_dir: str,
    image_height: int = 256,
    image_width: int = 256,
    metrics_to_compare: list[str] | None = None,
    enhanced: bool = False,
    detailed_csv_name: str = "detailed_model_comparison_results.csv",
    stats_csv_name: str = "performance_statistics.csv",
    pivoted_csv_name: str = "pivoted_mean_std_performance.csv",
    model_summary_csv_name: str = "model_performance_summary.csv",
    summary_txt_name: str = "comparison_summary.txt",
) -> dict[str, Any]:
    """
    End-to-end GT vs multi-model comparison pipeline.

    Steps
    -----
    1. Load GT JSON and model prediction JSONs
    2. Compare every GT image against every model
    3. Flatten results to one table
    4. Ensure missing combinations are filled with penalties
    5. Compute aggregated statistics
    6. Save all CSV/text outputs

    Returns
    -------
    dict
        Structured outputs and saved paths.
    """
    if metrics_to_compare is None:
        metrics_to_compare = list(DEFAULT_METRICS_TO_COMPARE)

    os.makedirs(dst_dir, exist_ok=True)

    gt_json = read_json(gt_json_path)
    model_jsons = load_model_prediction_jsons(model_dirs)
    model_names = list(model_dirs.keys())

    all_comparison_results: list[dict[str, Any]] = []

    for img in gt_json.get("images", []):
        filename = img.get("file_name")
        image_id = img.get("id")

        image_result = compare_one_image_against_models(
            image_id=image_id,
            filename=filename,
            gt_json=gt_json,
            model_jsons=model_jsons,
            image_height=image_height,
            image_width=image_width,
            metrics_to_compare=metrics_to_compare,
            enhanced=enhanced,
        )
        all_comparison_results.append(image_result)

    comparison_df = flatten_comparison_results(all_comparison_results, model_names)
    comparison_df = ensure_complete_comparison_table(
        comparison_df,
        gt_json=gt_json,
        model_names=model_names,
        metrics_to_compare=metrics_to_compare,
    )

    performance_stats = compute_performance_statistics(comparison_df)
    pivoted_stats = compute_pivoted_performance_statistics(performance_stats)
    model_summary = compute_model_summary(performance_stats)
    model_coverage = compute_model_coverage(all_comparison_results, model_names)

    detailed_csv_path = os.path.join(dst_dir, detailed_csv_name)
    stats_csv_path = os.path.join(dst_dir, stats_csv_name)
    pivoted_csv_path = os.path.join(dst_dir, pivoted_csv_name)
    model_summary_csv_path = os.path.join(dst_dir, model_summary_csv_name)
    summary_txt_path = os.path.join(dst_dir, summary_txt_name)

    comparison_df.to_csv(detailed_csv_path, index=False)
    performance_stats.to_csv(stats_csv_path, index=False)
    pivoted_stats.to_csv(pivoted_csv_path, index=False)
    model_summary.to_csv(model_summary_csv_path, index=False)

    no_prediction_percent = comparison_df.groupby("model")["no_predictions"].mean() * 100.0
    summary_text = create_comparison_summary_text(
        num_images=len(gt_json.get("images", [])),
        num_rows=len(comparison_df),
        model_names=model_names,
        model_coverage=model_coverage,
        no_prediction_percent=no_prediction_percent,
    )
    write_text(summary_text, summary_txt_path)

    return {
        "gt_json": gt_json,
        "model_jsons": model_jsons,
        "all_comparison_results": all_comparison_results,
        "comparison_df": comparison_df,
        "performance_stats": performance_stats,
        "pivoted_stats": pivoted_stats,
        "model_summary": model_summary,
        "model_coverage": model_coverage,
        "paths": {
            "detailed_csv": detailed_csv_path,
            "stats_csv": stats_csv_path,
            "pivoted_csv": pivoted_csv_path,
            "model_summary_csv": model_summary_csv_path,
            "summary_txt": summary_txt_path,
        },
    }


def build_single_image_results_for_plotting(
    *,
    filename: str,
    gt_json_path: str,
    model_dirs: dict[str, str],
    image_height: int = 256,
    image_width: int = 256,
) -> dict[str, Any]:
    """
    Build all_results structure for a single image, ready for plot_feature_grid().
    """
    gt_json = read_json(gt_json_path)
    model_jsons = load_model_prediction_jsons(model_dirs)

    image_id = None
    for img in gt_json.get("images", []):
        if img.get("file_name") == filename:
            image_id = img.get("id")
            break

    if image_id is None:
        raise ValueError(f"Filename '{filename}' not found in GT JSON.")

    image_annotations_gt = get_annotations_for_image_id(gt_json, image_id)
    gt_results = build_image_result(
        image_annotations_gt,
        image_height=image_height,
        image_width=image_width,
    )

    all_results = {"ground_truth": gt_results}

    for model_name, pred_json in model_jsons.items():
        if pred_json is None:
            continue
        image_annotations_pred = get_annotations_for_image_id(pred_json, image_id)
        all_results[model_name] = build_image_result(
            image_annotations_pred,
            image_height=image_height,
            image_width=image_width,
        )

    return all_results
