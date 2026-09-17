from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from cnt_project.evaluation.pipelines.density_evaluation import (
    normalize_figure_formats,
)
from cnt_project.evaluation.pipelines.dsb_density_evaluation import (
    run_dsb_by_density,
)
from cnt_project.evaluation.plotting.smoothing_ablation_density import (
    display_density_label,
    save_smoothing_ablation_object_metric_figures,
)
from cnt_project.evaluation.utils import (
    display_density_label,
    infer_smoothing_model_base,
    infer_smoothing_variant,
)
from cnt_project.io.paths import ProjectPaths


DENSITIES = ("LOW_DENSITY", "MID_DENSITY", "HIGH_DENSITY")



def _read_density_outputs(
    *,
    eval_run_name: str,
    pred_run: str,
) -> list[dict]:
    P = ProjectPaths.from_here(__file__)
    out_dir = P.eval_subdir(eval_run_name, "dsb", "density_eval")

    rows: list[dict] = []

    for density in DENSITIES:
        csv_path = out_dir / f"ap_dice_results_poly_{density}.csv"
        if not csv_path.exists():
            print(f"[{pred_run}] Missing DSB density CSV: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        for _, row in df.iterrows():
            rows.append(
                {
                    "run_name": pred_run,
                    "eval_run_name": eval_run_name,
                    "model_base": infer_smoothing_model_base(pred_run),
                    "smoothing_variant": infer_smoothing_variant(pred_run),
                    "density": display_density_label(density),
                    "filename": row.get("filename", ""),
                    "num_gt_objects": row.get("num_gt_objects"),
                    "object_mAP": float(row["mAP"]),
                    "object_mean_dice": float(row["mean_dice"]),
                }
            )

    return rows


def run_smoothing_ablation_dsb_by_density(
    *,
    runs: list[str],
    split_manifest_path: str | Path,
    subset: str = "test",
    density_source: str = "manifest",
    density_column: str = "density_class",
    legacy_density_csv: str | Path | None = None,
    pred_filename: str = "predicted_annotations_poly.json",
    gt_json_path: str | Path | None = None,
    run_label_map: dict[str, str] | None = None,
    figure_formats: list[str] | None = None,
    figure_dpi: int = 300,
    report_name: str | None = None,
    skip_eval: bool = False,
) -> Path:
    """
    Compare existing smoothing-ablation prediction JSONs using existing DSB density eval.

    This runner does not modify DSB logic. It calls run_dsb_by_density(...)
    and aggregates the CSVs already produced by that evaluator.
    """
    P = ProjectPaths.from_here(__file__)
    P.ensure_outputs()

    density_source = str(density_source).strip().lower()

    if density_source == "manifest" and split_manifest_path is None:
        raise ValueError(
            "split_manifest_path is required when "
            "density_source='manifest'."
        )

    if density_source == "legacy_metadata" and legacy_density_csv is None:
        raise ValueError(
            "legacy_density_csv is required when "
            "density_source='legacy_metadata'."
        )

    figure_formats_tuple = normalize_figure_formats(figure_formats)

    if report_name is None:
        report_name = f"smoothing_ablation_dsb_density_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    report_dir = P.report_dir(report_name, "smoothing_ablation_dsb_density")
    report_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for pred_run in runs:
        pred_json = P.predicted_poly_json( pred_run, filename=pred_filename, )
        if not pred_json.exists():
            raise FileNotFoundError(f"Prediction JSON not found for run '{pred_run}': {pred_json}")

        eval_run_name = f"{report_name}__{pred_run}"

        if not skip_eval:
            print(f"[{pred_run}] Running DSB density evaluation...")
            run_dsb_by_density(
                selected_density="ALL",
                run_name=eval_run_name,
                pred_run=pred_run,
                pred_filename=pred_filename,
                gt_json_path=(
                    str(gt_json_path)
                    if gt_json_path is not None
                    else None
                ),
                eval_tag="density_eval",
                split_manifest_path=split_manifest_path,
                subset=subset,
                density_source=density_source,
                density_column=density_column,
                legacy_density_csv=legacy_density_csv,
            )
        else:
            print(f"[{pred_run}] Reusing existing DSB density CSVs...")

        all_rows.extend(
            _read_density_outputs(
                eval_run_name=eval_run_name,
                pred_run=pred_run,
            )
        )

    long_df = pd.DataFrame(all_rows)
    long_csv = report_dir / "object_metrics_by_density_long.csv"
    long_df.to_csv(long_csv, index=False)

    if not long_df.empty:
        summary_df = (
            long_df.groupby(["run_name", "model_base", "smoothing_variant", "density"], as_index=False)
            .agg(
                n_images=("filename", "count"),
                object_mAP_mean=("object_mAP", "mean"),
                object_mAP_std=("object_mAP", "std"),
                object_mean_dice_mean=("object_mean_dice", "mean"),
                object_mean_dice_std=("object_mean_dice", "std"),
            )
        )
    else:
        summary_df = pd.DataFrame()

    summary_csv = report_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    save_smoothing_ablation_object_metric_figures(
        all_rows,
        run_names=runs,
        run_label_map=run_label_map,
        output_dir=report_dir,
        figure_formats=figure_formats_tuple,
        figure_dpi=figure_dpi,
    )

    print(f"Saved smoothing ablation DSB density comparison to: {report_dir}")
    return report_dir
