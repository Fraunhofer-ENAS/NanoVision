from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from cnt_project.coco.masks import coco_polygons_json_to_instance_masks
from cnt_project.evaluation.inputs.evaluation_grouping import (
    load_density_groups,
    normalize_evaluation_filename,
)
from cnt_project.evaluation.pipelines.loader_mask_evaluation import main_evaluation_loop
from cnt_project.io.paths import ProjectPaths
from cnt_project.preprocessing.dataset.dataset_loader import CNTDataset
from cnt_project.evaluation.outputs.density_reports import (
    append_summary_and_pixel_rows_from_density_dir,
)
from cnt_project.evaluation.plotting.density_metrics import (
    normalize_figure_formats,
    save_density_metric_figures,
)




DEFAULT_DENSITY_RUNNER_OUTPUTS = (
    "image_metrics",
    "metric_figures",
)

VALID_DENSITY_RUNNER_OUTPUTS = {
    "image_metrics",
    "iou_metrics",
    "properties",
    "distribution",
    "distribution_csv",
    "distribution_plots",
    "performance_plot",
    "summary",
    "pixel_metrics_long",
    "metric_figures",
}


class _SubsetLoader:
    """Minimal loader adapter exposing only the subset to evaluate."""

    def __init__(self, samples):
        self._samples = list(samples)

    def get_test_data(self):
        return self._samples




def _normalize_density_tokens(tokens: Iterable[str] | None) -> list[str]:
    if not tokens:
        return ["LOW_DENSITY", "MID_DENSITY", "HIGH_DENSITY"]

    aliases = {
        "LOW": "LOW_DENSITY",
        "LOW_DENSITY": "LOW_DENSITY",
        "MID": "MID_DENSITY",
        "MIDDLE": "MID_DENSITY",
        "MID_DENSITY": "MID_DENSITY",
        "HIGH": "HIGH_DENSITY",
        "HIGH_DENSITY": "HIGH_DENSITY",
        "ALL": "ALL",
    }

    resolved: list[str] = []
    for token in tokens:
        key = str(token).strip().upper()
        val = aliases.get(key)
        if val is None:
            raise ValueError(
                f"Unsupported density token '{token}'. "
                "Use one or more of: Low, Mid, High, All."
            )
        resolved.append(val)

    if "ALL" in resolved:
        return ["LOW_DENSITY", "MID_DENSITY", "HIGH_DENSITY"]

    # Preserve order, remove duplicates.
    out: list[str] = []
    for d in resolved:
        if d not in out:
            out.append(d)
    return out


def _algorithms_from_mode(mode: str) -> tuple[str, ...]:
    mode = str(mode).strip().lower()
    if mode == "both":
        return ("greedy", "hungarian")
    if mode in {"greedy", "hungarian"}:
        return (mode,)
    raise ValueError("object_matching_algorithm must be one of: greedy, hungarian, both")



def _canonical_pixel_metrics_algorithm(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode == "both":
        return "greedy"
    if mode in {"greedy", "hungarian"}:
        return mode
    raise ValueError("object_matching_algorithm must be one of: greedy, hungarian, both")




def _normalize_selected_outputs(outputs: list[str] | None) -> set[str]:
    selected = set(outputs or DEFAULT_DENSITY_RUNNER_OUTPUTS)
    unknown = selected - VALID_DENSITY_RUNNER_OUTPUTS
    if unknown:
        raise ValueError(f"Unsupported outputs: {sorted(unknown)}")
    return selected


def _evaluation_outputs_from_selected_outputs(selected_outputs: set[str]) -> tuple[str, ...]:
    mapping_order = (
        "image_metrics",
        "iou_metrics",
        "properties",
        "distribution",
        "distribution_csv",
        "distribution_plots",
        "performance_plot",
    )
    return tuple(name for name in mapping_order if name in selected_outputs)



def run_loader_metrics_by_density(
    *,
    run_names: list[str],
    dataset_root: str | Path,
    split_manifest_path: str | Path,
    subset: str = "test",
    density_source: str = "manifest",
    density_column: str = "density_class",
    legacy_density_csv: str | Path | None = None,
    run_label_map: dict[str, str] | None = None,
    selected_outputs: list[str] | None = None,
    figure_formats: list[str] | None = None,
    figure_dpi: int = 300,
    reuse_existing_image_metrics: bool = False,
    densities: list[str] | None = None,
    pred_filename: str = "predicted_annotations_poly.json",
    grayscale: bool = True,
    rewrite: bool = False,
    fig: bool = False,
    object_matching_algorithm: str = "both",
    debug_pixel_metrics: bool = False,
    debug_pixel_metrics_dirname: str = "debug_pixel_metrics",
    debug_pixel_metrics_max_images: int | None = None,
    report_name: str | None = None,
) -> Path:
    """
    Evaluate loader metrics on density subsets for multiple runs, then aggregate.

    Outputs per run/density under:
      global_outputs/runs/<RUN>/eval/loader_masks/density_eval/<DENSITY>/

    Aggregated summary under:
      global_outputs/reports/<REPORT>/loader_density_comparison/summary.csv
    """
    P = ProjectPaths.from_here(__file__)
    P.ensure_outputs()

    density_groups = load_density_groups(
        density_source=density_source,
        split_manifest_path=split_manifest_path,
        subset=subset,
        density_column=density_column,
        legacy_density_csv=legacy_density_csv,
    )
    selected_densities = _normalize_density_tokens(densities)
    selected_outputs_set = _normalize_selected_outputs(selected_outputs)
    selected_figure_formats = normalize_figure_formats(figure_formats)
    evaluation_outputs = _evaluation_outputs_from_selected_outputs(selected_outputs_set)
    need_summary = "summary" in selected_outputs_set
    need_pixel_rows = bool({"pixel_metrics_long", "metric_figures"} & selected_outputs_set)

    if reuse_existing_image_metrics:
        forbidden = {"iou_metrics", "properties", "distribution", "distribution_csv", "distribution_plots", "performance_plot"}
        unsupported = forbidden & selected_outputs_set
        if unsupported:
            raise ValueError(
                "reuse_existing_image_metrics mode supports only outputs derived from image_metrics. "
                f"Unsupported requested outputs: {sorted(unsupported)}"
            )

    missing_groups = [d for d in selected_densities if d not in density_groups]
    if missing_groups:
        raise ValueError(
            f"Density groups {missing_groups} are not available from "
            f"density source '{density_source}'. "
            f"Available: {sorted(density_groups.keys())}"
        )

    algorithms = _algorithms_from_mode(object_matching_algorithm)
    pixel_metrics_algorithm = _canonical_pixel_metrics_algorithm(object_matching_algorithm)

    summary_rows: list[dict] = []
    pixel_metric_rows: list[dict] = []

    for run_name in run_names:
        if not reuse_existing_image_metrics:
            pred_json_path = P.predicted_poly_json(run_name, filename=pred_filename)
            if not pred_json_path.exists():
                raise FileNotFoundError(
                    f"Prediction JSON not found for run '{run_name}': {pred_json_path}"
                )

            dataset = CNTDataset(
                dataset_root=dataset_root,
                split_manifest_path=split_manifest_path,
                subset=subset,
                grayscale=grayscale,
                require_masks=True,
            )

            samples = dataset.get_test_data()

            stems = [
                sample.X_fn
                for sample in samples
            ]

            shapes = [
                sample.X.shape
                for sample in samples
            ]

            pred_masks = coco_polygons_json_to_instance_masks(
                str(pred_json_path),
                image_stems=stems,
                image_shapes=shapes,
            )

            dataset.set_predictions(pred_masks)

            sample_key_to_sample = {
                normalize_evaluation_filename(sample.X_fn): sample
                for sample in samples
            }

            sample_keys = [
                normalize_evaluation_filename(sample.X_fn)
                for sample in samples
            ]
        else:
            sample_key_to_sample = None
            sample_keys = []

        for density in selected_densities:
            out_dir = P.eval_subdir(
                run_name,
                "loader_masks",
                "density_eval",
                density.lower(),
            )

            if not reuse_existing_image_metrics:
                target_keys = { normalize_evaluation_filename(name) for name in density_groups[density] }
                assert sample_key_to_sample is not None
                subset_samples = [ sample_key_to_sample[key] for key in sample_keys if key in target_keys ]


                if not subset_samples:
                    print(f"[{run_name}] {density}: no matching test samples found. Skipping.")
                    continue

                subset_loader = _SubsetLoader(subset_samples)
                print(f"[{run_name}] {density}: evaluating {len(subset_samples)} images...")

                main_evaluation_loop(
                    subset_loader,
                    str(out_dir),
                    rewrite=rewrite,
                    fig=fig,
                    object_matching_algorithm=object_matching_algorithm,
                    include_outputs=evaluation_outputs,
                    debug_pixel_metrics=debug_pixel_metrics,
                    debug_pixel_metrics_dirname=debug_pixel_metrics_dirname,
                    debug_pixel_metrics_max_images=debug_pixel_metrics_max_images,
                )
            else:
                print(f"[{run_name}] {density}: reusing existing image_metrics CSVs...")

            append_summary_and_pixel_rows_from_density_dir(
                out_dir=out_dir,
                run_name=run_name,
                density=density,
                algorithms=algorithms,
                pixel_metrics_algorithm=pixel_metrics_algorithm,
                need_summary=need_summary,
                need_pixel_rows=need_pixel_rows,
                summary_rows=summary_rows,
                pixel_metric_rows=pixel_metric_rows,
            )

    if report_name is None:
        report_name = f"loader_density_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    report_dir = P.report_dir(report_name, "loader_density_comparison")
    summary_path = report_dir / "summary.csv"
    return_path = report_dir

    if "summary" in selected_outputs_set:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(summary_path, index=False)
        return_path = summary_path

    if {"pixel_metrics_long", "metric_figures"} & selected_outputs_set:
        save_density_metric_figures(
            pixel_metric_rows,
            selected_densities=selected_densities,
            run_names=run_names,
            run_label_map=run_label_map,
            output_dir=report_dir,
            figure_formats=selected_figure_formats,
            figure_dpi=figure_dpi,
            write_long_csv=("pixel_metrics_long" in selected_outputs_set),
        )

    print(f"Saved loader density comparison outputs to: {report_dir}")
    return return_path

