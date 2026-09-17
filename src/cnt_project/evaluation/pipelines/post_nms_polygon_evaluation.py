from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.draw import polygon as draw_polygon

from cnt_project.evaluation.inputs.evaluation_grouping import (
    load_density_groups,
    normalize_evaluation_filename,
)
from cnt_project.evaluation.outputs.density_reports import (
    append_summary_and_pixel_rows_from_density_dir,
)
from cnt_project.evaluation.pipelines.density_evaluation import (
    _SubsetLoader,
    _algorithms_from_mode,
    _canonical_pixel_metrics_algorithm,
    _evaluation_outputs_from_selected_outputs,
    _normalize_density_tokens,
    _normalize_selected_outputs,
    normalize_figure_formats,
    save_density_metric_figures,
)
from cnt_project.evaluation.pipelines.loader_mask_evaluation import (
    main_evaluation_loop,
)
from cnt_project.io.paths import ProjectPaths
from cnt_project.io.run_manager import (
    create_run,
    make_run_id,
    timestamp_now,
)
from cnt_project.model_development.model_compat import (
    check_model_compatibility,
    recommended_grayscale_setting,
    infer_model_input_channels,
)
from cnt_project.model_development.stardist_patched.stardist_model_configuration import (
    MyStarDist2D,
)
from cnt_project.preprocessing.dataset.dataset_loader import (
    CNTDataset,
)

VALID_STAGES = {"after_nms", "after_smoothing"}


def polygons_to_binary_mask(
    *,
    polygons: list,
    image_shape: tuple[int, ...],
) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for poly in polygons:
        if poly is None or poly.is_empty or not hasattr(poly, "exterior"):
            continue

        x_coords, y_coords = poly.exterior.xy
        rr, cc = draw_polygon(
            np.asarray(y_coords, dtype=np.float32),
            np.asarray(x_coords, dtype=np.float32),
            shape=(h, w),
        )
        mask[rr, cc] = 1

    return mask


def run_post_nms_stage_inference(
    *,
    loader: CNTDataset,
    model_name: str,
    basedir: str,
    stages: list[str],
) -> dict[str, list[np.ndarray]]:
    model = MyStarDist2D(None, name=model_name, basedir=basedir)

    stage_masks: dict[str, list[np.ndarray]] = {stage: [] for stage in stages}
    samples = loader.get_test_data()

    for i, sample in enumerate(samples):
        print(f"[{model_name}] stage inference {i + 1}/{len(samples)} - {sample.X_fn}")

        y_pred, details, final_polygons, final_scores, final_coords = (
            model.predict_instances_with_stage_outputs(
                sample.X,
                n_tiles=model._guess_n_tiles(sample.X),
                show_tile_progress=False,
                fname=sample.X_fn,
            )
        )

        stage_outputs = details.get("stage_outputs")
        if stage_outputs is None:
            raise RuntimeError(
                "stage_outputs not found in prediction details. "
                "Check that predict_instances_with_stage_outputs() forwards "
                "return_stage_outputs=True through the adapter and pipeline."
            )

        for stage in stages:
            if stage not in stage_outputs:
                raise RuntimeError(
                    f"Stage '{stage}' not found in stage_outputs for image {sample.X_fn}. "
                    f"Available stages: {sorted(stage_outputs.keys())}"
                )

            polygons = stage_outputs[stage]["polygons"]
            mask = polygons_to_binary_mask(
                polygons=polygons,
                image_shape=sample.X.shape,
            )
            stage_masks[stage].append(mask)

    return stage_masks


def run_post_nms_stage_metrics_by_density(
    *,
    model_names: list[str],
    dataset_root: str | Path,
    split_manifest_path: str | Path,
    subset: str = "test",
    density_source: str = "manifest",
    density_column: str = "density_class",
    legacy_density_csv: str | Path | None = None,
    run_label_map: dict[str, str] | None = None,
    stages: list[str] | None = None,
    selected_outputs: list[str] | None = None,
    figure_formats: list[str] | None = None,
    figure_dpi: int = 300,
    densities: list[str] | None = None,
    grayscale: bool | None = None,
    rewrite: bool = False,
    fig: bool = False,
    object_matching_algorithm: str = "both",
    max_samples: int | None = None,
    debug_pixel_metrics: bool = False,
    debug_pixel_metrics_dirname: str = "debug_pixel_metrics",
    debug_pixel_metrics_max_images: int | None = None,
    report_name: str | None = None,
) -> Path:
    """
    Evaluate post-NMS polygon-mask stages by density for one or more models.

    Supported stages:
    - after_nms
    - after_smoothing

    Each model evaluation is stored in a timestamped run:

    global_outputs/runs/
    <MODEL>__post-nms-polygon-density-eval__<TIMESTAMP>/
    eval/post_nms_polygon_masks/<STAGE>/density_eval/<DENSITY>/

    Aggregated cross-model/stage reports are written under:

    global_outputs/reports/<REPORT>/
    post_nms_polygon_density_comparison/
    """
    P = ProjectPaths.from_here(__file__)
    P.ensure_outputs()

    basedir = str(P.project_root / "models")

    selected_stages = stages or ["after_nms", "after_smoothing"]
    unknown_stages = set(selected_stages) - VALID_STAGES
    if unknown_stages:
        raise ValueError(f"Unsupported stages: {sorted(unknown_stages)}")

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

    missing_groups = [d for d in selected_densities if d not in density_groups]
    if missing_groups:
        raise ValueError(
            f"Density groups {missing_groups} not available. "
            f"Available: {sorted(density_groups.keys())}"
        )

    algorithms = _algorithms_from_mode(object_matching_algorithm)
    pixel_metrics_algorithm = _canonical_pixel_metrics_algorithm(object_matching_algorithm)

    summary_rows: list[dict] = []
    pixel_metric_rows: list[dict] = []
    comparison_names: list[str] = []
    evaluation_timestamp = timestamp_now()

    for model_name in model_names:
        evaluation_run_id = make_run_id( model_name, "post-nms-polygon-density-eval", ts=evaluation_timestamp, )

        run_paths = create_run( project_root=P.project_root, run_id=evaluation_run_id, )

        model_n_channels = infer_model_input_channels(basedir, model_name)
        resolved_grayscale = recommended_grayscale_setting(
            current_grayscale=False if grayscale is None else grayscale,
            model_name=model_name,
            model_n_channels=model_n_channels,
        )

        check_model_compatibility(
            grayscale=resolved_grayscale,
            model_name=model_name,
            model_n_channels=model_n_channels,
        )

        dataset = CNTDataset(
            dataset_root=dataset_root,
            split_manifest_path=split_manifest_path,
            subset=subset,
            grayscale=resolved_grayscale,
            require_masks=True,
            max_samples=max_samples,
        )

        samples = dataset.get_test_data()
        stems = [ sample.X_fn for sample in samples ]
        stem_to_sample = { sample.X_fn: sample for sample in samples }

        stage_masks = run_post_nms_stage_inference(
            loader=dataset,
            model_name=model_name,
            basedir=basedir,
            stages=selected_stages,
        )

        for stage in selected_stages:
            comparison_name = f"{model_name}__{stage}"
            comparison_names.append(comparison_name)

            dataset.set_predictions(stage_masks[stage])

            samples = dataset.get_test_data()

            stem_to_sample = { sample.X_fn: sample for sample in samples }

            for density in selected_densities:
                out_dir = ( run_paths.eval_dir / "post_nms_polygon_masks" / stage / "density_eval" / density.lower() )

                out_dir.mkdir( parents=True, exist_ok=True, )

                target_keys = { normalize_evaluation_filename(name) for name in density_groups[density] }
                subset_samples = [
                    stem_to_sample[s]
                    for s in stems
                    if normalize_evaluation_filename(s) in target_keys
                    and s in stem_to_sample
                ]

                if not subset_samples:
                    print(f"[{comparison_name}] {density}: no matching test samples found. Skipping.")
                    continue

                subset_loader = _SubsetLoader(subset_samples)
                print(
                    f"[{comparison_name}] {density}: "
                    f"evaluating {len(subset_samples)} images..."
                )

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

                append_summary_and_pixel_rows_from_density_dir(
                    out_dir=out_dir,
                    run_name=comparison_name,
                    density=density,
                    algorithms=algorithms,
                    pixel_metrics_algorithm=pixel_metrics_algorithm,
                    need_summary=need_summary,
                    need_pixel_rows=need_pixel_rows,
                    summary_rows=summary_rows,
                    pixel_metric_rows=pixel_metric_rows,
                )
        print( f"[{model_name}] evaluation run: " f"{evaluation_run_id}" )

        print( f"[{model_name}] outputs: " f"{run_paths.eval_dir / 'post_nms_polygon_masks'}" )

    if report_name is None:
        report_name = f"post_nms_polygon_density_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    report_dir = P.report_dir(report_name, "post_nms_polygon_density_comparison")
    report_dir.mkdir(parents=True, exist_ok=True)

    return_path = report_dir

    if "summary" in selected_outputs_set:
        summary_path = report_dir / "summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        return_path = summary_path

    if {"pixel_metrics_long", "metric_figures"} & selected_outputs_set:
        pixel_df = pd.DataFrame(pixel_metric_rows)

        for stage in selected_stages:
            stage_suffix = f"__{stage}"
            stage_rows = pixel_df[
                pixel_df["run_name"].astype(str).str.endswith(stage_suffix)
            ].copy()

            if stage_rows.empty:
                print(f"No pixel metric rows found for stage '{stage}'. Skipping figures.")
                continue

            stage_dir = report_dir / stage
            stage_dir.mkdir(parents=True, exist_ok=True)

            # Convert comparison names back to model names for clean labels/colors.
            stage_rows["run_name"] = (
                stage_rows["run_name"]
                .astype(str)
                .str.replace(stage_suffix, "", regex=False)
            )

            stage_run_names = [
                name.replace(stage_suffix, "")
                for name in comparison_names
                if name.endswith(stage_suffix)
            ]

            stage_run_label_map = None
            if run_label_map:
                stage_run_label_map = {}
                for comparison_name in comparison_names:
                    if not comparison_name.endswith(stage_suffix):
                        continue

                    model_name = comparison_name.replace(stage_suffix, "")
                    stage_run_label_map[model_name] = run_label_map.get(
                        comparison_name,
                        run_label_map.get(model_name, model_name),
                    )

            save_density_metric_figures(
                stage_rows.to_dict("records"),
                selected_densities=selected_densities,
                run_names=stage_run_names,
                run_label_map=stage_run_label_map,
                output_dir=stage_dir,
                figure_formats=selected_figure_formats,
                figure_dpi=figure_dpi,
                write_long_csv=("pixel_metrics_long" in selected_outputs_set),
            )

    print(f"Saved post-NMS polygon density comparison outputs to: {report_dir}")
    return return_path

