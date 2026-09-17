from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

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

def point_scores_to_binary_mask(
    *,
    points: np.ndarray,
    scores: np.ndarray,
    image_shape: tuple[int, ...],
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Convert raw pre-NMS StarDist candidate points into a binary point mask.

    points are expected in y/x order.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if points is None or scores is None:
        return mask

    points = np.asarray(points)
    scores = np.asarray(scores)

    if points.size == 0 or scores.size == 0:
        return mask

    ys = np.clip(np.rint(points[:, 0]).astype(int), 0, h - 1)
    xs = np.clip(np.rint(points[:, 1]).astype(int), 0, w - 1)

    keep = scores > threshold
    mask[ys[keep], xs[keep]] = 1

    return mask


def run_raw_point_inference(
    *,
    loader: CNTDataset,
    model_name: str,
    basedir: str,
    score_threshold: float = 0.0,
) -> list[np.ndarray]:
    """
    Run model inference and return binary masks made from raw pre-NMS points.

    This uses model.predict_raw_points(), which wraps predict_sparse().
    Therefore the points/scores are taken before _instances_from_prediction(),
    before NMS, and before polygon merging.
    """
    model = MyStarDist2D(None, name=model_name, basedir=basedir)

    point_masks: list[np.ndarray] = []

    samples = loader.get_test_data()
    for i, sample in enumerate(samples):
        print(f"[{model_name}] raw-point inference {i + 1}/{len(samples)} - {sample.X_fn}")

        raw = model.predict_raw_points(
            sample.X,
            n_tiles=model._guess_n_tiles(sample.X),
            show_tile_progress=False,
        )

        pred_mask = point_scores_to_binary_mask(
            points=raw["points"],
            scores=raw["scores"],
            image_shape=sample.X.shape,
            threshold=score_threshold,
        )

        point_masks.append(pred_mask)

    return point_masks


def run_raw_point_metrics_by_density(
    *,
    model_names: list[str],
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
    densities: list[str] | None = None,
    grayscale: bool | None = None,
    rewrite: bool = False,
    fig: bool = False,
    object_matching_algorithm: str = "both",
    score_threshold: float = 0.0,
    max_samples: int | None = None,
    debug_pixel_metrics: bool = False,
    debug_pixel_metrics_dirname: str = "debug_pixel_metrics",
    debug_pixel_metrics_max_images: int | None = None,
    report_name: str | None = None,
) -> Path:
    """
    Evaluate raw pre-NMS point-score masks by density for one or more models.

    Outputs per model/density under:
      global_outputs/runs/<MODEL>__raw-point-density-eval__<TIMESTAMP>/eval/raw_point_masks/density_eval/<DENSITY>/

    Aggregated outputs under:
      global_outputs/reports/<REPORT>/raw_point_density_comparison/
    """
    P = ProjectPaths.from_here(__file__)
    P.ensure_outputs()

    basedir = str(P.project_root / "models")

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

    missing_groups = [ d for d in selected_densities if d not in density_groups ]
    if missing_groups:
        raise ValueError(
            f"Density groups {missing_groups} not available. "
            f"Available: {sorted(density_groups.keys())}"
        )

    algorithms = _algorithms_from_mode(object_matching_algorithm)
    pixel_metrics_algorithm = _canonical_pixel_metrics_algorithm(object_matching_algorithm)

    summary_rows: list[dict] = []
    pixel_metric_rows: list[dict] = []
    evaluation_timestamp = timestamp_now()


    for model_name in model_names:
        evaluation_run_id = make_run_id( model_name, "raw-point-density-eval", ts=evaluation_timestamp, )

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

        point_masks = run_raw_point_inference( loader=dataset, model_name=model_name, basedir=basedir, score_threshold=score_threshold, )

        dataset.set_predictions(point_masks)

        samples = dataset.get_test_data()
        stem_to_sample = {sample.X_fn: sample for sample in samples}

        for density in selected_densities:
            out_dir = ( run_paths.eval_dir / "raw_point_masks" / "density_eval" / density.lower() )

            out_dir.mkdir( parents=True, exist_ok=True, )

            target_keys = { normalize_evaluation_filename(name) for name in density_groups[density] }
            subset_samples = [
                stem_to_sample[s]
                for s in stems
                if normalize_evaluation_filename(s) in target_keys
                and s in stem_to_sample
            ]

            if not subset_samples:
                print( f"[{model_name}] {density}: " "no matching test samples found. Skipping." )
                continue

            subset_loader = _SubsetLoader(subset_samples)

            print( f"[{model_name}] {density}: " f"evaluating {len(subset_samples)} raw-point masks..." )

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
                run_name=model_name,
                density=density,
                algorithms=algorithms,
                pixel_metrics_algorithm=pixel_metrics_algorithm,
                need_summary=need_summary,
                need_pixel_rows=need_pixel_rows,
                summary_rows=summary_rows,
                pixel_metric_rows=pixel_metric_rows,
            )
        print( f"[{model_name}] evaluation run: " f"{evaluation_run_id}" )

        print( f"[{model_name}] outputs: " f"{run_paths.eval_dir / 'raw_point_masks' / 'density_eval'}" )

    if report_name is None:
        report_name = f"raw_point_density_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    report_dir = P.report_dir(report_name, "raw_point_density_comparison")
    report_dir.mkdir(parents=True, exist_ok=True)

    return_path = report_dir

    if "summary" in selected_outputs_set:
        summary_path = report_dir / "summary.csv"
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(summary_path, index=False)
        return_path = summary_path

    if {"pixel_metrics_long", "metric_figures"} & selected_outputs_set:
        save_density_metric_figures(
            pixel_metric_rows,
            selected_densities=selected_densities,
            run_names=model_names,
            run_label_map=run_label_map,
            output_dir=report_dir,
            figure_formats=selected_figure_formats,
            figure_dpi=figure_dpi,
            write_long_csv=("pixel_metrics_long" in selected_outputs_set),
        )

    print(f"Saved raw-point density comparison outputs to: {report_dir}")
    return return_path
