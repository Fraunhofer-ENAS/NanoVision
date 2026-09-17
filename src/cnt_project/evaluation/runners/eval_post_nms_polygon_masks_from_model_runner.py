from __future__ import annotations

import argparse
from pathlib import Path

from cnt_project.evaluation.pipelines.density_evaluation import (
    DEFAULT_DENSITY_RUNNER_OUTPUTS,
)
from cnt_project.evaluation.pipelines.post_nms_polygon_evaluation import (
    VALID_STAGES,
    run_post_nms_stage_metrics_by_density,
)
from cnt_project.evaluation.runners.eval_loader_masks_by_density_runner import (
    _parse_run_label_mappings,
)
from cnt_project.model_development.model_compat import (
    parse_grayscale_arg,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate intermediate postprocessing polygon masks from model inference. "
            "Uses existing pipeline stage outputs after NMS and after smoothing."
        )
    )
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--run-labels", nargs="*", default=None)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["after_nms", "after_smoothing"],
        choices=sorted(VALID_STAGES),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--split-manifest-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--subset",
        choices=("train", "val", "test"),
        default="test",
    )

    parser.add_argument(
        "--density-source",
        choices=("manifest", "legacy_metadata"),
        default="manifest",
    )

    parser.add_argument(
        "--legacy-density-csv",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--density-column",
        default="density_class",
    )
    parser.add_argument("--densities", nargs="+", default=["ALL"])
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=list(DEFAULT_DENSITY_RUNNER_OUTPUTS),
    )
    parser.add_argument("--figure-formats", nargs="+", default=["png"])
    parser.add_argument("--figure-dpi", type=int, default=300)
    parser.add_argument(
        "--grayscale",
        type=parse_grayscale_arg,
        default=None,
        help="Use auto, true, or false.",
    )
    parser.add_argument("--rewrite", action="store_true")
    parser.add_argument("--no-fig", action="store_true")
    parser.add_argument(
        "--object-matching",
        default="both",
        choices=["greedy", "hungarian", "both"],
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Optional maximum number of samples loaded from the selected "
            "dataset subset. Intended primarily for debugging and smoke tests."
        ),
    )
    parser.add_argument("--debug-pixel-metrics", action="store_true")
    parser.add_argument("--debug-pixel-dirname", default="debug_pixel_metrics")
    parser.add_argument("--debug-pixel-max-images", type=int, default=None)
    parser.add_argument("--report-name", default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_post_nms_stage_metrics_by_density(
        model_names=args.models,
        dataset_root=args.dataset_root,
        split_manifest_path=args.split_manifest_path,
        subset=args.subset,
        density_source=args.density_source,
        density_column=args.density_column,
        legacy_density_csv=args.legacy_density_csv,
        run_label_map=_parse_run_label_mappings(args.run_labels),
        stages=args.stages,
        selected_outputs=args.outputs,
        figure_formats=args.figure_formats,
        figure_dpi=args.figure_dpi,
        densities=args.densities,
        grayscale=args.grayscale,
        rewrite=args.rewrite,
        fig=not args.no_fig,
        object_matching_algorithm=args.object_matching,
        max_samples=args.max_samples,
        debug_pixel_metrics=args.debug_pixel_metrics,
        debug_pixel_metrics_dirname=args.debug_pixel_dirname,
        debug_pixel_metrics_max_images=args.debug_pixel_max_images,
        report_name=args.report_name,
    )


if __name__ == "__main__":
    main()