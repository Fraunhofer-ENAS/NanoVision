from __future__ import annotations

import argparse
from pathlib import Path

from cnt_project.evaluation.pipelines.density_evaluation import (
    DEFAULT_DENSITY_RUNNER_OUTPUTS,
)
from cnt_project.evaluation.pipelines.raw_point_evaluation import (
    run_raw_point_metrics_by_density,
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
            "Run raw pre-NMS point-score inference for one or more models, "
            "convert raw points to binary masks, and evaluate by density."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more model folder names under models/.",
    )
    parser.add_argument(
        "--run-labels",
        nargs="*",
        default=None,
        help=(
            "Optional model/run-to-legend mappings in the form MODEL_NAME=LABEL. "
            "Labels matching SYM, EDT, EDT-SYM, or EDT-FULL use the scoring blue palette."
        ),
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help=(
            "Canonical unsplit dataset root containing images/, masks/, "
            "metadata/, splits/, and COCO_mask/."
        ),
    )

    parser.add_argument(
        "--split-manifest-path",
        type=Path,
        required=True,
        help="Path to the canonical split manifest CSV.",
    )

    parser.add_argument(
        "--subset",
        choices=("train", "val", "test"),
        default="test",
        help="Manifest subset to evaluate. Default: test.",
    )

    parser.add_argument(
        "--density-source",
        choices=("manifest", "legacy_metadata"),
        default="manifest",
        help=(
            "Density classification source. "
            "'manifest' uses canonical split-manifest density labels. "
            "'legacy_metadata' uses the frozen historical classification."
        ),
    )

    parser.add_argument(
        "--legacy-density-csv",
        type=Path,
        default=None,
        help=(
            "Frozen historical filename-to-density CSV. "
            "Required when --density-source legacy_metadata."
        ),
    )

    parser.add_argument(
        "--density-column",
        default="density_class",
        help=(
            "Split manifest column containing density labels. "
            "Default: density_class."
        ),
    )



    parser.add_argument(
        "--densities",
        nargs="+",
        default=["ALL"],
        help="Any combination of: Low Mid High All.",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=list(DEFAULT_DENSITY_RUNNER_OUTPUTS),
        help=(
            "Select which files to generate. Recommended for raw points: "
            "image_metrics pixel_metrics_long metric_figures summary"
        ),
    )
    parser.add_argument(
        "--figure-formats",
        nargs="+",
        default=["png"],
        help="Figure formats for metric_figures. Choices: png svg",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=300,
        help="DPI used when saving metric figures.",
    )
    parser.add_argument(
        "--grayscale",
        type=parse_grayscale_arg,
        default=None,
        help="Use auto, true, or false. Default: auto from model config/name.",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Rewrite CSV files instead of appending.",
    )
    parser.add_argument(
        "--no-fig",
        action="store_true",
        help="Disable per-image figure output.",
    )
    parser.add_argument(
        "--object-matching",
        default="both",
        choices=["greedy", "hungarian", "both"],
        help="Object-level matching algorithm mode.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Raw point score threshold. Points with score > threshold become 1.",
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
    parser.add_argument(
        "--debug-pixel-metrics",
        action="store_true",
        help="Save debug panels showing GT/pred label images and binary masks.",
    )
    parser.add_argument(
        "--debug-pixel-dirname",
        default="debug_pixel_metrics",
        help="Subfolder name under each density output folder for debug pixel panels.",
    )
    parser.add_argument(
        "--debug-pixel-max-images",
        type=int,
        default=None,
        help="Optional cap on number of debug pixel panels saved per density subset.",
    )
    parser.add_argument(
        "--report-name",
        default=None,
        help="Optional report folder name under global_outputs/reports.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_raw_point_metrics_by_density(
        model_names=args.models,
        dataset_root=args.dataset_root,
        split_manifest_path=args.split_manifest_path,
        subset=args.subset,
        density_source=args.density_source,
        density_column=args.density_column,
        legacy_density_csv=args.legacy_density_csv,
        run_label_map=_parse_run_label_mappings(args.run_labels),
        selected_outputs=args.outputs,
        figure_formats=args.figure_formats,
        figure_dpi=args.figure_dpi,
        densities=args.densities,
        grayscale=args.grayscale,
        rewrite=args.rewrite,
        fig=not args.no_fig,
        object_matching_algorithm=args.object_matching,
        score_threshold=args.score_threshold,
        max_samples=args.max_samples,
        debug_pixel_metrics=args.debug_pixel_metrics,
        debug_pixel_metrics_dirname=args.debug_pixel_dirname,
        debug_pixel_metrics_max_images=args.debug_pixel_max_images,
        report_name=args.report_name,
    )


if __name__ == "__main__":
    main()