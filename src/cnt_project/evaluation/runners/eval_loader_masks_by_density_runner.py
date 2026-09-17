from __future__ import annotations

import argparse
from pathlib import Path
from cnt_project.evaluation.pipelines.density_evaluation import (
    DEFAULT_DENSITY_RUNNER_OUTPUTS,
    run_loader_metrics_by_density,
)


def _parse_run_label_mappings(mappings: list[str] | None) -> dict[str, str]:
    if not mappings:
        return {}

    parsed: dict[str, str] = {}
    for item in mappings:
        if "=" not in item:
            raise ValueError(
                f"Invalid run label mapping '{item}'. Use RUN_NAME=LABEL."
            )
        run_name, label = item.split("=", 1)
        run_name = run_name.strip()
        label = label.strip()
        if not run_name or not label:
            raise ValueError(
                f"Invalid run label mapping '{item}'. Use RUN_NAME=LABEL."
            )
        parsed[run_name] = label
    return parsed

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run loader-mask evaluation on selected density subsets for one or more runs, "
            "and save an aggregated cross-run summary."
        )
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="One or more run names under global_outputs/runs.",
    )
    parser.add_argument(
        "--run-labels",
        nargs="*",
        default=None,
        help=(
            "Optional run-to-legend mappings in the form RUN_NAME=LABEL. "
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
            "'manifest' uses density labels from the canonical split manifest. "
            "'legacy_metadata' uses the frozen historical density classification."
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
        type=str,
        default="density_class",
        help=(
            "Split manifest column containing Low/Mid/High density labels. "
            "Default: density_class."
        ),
    )


    parser.add_argument(
        "--densities",
        nargs="+",
        default=["ALL"],
        help="Any combination of: Low Mid High All (case-insensitive).",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=list(DEFAULT_DENSITY_RUNNER_OUTPUTS),
        help=(
            "Select which files to generate. Choices: image_metrics iou_metrics properties "
            "distribution distribution_csv distribution_plots performance_plot summary "
            "pixel_metrics_long metric_figures"
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
        help="DPI used when saving metric figures (applies to raster formats like PNG).",
    )
    parser.add_argument(
        "--reuse-existing-image-metrics",
        action="store_true",
        help=(
            "Skip evaluation and reuse existing image_metrics_<algo>.csv files under each "
            "run/density folder to build summaries and metric figures."
        ),
    )
    parser.add_argument(
        "--pred-filename",
        default="predicted_annotations_poly.json",
        help="Prediction JSON filename inside each run inference folder.",
    )
    parser.add_argument(
        "--grayscale",
        default="true",
        choices=["true", "false"],
        help="Whether to load test images as grayscale for loader alignment.",
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
        "--debug-pixel-metrics",
        action="store_true",
        help=(
            "Save debug panels showing GT/pred label images and their binary masks "
            "before pixel metric computation."
        ),
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

    run_loader_metrics_by_density(
        run_names=args.runs,
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
        reuse_existing_image_metrics=args.reuse_existing_image_metrics,
        densities=args.densities,
        pred_filename=args.pred_filename,
        grayscale=(args.grayscale == "true"),
        rewrite=args.rewrite,
        fig=not args.no_fig,
        object_matching_algorithm=args.object_matching,
        debug_pixel_metrics=args.debug_pixel_metrics,
        debug_pixel_metrics_dirname=args.debug_pixel_dirname,
        debug_pixel_metrics_max_images=args.debug_pixel_max_images,
        report_name=args.report_name,
    )


if __name__ == "__main__":
    main()
