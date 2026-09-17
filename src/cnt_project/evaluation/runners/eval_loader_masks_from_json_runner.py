from __future__ import annotations

import argparse
from pathlib import Path

from cnt_project.evaluation.pipelines.loader_json_evaluation import (
    run_loader_metrics_from_json,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run loader-based pixel/feature metrics from an existing prediction JSON.",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Run folder under global_outputs/runs containing prediction JSON.",
    )
    parser.add_argument(
        "--pred-filename",
        default="predicted_annotations_poly.json",
        help="Prediction JSON filename inside run inference folder.",
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
        default="density_class",
        help=(
            "Split manifest column containing density labels. "
            "Default: density_class."
        ),
    )

    parser.add_argument(
        "--length-source",
        choices=("canonical_metadata", "legacy_metadata"),
        default="canonical_metadata",
        help=(
            "Length classification source. "
            "'canonical_metadata' derives classes from canonical length metadata; "
            "'legacy_metadata' uses the frozen historical classification."
        ),
    )

    parser.add_argument(
        "--length-measurement",
        choices=("skeleton", "geodesic_px", "geodesic_um"),
        default="skeleton",
        help=(
            "Length measurement used with canonical metadata. "
            "Default: skeleton."
        ),
    )

    parser.add_argument(
        "--length-percentile",
        type=float,
        default=98.5,
        help=(
            "Object-length percentile used to define the long-object threshold "
            "for canonical metadata. Default: 98.5."
        ),
    )

    parser.add_argument(
        "--legacy-length-csv",
        type=Path,
        default=None,
        help=(
            "Frozen historical image-length classification CSV. "
            "Required when --length-source legacy_metadata."
        ),
    )


    parser.add_argument(
        "--mode",
        default="full",
        choices=["full", "density", "length", "all"],
        help="Which loader evaluation subset mode to run.",
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
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_loader_metrics_from_json(
        run_name=args.run_name,
        dataset_root=args.dataset_root,
        split_manifest_path=args.split_manifest_path,
        subset=args.subset,
        density_source=args.density_source,
        density_column=args.density_column,
        legacy_density_csv=args.legacy_density_csv,
        pred_filename=args.pred_filename,
        length_source=args.length_source,
        length_measurement=args.length_measurement,
        length_percentile=args.length_percentile,
        legacy_length_csv=args.legacy_length_csv,
        mode=args.mode,
        grayscale=(args.grayscale == "true"),
        rewrite=args.rewrite,
        fig=not args.no_fig,
    )


if __name__ == "__main__":
    main()