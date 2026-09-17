from __future__ import annotations

import argparse
from pathlib import Path

from cnt_project.evaluation.pipelines.smoothing_ablation_dsb_evaluation import (
    run_smoothing_ablation_dsb_by_density,
)
from cnt_project.evaluation.runners.eval_loader_masks_by_density_runner import (
    _parse_run_label_mappings,
)




def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare smoothing-ablation COCO polygon prediction JSONs using existing "
            "DSB density object-level evaluation outputs."
        )
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Prediction run names under global_outputs/runs.",
    )
    parser.add_argument(
        "--run-labels",
        nargs="*",
        default=None,
        help="Optional RUN=LABEL mappings for plot legends.",
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
        help="Dataset subset used for density grouping. Default: test.",
    )

    parser.add_argument(
        "--density-source",
        choices=("manifest", "legacy_metadata"),
        default="manifest",
        help=(
            "Density classification source. "
            "'manifest' uses canonical split-manifest labels; "
            "'legacy_metadata' uses the frozen historical CSV."
        ),
    )

    parser.add_argument(
        "--density-column",
        default="density_class",
        help="Density column in the split manifest. Default: density_class.",
    )

    parser.add_argument(
        "--legacy-density-csv",
        type=Path,
        default=None,
        help=(
            "Frozen historical density-classification CSV. "
            "Required when --density-source legacy_metadata."
        ),
    )

    parser.add_argument(
        "--pred-filename",
        default="predicted_annotations_poly.json",
        help="Prediction JSON filename inside each prediction run.",
    )

    parser.add_argument(
        "--gt-json-path",
        type=Path,
        default=None,
        help=(
            "Optional explicit GT COCO JSON. "
            "When omitted, the DSB evaluator uses its configured default."
        ),
    )
    parser.add_argument(
        "--figure-formats",
        nargs="+",
        default=["png"],
        help="Figure formats. Choices: png svg.",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--report-name",
        default=None,
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Reuse existing DSB density CSVs instead of running DSB evaluation.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_smoothing_ablation_dsb_by_density(
        runs=args.runs,
        split_manifest_path=args.split_manifest_path,
        subset=args.subset,
        density_source=args.density_source,
        density_column=args.density_column,
        legacy_density_csv=args.legacy_density_csv,
        pred_filename=args.pred_filename,
        gt_json_path=args.gt_json_path,
        run_label_map=_parse_run_label_mappings(args.run_labels),
        figure_formats=args.figure_formats,
        figure_dpi=args.figure_dpi,
        report_name=args.report_name,
        skip_eval=args.skip_eval,
    )


if __name__ == "__main__":
    main()