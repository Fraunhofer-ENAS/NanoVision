from __future__ import annotations

import argparse
from pathlib import Path

from cnt_project.evaluation.pipelines.dsb_density_evaluation import (
    run_dsb_by_density,
)
from cnt_project.evaluation.pipelines.dsb_diagnostics import (
    run_dsb_full_dataset_diagnostics,
)
from cnt_project.evaluation.pipelines.dsb_length_evaluation import (
    run_dsb_by_length_class,
)
from cnt_project.evaluation.pipelines.dsb_noise_evaluation import (
    run_dsb_by_noise_class,
)

"""This is a dataset-level evaluation runner for DSB-style instance segmentation metrics.
it is mask-based instance evaluation family, even though the inputs are COCO polygon JSONs."""

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run DSB-style instance-segmentation evaluation by density, "
            "noise class, length class, or full-dataset diagnostics."
        )
    )

    parser.add_argument(
        "--mode",
        default="diagnostics",
        choices=["density", "length", "noise", "diagnostics"],
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--pred-run", required=True)
    parser.add_argument(
        "--pred-filename",
        default="predicted_annotations_poly.json",
    )
    parser.add_argument(
        "--global-outputs-root",
        type=Path,
        default=None,
        help=(
            "Root directory containing CNTLib generated outputs. "
            "Predictions are read from runs/<pred-run>/inference/ "
            "and evaluation results are written under "
            "runs/<run-name>/eval/. "
            "When omitted, the source-repository global_outputs "
            "directory is used."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Canonical dataset root containing images/, masks/, "
            "metadata/, splits/, and COCO_mask/. "
            "Required for canonical length evaluation."
        ),
    )
    parser.add_argument("--gt-json-path", default=None)
    parser.add_argument("--eval-tag", default="diagnostics")
    parser.add_argument("--density-group", default=None)
    parser.add_argument(
        "--split-manifest-path",
        type=Path,
        default=None,
        help=(
            "Canonical split manifest used for evaluation grouping. "
            "Required for manifest-based density grouping and "
            "canonical length grouping."
        ),
    )
    parser.add_argument(
        "--length-source",
        choices=("canonical_metadata", "legacy_metadata"),
        default="canonical_metadata",
        help=(
            "Length classification source. "
            "'canonical_metadata' derives classes from canonical "
            "object/image length metadata; 'legacy_metadata' uses "
            "the frozen historical classification CSV."
        ),
    )

    parser.add_argument(
        "--length-measurement",
        choices=("skeleton", "geodesic_px", "geodesic_um"),
        default="geodesic_px",
        help=(
            "Length measurement used with canonical metadata. "
            "Default: geodesic_px."
        ),
    )

    parser.add_argument(
        "--length-percentile",
        type=float,
        default=98.5,
        help=(
            "Object-length percentile defining the long-object "
            "threshold for canonical metadata. Default: 98.5."
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
        "--subset",
        choices=("train", "val", "test"),
        default="test",
        help="Dataset subset used for evaluation grouping. Default: test.",
    )

    parser.add_argument(
        "--density-source",
        choices=("manifest", "legacy_metadata"),
        default="manifest",
        help=(
            "Density classification source. "
            "'manifest' uses canonical density labels from the split manifest; "
            "'legacy_metadata' uses the frozen historical density CSV."
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
    parser.add_argument("--noise-group", default=None)
    parser.add_argument("--noise-csv-path", default=None)
    parser.add_argument(
        "--noise-label-column",
        default="noise_class",
    )

    return parser
# -----------------------------------------------------------------------------
# Entry point: choose which behavior to run
# -----------------------------------------------------------------------------
def main(
    mode: str = "diagnostics",
    density_group: str | None = None,
    noise_group: str | None = None,
    *,
    run_name: str = "evaluation_runner_test",
    pred_run: str = "evaluation_runner_test",
    pred_filename: str = "predicted_annotations_poly.json",
    global_outputs_root: str | Path | None = None,
    split_manifest_path: str | Path | None = None,
    subset: str = "test",
    density_source: str = "manifest",
    density_column: str = "density_class",
    legacy_density_csv: str | Path | None = None,
    gt_json_path: str | None = None,
    dataset_root: str | Path | None = None,
    length_source: str = "canonical_metadata",
    length_measurement: str = "geodesic_px",
    length_percentile: float = 98.5,
    legacy_length_csv: str | Path | None = None,
    eval_tag: str = "diagnostics",
    noise_csv_path: str | None = None,
    noise_label_column: str = "noise_class",
) -> None:
    if mode == "density":
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

        run_dsb_by_density(
            selected_density=density_group,
            run_name=run_name,
            pred_run=pred_run,
            pred_filename=pred_filename,
            gt_json_path=gt_json_path,
            eval_tag=eval_tag,
            split_manifest_path=split_manifest_path,
            subset=subset,
            density_source=density_source,
            density_column=density_column,
            legacy_density_csv=legacy_density_csv,
            global_outputs_root=global_outputs_root,
        )
    elif mode == "length":
        if dataset_root is None:
            raise ValueError(
                "dataset_root is required for length evaluation."
            )

        if split_manifest_path is None:
            raise ValueError(
                "split_manifest_path is required for length evaluation."
            )

        if (
            length_source == "legacy_metadata"
            and legacy_length_csv is None
        ):
            raise ValueError(
                "legacy_length_csv is required when "
                "length_source='legacy_metadata'."
            )

        run_dsb_by_length_class(
            run_name=run_name,
            pred_run=pred_run,
            pred_filename=pred_filename,
            gt_json_path=gt_json_path,
            eval_tag=eval_tag,
            dataset_root=dataset_root,
            split_manifest_path=split_manifest_path,
            subset=subset,
            length_source=length_source,
            length_measurement=length_measurement,
            length_percentile=length_percentile,
            legacy_length_csv=legacy_length_csv,
            global_outputs_root=global_outputs_root,
        )
    elif mode == "noise":
        run_dsb_by_noise_class(
            selected_noise=noise_group,
            run_name=run_name,
            pred_run=pred_run,
            pred_filename=pred_filename,
            gt_json_path=gt_json_path,
            eval_tag=eval_tag,
            noise_csv_path=noise_csv_path,
            noise_label_column=noise_label_column,
            global_outputs_root=global_outputs_root,
        )
    elif mode == "diagnostics":
        run_dsb_full_dataset_diagnostics(
            run_name=run_name,
            pred_run=pred_run,
            pred_filename=pred_filename,
            gt_json_path=gt_json_path,
            eval_tag=eval_tag,
            global_outputs_root=global_outputs_root,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    main(
        mode=args.mode,
        density_group=args.density_group,
        noise_group=args.noise_group,
        run_name=args.run_name,
        pred_run=args.pred_run,
        pred_filename=args.pred_filename,
        global_outputs_root=args.global_outputs_root,
        gt_json_path=args.gt_json_path,
        eval_tag=args.eval_tag,
        dataset_root=args.dataset_root,
        length_source=args.length_source,
        length_measurement=args.length_measurement,
        length_percentile=args.length_percentile,
        legacy_length_csv=args.legacy_length_csv,
        noise_csv_path=args.noise_csv_path,
        noise_label_column=args.noise_label_column,
        split_manifest_path=args.split_manifest_path,
        subset=args.subset,
        density_source=args.density_source,
        density_column=args.density_column,
        legacy_density_csv=args.legacy_density_csv,
    )