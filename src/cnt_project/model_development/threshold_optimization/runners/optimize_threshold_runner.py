from __future__ import annotations

import argparse
from pathlib import Path

from cnt_project.io.paths import ProjectPaths
from cnt_project.io.run_manager import (
    create_run,
    make_run_id,
)
from cnt_project.model_development.inference.runners.infer_runner import (
    _parse_optional_bool,
)
from cnt_project.model_development.model_compat import (
    check_model_compatibility,
)
from cnt_project.model_development.stardist_patched.stardist_model_configuration import (
    MyStarDist2D,
)
from cnt_project.model_development.threshold_optimization.optimizer import (
    optimize_probability_threshold,
    save_probability_threshold,
)
from cnt_project.preprocessing.dataset.dataset_loader import (
    CNTDataset,
)

def _parse_threshold(value: str) -> float:
    threshold = float(value)

    if not 0.0 < threshold < 1.0:
        raise argparse.ArgumentTypeError(
            "Probability thresholds must be strictly between 0 and 1."
        )

    return threshold

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize the CNT StarDist probability threshold using the "
            "canonical inference/postprocessing pipeline."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Canonical CNT dataset root.",
    )

    parser.add_argument(
        "--split-manifest-path",
        type=Path,
        required=True,
        help="Canonical split manifest CSV.",
    )

    parser.add_argument(
        "--subset",
        choices=("train", "val", "test"),
        default="val",
        help=(
            "Dataset subset used for threshold optimization. "
            "Default: val."
        ),
    )

    parser.add_argument(
        "--gt-json-path",
        type=Path,
        required=True,
        help=(
            "COCO ground-truth JSON corresponding to the selected subset."
        ),
    )

    parser.add_argument(
        "--model-name",
        required=True,
        help="Model folder name under --model-basedir.",
    )

    parser.add_argument(
        "--model-basedir",
        type=Path,
        default=None,
        help=(
            "Directory containing model folders. "
            "Defaults to <project_root>/models."
        ),
    )

    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=_parse_threshold,
        required=True,
        help=(
            "Probability thresholds to evaluate. "
            "Example: --thresholds 0.05 0.10 0.15 0.20"
        ),
    )

    parser.add_argument(
        "--objective",
        type=str,
        default="dsb_map",
        help=(
            "Objective used to recommend the best threshold. "
            "Supported examples: dsb_map, mean_dice, "
            "dsb_ap@0.50, object_f1@0.50, "
            "cldice_object_f1@0.30, "
            "coco_ap, coco_ap50, coco_ap75."
        ),
    )

    parser.add_argument(
        "--grayscale",
        type=_parse_optional_bool,
        default=None,
        help=(
            "Use true, false, or omit for automatic model-compatible "
            "selection."
        ),
    )

    parser.add_argument(
        "--apply-smoothing",
        type=_parse_optional_bool,
        default=None,
        help=(
            "Optionally override CNT polygon smoothing. "
            "When omitted, model/postprocessing default behavior is used."
        ),
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Optional cap on the selected dataset subset."
        ),
    )
    parser.add_argument(
        "--save-best-threshold",
        action="store_true",
        help=(
            "Persist the automatically selected probability threshold to the "
            "model's thresholds.json. The existing/default NMS threshold is "
            "preserved. Without this flag, optimization is report-only."
        ),
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Optional explicit run identifier. "
            "When omitted, a timestamped run ID is generated."
        ),
    )

    return parser

def _resolve_grayscale(
    *,
    requested_grayscale: bool | None,
    model: MyStarDist2D,
) -> bool:
    model_n_channels = int(
        model.config.n_channel_in
    )

    if requested_grayscale is not None:
        return bool(requested_grayscale)

    return model_n_channels == 1


def main() -> None:
    args = build_arg_parser().parse_args()

    paths = ProjectPaths.from_here(__file__)
    paths.ensure_outputs()

    model_basedir = (
        args.model_basedir
        if args.model_basedir is not None
        else paths.project_root / "models"
    )

    model = MyStarDist2D(
        None,
        name=args.model_name,
        basedir=str(model_basedir),
    )

    resolved_grayscale = _resolve_grayscale(
        requested_grayscale=args.grayscale,
        model=model,
    )

    print(
        f"Resolved grayscale: {resolved_grayscale}"
    )

    dataset = CNTDataset(
        dataset_root=args.dataset_root,
        split_manifest_path=args.split_manifest_path,
        subset=args.subset,
        grayscale=resolved_grayscale,
        require_masks=True,
        max_samples=args.max_samples,
    )

    samples = dataset.get_test_data()

    if not samples:
        raise RuntimeError(
            f"No samples found for subset {args.subset!r}."
        )

    dataset_n_channels = int(
        samples[0].n_channel
    )

    model_n_channels = int(
        model.config.n_channel_in
    )

    check_model_compatibility(
        grayscale=resolved_grayscale,
        model_name=args.model_name,
        dataset_n_channels=dataset_n_channels,
        model_n_channels=model_n_channels,
    )

    images = [
        sample.X
        for sample in samples
    ]

    filenames = [
        sample.X_fn
        for sample in samples
    ]

    if args.run_name is None:
        run_id = make_run_id(
            args.model_name,
            "prob-threshold-optimization",
            args.subset,
        )
    else:
        run_id = args.run_name

    run_paths = create_run(
        project_root=paths.project_root,
        run_id=run_id,
    )

    predictions_dir = (
        run_paths.inference_dir
        / "threshold_optimization"
    )

    results_dir = (
        run_paths.eval_dir
        / "threshold_optimization"
    )

    print(
        f"Threshold optimization run: "
        f"{run_paths.run_id}"
    )

    print(
        f"Subset: {args.subset}"
    )

    print(
        f"Samples: {len(samples)}"
    )

    print(
        f"Thresholds: {args.thresholds}"
    )

    print(
        f"Objective: {args.objective}"
    )

    result = optimize_probability_threshold(
        thresholds=args.thresholds,
        model=model,
        images=images,
        filenames=filenames,
        gt_json_path=args.gt_json_path,
        predictions_dir=predictions_dir,
        results_dir=results_dir,
        objective=args.objective,
        apply_smoothing=args.apply_smoothing,
    )

    print("")
    print("Recommended probability threshold")
    print("---------------------------------")
    print(
        f"Threshold: {result.best_threshold:.6f}"
    )
    print(
        f"Objective: {result.objective}"
    )
    print(
        f"Metric:    {result.best_metric:.6f}"
    )
    print(
        f"CSV:       {result.results_csv_path}"
    )
    if args.save_best_threshold:
        thresholds_path = save_probability_threshold(
            model=model,
            model_dir=(
                Path(model_basedir)
                / args.model_name
            ),
            probability_threshold=result.best_threshold,
        )

        print(
            "Saved optimized probability threshold to: "
            f"{thresholds_path}"
        )
    else:
        print(
            "Model thresholds were not modified. "
            "Use --save-best-threshold to persist the recommendation."
        )


if __name__ == "__main__":
    main()