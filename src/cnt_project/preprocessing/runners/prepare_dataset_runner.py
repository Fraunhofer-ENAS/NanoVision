from __future__ import annotations

import argparse
from pathlib import Path

from cnt_project.preprocessing.dataset import (
    PreparationGenerationParameters,
    prepare_dataset,
)


def _default_dataset_root() -> Path:
    # .../src/cnt_project/preprocessing/runners/prepare_dataset_runner.py
    return Path(__file__).resolve().parents[4] / "data" / "dataset"


def _default_legacy_manual_root() -> Path:
    return Path(__file__).resolve().parents[4] / "data" / "annotations_uniques"


def _parse_shape(raw: str | None) -> tuple[int, int] | None:
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise ValueError("--noise-expected-shape must be formatted as 'H,W', for example: 256,256")
    return int(parts[0]), int(parts[1])


def _parse_subsets(raw: str) -> tuple[str, ...]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("--coco-subsets must include at least one subset.")
    return tuple(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare canonical dataset artifacts from unsplit images/masks via one orchestration entry point."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_default_dataset_root(),
        help="Dataset root containing images/, masks/, metadata/, splits/, and COCO_mask/.",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="default_split",
        help="Split identifier used for splits/<split_name>.csv|yaml and COCO_mask/<split_name>/...",
    )
    parser.add_argument(
        "--policy",
        choices=("validate", "regenerate"),
        default="validate",
        help="validate: keep existing artifacts and generate only missing downstream; regenerate: overwrite explicitly.",
    )

    parser.add_argument("--density-method", choices=("tertile", "kmeans"), default="kmeans")
    parser.add_argument("--density-min-mask-area", type=int, default=1)
    parser.add_argument("--noise-method", choices=("otsu", "kmeans"), default="otsu")
    parser.add_argument("--noise-wavelet", type=str, default="db1")
    parser.add_argument(
        "--noise-expected-shape",
        type=str,
        default=None,
        help="Optional strict image shape as H,W (example: 256,256).",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.85,
        help=(
            "Training fraction of the non-test development pool. "
            "Must sum to 1.0 with --val-fraction. Default: 0.85."
        ),
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help=(
            "Validation fraction of the non-test development pool. "
            "Must sum to 1.0 with --train-fraction. Default: 0.15."
        ),
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=30 / 130,
        help=(
            "Test fraction of the complete dataset. "
            "Default preserves the historical 30-of-130 test size."
        ),
    )
    parser.add_argument("--density-column", type=str, default="density_class_tertile")
    parser.add_argument("--noise-column", type=str, default="noise_class_otsu")

    parser.add_argument(
        "--reproduce-manual-split",
        action="store_true",
        help="Preserve the legacy manual test set exactly and create a seeded "
            "density/noise-stratified train/validation split from the legacy "
            "training pool.",
    )
    parser.add_argument(
        "--manual-split-root",
        type=Path,
        default=_default_legacy_manual_root(),
        help="Root containing the legacy training pool under images/ and the "
            "fixed legacy test set under test/images/.",
    )
    parser.add_argument(
        "--legacy-val-fraction",
        type=float,
        default=0.15,
        help=(
            "Validation fraction taken from the legacy training pool when "
            "--reproduce-manual-split is enabled. The legacy test set remains "
            "fixed. Default: 0.15."
        ),
    )
    parser.add_argument(
        "--allow-stratification-fallback",
        action="store_true",
        help="Allow small density/noise groups to skip strict per-subset representation.",
    )

    parser.add_argument(
        "--single-default-json",
        action="store_true",
        help="Generate/validate only annotations.json instead of dual chain approximation variants.",
    )
    parser.add_argument(
        "--coco-subsets",
        type=str,
        default="train,val,test",
        help="Comma-separated subset order for COCO orchestration (example: train,val,test).",
    )
    parser.add_argument(
        "--coco-missing-subset-policy",
        choices=("skip", "reject"),
        default="reject",
        help="Behavior when requested subset is empty in split manifest.",
    )

    parser.add_argument(
        "--no-density-comparison-columns",
        action="store_true",
        help="When generating density metadata, write canonical columns only.",
    )
    parser.add_argument(
        "--no-noise-comparison-columns",
        action="store_true",
        help="When generating noise metadata, write canonical columns only.",
    )

    args = parser.parse_args()

    parameters = PreparationGenerationParameters(
        density_method=args.density_method,
        density_min_area_px=args.density_min_mask_area,
        density_include_comparison_columns=not args.no_density_comparison_columns,
        noise_method=args.noise_method,
        noise_wavelet=args.noise_wavelet,
        noise_expected_shape=_parse_shape(args.noise_expected_shape),
        noise_include_comparison_columns=not args.no_noise_comparison_columns,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        allow_stratification_fallback=args.allow_stratification_fallback,
        reproduce_manual_split=args.reproduce_manual_split,
        manual_split_root=args.manual_split_root,
        legacy_val_fraction=args.legacy_val_fraction,
        split_density_column=args.density_column,
        split_noise_column=args.noise_column,
        generate_coco_variants=not args.single_default_json,
        coco_missing_subset_policy=args.coco_missing_subset_policy,
        coco_subsets=_parse_subsets(args.coco_subsets),
    )

    result = prepare_dataset(
        dataset_root=args.dataset_root,
        split_name=args.split_name,
        policy=args.policy,
        generation_parameters=parameters,
    )

    print("Prepared dataset artifacts:")
    print(f"  dataset_root: {result.dataset_root}")
    print(f"  split_name: {result.split_name}")
    print(f"  policy: {result.policy}")
    print(f"  samples: {len(result.sample_filenames)}")
    print(f"  density_metadata_csv: {result.density_metadata_csv}")
    print(f"  noise_metadata_csv: {result.noise_metadata_csv}")
    print(f"  split_manifest_csv: {result.split_manifest_csv}")
    print(f"  split_metadata_yaml: {result.split_metadata_yaml}")
    print("  coco_json_paths:")
    for subset, variants in sorted(result.coco_json_paths.items()):
        for variant_name, output_path in sorted(variants.items()):
            print(f"    {subset}.{variant_name}: {output_path}")


if __name__ == "__main__":
    main()
