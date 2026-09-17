from __future__ import annotations

import argparse
from pathlib import Path

from cnt_project.preprocessing.dataset_splitting.generate_split import generate_split_manifest


def _default_dataset_root() -> Path:
    # .../src/cnt_project/preprocessing/runners/generate_dataset_split_runner.py
    return Path(__file__).resolve().parents[4] / "data" / "dataset"


def _default_legacy_manual_root() -> Path:
    return Path(__file__).resolve().parents[4] / "data" / "annotations_uniques"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a train/val/test split manifest from full-dataset metadata.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_default_dataset_root(),
        help="Dataset root containing images/, masks/, metadata/, splits/, and COCO_mask/.",
    )
    parser.add_argument("--split-name", type=str, default="default_split")

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
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--density-metadata-path",
        type=Path,
        default=None,
        help="Defaults to <dataset_root>/metadata/density_classified_filenames.csv",
    )
    parser.add_argument(
        "--noise-metadata-path",
        type=Path,
        default=None,
        help="Defaults to <dataset_root>/metadata/noise_classification.csv",
    )

    parser.add_argument("--density-column", type=str, default="density_class_tertile")
    parser.add_argument("--noise-column", type=str, default="noise_class_otsu")

    legacy_mode = parser.add_mutually_exclusive_group()

    legacy_mode.add_argument(
        "--reproduce-manual-split",
        action="store_true",
        help=(
            "Preserve the legacy manual test set exactly and create a "
            "seeded density/noise-stratified train/validation split from "
            "the legacy training pool."
        ),
    )

    legacy_mode.add_argument(
        "--reproduce-legacy-dataloader-split",
        action="store_true",
        help=(
            "Reproduce the historical Modified_Stardist DataLoader split "
            "exactly: preserve the legacy test set and split the historical "
            "development pool using RandomState(seed) without stratification."
        ),
    )
    parser.add_argument(
        "--manual-split-root",
        type=Path,
        default=_default_legacy_manual_root(),
        help="Root containing the legacy training pool under images/ "
            "and the fixed legacy test set under test/images/.",
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
        help=(
            "Allow small combined groups to be allocated without representation in all subsets. "
            "By default, exact stratification is required and tiny groups raise an error."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split CSV/YAML if they already exist.",
    )

    args = parser.parse_args()

    result = generate_split_manifest(
        dataset_root=args.dataset_root,
        split_name=args.split_name,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        density_metadata_path=args.density_metadata_path,
        noise_metadata_path=args.noise_metadata_path,
        density_column=args.density_column,
        noise_column=args.noise_column,
        allow_stratification_fallback=args.allow_stratification_fallback,
        overwrite=args.overwrite,
        reproduce_manual_split=args.reproduce_manual_split,
        reproduce_legacy_dataloader_split=args.reproduce_legacy_dataloader_split,
        manual_split_root=args.manual_split_root,
        legacy_val_fraction=args.legacy_val_fraction,
    )

    print(f"Saved split manifest: {result.paths.manifest_csv}")
    print(f"Saved split metadata: {result.paths.metadata_yaml}")
    print("Subset counts:")
    print(result.split_df["subset"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
