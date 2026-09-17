from __future__ import annotations

import argparse
from pathlib import Path

from cnt_project.coco.generate import generate_coco_for_split, generate_coco_for_subset


def _default_dataset_root() -> Path:
    return Path(__file__).resolve().parents[4] / "data" / "dataset"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate split-aware COCO JSON files from instance-indexed TIFF masks.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_default_dataset_root(),
        help="Dataset root containing images/, masks/, metadata/, splits/, and COCO_mask/.",
    )
    parser.add_argument(
        "--split-manifest-path",
        type=Path,
        required=True,
        help="Path to the split manifest CSV.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="If provided, generate COCO only for this subset. Otherwise generate for all subsets present.",
    )
    parser.add_argument(
        "--missing-subset-policy",
        choices=("skip", "reject"),
        default="reject",
        help="Policy for expected train/val/test subsets when generating all subset COCO files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing annotations.json outputs if present.",
    )
    parser.add_argument(
        "--single-default-json",
        action="store_true",
        help="Generate only annotations.json instead of chain_approx_simple and chain_approx_none variants.",
    )

    args = parser.parse_args()

    if args.subset:
        result = generate_coco_for_subset(
            dataset_root=args.dataset_root,
            split_manifest_path=args.split_manifest_path,
            subset=args.subset,
            generate_variants=not args.single_default_json,
            overwrite=args.overwrite,
        )
        for variant_name, output_path in result.output_json_paths.items():
            print(f"Saved COCO for subset '{result.subset}' ({variant_name}): {output_path}")
        print(f"Converted masks: {len(result.filenames)}")
        return

    outputs = generate_coco_for_split(
        dataset_root=args.dataset_root,
        split_manifest_path=args.split_manifest_path,
        missing_subset_policy=args.missing_subset_policy,
        generate_variants=not args.single_default_json,
        overwrite=args.overwrite,
    )
    for subset, variant_paths in outputs.items():
        for variant_name, output_path in variant_paths.items():
            print(f"Saved COCO for subset '{subset}' ({variant_name}): {output_path}")


if __name__ == "__main__":
    main()
