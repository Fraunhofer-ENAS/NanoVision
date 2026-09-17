from __future__ import annotations

import argparse
from pathlib import Path

from cnt_project.preprocessing.metadata import generate_density_metadata


def _default_dataset_root() -> Path:
    # .../src/cnt_project/preprocessing/runners/generate_density_metadata_runner.py
    return Path(__file__).resolve().parents[4] / "data" / "dataset"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate canonical density metadata from the unsplit dataset.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_default_dataset_root(),
        help="Dataset root containing images/, masks/, metadata/, splits/, and COCO_mask/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <dataset_root>/metadata/density_classified_filenames.csv.",
    )
    parser.add_argument("--density-method", choices=("tertile", "kmeans"), default="kmeans")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-mask-area", type=int, default=1)
    parser.add_argument(
        "--no-comparison-columns",
        action="store_true",
        help="If set, write only canonical columns (drop method-comparison columns).",
    )

    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_csv = (args.output or (dataset_root / "metadata" / "density_classified_filenames.csv")).resolve()

    df = generate_density_metadata(
        dataset_root=dataset_root,
        output_csv=output_csv,
        method=args.density_method,
        random_seed=args.seed,
        min_area_px=args.min_mask_area,
        include_comparison_columns=not args.no_comparison_columns,
    )
    print(f"Saved density metadata ({len(df)} rows): {output_csv}")


if __name__ == "__main__":
    main()
