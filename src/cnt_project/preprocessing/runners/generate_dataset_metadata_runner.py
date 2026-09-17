from __future__ import annotations

import argparse
from pathlib import Path

from cnt_project.preprocessing.metadata import generate_density_metadata, generate_noise_metadata


def _default_dataset_root() -> Path:
    # .../src/cnt_project/preprocessing/runners/generate_dataset_metadata_runner.py -> repo root is parents[4]
    return Path(__file__).resolve().parents[4] / "data" / "dataset"


def _parse_shape(raw: str | None) -> tuple[int, int] | None:
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ValueError("--expected-shape must be formatted as 'H,W', for example: 256,256")
    return int(parts[0]), int(parts[1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility runner for metadata generation. "
            "Prefer generate_density_metadata_runner.py or generate_noise_metadata_runner.py."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_default_dataset_root(),
        help="Dataset root containing images/, masks/, metadata/, splits/, and COCO_mask/.",
    )

    parser.add_argument(
        "--density-output",
        type=Path,
        default=None,
        help="Output CSV for density metadata. Defaults to <dataset_root>/metadata/density_classified_filenames.csv.",
    )
    parser.add_argument(
        "--noise-output",
        type=Path,
        default=None,
        help="Output CSV for noise metadata. Defaults to <dataset_root>/metadata/noise_classification.csv.",
    )

    parser.add_argument(
        "--mode",
        choices=("density", "noise", "both"),
        default="both",
        help="Select which metadata to generate.",
    )

    parser.add_argument("--density-method", choices=("tertile", "kmeans"), default="kmeans")
    parser.add_argument("--noise-method", choices=("otsu", "kmeans"), default="otsu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wavelet", type=str, default="db1")
    parser.add_argument("--min-mask-area", type=int, default=1)
    parser.add_argument(
        "--expected-shape",
        type=str,
        default=None,
        help="Optional strict image shape as H,W (example: 256,256).",
    )
    parser.add_argument(
        "--no-comparison-columns",
        action="store_true",
        help="If set, write only canonical columns (drop method-comparison columns).",
    )

    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    density_output = (args.density_output or (dataset_root / "metadata" / "density_classified_filenames.csv")).resolve()
    noise_output = (args.noise_output or (dataset_root / "metadata" / "noise_classification.csv")).resolve()

    include_comparison_columns = not args.no_comparison_columns
    expected_shape = _parse_shape(args.expected_shape)

    if args.mode in {"density", "both"}:
        density_df = generate_density_metadata(
            dataset_root=dataset_root,
            output_csv=density_output,
            method=args.density_method,
            random_seed=args.seed,
            min_area_px=args.min_mask_area,
            include_comparison_columns=include_comparison_columns,
        )
        print(f"Saved density metadata ({len(density_df)} rows): {density_output}")

    if args.mode in {"noise", "both"}:
        noise_df = generate_noise_metadata(
            dataset_root=dataset_root,
            output_csv=noise_output,
            method=args.noise_method,
            random_seed=args.seed,
            wavelet=args.wavelet,
            expected_shape=expected_shape,
            include_comparison_columns=include_comparison_columns,
        )
        print(f"Saved noise metadata ({len(noise_df)} rows): {noise_output}")


if __name__ == "__main__":
    main()
