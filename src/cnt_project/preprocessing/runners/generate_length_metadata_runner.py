from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd

from cnt_project.preprocessing.metadata.length import (
    aggregate_object_lengths_by_image,
    generate_object_length_metadata,
)
from cnt_project.preprocessing.metadata.schemas import (
    FILENAME_COL,
)


DEFAULT_FIELD_SIZE_UM = 5.0
DEFAULT_IMAGE_SIZE_PX = 256


def _default_dataset_root() -> Path:
    """
    Return the canonical unsplit dataset root.
    """
    return (
        Path(__file__).resolve().parents[4]
        / "data"
        / "cnt_segmentation"
    )


def _parse_positive_float(raw: str) -> float:
    """
    Parse a strictly positive floating-point command-line value.
    """
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a floating-point value, got {raw!r}."
        ) from exc

    if value <= 0:
        raise argparse.ArgumentTypeError(
            f"Expected a value greater than zero, got {value}."
        )

    return value


def _parse_positive_int(raw: str) -> int:
    """
    Parse a strictly positive integer command-line value.
    """
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected an integer value, got {raw!r}."
        ) from exc

    if value <= 0:
        raise argparse.ArgumentTypeError(
            f"Expected a value greater than zero, got {value}."
        )

    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate canonical object-level and image-level CNT length "
            "metadata from instance-indexed TIFF masks."
        ),
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_default_dataset_root(),
        help=(
            "Canonical dataset root containing images/, masks/, and metadata/. "
            "Defaults to data/cnt_segmentation."
        ),
    )

    parser.add_argument(
        "--output-level",
        choices=("object", "image", "both"),
        default="both",
        help=(
            "Metadata level to generate. "
            "'object' writes one row per CNT instance; "
            "'image' writes one aggregated row per image; "
            "'both' writes both files. Default: both."
        ),
    )

    parser.add_argument(
        "--object-output-csv",
        type=Path,
        default=None,
        help=(
            "Optional object-level CSV output path. "
            "Defaults to <dataset-root>/metadata/object_lengths.csv."
        ),
    )

    parser.add_argument(
        "--image-output-csv",
        type=Path,
        default=None,
        help=(
            "Optional image-level CSV output path. "
            "Defaults to <dataset-root>/metadata/image_lengths.csv."
        ),
    )

    parser.add_argument(
        "--field-size-um",
        type=_parse_positive_float,
        default=DEFAULT_FIELD_SIZE_UM,
        help=(
            "Physical image width in micrometres. "
            "Used with --image-size-px to calculate micrometres per pixel. "
            "Default: 5.0."
        ),
    )

    parser.add_argument(
        "--image-size-px",
        type=_parse_positive_int,
        default=DEFAULT_IMAGE_SIZE_PX,
        help=(
            "Image width in pixels used for physical-unit conversion. "
            "Default: 256."
        ),
    )

    parser.add_argument(
        "--microns-per-pixel",
        type=_parse_positive_float,
        default=None,
        help=(
            "Optional explicit physical pixel size. When supplied, this "
            "overrides --field-size-um / --image-size-px."
        ),
    )

    parser.add_argument(
        "--min-area-px",
        type=_parse_positive_int,
        default=1,
        help=(
            "Minimum total instance area required for inclusion. Default: 1."
        ),
    )

    parser.add_argument(
        "--geodesic-alpha",
        type=float,
        default=2.0,
        help=(
            "Boundary-penalty strength for the weighted geodesic algorithm. "
            "Must be greater than or equal to zero. Default: 2.0."
        ),
    )

    parser.add_argument(
        "--geodesic-eps",
        type=_parse_positive_float,
        default=1e-3,
        help=(
            "Positive numerical stabilizer for the weighted geodesic "
            "algorithm. Default: 0.001."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing output CSV files to be replaced.",
    )

    return parser


def _resolve_output_path(
    explicit_path: Path | None,
    *,
    default_path: Path,
) -> Path:
    """
    Resolve an optional output path against the current working directory.
    """
    if explicit_path is None:
        return default_path.resolve()

    return explicit_path.resolve()


def _validate_output_path(
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    """
    Prevent accidental replacement of existing metadata files.
    """
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            "Use --overwrite to replace it."
        )


def _save_image_metadata(
    object_lengths_df: pd.DataFrame,
    output_csv: Path,
) -> pd.DataFrame:
    """
    Aggregate object measurements and save one row per image.
    """
    image_lengths_df = aggregate_object_lengths_by_image(
        object_lengths_df
    )

    image_lengths_df = (
        image_lengths_df
        .sort_values(FILENAME_COL)
        .reset_index(drop=True)
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_lengths_df.to_csv(
        output_csv,
        index=False,
    )

    return image_lengths_df


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.geodesic_alpha < 0:
        raise ValueError(
            "geodesic_alpha must be greater than or equal to zero, "
            f"got {args.geodesic_alpha}."
        )

    dataset_root = args.dataset_root.resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {dataset_root}"
        )

    metadata_root = dataset_root / "metadata"

    object_output_csv = _resolve_output_path(
        args.object_output_csv,
        default_path=metadata_root / "object_lengths.csv",
    )

    image_output_csv = _resolve_output_path(
        args.image_output_csv,
        default_path=metadata_root / "image_lengths.csv",
    )

    selected_outputs = {
        "object",
        "image",
    } if args.output_level == "both" else {
        args.output_level
    }

    if "object" in selected_outputs:
        _validate_output_path(
            object_output_csv,
            overwrite=args.overwrite,
        )

    if "image" in selected_outputs:
        _validate_output_path(
            image_output_csv,
            overwrite=args.overwrite,
        )

    microns_per_pixel = (
        float(args.microns_per_pixel)
        if args.microns_per_pixel is not None
        else float(args.field_size_um)
        / int(args.image_size_px)
    )

    print("")
    print("Length metadata configuration")
    print("-----------------------------")
    print(f"Dataset root:          {dataset_root}")
    print(f"Output level:          {args.output_level}")
    print(f"Micrometres per pixel: {microns_per_pixel}")
    print(f"Minimum area:          {args.min_area_px} px")
    print(f"Geodesic alpha:        {args.geodesic_alpha}")
    print(f"Geodesic epsilon:      {args.geodesic_eps}")

    if "object" in selected_outputs:
        print(f"Object output:         {object_output_csv}")

    if "image" in selected_outputs:
        print(f"Image output:          {image_output_csv}")

    print("")

    started_at = perf_counter()

    # Object-level measurements are always generated in memory because
    # image-level metadata depends on them. They are written only when the
    # selected output level includes "object".
    object_lengths_df = generate_object_length_metadata(
        dataset_root=dataset_root,
        output_csv=(
            object_output_csv
            if "object" in selected_outputs
            else None
        ),
        microns_per_pixel=microns_per_pixel,
        min_area_px=args.min_area_px,
        geodesic_alpha=args.geodesic_alpha,
        geodesic_eps=args.geodesic_eps,
    )

    image_lengths_df: pd.DataFrame | None = None

    if "image" in selected_outputs:
        image_lengths_df = _save_image_metadata(
            object_lengths_df,
            image_output_csv,
        )

    elapsed_seconds = perf_counter() - started_at

    print("")
    print("Length metadata generation completed")
    print("------------------------------------")
    print(f"Object instances measured: {len(object_lengths_df)}")
    print(
        "Images represented:       "
        f"{object_lengths_df[FILENAME_COL].nunique()}"
    )

    if "object" in selected_outputs:
        print(f"Object CSV:              {object_output_csv}")

    if image_lengths_df is not None:
        print(f"Image rows generated:    {len(image_lengths_df)}")
        print(f"Image CSV:               {image_output_csv}")

    print(f"Elapsed time:            {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    main()