"""Prepare Cellpose folders for the five stratified splits."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANONICAL_ROOT = (
    PROJECT_ROOT
    / "data"
    / "cnt_segmentation"
)

IMAGE_SOURCE = (
    CANONICAL_ROOT
    / "images"
)

MASK_SOURCE = (
    CANONICAL_ROOT
    / "masks"
)

SPLIT_DIRECTORY = (
    CANONICAL_ROOT
    / "splits"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "cellpose_dataset"
    / "stratified_splits"
)

SEEDS = (
    0,
    1,
    2,
    42,
    123,
)

EXPECTED_COUNTS = {
    "train": 85,
    "val": 15,
    "test": 30,
}

SUBSET_DIRECTORY_NAMES = {
    "train": "train",
    "val": "validation",
    "test": "test",
}


def prepare_split(
    seed: int,
) -> None:
    manifest_path = (
        SPLIT_DIRECTORY
        / f"stratified_seed_{seed}.csv"
    )

    manifest = pd.read_csv(
        manifest_path
    )

    if len(manifest) != 130:
        raise ValueError(
            f"{manifest_path.name} contains "
            f"{len(manifest)} rows instead of 130."
        )

    if manifest["filename"].duplicated().any():
        raise ValueError(
            f"{manifest_path.name} contains "
            "duplicate filenames."
        )

    split_root = (
        OUTPUT_ROOT
        / f"seed_{seed}"
    )

    for subset, expected_count in (
        EXPECTED_COUNTS.items()
    ):
        subset_rows = manifest.loc[
            manifest["subset"] == subset
        ]

        if len(subset_rows) != expected_count:
            raise ValueError(
                f"Seed {seed}, subset {subset}: "
                f"expected {expected_count} images, "
                f"found {len(subset_rows)}."
            )

        output_directory = (
            split_root
            / SUBSET_DIRECTORY_NAMES[subset]
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        for filename in subset_rows[
            "filename"
        ]:
            image_path = (
                IMAGE_SOURCE
                / filename
            )

            mask_path = (
                MASK_SOURCE
                / filename
            )

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Mask not found: {mask_path}"
                )

            image = tifffile.imread(
                image_path
            )

            mask = tifffile.imread(
                mask_path
            )

            if image.shape != (
                256,
                256,
                3,
            ):
                raise ValueError(
                    f"Unexpected image shape for "
                    f"{filename}: {image.shape}"
                )

            if mask.shape != (
                256,
                256,
            ):
                raise ValueError(
                    f"Unexpected mask shape for "
                    f"{filename}: {mask.shape}"
                )

            if not np.issubdtype(
                mask.dtype,
                np.integer,
            ):
                raise TypeError(
                    f"Mask is not integer-labelled: "
                    f"{filename}, dtype={mask.dtype}"
                )

            if mask.max() == 0:
                raise ValueError(
                    f"Mask contains no objects: "
                    f"{filename}"
                )

            output_image_path = (
                output_directory
                / filename
            )

            output_seg_path = (
                output_directory
                / (
                    f"{Path(filename).stem}"
                    "_seg.npy"
                )
            )

            shutil.copy2(
                image_path,
                output_image_path,
            )

            np.save(
                output_seg_path,
                {
                    "masks": mask,
                },
                allow_pickle=True,
            )

        print(
            f"Seed {seed}: "
            f"{subset}={len(subset_rows)}"
        )


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            "Prepared split directory already exists: "
            f"{OUTPUT_ROOT}. Move or remove it before "
            "preparing the splits again."
        )

    for seed in SEEDS:
        prepare_split(
            seed
        )

    print()
    print(
        "Prepared Cellpose splits:",
        OUTPUT_ROOT,
    )


if __name__ == "__main__":
    main()