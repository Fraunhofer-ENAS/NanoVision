from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from cellpose import io, models
from skimage.segmentation import find_boundaries


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_ROOT = (
    PROJECT_ROOT
    / "data"
    / "cellpose_dataset"
    / "test"
)

TEST_IMAGES_DIR = TEST_ROOT / "images"
TEST_MASKS_DIR = TEST_ROOT / "masks"

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "cellpose_dataset"
    / "training_runs"
    / "cellpose_cyto_original_legacy_seed42_150epochs_patience40"
    / "models"
    / (
        "cellpose_cyto_original_legacy_seed42_"
        "150epochs_patience40_best_validation"
    )
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "cellpose_dataset"
    / "inference_runs"
    / (
        "cellpose_cyto_original_legacy_seed42_"
        "best_validation_optimized"
    )
)

PREDICTED_MASKS_DIR = (
    OUTPUT_DIR
    / "predicted_masks"
)

OVERLAYS_DIR = (
    OUTPUT_DIR
    / "overlays"
)

FLOW_THRESHOLD = 0.8
CELLPROB_THRESHOLD = -1.0
MIN_SIZE = 15
BATCH_SIZE = 1
BSIZE = 256


def count_objects(
    mask: np.ndarray,
) -> int:
    object_ids = np.unique(mask)
    return int(
        np.count_nonzero(object_ids != 0)
    )


def prepare_display_image(
    image: np.ndarray,
) -> np.ndarray:
    image = np.asarray(
        image,
        dtype=np.float32,
    )

    if image.ndim == 2:
        image = np.repeat(
            image[..., None],
            3,
            axis=-1,
        )

    if image.ndim == 3 and image.shape[-1] > 3:
        image = image[..., :3]

    finite = image[
        np.isfinite(image)
    ]

    if finite.size == 0:
        return np.zeros(
            (*image.shape[:2], 3),
            dtype=np.float32,
        )

    low, high = np.percentile(
        finite,
        [1.0, 99.0],
    )

    if high > low:
        image = (
            image - low
        ) / (
            high - low
        )
    else:
        image = np.zeros_like(
            image
        )

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    image[
        ~np.isfinite(image)
    ] = 0.0

    return image


def create_comparison_overlay(
    image: np.ndarray,
    gt_mask: np.ndarray,
    predicted_mask: np.ndarray,
) -> np.ndarray:
    overlay = prepare_display_image(
        image
    ).copy()

    gt_boundary = find_boundaries(
        gt_mask,
        mode="outer",
    )

    predicted_boundary = find_boundaries(
        predicted_mask,
        mode="outer",
    )

    common = (
        gt_boundary
        & predicted_boundary
    )

    gt_only = (
        gt_boundary
        & ~predicted_boundary
    )

    prediction_only = (
        predicted_boundary
        & ~gt_boundary
    )

    # Green: GT boundary
    overlay[gt_only] = [
        0.0,
        1.0,
        0.0,
    ]

    # Red: predicted boundary
    overlay[prediction_only] = [
        1.0,
        0.0,
        0.0,
    ]

    # Yellow: coincident boundaries
    overlay[common] = [
        1.0,
        1.0,
        0.0,
    ]

    return overlay


def save_visualization(
    *,
    image: np.ndarray,
    gt_mask: np.ndarray,
    predicted_mask: np.ndarray,
    image_name: str,
    output_path: Path,
) -> None:
    display_image = prepare_display_image(
        image
    )

    comparison_overlay = (
        create_comparison_overlay(
            image,
            gt_mask,
            predicted_mask,
        )
    )

    gt_count = count_objects(
        gt_mask
    )

    predicted_count = count_objects(
        predicted_mask
    )

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4),
    )

    axes[0].imshow(
        display_image
    )
    axes[0].set_title(
        "AFM image"
    )

    axes[1].imshow(
        gt_mask,
        cmap="nipy_spectral",
        interpolation="nearest",
    )
    axes[1].set_title(
        f"Ground truth\n{gt_count} objects"
    )

    axes[2].imshow(
        predicted_mask,
        cmap="nipy_spectral",
        interpolation="nearest",
    )
    axes[2].set_title(
        f"Prediction\n{predicted_count} objects"
    )

    axes[3].imshow(
        comparison_overlay
    )
    axes[3].set_title(
        "Boundaries\nGT: green, prediction: red"
    )

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        image_name
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            MODEL_PATH
        )

    image_paths = sorted(
        path
        for path in TEST_IMAGES_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in {".tif", ".tiff"}
        )
    )

    if not image_paths:
        raise RuntimeError(
            "No test images found."
        )

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            "Output directory already exists: "
            f"{OUTPUT_DIR}"
        )

    PREDICTED_MASKS_DIR.mkdir(
        parents=True,
    )

    OVERLAYS_DIR.mkdir(
        parents=True,
    )

    print(
        f"Model: {MODEL_PATH}"
    )
    print(
        f"Test images: {len(image_paths)}"
    )
    print(
        f"Output: {OUTPUT_DIR}"
    )

    model = models.CellposeModel(
        gpu=True,
        pretrained_model=str(
            MODEL_PATH
        ),
    )

    rows = []

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        mask_path = (
            TEST_MASKS_DIR
            / image_path.name
        )

        if not mask_path.is_file():
            raise FileNotFoundError(
                mask_path
            )

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        image = io.imread(
            str(image_path)
        )

        gt_mask = np.squeeze(
            tifffile.imread(
                mask_path
            )
        )

        predicted_mask, flows, styles = (
            model.eval(
                image,
                batch_size=BATCH_SIZE,
                channels=[0, 0],
                channel_axis=-1,
                normalize=True,
                diameter=None,
                flow_threshold=(
                    FLOW_THRESHOLD
                ),
                cellprob_threshold=(
                    CELLPROB_THRESHOLD
                ),
                min_size=MIN_SIZE,
                bsize=BSIZE,
            )
        )

        predicted_mask = np.asarray(
            predicted_mask
        )

        if predicted_mask.shape != gt_mask.shape:
            raise ValueError(
                "Prediction and GT shapes differ: "
                f"{predicted_mask.shape} and "
                f"{gt_mask.shape}"
            )

        prediction_path = (
            PREDICTED_MASKS_DIR
            / (
                f"{image_path.stem}"
                "_predicted_mask.tif"
            )
        )

        tifffile.imwrite(
            prediction_path,
            predicted_mask.astype(
                np.uint32
            ),
        )

        overlay_path = (
            OVERLAYS_DIR
            / (
                f"{image_path.stem}"
                "_comparison.png"
            )
        )

        save_visualization(
            image=image,
            gt_mask=gt_mask,
            predicted_mask=predicted_mask,
            image_name=image_path.name,
            output_path=overlay_path,
        )

        gt_count = count_objects(
            gt_mask
        )

        predicted_count = count_objects(
            predicted_mask
        )

        rows.append(
            {
                "image": image_path.name,
                "gt_object_count": gt_count,
                "predicted_object_count": (
                    predicted_count
                ),
                "object_count_difference": (
                    predicted_count
                    - gt_count
                ),
                "predicted_mask": str(
                    prediction_path
                ),
                "overlay": str(
                    overlay_path
                ),
            }
        )

        print(
            f"    GT objects: {gt_count}; "
            f"predicted: {predicted_count}"
        )

    manifest_path = (
        OUTPUT_DIR
        / "inference_manifest.csv"
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    summary_path = (
        OUTPUT_DIR
        / "inference_summary.txt"
    )

    summary_path.write_text(
        "\n".join(
            [
                f"model={MODEL_PATH}",
                (
                    f"test_images="
                    f"{len(image_paths)}"
                ),
                (
                    "flow_threshold="
                    f"{FLOW_THRESHOLD}"
                ),
                (
                    "cellprob_threshold="
                    f"{CELLPROB_THRESHOLD}"
                ),
                f"min_size={MIN_SIZE}",
                "channels=[0, 0]",
                "channel_axis=-1",
                "normalize=True",
                "diameter=None",
                (
                    "total_gt_objects="
                    f"{sum(row['gt_object_count'] for row in rows)}"
                ),
                (
                    "total_predicted_objects="
                    f"{sum(row['predicted_object_count'] for row in rows)}"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Inference completed.")
    print(
        f"Manifest: {manifest_path}"
    )
    print(
        f"Summary: {summary_path}"
    )
    print(
        f"Predicted masks: "
        f"{PREDICTED_MASKS_DIR}"
    )
    print(
        f"Overlays: {OVERLAYS_DIR}"
    )


if __name__ == "__main__":
    main()