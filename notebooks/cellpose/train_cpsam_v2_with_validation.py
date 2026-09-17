from __future__ import annotations

import argparse
import random
import time
import logging
import sys
from pathlib import Path

import scienceplots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch

from cellpose import models, train


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "cellpose_dataset"
)

OUTPUT_ROOT = (
    DATASET_ROOT
    / "training_runs"
)

EXPECTED_TRAIN_IMAGES = 85
EXPECTED_VALIDATION_IMAGES = 15

BATCH_SIZE = 1
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune Cellpose cpsam_v2 with separate "
            "training and validation sets."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        choices=(
            0,
            1,
            2,
            42,
            123,
        ),
        help=(
            "Dataset split and training random seed."
        ),
    )

    return parser.parse_args()


def find_tif_images(
    folder: Path,
) -> list[Path]:
    """
    Find original input TIFF images while ignoring files generated
    by Cellpose.
    """
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}"
        )

    excluded_suffixes = (
        "_flows",
        "_masks",
        "_outlines",
    )

    return sorted(
        path
        for path in folder.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in {".tif", ".tiff"}
            and not path.stem.lower().endswith(
                excluded_suffixes
            )
        )
    )


def load_cellpose_folder(
    folder: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    image_paths = find_tif_images(folder)

    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    filenames: list[str] = []

    for image_path in image_paths:
        seg_path = (
            image_path.parent
            / f"{image_path.stem}_seg.npy"
        )

        if not seg_path.is_file():
            raise FileNotFoundError(
                                              f"Missing Cellpose annotation: {seg_path}"
            )

        image = np.asarray(
            tifffile.imread(image_path)
        )

        seg_data = np.load(
            seg_path,
            allow_pickle=True,
        ).item()

        if "masks" not in seg_data:
            raise KeyError(
                f"'masks' not found in {seg_path}"
            )

        mask = np.asarray(
            seg_data["masks"]
        )

        if image.shape != (256, 256, 3):
            raise ValueError(
                f"Unexpected image shape for {image_path}: "
                f"{image.shape}"
            )

        if mask.shape != image.shape[:2]:
            raise ValueError(
                f"Image/mask shape mismatch for {image_path}: "
                f"image={image.shape}, mask={mask.shape}"
            )

        if not np.issubdtype(
            mask.dtype,
            np.integer,
        ):
            raise TypeError(
                f"Mask is not integer-labelled: "
                f"{seg_path}, dtype={mask.dtype}"
            )

        if mask.max() == 0:
            raise ValueError(
                f"Mask contains no annotated objects: {seg_path}"
            )

        images.append(image)
        masks.append(mask)
        filenames.append(str(image_path))

    return images, masks, filenames


def save_loss_history(
    *,
    train_losses: np.ndarray,
    validation_losses: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Save the complete loss history and create a SciencePlots figure.
    """
    epochs = np.arange(
        1,
        len(train_losses) + 1,
    )

    history = pd.DataFrame(
        {
            "epoch": epochs,
            "train_loss": train_losses,
            "validation_loss": validation_losses,
        }
    )

    csv_path = (
        output_dir
        / "loss_history.csv"
    )

    history.to_csv(
        csv_path,
        index=False,
    )

    npz_path = (
        output_dir
        / "loss_history.npz"
    )

    np.savez_compressed(
        npz_path,
        epoch=epochs,
        train_loss=train_losses,
        validation_loss=validation_losses,
    )

    png_path = (
        output_dir
        / "loss_history.png"
    )

    pdf_path = (
        output_dir
        / "loss_history.pdf"
    )

    # "no-latex" avoids requiring a separate LaTeX installation.
    with plt.style.context(
        [
            "science",
            "no-latex",
        ]
    ):
        fig, ax = plt.subplots(
            figsize=(7.0, 4.5)
        )

        ax.plot(
            epochs,
            train_losses,
            label="Training loss",
            linewidth=1.8,
        )

        ax.plot(
            epochs,
            validation_losses,
            label="Validation loss",
            linewidth=1.8,
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cellpose loss")
        ax.set_title(
            "Cellpose-SAM v2 fine-tuning"
        )

        ax.legend(
            frameon=True
        )

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
        )

        fig.savefig(
            pdf_path,
            bbox_inches="tight",
        )

        plt.close(fig)

    print(f"Loss CSV:   {csv_path}")
    print(f"Loss NumPy: {npz_path}")
    print(f"Loss PNG:   {png_path}")
    print(f"Loss PDF:   {pdf_path}")


def configure_logging(
    output_dir: Path,
) -> Path:
    """
    Show Cellpose INFO messages in the terminal and save them to a file.

    Because Cellpose validation was patched to run every epoch,
    this records one training and validation loss line per epoch.
    """
    log_path = output_dir / "training.log"

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "[%(name)s %(levelname)s] "
            "%(message)s"
        ),
        handlers=[
            logging.StreamHandler(
                sys.stdout
            ),
            logging.FileHandler(
                log_path,
                mode="w",
                encoding="utf-8",
            ),
        ],
        force=True,
    )

    logging.getLogger(
        "cellpose"
    ).setLevel(logging.INFO)

    return log_path

def main() -> None:
    args = parse_arguments()

    seed = int(
        args.seed
    )

    split_root = (
        DATASET_ROOT
        / "stratified_splits"
        / f"seed_{seed}"
    )

    train_dir = (
        split_root
        / "train"
    )

    validation_dir = (
        split_root
        / "validation"
    )

    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be greater than zero."
        )

    if not args.model_name.strip():
        raise ValueError(
            "--model-name cannot be empty."
        )

    output_dir = (
        OUTPUT_ROOT
        / args.model_name
    )

    if output_dir.exists():
        raise FileExistsError(
            "Output directory already exists. "
            "Choose a new --model-name or move the existing run: "
            f"{output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    log_path = configure_logging(
        output_dir
    )

    print(f"Training log: {log_path}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in .venv_cellpose."
        )

    device_name = torch.cuda.get_device_name(0)

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {device_name}")

    # Force a real CUDA operation before starting the long run.
    test_tensor = torch.ones(
        1,
        device="cuda",
    )

    print(
        "CUDA tensor test:",
        test_tensor.item(),
    )

    (
        train_images,
        train_masks,
        train_files,
    ) = load_cellpose_folder(train_dir)

    (
        validation_images,
        validation_masks,
        validation_files,
    ) = load_cellpose_folder(
        validation_dir
    )

    print(f"Split seed: {seed}")
    print(
        f"Training directory: "
        f"{train_dir}"
    )
    print(
        f"Validation directory: "
        f"{validation_dir}"
    )

    if len(train_images) != EXPECTED_TRAIN_IMAGES:
        raise ValueError(
            "Unexpected number of training images: "
            f"{len(train_images)}"
        )

    if (
        len(validation_images)
        != EXPECTED_VALIDATION_IMAGES
    ):
        raise ValueError(
            "Unexpected number of validation images: "
            f"{len(validation_images)}"
        )

    print(f"Training images:   {len(train_images)}")
    print(
        f"Validation images: "
        f"{len(validation_images)}"
    )

    model = models.CellposeModel(
        gpu=True,
        pretrained_model="cpsam_v2",
    )

    print("Initial model: cpsam_v2")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Batch size: {BATCH_SIZE}")
    print("Channel axis: -1")
    print(f"Output directory: {output_dir}")

    start_time = time.time()

    (
        model_path,
        train_losses,
        validation_losses,
    ) = train.train_seg(
        model.net,
        train_data=train_images,
        train_labels=train_masks,
        # train_files=train_files,
        test_data=validation_images,
        test_labels=validation_masks,
        # test_files=validation_files,
        channel_axis=-1,
        load_files=True,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        n_epochs=args.epochs,
        weight_decay=WEIGHT_DECAY,
        normalize=True,
        compute_flows=False,
        save_path=str(output_dir),
        save_every=10,
        save_each=True,
        early_stopping_patience=15,
        early_stopping_min_delta=0.0,
        rescale=False,
        bsize=256,
        # CNT images with fewer than five objects are still valid.
        min_train_masks=1,
        model_name=args.model_name,
        random_seed=seed,
    )

    elapsed_seconds = (
        time.time() - start_time
    )

    train_losses = np.asarray(
        train_losses,
        dtype=float,
    )

    validation_losses = np.asarray(
        validation_losses,
        dtype=float,
    )

    completed_epochs = len(
        train_losses
    )

    if completed_epochs == 0:
        raise RuntimeError(
            "Cellpose returned an empty training history."
        )

    if completed_epochs > args.epochs:
        raise RuntimeError(
            "Cellpose returned more training epochs than requested: "
            f"{completed_epochs} > {args.epochs}"
        )

    if len(validation_losses) != completed_epochs:
        raise RuntimeError(
            "Training and validation histories have different lengths: "
            f"{completed_epochs} and "
            f"{len(validation_losses)}"
        )

    if not np.isfinite(train_losses).all():
        raise RuntimeError(
            "Training history contains non-finite values."
        )

    if not np.isfinite(
        validation_losses
    ).all():
        raise RuntimeError(
            "Validation history contains non-finite values."
        )

    best_epoch_index = int(
        np.argmin(validation_losses)
    )

    best_epoch = (
        best_epoch_index + 1
    )

    best_validation_loss = float(
        validation_losses[
            best_epoch_index
        ]
    )

    best_model_path = (
        output_dir
        / "models"
        / (
            f"{args.model_name}"
            "_best_validation"
        )
    )

    best_record_path = (
        output_dir
        / "models"
        / (
            f"{args.model_name}"
            "_best_validation.txt"
        )
    )

    if not best_model_path.is_file():
        raise RuntimeError(
            "Best-validation model was not created: "
            f"{best_model_path}"
        )

    if not best_record_path.is_file():
        raise RuntimeError(
            "Best-validation metadata was not created: "
            f"{best_record_path}"
        )

    save_loss_history(
        train_losses=train_losses,
        validation_losses=validation_losses,
        output_dir=output_dir,
    )

    summary_path = (
        output_dir
        / "training_summary.txt"
    )

    summary_path.write_text(
        "\n".join(
            [
                f"initial_model=cpsam_v2",
                f"saved_model={model_path}",
                f"epochs_requested={args.epochs}",
                f"epochs_completed={completed_epochs}",
                f"stopped_early={completed_epochs < args.epochs}",
                f"early_stopping_patience=15",
                f"early_stopping_min_delta=0.0",
                f"best_epoch={best_epoch}",
                f"best_validation_loss={best_validation_loss:.8f}",
                f"best_model={best_model_path}",
                f"best_model_record={best_record_path}",
                f"training_log={log_path}",
                f"training_images={len(train_images)}",
                (
                    "validation_images="
                    f"{len(validation_images)}"
                ),
                f"learning_rate={LEARNING_RATE}",
                f"weight_decay={WEIGHT_DECAY}",
                f"batch_size={BATCH_SIZE}",
                f"channel_axis=-1",
                f"seed={seed}",
                f"split_manifest=stratified_seed_{seed}.csv",
                f"train_directory={train_dir}",
                f"validation_directory={validation_dir}",
                (
                    "elapsed_seconds="
                    f"{elapsed_seconds:.2f}"
                ),
                (
                    "final_train_loss="
                    f"{train_losses[-1]:.8f}"
                ),
                (
                    "final_validation_loss="
                    f"{validation_losses[-1]:.8f}"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Training completed.")
    print(f"Saved model: {model_path}")
    print(
        f"Completed epochs: "
        f"{completed_epochs}/{args.epochs}"
    )

    print(
        f"Best validation epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.8f}"
    )

    print(
        f"Best validation model: "
        f"{best_model_path}"
    )
    print(
        "Elapsed hours:",
        elapsed_seconds / 3600,
    )
    print(f"Training log: {log_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()