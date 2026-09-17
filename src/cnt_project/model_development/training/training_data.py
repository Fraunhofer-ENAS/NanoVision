from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cnt_project.preprocessing.dataset.dataset_loader import CNTDataset


@dataclass(frozen=True)
class TrainingDataBundle:
    """
    Validated StarDist training and validation data.

    Attributes
    ----------
    X_train:
        Training input images.
    Y_train:
        Training instance masks.
    X_val:
        Validation input images.
    Y_val:
        Validation instance masks.
    train_filenames:
        Training sample filename stems.
    val_filenames:
        Validation sample filename stems.
    n_channels:
        Effective number of input channels shared by train and validation data.
    """

    X_train: list[np.ndarray]
    Y_train: list[np.ndarray]
    X_val: list[np.ndarray]
    Y_val: list[np.ndarray]
    train_filenames: list[str]
    val_filenames: list[str]
    n_channels: int
    grayscale: bool


def _validate_loaded_subset(
    dataset: CNTDataset,
    *,
    subset_name: str,
) -> None:
    """
    Validate that a training subset is non-empty and fully labeled.
    """
    if len(dataset) == 0:
        raise RuntimeError(
            f"The '{subset_name}' subset is empty."
        )

    missing_masks = [
        sample.filename
        for sample in dataset
        if sample.Y is None
    ]

    if missing_masks:
        raise RuntimeError(
            f"The '{subset_name}' subset contains samples without masks: "
            f"{missing_masks[:10]}"
        )


def _extract_training_arrays(
    dataset: CNTDataset,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[str],
]:
    """
    Extract StarDist inputs, labels, and filename stems from a CNTDataset.
    """
    images = [
        sample.X
        for sample in dataset
    ]

    masks = [
        sample.Y
        for sample in dataset
    ]

    filenames = [
        sample.X_fn
        for sample in dataset
    ]

    if any(mask is None for mask in masks):
        raise RuntimeError(
            "Cannot build training arrays because at least one mask is missing."
        )

    labeled_masks = [
        mask
        for mask in masks
        if mask is not None
    ]

    return (
        images,
        labeled_masks,
        filenames,
    )


def load_training_data_bundle(
    *,
    dataset_root: Path,
    split_manifest_path: Path,
    grayscale: bool,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> TrainingDataBundle:
    """
    Load and validate manifest-defined train and validation subsets.

    This function owns the conversion from canonical CNTDataset subsets into
    StarDist-ready training arrays.

    Parameters
    ----------
    dataset_root:
        Canonical unsplit dataset root.
    split_manifest_path:
        Manifest defining train/val/test membership.
    grayscale:
        Whether CNTDataset should expose one-channel images.
    max_train_samples:
        Optional cap on the number of training samples.
    max_val_samples:
        Optional cap on the number of validation samples.

    Returns
    -------
    TrainingDataBundle
        Validated training and validation arrays, filenames, and channel count.
    """
    print("Loading manifest-defined training subset...")

    train_dataset = CNTDataset(
        dataset_root=dataset_root,
        split_manifest_path=split_manifest_path,
        subset="train",
        grayscale=grayscale,
        require_masks=True,
        max_samples=max_train_samples,
    )

    print("Loading manifest-defined validation subset...")

    val_dataset = CNTDataset(
        dataset_root=dataset_root,
        split_manifest_path=split_manifest_path,
        subset="val",
        grayscale=grayscale,
        require_masks=True,
        max_samples=max_val_samples,
    )

    _validate_loaded_subset(
        train_dataset,
        subset_name="train",
    )

    _validate_loaded_subset(
        val_dataset,
        subset_name="val",
    )

    (
        X_train,
        Y_train,
        train_filenames,
    ) = _extract_training_arrays(
        train_dataset
    )

    (
        X_val,
        Y_val,
        val_filenames,
    ) = _extract_training_arrays(
        val_dataset
    )

    train_n_channels = int(
        train_dataset.samples[0].n_channel
    )

    val_n_channels = int(
        val_dataset.samples[0].n_channel
    )

    if train_n_channels != val_n_channels:
        raise RuntimeError(
            "Training and validation subsets have different channel counts: "
            f"train={train_n_channels}, "
            f"val={val_n_channels}."
        )

    return TrainingDataBundle(
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        train_filenames=train_filenames,
        val_filenames=val_filenames,
        n_channels=train_n_channels,
        grayscale=bool(grayscale),
    )