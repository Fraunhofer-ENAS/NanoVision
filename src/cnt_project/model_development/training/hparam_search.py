from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from cnt_project.model_development.model_compat import (
    check_model_compatibility,
)
from cnt_project.model_development.training.dataset_augmentation import (
    augmenter,
)
from cnt_project.model_development.training.stardist_training import (
    initialize_stardist_model,
    prepare_stardist_model_for_training,
    validate_model_directory_for_initialization,
)
from cnt_project.model_development.training.training_data import (
    TrainingDataBundle,
)

@dataclass(frozen=True)
class StarDistHParamTrial:
    """
    Hyperparameter configuration for one StarDist search trial.

    Version 1 intentionally contains only fine-tuning parameters that do not
    change the pretrained model architecture.
    """

    learning_rate: float
    trainable_last_n_layers: int | None


@dataclass(frozen=True)
class HParamTrialResult:
    """
    Result produced by one completed hyperparameter-search trial.
    """

    trial: StarDistHParamTrial
    model_name: str
    validation_loss: float
    training_time_s: float
    model_dir: Path
    history: dict[str, list[float]]


def iter_hparam_combinations(
    *,
    learning_rates: Iterable[float],
    trainable_last_n_layers_values: Iterable[int | None],
) -> list[StarDistHParamTrial]:
    """
    Generate the Cartesian product of the configured fine-tuning search space.
    """
    learning_rates = list(learning_rates)
    trainable_values = list(trainable_last_n_layers_values)

    if not learning_rates:
        raise ValueError(
            "learning_rates must contain at least one value."
        )

    if not trainable_values:
        raise ValueError(
            "trainable_last_n_layers_values must contain at least one value."
        )

    trials: list[StarDistHParamTrial] = []

    for learning_rate, trainable_last_n_layers in product(
        learning_rates,
        trainable_values,
    ):
        learning_rate = float(learning_rate)

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than zero, "
                f"got {learning_rate}."
            )

        if (
            trainable_last_n_layers is not None
            and int(trainable_last_n_layers) <= 0
        ):
            raise ValueError(
                "trainable_last_n_layers must be greater than zero "
                "when supplied."
            )

        trials.append(
            StarDistHParamTrial(
                learning_rate=learning_rate,
                trainable_last_n_layers=(
                    None
                    if trainable_last_n_layers is None
                    else int(trainable_last_n_layers)
                ),
            )
        )

    return trials


def select_best_trial(
    results: Iterable[HParamTrialResult],
) -> HParamTrialResult:
    """
    Select the completed trial with the lowest validation loss.
    """
    results = list(results)

    if not results:
        raise ValueError(
            "Cannot select a best trial from an empty result collection."
        )

    return min(
        results,
        key=lambda result: result.validation_loss,
    )


def build_trial_model_name(
    *,
    search_name: str,
    trial_index: int,
    trial: StarDistHParamTrial,
) -> str:
    """
    Build a deterministic local model name for one HPO trial.
    """
    learning_rate_tag = format(
        trial.learning_rate,
        ".0e",
    ).replace(
        "-",
        "m",
    )

    trainable_tag = (
        "all"
        if trial.trainable_last_n_layers is None
        else f"last{trial.trainable_last_n_layers}"
    )

    return (
        f"{search_name}"
        f"__trial-{trial_index:03d}"
        f"__lr-{learning_rate_tag}"
        f"__trainable-{trainable_tag}"
    )

def run_hparam_trial(
    *,
    trial: StarDistHParamTrial,
    trial_index: int,
    search_name: str,
    training_data: TrainingDataBundle,
    model_basedir: Path,
    source_model_name: str,
    train_patch_size: tuple[int, int],
    epochs: int,
    steps_per_epoch: int,
    training_mode: str = "standard",
    augmentation_enabled: bool = True,
    debug_plots: bool = False,
) -> HParamTrialResult:
    """
    Execute one local-pretrained StarDist fine-tuning trial.

    The source model remains unchanged. Each trial creates a new local target
    model initialized from the source model, applies the trial-specific
    learning rate and layer-trainability policy, trains it, and reports the
    minimum validation loss.

    Parameters
    ----------
    trial:
        Hyperparameter configuration for this trial.
    trial_index:
        Stable integer index used in the generated model name.
    search_name:
        Prefix identifying the hyperparameter-search experiment.
    training_data:
        Canonical train/validation arrays loaded once for the whole search.
    model_basedir:
        Directory containing StarDist model folders.
    source_model_name:
        Existing local pretrained model used as the immutable source.
    train_patch_size:
        Spatial StarDist training patch size.
    epochs:
        Number of epochs for this search trial.
    steps_per_epoch:
        Training steps per epoch.
    training_mode:
        Patched StarDist scoring/preprocessing mode.
    augmentation_enabled:
        Whether to use the canonical StarDist augmenter.
    debug_plots:
        Whether patched preprocessing debug plots are enabled.

    Returns
    -------
    HParamTrialResult
        Structured result containing the trial configuration, model location,
        validation loss, and training duration.
    """
    if epochs <= 0:
        raise ValueError(
            f"epochs must be greater than zero, got {epochs}."
        )

    if steps_per_epoch <= 0:
        raise ValueError(
            "steps_per_epoch must be greater than zero, "
            f"got {steps_per_epoch}."
        )

    model_name = build_trial_model_name(
        search_name=search_name,
        trial_index=trial_index,
        trial=trial,
    )

    model_dir = (
        model_basedir
        / model_name
    )

    validate_model_directory_for_initialization(
        model_dir,
        initialization_mode="local_pretrained",
    )

    initialized_model = initialize_stardist_model(
        initialization_mode="local_pretrained",
        model_name=model_name,
        model_basedir=model_basedir,
        train_n_channels=training_data.n_channels,

        # These architecture arguments are irrelevant to local_pretrained,
        # because the source architecture is preserved.
        n_rays=1,
        grid=(1, 1),
        use_gpu=False,
        distance_loss="mae",
        epochs=epochs,
        tensorboard=False,

        train_patch_size=train_patch_size,
        learning_rate=None,
        unet_depth=None,
        pretrained_model_name=None,
        source_model_name=source_model_name,
    )

    prepared_model = prepare_stardist_model_for_training(
        initialized_model=initialized_model,
        learning_rate=trial.learning_rate,
        trainable_last_n_layers=(
            trial.trainable_last_n_layers
        ),
    )

    model = prepared_model.model

    check_model_compatibility(
        grayscale=training_data.grayscale,
        model_name=model_name,
        dataset_n_channels=training_data.n_channels,
        model_n_channels=prepared_model.model_n_channels,
    )

    train_kwargs: dict[str, object] = {
        "validation_data": (
            training_data.X_val,
            training_data.Y_val,
        ),
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "mode": training_mode,
        "debug_plots": debug_plots,
        "debug_run_name": model_name,
    }

    if augmentation_enabled:
        train_kwargs["augmenter"] = augmenter

    started_at = perf_counter()

    history = model.train(
        training_data.X_train,
        training_data.Y_train,
        **train_kwargs,
    )

    training_time_s = (
        perf_counter()
        - started_at
    )

    history_data = getattr(
        history,
        "history",
        history,
    )

    if (
        not isinstance(history_data, dict)
        or "val_loss" not in history_data
    ):
        raise RuntimeError(
            "StarDist training history does not contain 'val_loss'."
        )

    validation_losses = np.asarray(
        history_data["val_loss"],
        dtype=float,
    )

    normalized_history: dict[str, list[float]] = {}

    for metric_name, metric_values in history_data.items():
        normalized_history[str(metric_name)] = [
            float(value)
            for value in metric_values
        ]

    if validation_losses.size == 0:
        raise RuntimeError(
            "StarDist training returned an empty validation-loss history."
        )

    validation_loss = float(
        np.min(validation_losses)
    )

    return HParamTrialResult(
        trial=trial,
        model_name=model_name,
        validation_loss=validation_loss,
        training_time_s=float(training_time_s),
        model_dir=model_dir,
        history=normalized_history,
    )