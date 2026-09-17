from __future__ import annotations

from typing import Iterable
from dataclasses import dataclass
from pathlib import Path

from cnt_project.model_development.training.hparam_search import (
    HParamTrialResult,
    StarDistHParamTrial,
    run_hparam_trial,
)
from cnt_project.model_development.training.training_data import (
    TrainingDataBundle,
)

def trial_from_wandb_config(
    *,
    learning_rate: float,
    trainable_last_n_layers: int | str,
) -> StarDistHParamTrial:
    """
    Convert one W&B sweep configuration into a canonical HPO trial.
    """
    if isinstance(
        trainable_last_n_layers,
        str,
    ):
        normalized = (
            trainable_last_n_layers
            .strip()
            .lower()
        )

        if normalized != "all":
            raise ValueError(
                "String trainable_last_n_layers values must be 'all'."
            )

        trainable_value = None

    else:
        trainable_value = int(
            trainable_last_n_layers
        )

    return StarDistHParamTrial(
        learning_rate=float(
            learning_rate
        ),
        trainable_last_n_layers=trainable_value,
    )

def build_wandb_finetuning_sweep_config(
    *,
    learning_rates: Iterable[float],
    trainable_last_n_layers_values: Iterable[int | None],
) -> dict:
    """
    Build a W&B grid-sweep configuration for canonical StarDist fine-tuning.
    """
    learning_rates = [
        float(value)
        for value in learning_rates
    ]

    trainable_values = [
        (
            "all"
            if value is None
            else int(value)
        )
        for value in trainable_last_n_layers_values
    ]

    if not learning_rates:
        raise ValueError(
            "learning_rates must contain at least one value."
        )

    if not trainable_values:
        raise ValueError(
            "trainable_last_n_layers_values must contain at least one value."
        )

    return {
        "method": "grid",
        "metric": {
            "name": "best_validation_loss",
            "goal": "minimize",
        },
        "parameters": {
            "learning_rate": {
                "values": learning_rates,
            },
            "trainable_last_n_layers": {
                "values": trainable_values,
            },
        },
    }

@dataclass(frozen=True)
class WandbTrialResult:
    """
    Canonical HPO trial result with its corresponding W&B run identity.
    """

    wandb_run_id: str
    wandb_run_name: str
    trial_result: HParamTrialResult

@dataclass(frozen=True)
class WandbHParamSearchResult:
    """
    Result of one completed W&B hyperparameter sweep.
    """

    sweep_id: str
    trial_results: list[WandbTrialResult]

def run_wandb_hparam_search(
    *,
    search_name: str,
    project: str,
    entity: str | None,
    sweep_config: dict,
    training_data: TrainingDataBundle,
    model_basedir: Path,
    source_model_name: str,
    train_patch_size: tuple[int, int],
    epochs: int,
    steps_per_epoch: int,
    training_mode: str = "standard",
    augmentation_enabled: bool = True,
    debug_plots: bool = False,
    wandb_mode: str = "online",
    log_model_artifacts: bool = False,
    log_sample_images: bool = False,
    sample_image_index: int = 0,
) -> WandbHParamSearchResult:
    """
    Execute a W&B-managed StarDist fine-tuning sweep.

    W&B owns sweep scheduling and experiment logging. Actual StarDist
    fine-tuning is delegated to the canonical ``run_hparam_trial`` function.
    """
    import wandb

    if not search_name.strip():
        raise ValueError(
            "search_name must not be empty."
        )

    if not project.strip():
        raise ValueError(
            "project must not be empty."
        )

    if log_sample_images:
        if sample_image_index < 0:
            raise ValueError(
                "sample_image_index must be greater than or equal to zero."
            )

        if sample_image_index >= len(training_data.X_train):
            raise ValueError(
                "sample_image_index exceeds the available training samples: "
                f"requested={sample_image_index}, "
                f"training_samples={len(training_data.X_train)}."
            )

    trial_results: list[WandbTrialResult] = []

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project,
        entity=entity,
    )

    trial_counter = 0

    def train_sweep_trial() -> None:
        nonlocal trial_counter

        with wandb.init(
            project=project,
            entity=entity,
            mode=wandb_mode,
        ) as run:
            trial_counter += 1

            if log_sample_images:
                sample_image = training_data.X_train[
                    sample_image_index
                ]

                sample_label = training_data.Y_train[
                    sample_image_index
                ]

                sample_filename = training_data.train_filenames[
                    sample_image_index
                ]

                run.log(
                    {
                        "samples/input_image": wandb.Image(
                            sample_image,
                            caption=(
                                f"Training input: {sample_filename}"
                            ),
                        ),
                        "samples/ground_truth_label": wandb.Image(
                            sample_label,
                            caption=(
                                f"Ground-truth instance mask: "
                                f"{sample_filename}"
                            ),
                        ),
                    },
                    step=0,
                )

            trial = trial_from_wandb_config(
                learning_rate=run.config.learning_rate,
                trainable_last_n_layers=(
                    run.config.trainable_last_n_layers
                ),
            )

            result = run_hparam_trial(
                trial=trial,
                trial_index=trial_counter,
                search_name=search_name,
                training_data=training_data,
                model_basedir=model_basedir,
                source_model_name=source_model_name,
                train_patch_size=train_patch_size,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                training_mode=training_mode,
                augmentation_enabled=augmentation_enabled,
                debug_plots=debug_plots,
            )

            trial_results.append(
                WandbTrialResult(
                    wandb_run_id=str(run.id),
                    wandb_run_name=str(run.name),
                    trial_result=result,
                )
            )

            history = result.history

            epoch_count = max(
                len(values)
                for values in history.values()
            )

            for epoch_index in range(epoch_count):
                epoch_metrics: dict[str, float | int] = {
                    "epoch": epoch_index + 1,
                }

                for metric_name, values in history.items():
                    if epoch_index < len(values):
                        epoch_metrics[metric_name] = values[
                            epoch_index
                        ]

                run.log(
                    epoch_metrics,
                    step=epoch_index + 1,
                )

            run.summary["best_validation_loss"] = (
                result.validation_loss
            )

            run.summary["training_time_s"] = (
                result.training_time_s
            )

            run.summary["local_model_name"] = (
                result.model_name
            )

            run.summary["local_model_dir"] = str(
                result.model_dir
            )

            if log_model_artifacts:
                artifact = wandb.Artifact(
                    name=f"model-{run.id}",
                    type="model",
                    metadata={
                        "search_name": search_name,
                        "learning_rate": (
                            trial.learning_rate
                        ),
                        "trainable_last_n_layers": (
                            trial.trainable_last_n_layers
                        ),
                        "best_validation_loss": (
                            result.validation_loss
                        ),
                    },
                )

                artifact.add_dir(
                    local_path=str(
                        result.model_dir
                    )
                )

                run.log_artifact(
                    artifact
                )

    wandb.agent(
        sweep_id,
        function=train_sweep_trial,
        count=None,
        project=project,
        entity=entity,
    )

    return WandbHParamSearchResult(
        sweep_id=str(sweep_id),
        trial_results=trial_results,
    )