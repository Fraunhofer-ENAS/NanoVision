"""
Training artifact helpers.

Current status
--------------
This module currently contains only a minimal set of helper functions and is
not yet part of the canonical training pipeline.

Future responsibility
---------------------
Once the training runner is implemented, this module should become the
central location responsible for persisting training-specific artifacts,
for example:

- training history;
- loss and metric plots;
- learning-curve figures;
- training summaries.

This module should not generate run names, create run directories, or manage
experiment runs. Those responsibilities belong exclusively to
``cnt_project.io.run_manager``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from datetime import datetime

import matplotlib.pyplot as plt

from cnt_project.io.run_manager import safe_run_name



def save_training_metric_plots(
    history: object,
    output_dir: Path,
    *,
    dpi: int = 300,
) -> list[Path]:
    """
    Save one figure per metric from a Keras-style training history.

    Metrics unavailable in the returned history are naturally skipped because
    this function iterates only over metrics that actually exist.
    """
    history_data = getattr(history, "history", history)

    if not isinstance(history_data, dict):
        raise TypeError(
            "history must be a mapping or expose a mapping through "
            "history.history."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    for metric_name, metric_values in history_data.items():
        values = list(metric_values)

        if not values:
            continue

        figure, axis = plt.subplots()

        axis.plot( range(1, len(values) + 1), values, label=str(metric_name), )

        axis.set_xlabel("Epoch")
        axis.set_ylabel(str(metric_name))
        axis.set_title(f"{metric_name} over epochs")
        axis.legend()

        output_path = ( output_dir / f"{safe_run_name(str(metric_name))}.png" )

        figure.savefig( output_path, dpi=int(dpi), bbox_inches="tight", )
        plt.close(figure)

        saved_paths.append(output_path)

    return saved_paths


def save_training_manifest(
        output_dir: Path,
        *,
        dataset_root: Path,
        split_manifest_path: Path,
        model_name: str,
        model_basedir: Path,
        initialization_mode: str,
        pretrained_model_name: str | None,
        source_model_name: str | None,
        train_filenames: list[str],
        val_filenames: list[str],
        max_train_samples: int | None,
        max_val_samples: int | None,
        grayscale: bool,
        dataset_n_channels: int,
        model_n_channels: int,
        n_rays: int,
        grid: tuple[int, int],
        distance_loss: str,
        training_mode: str,
        train_patch_size: tuple[int, int],
        learning_rate: float,

        train_loss_weights: tuple[float, float],
        batch_size: int,
        reduce_lr: dict[str, object],

        unet_depth: int,
        epochs: int,
        steps_per_epoch: int,
        seed: int,
        trainable_last_n_layers: int | None,
        total_model_layers: int,
        trainable_model_layers: int,
        frozen_model_layers: int,
        augmentation_enabled: bool,
        threshold_optimization_enabled: bool,
        overwrite: bool = False,
    ) -> Path:
        """
        Save the exact dataset membership and training invocation parameters.

        This artifact describes one training invocation. It does not replace the
        StarDist model configuration stored under the model directory.
        """
        payload = {
            "dataset": {
                "dataset_root": str(dataset_root),
                "split_manifest_path": str(split_manifest_path),
                "train_subset": "train",
                "validation_subset": "val",
                "max_train_samples": max_train_samples,
                "max_val_samples": max_val_samples,
                "train_sample_count": len(train_filenames),
                "validation_sample_count": len(val_filenames),
                "train_filenames": train_filenames,
                "validation_filenames": val_filenames,
            },
            "model": {
                "model_name": model_name,
                "model_basedir": str(model_basedir),
                "initialization_mode": initialization_mode,
                "pretrained_model_name": pretrained_model_name,
                "source_model_name": source_model_name,
                "grayscale": grayscale,
                "dataset_n_channels": dataset_n_channels,
                "model_n_channels": model_n_channels,
                "n_rays": n_rays,
                "grid": list(grid),
                "distance_loss": distance_loss,
                "unet_depth": unet_depth,
            },
            "training": {
                "mode": training_mode,
                "epochs_this_run": epochs,
                "train_patch_size": list(train_patch_size),
                "learning_rate": learning_rate,

                "train_loss_weights": list(train_loss_weights),
                "batch_size": batch_size,
                "reduce_lr": reduce_lr,

                "steps_per_epoch": steps_per_epoch,
                "seed": seed,
                "trainable_last_n_layers": trainable_last_n_layers,
                "total_model_layers": total_model_layers,
                "trainable_model_layers": trainable_model_layers,
                "frozen_model_layers": frozen_model_layers,
                "augmentation_enabled": augmentation_enabled,
                "threshold_optimization_enabled": (
                    threshold_optimization_enabled
                ),
            },
        }

        return save_config(
            output_dir,
            payload,
            filename="training_manifest.json",
            overwrite=overwrite,
        )


def save_training_status(
    output_dir: Path,
    *,
    run_id: str,
    model_name: str,
    initialization_mode: str,
    status: str,
    phase: str,
    started_at: str,
    error: BaseException | None = None,
) -> Path:
    """
    Save the current state of one training invocation.

    The same ``status.json`` file is overwritten as the invocation progresses.

    Supported status values are:

    - ``running``;
    - ``completed``;
    - ``failed``.
    """
    allowed_statuses = {
        "running",
        "completed",
        "failed",
    }

    if status not in allowed_statuses:
        raise ValueError(
            f"Unsupported training status {status!r}. "
            f"Expected one of: {sorted(allowed_statuses)}"
        )

    updated_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    payload: dict[str, object] = {
        "run_id": run_id,
        "model_name": model_name,
        "initialization_mode": initialization_mode,
        "status": status,
        "phase": phase,
        "started_at": started_at,
        "updated_at": updated_at,
    }

    if status == "completed":
        payload["completed_at"] = updated_at

    if status == "failed":
        payload["failed_at"] = updated_at

        if error is not None:
            payload["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }

    return save_config(
        output_dir,
        payload,
        filename="status.json",
        overwrite=True,
    )



def _to_json_compatible(value: Any) -> Any:
    """
    Convert common configuration values into JSON-compatible objects.

    This supports:

    - dataclasses;
    - dictionaries;
    - lists and tuples;
    - ``Path`` objects;
    - primitive JSON values;
    - regular Python objects exposing ``__dict__``.

    Unsupported values are represented using ``str(value)``.
    """
    if is_dataclass(value):
        return _to_json_compatible(asdict(value))

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): _to_json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _to_json_compatible(item)
            for item in value
        ]

    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if hasattr(value, "__dict__"):
        return {
            str(key): _to_json_compatible(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }

    return str(value)


def save_config(
    output_dir: str | Path,
    config: Any,
    *,
    filename: str = "config.json",
    overwrite: bool = False,
) -> Path:
    """
    Serialize a training configuration as JSON.

    Parameters
    ----------
    output_dir
        Directory in which the configuration file will be written.

    config
        Configuration object. Dataclasses, dictionaries, ``Path`` objects,
        sequences, primitive values, and regular objects with ``__dict__`` are
        supported.

    filename
        Output filename.

    overwrite
        Whether an existing file may be replaced.

    Returns
    -------
    Path
        Path to the generated JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_filename = safe_run_name(filename)

    if not safe_filename:
        raise ValueError(
            "filename is empty after filesystem sanitization."
        )

    if not safe_filename.lower().endswith(".json"):
        safe_filename = f"{safe_filename}.json"

    output_path = output_dir / safe_filename

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Configuration file already exists: {output_path}"
        )

    payload = _to_json_compatible(config)

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


def save_training_history(
    output_dir: str | Path,
    history: Any,
    *,
    filename: str = "history.json",
    overwrite: bool = False,
) -> Path:
    """
    Save a Keras-style training history as JSON.

    ``history`` may be:

    - a Keras ``History`` object exposing ``history``;
    - a dictionary of metric lists.

    Parameters
    ----------
    output_dir
        Training artifact directory.

    history
        Training history object or dictionary.

    filename
        Output JSON filename.

    overwrite
        Whether an existing history file may be replaced.

    Returns
    -------
    Path
        Path to the generated history JSON.
    """
    history_payload = getattr(
        history,
        "history",
        history,
    )

    if not isinstance(history_payload, dict):
        raise TypeError(
            "history must be a mapping or expose a dictionary through "
            "history.history."
        )

    return save_config(
        output_dir,
        history_payload,
        filename=filename,
        overwrite=overwrite,
    )