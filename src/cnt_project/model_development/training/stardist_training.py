from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass

from stardist.models import StarDist2D, Config2D

from cnt_project.model_development.stardist_patched.stardist_model_configuration import (
    MyStarDist2D,
)

def _create_fresh_stardist_model(
    *,
    model_name: str,
    model_basedir: Path,
    train_n_channels: int,
    n_rays: int,
    grid: tuple[int, int],
    use_gpu: bool,
    distance_loss: str,
    train_loss_weights: tuple[float, float],
    batch_size: int,
    reduce_lr_factor: float,
    reduce_lr_patience: int,
    reduce_lr_min_delta: float,
    epochs: int,
    tensorboard: bool,
    train_patch_size: tuple[int, int],
    learning_rate: float | None,
    unet_depth: int | None,
) -> MyStarDist2D:
    """
    Create a new randomly initialized patched StarDist model from the
    requested training configuration.
    """
    config_kwargs: dict[str, object] = {
        "n_rays": n_rays,
        "grid": grid,
        "use_gpu": use_gpu,
        "train_dist_loss": distance_loss,
        "train_loss_weights": tuple(train_loss_weights),
        "n_channel_in": train_n_channels,
        "train_epochs": epochs,
        "train_tensorboard": tensorboard,
        "train_patch_size": train_patch_size,
        "train_batch_size": int(batch_size),
        "train_reduce_lr": {
            "factor": float(reduce_lr_factor),
            "patience": int(reduce_lr_patience),
            "min_delta": float(reduce_lr_min_delta),
        },
    }

    if learning_rate is not None:
        config_kwargs["train_learning_rate"] = learning_rate

    if unet_depth is not None:
        config_kwargs["unet_n_depth"] = unet_depth

    config = Config2D(
        **config_kwargs,
    )

    return MyStarDist2D(
        config,
        name=model_name,
        basedir=str(model_basedir),
    )

def _create_local_model_from_source(
    *,
    source_model: object,
    local_model_name: str,
    model_basedir: Path,
    train_patch_size: tuple[int, int],
    source_description: str,
) -> MyStarDist2D:
    """
    Create a new local patched StarDist model from an initialized source model.

    The source model configuration is copied and its network weights are
    transferred into a new ``MyStarDist2D`` instance.

    The source model remains unchanged.

    Parameters
    ----------
    source_model:
        Initialized StarDist-compatible source model exposing ``config`` and
        ``keras_model``.
    local_model_name:
        Name of the new local target model.
    model_basedir:
        Directory containing local StarDist models.
    train_patch_size:
        Training patch size applied to the copied configuration.
    source_description:
        Human-readable source description used in validation errors.

    Returns
    -------
    MyStarDist2D
        Newly created local patched model initialized from the source weights.
    """
    local_config = deepcopy(
        source_model.config
    )

    local_config.train_patch_size = tuple(
        int(value)
        for value in train_patch_size
    )

    local_model = MyStarDist2D(
        local_config,
        name=local_model_name,
        basedir=str(model_basedir),
    )

    source_weights = (
        source_model.keras_model.get_weights()
    )

    local_weights = (
        local_model.keras_model.get_weights()
    )

    if len(source_weights) != len(local_weights):
        raise RuntimeError(
            f"{source_description} and target model architectures have "
            "different numbers of weight tensors: "
            f"source={len(source_weights)}, "
            f"target={len(local_weights)}."
        )

    incompatible_shapes: list[str] = []

    for index, (
        source_weight,
        local_weight,
    ) in enumerate(
        zip(source_weights, local_weights)
    ):
        if source_weight.shape != local_weight.shape:
            incompatible_shapes.append(
                f"tensor {index}: "
                f"source={source_weight.shape}, "
                f"target={local_weight.shape}"
            )

    if incompatible_shapes:
        raise RuntimeError(
            f"{source_description} weights are incompatible with the "
            "new local patched model architecture. First mismatches: "
            + "; ".join(incompatible_shapes[:10])
        )

    local_model.keras_model.set_weights(
        source_weights
    )

    return local_model

def _create_local_model_from_official_pretrained(
        *,
        pretrained_model_name: str,
        local_model_name: str,
        model_basedir: Path,
        train_patch_size: tuple[int, int],
    ) -> tuple[MyStarDist2D, object]:
        """
        Create a new local patched model initialized from official StarDist weights.

        The official model is loaded temporarily and remains unchanged. Its
        configuration and network weights are copied into a new MyStarDist2D
        instance stored under ``model_basedir / local_model_name``.

        Returns
        -------
        tuple[MyStarDist2D, object]
            The new local patched model and the temporary official source model.
            The source model is returned temporarily so the caller can inspect or
            report its configuration if needed.
        """
        source_name = str(pretrained_model_name).strip()

        if not source_name:
            raise ValueError(
                "pretrained_model_name must not be empty."
            )

        print(
            f"Loading official StarDist pretrained model: {source_name}"
        )

        official_model = StarDist2D.from_pretrained(
            source_name
        )

        local_model = _create_local_model_from_source(
            source_model=official_model,
            local_model_name=local_model_name,
            model_basedir=model_basedir,
            train_patch_size=train_patch_size,
            source_description="Official pretrained StarDist model",
        )

        print(
            "Official pretrained weights were transferred to the "
            "new local patched model."
        )

        return local_model, official_model


def _create_local_model_from_local_pretrained(
    *,
    source_model_name: str,
    local_model_name: str,
    model_basedir: Path,
    train_patch_size: tuple[int, int],
) -> tuple[MyStarDist2D, MyStarDist2D]:
    """
    Create a new local patched model initialized from another local model.

    The source model is loaded from ``model_basedir / source_model_name`` and
    remains unchanged. Its architecture and network weights are copied into a
    new ``MyStarDist2D`` instance stored under
    ``model_basedir / local_model_name``.

    The training patch size may be adapted because it does not change the
    network architecture.
    """
    source_name = str(source_model_name).strip()
    target_name = str(local_model_name).strip()

    if not source_name:
        raise ValueError(
            "source_model_name must not be empty."
        )

    if not target_name:
        raise ValueError(
            "local_model_name must not be empty."
        )

    if source_name == target_name:
        raise ValueError(
            "Local-pretrained initialization requires different source and "
            "target model names. Use resume_local to continue training the "
            "same model."
        )

    source_dir = model_basedir / source_name

    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(
            f"Local pretrained source model directory does not exist: "
            f"{source_dir}"
        )

    print(
        f"Loading local pretrained source model: {source_name}"
    )

    source_model = MyStarDist2D(
        None,
        name=source_name,
        basedir=str(model_basedir),
    )

    local_model = _create_local_model_from_source(
        source_model=source_model,
        local_model_name=target_name,
        model_basedir=model_basedir,
        train_patch_size=train_patch_size,
        source_description="Local pretrained StarDist model",
    )

    print(
        "Local pretrained weights were transferred to the "
        "new local patched model."
    )

    return local_model, source_model


def validate_model_directory_for_initialization(
        model_dir: Path,
        *,
        initialization_mode: str,
    ) -> None:
        """
        Validate the local model directory for the requested initialization mode.

        Fresh, official-pretrained, and local-pretrained initialization create
        a new local model. Resume-local initialization requires an existing
        saved local model.
        """
        if initialization_mode in {
            "fresh_config",
            "official_pretrained",
            "local_pretrained",
        }:
            mode_descriptions = {
                "fresh_config": "Fresh-config",
                "official_pretrained": "Official-pretrained",
                "local_pretrained": "Local-pretrained",
            }

            if model_dir.exists() and any(model_dir.iterdir()):
                mode_description = mode_descriptions[
                    initialization_mode
                ]

                raise FileExistsError(
                    f"{mode_description} training requires a new or empty "
                    f"local model directory. The target already contains "
                    f"files: {model_dir}"
                )

            return

        if initialization_mode == "resume_local":
            if not model_dir.exists() or not model_dir.is_dir():
                raise FileNotFoundError(
                    "Resume-local training requires an existing model "
                    f"directory: {model_dir}"
                )

            if not any(model_dir.iterdir()):
                raise FileNotFoundError(
                    "Resume-local training requires a non-empty model "
                    f"directory: {model_dir}"
                )

            config_path = model_dir / "config.json"

            if not config_path.exists():
                raise FileNotFoundError(
                    "Resume-local training could not find the saved StarDist "
                    f"configuration: {config_path}"
                )

            weight_candidates = (
                model_dir / "weights_best.h5",
                model_dir / "weights_last.h5",
                model_dir / "weights_now.h5",
            )

            if not any(path.exists() for path in weight_candidates):
                raise FileNotFoundError(
                    "Resume-local training could not find a supported weights "
                    f"file in: {model_dir}"
                )

            return

        raise ValueError(
            "initialization_mode must be one of: "
            "'fresh_config', 'resume_local', "
            "'official_pretrained', 'local_pretrained'."
        )


def validate_stardist_training_arguments(
    *,
    initialization_mode: str,
    pretrained_model_name: str | None,
    source_model_name: str | None,
    unet_depth: int | None,
    trainable_last_n_layers: int | None,
) -> None:
    """
    Validate model and fine-tuning arguments whose meaning depends on the
    selected StarDist initialization mode.
    """
    if initialization_mode == "official_pretrained":
        if (
            pretrained_model_name is None
            or not pretrained_model_name.strip()
        ):
            raise ValueError(
                "--pretrained-model-name is required when "
                "--initialization-mode official_pretrained is selected."
            )

        if source_model_name is not None:
            raise ValueError(
                "--source-model-name may only be used with "
                "--initialization-mode local_pretrained."
            )

    elif initialization_mode == "local_pretrained":
        if (
            source_model_name is None
            or not source_model_name.strip()
        ):
            raise ValueError(
                "--source-model-name is required when "
                "--initialization-mode local_pretrained is selected."
            )

        if pretrained_model_name is not None:
            raise ValueError(
                "--pretrained-model-name may only be used with "
                "--initialization-mode official_pretrained."
            )

    else:
        if pretrained_model_name is not None:
            raise ValueError(
                "--pretrained-model-name may only be used with "
                "--initialization-mode official_pretrained."
            )

        if source_model_name is not None:
            raise ValueError(
                "--source-model-name may only be used with "
                "--initialization-mode local_pretrained."
            )

    if (
        unet_depth is not None
        and initialization_mode not in {
            "fresh_config",
            "official_pretrained",
        }
    ):
        raise ValueError(
            "--unet-depth may only be used with "
            "--initialization-mode fresh_config or official_pretrained."
        )

    if (
        trainable_last_n_layers is not None
        and initialization_mode == "fresh_config"
    ):
        raise ValueError(
            "--trainable-last-n-layers may not be used with "
            "--initialization-mode fresh_config because selectively freezing "
            "randomly initialized layers is not a supported fine-tuning mode."
        )

@dataclass(frozen=True)
class TrainabilitySummary:
    """
    Summary of the layer trainability policy applied to a StarDist model.
    """

    total_layers: int
    trainable_layers: int
    frozen_layers: int
    trainable_last_n_layers: int | None

def apply_trainability_policy(
    model: MyStarDist2D,
    *,
    trainable_last_n_layers: int | None,
) -> TrainabilitySummary:
    """
    Apply an optional fine-tuning policy to the Keras layers of a StarDist model.

    Parameters
    ----------
    model:
        Initialized patched StarDist model.
    trainable_last_n_layers:
        If ``None``, leave the model's existing layer trainability unchanged.

        If an integer ``N`` is supplied, freeze every layer except the final
        ``N`` layers.

    Returns
    -------
    TrainabilitySummary
        Counts describing the resulting layer trainability state.

    Raises
    ------
    ValueError
        If ``trainable_last_n_layers`` is not positive or exceeds the number
        of layers in the model.
    """
    layers = list(model.keras_model.layers)
    total_layers = len(layers)

    if trainable_last_n_layers is not None:
        n_trainable = int(trainable_last_n_layers)

        if n_trainable <= 0:
            raise ValueError(
                "trainable_last_n_layers must be greater than zero."
            )

        if n_trainable > total_layers:
            raise ValueError(
                "trainable_last_n_layers cannot exceed the number of "
                f"model layers: requested={n_trainable}, "
                f"total_layers={total_layers}."
            )

        freeze_until = total_layers - n_trainable

        for index, layer in enumerate(layers):
            layer.trainable = index >= freeze_until

    trainable_layers = sum(
        int(layer.trainable)
        for layer in layers
    )

    return TrainabilitySummary(
        total_layers=total_layers,
        trainable_layers=trainable_layers,
        frozen_layers=total_layers - trainable_layers,
        trainable_last_n_layers=trainable_last_n_layers,
    )


@dataclass(frozen=True)
class InitializedStarDistModel:
    model: MyStarDist2D
    official_source_model: object | None
    local_source_model: MyStarDist2D | None

def initialize_stardist_model(
    *,
    initialization_mode: str,
    model_name: str,
    model_basedir: Path,
    train_n_channels: int,
    n_rays: int,
    grid: tuple[int, int],
    use_gpu: bool,
    distance_loss: str,
    train_loss_weights: tuple[float, float],
    batch_size: int,
    reduce_lr_factor: float,
    reduce_lr_patience: int,
    reduce_lr_min_delta: float,
    epochs: int,
    tensorboard: bool,
    train_patch_size: tuple[int, int],
    learning_rate: float | None,
    unet_depth: int | None,
    pretrained_model_name: str | None,
    source_model_name: str | None,
) -> InitializedStarDistModel:
    """
    Initialize a patched StarDist model according to the requested mode.
    """
    official_source_model = None
    local_source_model = None

    if initialization_mode == "fresh_config":
        model = _create_fresh_stardist_model(
            model_name=model_name,
            model_basedir=model_basedir,
            train_n_channels=train_n_channels,
            n_rays=n_rays,
            grid=grid,
            use_gpu=use_gpu,
            distance_loss=distance_loss,
            train_loss_weights=train_loss_weights,
            batch_size=batch_size,
            reduce_lr_factor=reduce_lr_factor,
            reduce_lr_patience=reduce_lr_patience,
            reduce_lr_min_delta=reduce_lr_min_delta,
            epochs=epochs,
            tensorboard=tensorboard,
            train_patch_size=train_patch_size,
            learning_rate=learning_rate,
            unet_depth=unet_depth,
        )

    elif initialization_mode == "resume_local":
        model = MyStarDist2D(
            None,
            name=model_name,
            basedir=str(model_basedir),
        )

    elif initialization_mode == "official_pretrained":
        if pretrained_model_name is None:
            raise ValueError(
                "pretrained_model_name is required for "
                "official_pretrained initialization."
            )

        official_source_model = StarDist2D.from_pretrained(
            pretrained_model_name
        )

        model = _create_fresh_stardist_model(
            model_name=model_name,
            model_basedir=model_basedir,
            train_n_channels=train_n_channels,
            n_rays=n_rays,
            grid=grid,
            use_gpu=use_gpu,
            distance_loss=distance_loss,
            train_loss_weights=train_loss_weights,
            batch_size=batch_size,
            reduce_lr_factor=reduce_lr_factor,
            reduce_lr_patience=reduce_lr_patience,
            reduce_lr_min_delta=reduce_lr_min_delta,
            epochs=epochs,
            tensorboard=tensorboard,
            train_patch_size=train_patch_size,
            learning_rate=learning_rate,
            unet_depth=unet_depth,
        )

        source_weights = (
            official_source_model.keras_model.get_weights()
        )
        target_weights = model.keras_model.get_weights()

        if (
            len(source_weights) == len(target_weights)
            and all(
                source.shape == target.shape
                for source, target in zip(
                    source_weights,
                    target_weights,
                )
            )
        ):
            model.keras_model.set_weights(
                source_weights
            )

    elif initialization_mode == "local_pretrained":
        if source_model_name is None:
            raise ValueError(
                "source_model_name is required for "
                "local_pretrained initialization."
            )

        model, local_source_model = (
            _create_local_model_from_local_pretrained(
                source_model_name=source_model_name,
                local_model_name=model_name,
                model_basedir=model_basedir,
                train_patch_size=train_patch_size,
            )
        )

    else:
        raise ValueError(
            f"Unsupported initialization mode: {initialization_mode}"
        )

    return InitializedStarDistModel(
        model=model,
        official_source_model=official_source_model,
        local_source_model=local_source_model,
    )

@dataclass(frozen=True)
class PreparedStarDistModel:
    model: MyStarDist2D
    official_source_model: object | None
    local_source_model: MyStarDist2D | None

    effective_learning_rate: float
    effective_unet_depth: int
    model_n_channels: int
    trainability: TrainabilitySummary

def prepare_stardist_model_for_training(
    *,
    initialized_model: InitializedStarDistModel,
    learning_rate: float | None,
    train_loss_weights: tuple[float, float],
    batch_size: int,
    reduce_lr_factor: float,
    reduce_lr_patience: int,
    reduce_lr_min_delta: float,
    trainable_last_n_layers: int | None,
) -> PreparedStarDistModel:
    """
    Apply training-time configuration to an initialized StarDist model.

    This step is intentionally separate from model initialization. It applies
    training policies that may be changed independently of the model
    architecture or initialization source.

    Parameters
    ----------
    initialized_model:
        Result returned by ``initialize_stardist_model``.
    learning_rate:
        Optional learning-rate override. If ``None``, preserve the learning
        rate stored in the initialized model configuration.
    trainable_last_n_layers:
        Optional fine-tuning policy. If supplied, freeze all model layers
        except the final N layers.

    Returns
    -------
    PreparedStarDistModel
        The initialized model together with its effective training
        configuration and trainability summary.
    """
    model = initialized_model.model

    if learning_rate is not None:
        model.config.train_learning_rate = float(
            learning_rate
        )

    model.config.train_loss_weights = tuple(
        float(value)
        for value in train_loss_weights
    )

    model.config.train_batch_size = int(batch_size)

    model.config.train_reduce_lr = {
        "factor": float(reduce_lr_factor),
        "patience": int(reduce_lr_patience),
        "min_delta": float(reduce_lr_min_delta),
    }

    trainability = apply_trainability_policy(
        model,
        trainable_last_n_layers=trainable_last_n_layers,
    )
    return PreparedStarDistModel(
        model=model,
        official_source_model=(
            initialized_model.official_source_model
        ),
        local_source_model=(
            initialized_model.local_source_model
        ),
        effective_learning_rate=float(
            model.config.train_learning_rate
        ),
        effective_unet_depth=int(
            model.config.unet_n_depth
        ),
        model_n_channels=int(
            model.config.n_channel_in
        ),
        trainability=trainability,
    )