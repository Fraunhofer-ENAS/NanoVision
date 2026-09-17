from __future__ import annotations

import argparse
import json
from pathlib import Path


def infer_model_input_channels(
    basedir: str | Path,
    model_name: str,
) -> int | None:
    """
    Read the expected input-channel count from a saved model config.

    Parameters
    ----------
    basedir:
        Directory containing model folders.

    model_name:
        Model folder name.

    Returns
    -------
    int | None
        ``n_channel_in`` from the model's ``config.json`` when available,
        otherwise ``None``.
    """
    config_path = Path(basedir) / model_name / "config.json"

    if not config_path.exists():
        return None

    try:
        config = json.loads(
            config_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None

    n_channel_in = config.get("n_channel_in")

    return (
        int(n_channel_in)
        if n_channel_in is not None
        else None
    )


def parse_grayscale_arg(
    value: str,
) -> bool | None:
    """
    Parse a tri-state grayscale command-line argument.

    Accepted values
    ---------------
    auto
        Return ``None`` so the caller can choose the setting based on
        model compatibility.

    true, 1, yes, y
        Return ``True``.

    false, 0, no, n
        Return ``False``.
    """
    value_norm = str(value).strip().lower()

    if value_norm == "auto":
        return None

    if value_norm in {
        "true",
        "1",
        "yes",
        "y",
    }:
        return True

    if value_norm in {
        "false",
        "0",
        "no",
        "n",
    }:
        return False

    raise argparse.ArgumentTypeError(
        "--grayscale must be one of: auto, true, false"
    )

def infer_expected_model_channels(*, model_name: str, model_n_channels: int | None = None) -> int | None:
    """
    Infer the expected number of input channels for a model.

    Preference order:
    1. explicit ``model_n_channels`` from the loaded model config
    2. known official model naming conventions
    3. unknown -> ``None``
    """
    if model_n_channels is not None:
        return int(model_n_channels)

    model_name_lc = model_name.lower()
    if "versatile_fluo" in model_name_lc or "paper_dsb2018" in model_name_lc:
        return 1
    if "versatile_he" in model_name_lc:
        return 3
    return None


def infer_effective_dataset_channels(*, dataset_n_channels: int | None, grayscale: bool) -> int | None:
    """Return the channel count that will actually be seen by the model."""
    if dataset_n_channels is None:
        return 1 if grayscale else None
    return 1 if grayscale else int(dataset_n_channels)


def recommended_grayscale_setting(
    *,
    current_grayscale: bool,
    model_name: str,
    model_n_channels: int | None = None,
) -> bool:
    """
    Recommend the grayscale flag that best matches the selected model.

    If the model channel count is unknown, preserve the current setting.
    """
    expected_model_channels = infer_expected_model_channels(
        model_name=model_name,
        model_n_channels=model_n_channels,
    )
    if expected_model_channels == 1:
        return True
    if expected_model_channels == 3:
        return False
    return current_grayscale


def check_model_compatibility(
    *,
    grayscale: bool,
    model_name: str,
    dataset_n_channels: int | None = None,
    model_n_channels: int | None = None,
) -> None:
    """
    Guardrails for known incompatible model/data combinations.

    Parameters
    ----------
    grayscale:
        Whether the data loader will reduce images to a single channel.
    model_name:
        Model name or folder name for legacy heuristics and clearer error messages.
    dataset_n_channels:
        Number of channels in the raw dataset before any grayscale conversion.
    model_n_channels:
        Optional explicit expected input channel count from ``model.config.n_channel_in``.
    """
    model_name_lc = model_name.lower()
    effective_dataset_channels = infer_effective_dataset_channels(
        dataset_n_channels=dataset_n_channels,
        grayscale=grayscale,
    )
    expected_model_channels = infer_expected_model_channels(
        model_name=model_name,
        model_n_channels=model_n_channels,
    )

    if grayscale and "versatile_he" in model_name_lc:
        raise ValueError(
            "grayscale=True cannot be used when model_name contains 'versatile_he' because "
            "that pretrained family expects 3-channel RGB input."
        )

    if (
        effective_dataset_channels is not None
        and expected_model_channels is not None
        and effective_dataset_channels != expected_model_channels
    ):
        details = (
            f"Model/data channel mismatch for model '{model_name}': "
            f"model expects {expected_model_channels} channel(s), but the effective dataset "
            f"input has {effective_dataset_channels} channel(s) "
            f"(raw dataset channels={dataset_n_channels}, grayscale={grayscale})."
        )

        if expected_model_channels == 1 and effective_dataset_channels == 3:
            raise ValueError(
                details
                + " Use grayscale=True when loading data or convert the dataset to 1-channel "
                  "images before training/fine-tuning this model."
            )

        if expected_model_channels == 3 and effective_dataset_channels == 1:
            raise ValueError(
                details
                + " Use grayscale=False or choose a 1-channel pretrained model instead."
            )

        raise ValueError(details)