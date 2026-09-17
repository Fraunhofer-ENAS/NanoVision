from __future__ import annotations

import argparse
from pathlib import Path


def _default_dataset_root() -> Path:
    """
    Return the default canonical unsplit dataset root.
    """
    return ( Path(__file__).resolve().parents[4] / "data" / "cnt_segmentation" )


def _default_split_manifest_path() -> Path:
    """
    Return the default prepared split manifest.
    """
    return ( _default_dataset_root() / "splits" / "default_split.csv" )


def _parse_grid(raw: str) -> tuple[int, int]:
    """
    Parse a StarDist grid argument formatted as ``Y,X``.
    """
    parts = [ part.strip() for part in str(raw).split(",") ]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "--grid must be formatted as Y,X, for example: 2,2"
        )

    try:
        grid = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--grid values must be integers."
        ) from exc

    if any(value <= 0 for value in grid):
        raise argparse.ArgumentTypeError(
            "--grid values must be greater than zero."
        )

    return grid

def _parse_float_pair(raw: str) -> tuple[float, float]:
    """
    Parse two floating-point values formatted as A,B.
    """
    parts = [part.strip() for part in str(raw).split(",")]

    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "Expected two values formatted as A,B, for example: 10,1"
        )

    try:
        values = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Both values must be floating-point numbers."
        ) from exc

    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(
            "Both values must be greater than zero."
        )

    return values

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


def _parse_bool(raw: str) -> bool:
    """
    Parse an explicit true/false CLI value.
    """
    normalized = str(raw).strip().lower()

    if normalized in {"true", "1", "yes", "y"}:
        return True

    if normalized in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(
        "Expected one of: true, false."
    )



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train or fine-tune a patched StarDist model using canonical "
            "manifest-defined train and validation subsets."
        ),
    )

    # Dataset
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_default_dataset_root(),
        help=(
            "Canonical dataset root containing images/, masks/, metadata/, "
            "splits/, and COCO_mask/."
        ),
    )
    parser.add_argument(
        "--split-manifest-path",
        type=Path,
        default=_default_split_manifest_path(),
        help="Path to the canonical split manifest CSV.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on loaded training samples.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optional cap on loaded validation samples.",
    )
    parser.add_argument(
        "--grayscale",
        type=_parse_bool,
        default=False,
        help=(
            "Load images as one channel by selecting channel zero. "
            "Use true or false. Default: false."
        ),
    )

    # Model
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help=(
            "Name of the new StarDist model folder under --model-basedir."
        ),
    )
    parser.add_argument(
        "--model-basedir",
        type=Path,
        default=None,
        help=(
            "Directory containing StarDist model folders. "
            "When omitted in a source-repository workflow, defaults to "
            "<project_root>/models. External library consumers should "
            "provide this path explicitly."
        ),
    )
    parser.add_argument(
        "--initialization-mode",
        choices=(
            "fresh_config",
            "resume_local",
            "official_pretrained",
            "local_pretrained",
        ),
        default="fresh_config",
        help=(
            "Model initialization strategy. "
            "'fresh_config' creates a new randomly initialized local model. "
            "'resume_local' loads an existing local patched model and "
            "continues training it. "
            "'official_pretrained' creates a new local patched model whose "
            "initial weights come from an official StarDist pretrained model. "
            "'local_pretrained' creates a new local patched model whose "
            "initial architecture and weights come from another existing "
            "local model."
            "Default: fresh_config."
        ),
    )
    parser.add_argument(
        "--pretrained-model-name",
        type=str,
        default=None,
        help=(
            "Official StarDist pretrained model used as the source when "
            "--initialization-mode official_pretrained is selected. "
            "For example: 2D_versatile_fluo, 2D_paper_dsb2018, or "
            "2D_versatile_he. "
            "--model-name remains the name of the new local patched model."
        ),
    )
    parser.add_argument(
        "--source-model-name",
        type=str,
        default=None,
        help=(
            "Existing local StarDist model used as the source when "
            "--initialization-mode local_pretrained is selected. "
            "The source model remains unchanged and --model-name specifies "
            "the new local target model."
        ),
    )
    parser.add_argument(
        "--n-rays",
        type=int,
        default=64,
        help="Number of StarDist radial directions. Default: 64.",
    )
    parser.add_argument(
        "--unet-depth",
        type=_parse_positive_int,
        default=None,
        help=(
            "Optional U-Net depth for fresh model construction. "
            "This option is only valid with "
            "--initialization-mode fresh_config. "
            "Loaded or pretrained models preserve their existing architecture."
        ),
    )
    parser.add_argument(
        "--grid",
        type=_parse_grid,
        default=(2, 2),
        help="StarDist grid formatted as Y,X. Default: 2,2.",
    )
    parser.add_argument(
        "--distance-loss",
        choices=("mae", "mse"),
        default="mae",
        help="StarDist distance loss. Default: mae.",
    )
    parser.add_argument(
        "--train-loss-weights",
        type=_parse_float_pair,
        default=(10.0, 1.0),
        help="StarDist probability and distance loss weights.",
    )

    parser.add_argument(
        "--batch-size",
        type=_parse_positive_int,
        default=4,
        help="Training batch size. Default: 4.",
    )

    parser.add_argument(
        "--reduce-lr-factor",
        type=_parse_positive_float,
        default=0.5,
        help="ReduceLROnPlateau reduction factor. Default: 0.5.",
    )

    parser.add_argument(
        "--reduce-lr-patience",
        type=_parse_positive_int,
        default=40,
        help="ReduceLROnPlateau patience in epochs. Default: 40.",
    )

    parser.add_argument(
        "--reduce-lr-min-delta",
        type=float,
        default=0.0,
        help="ReduceLROnPlateau minimum improvement. Default: 0.0.",
    )
    
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable the StarDist gputools preprocessing path.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable StarDist/TensorBoard training logs.",
    )

    # Training
    parser.add_argument(
        "--learning-rate",
        type=_parse_positive_float,
        default=None,
        help=(
            "Optional training learning rate. "
            "When omitted, the learning rate from the created or loaded model "
            "configuration is preserved."
        ),
    )
    parser.add_argument(
        "--trainable-last-n-layers",
        type=_parse_positive_int,
        default=None,
        help=(
            "Optional selective fine-tuning policy. When supplied, all model "
            "layers are frozen except the final N layers. This option is intended "
            "for pretrained or resumed models and is not valid with "
            "--initialization-mode fresh_config. When omitted, existing layer "
            "trainability is left unchanged."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs. Default: 5.",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=10,
        help="Training steps per epoch. Default: 10.",
    )
    parser.add_argument(
        "--train-patch-size",
        type=_parse_grid,
        default=(256, 256),
        help=(
            "Spatial patch size used during training, formatted as Y,X. "
            "It must not exceed the image dimensions. Default: 256,256."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed applied to Python, NumPy, and TensorFlow before "
            "model initialization and training. Default: 42."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        help=(
            "Patched StarDist preprocessing/scoring mode. "
            "Default: standard."
        ),
    )
    parser.add_argument(
        "--disable-augmentation",
        action="store_true",
        help="Train without stochastic data augmentation.",
    )
    parser.add_argument(
        "--debug-plots",
        action="store_true",
        help="Enable patched StarDist preprocessing debug plots.",
    )

    # Run artifacts
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Optional explicit run identifier. When omitted, a detailed "
            "identifier containing key training parameters is generated."
        ),
    )
    parser.add_argument(
        "--global-outputs-root",
        type=Path,
        default=None,
        help=(
            "Root directory for CNTLib generated outputs. "
            "CNTLib manages runs/, reports/, and misc/ beneath this root. "
            "When omitted in a source-repository workflow, defaults to "
            "<project_root>/global_outputs."
        ),
    )
    parser.add_argument(
        "--skip-threshold-optimization",
        action="store_true",
        help=(
            "Skip post-training optimization of the CNT probability "
            "threshold on the validation subset. The CNT workflow does "
            "not optimize the native StarDist NMS threshold."
        ),
    )
    parser.add_argument(
        "--probability-thresholds",
        nargs="+",
        type=float,
        default=[ 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, ],
        help=(
            "Probability-threshold candidates evaluated after training. "
            "Default: 0.05 through 0.65 in steps of 0.05."
        ),
    )
    parser.add_argument(
        "--max-prediction-annotations",
        type=_parse_positive_int,
        default=None,
        help=(
            "Optional resource guard for probability-threshold optimization. "
            "If a candidate produces more than this number of prediction "
            "annotations, expensive metric evaluation is skipped for that "
            "candidate. Default: no limit."
        ),
    )

    parser.add_argument(
        "--threshold-objective",
        type=str,
        default="dsb_map",
        help=(
            "Metric used to recommend the best probability threshold. "
            "Supported examples include: dsb_map, mean_dice, "
            "dsb_ap@0.50, object_f1@0.50, coco_ap, coco_ap50, "
            "and coco_ap75. Default: dsb_map."
        ),
    )

    parser.add_argument(
        "--threshold-apply-smoothing",
        type=_parse_bool,
        default=True,
        help=(
            "Whether polygon smoothing is applied during post-training "
            "probability-threshold optimization. Use true or false. "
            "Default: true."
        ),
    )
    parser.add_argument(
        "--history-overwrite",
        action="store_true",
        help="Allow replacement of an existing training history file.",
    )
    parser.add_argument(
        "--metric-plot-dpi",
        type=int,
        default=300,
        help="DPI for training metric plots. Default: 300.",
    )
    parser.add_argument(
        "--print-filenames",
        action="store_true",
        help=(
            "Print every training and validation filename. "
            "By default, only subset counts are printed."
        ),
    )

    return parser
