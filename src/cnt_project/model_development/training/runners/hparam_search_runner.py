from __future__ import annotations

import argparse
from pathlib import Path
import json
import pandas as pd

from cnt_project.io.paths import ProjectPaths
from cnt_project.model_development.training.hparam_search import (
    HParamTrialResult,
    iter_hparam_combinations,
    run_hparam_trial,
    select_best_trial,
)
from cnt_project.model_development.training.training_data import (
    load_training_data_bundle,
)


def _save_hparam_search_manifest(
    *,
    output_path: Path,
    search_name: str,
    source_model_name: str,
    dataset_root: Path,
    split_manifest_path: Path,
    training_data,
    learning_rates: list[float],
    trainable_values: list[int | None],
    train_patch_size: tuple[int, int],
    epochs: int,
    steps_per_epoch: int,
    training_mode: str,
    augmentation_enabled: bool,
    results: list[HParamTrialResult],
    best_result: HParamTrialResult,
) -> Path:
    payload = {
        "search": {
            "search_name": search_name,
            "source_model_name": source_model_name,
            "trial_count": len(results),
        },
        "dataset": {
            "dataset_root": str(dataset_root),
            "split_manifest_path": str(split_manifest_path),
            "train_sample_count": len(training_data.X_train),
            "validation_sample_count": len(training_data.X_val),
            "train_filenames": training_data.train_filenames,
            "validation_filenames": training_data.val_filenames,
            "grayscale": training_data.grayscale,
            "n_channels": training_data.n_channels,
        },
        "search_space": {
            "learning_rates": [
                float(value)
                for value in learning_rates
            ],
            "trainable_last_n_layers": [
                (
                    None
                    if value is None
                    else int(value)
                )
                for value in trainable_values
            ],
        },
        "training": {
            "train_patch_size": list(train_patch_size),
            "epochs_per_trial": int(epochs),
            "steps_per_epoch": int(steps_per_epoch),
            "mode": training_mode,
            "augmentation_enabled": augmentation_enabled,
        },
        "best_result": {
            "model_name": best_result.model_name,
            "learning_rate": best_result.trial.learning_rate,
            "trainable_last_n_layers": (
                best_result.trial.trainable_last_n_layers
            ),
            "validation_loss": best_result.validation_loss,
            "training_time_s": best_result.training_time_s,
            "model_dir": str(best_result.model_dir),
        },
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    return output_path

def _parse_trainable_values(
    values: list[str],
) -> list[int | None]:
    parsed: list[int | None] = []

    for value in values:
        normalized = value.strip().lower()

        if normalized == "all":
            parsed.append(None)
            continue

        try:
            number = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "trainable-last-n-layers values must be positive integers "
                "or 'all'."
            ) from exc

        if number <= 0:
            raise argparse.ArgumentTypeError(
                "trainable-last-n-layers values must be greater than zero."
            )

        parsed.append(number)

    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a canonical StarDist local-pretrained hyperparameter search."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/cnt_segmentation"),
    )

    parser.add_argument(
        "--split-manifest-path",
        type=Path,
        default=Path(
            "data/cnt_segmentation/splits/default_split.csv"
        ),
    )

    parser.add_argument(
        "--source-model-name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--search-name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model-basedir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--trainable-last-n-layers",
        nargs="+",
        default=["7"],
        help=(
            "Values for the fine-tuning search. Use positive integers "
            "or 'all' to leave all layers trainable."
        ),
    )

    parser.add_argument(
        "--train-patch-size",
        type=str,
        default="256,256",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
    )

    parser.add_argument(
        "--grayscale",
        action="store_true",
    )

    parser.add_argument(
        "--disable-augmentation",
        action="store_true",
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
    )

    return parser


def _parse_patch_size(raw: str) -> tuple[int, int]:
    parts = [
        part.strip()
        for part in raw.split(",")
    ]

    if len(parts) != 2:
        raise ValueError(
            "--train-patch-size must be formatted as Y,X."
        )

    patch_size = (
        int(parts[0]),
        int(parts[1]),
    )

    if any(value <= 0 for value in patch_size):
        raise ValueError(
            "--train-patch-size values must be greater than zero."
        )

    return patch_size


def _results_to_dataframe(
    results: list[HParamTrialResult],
) -> pd.DataFrame:
    rows = []

    for result in results:
        rows.append(
            {
                "model_name": result.model_name,
                "learning_rate": result.trial.learning_rate,
                "trainable_last_n_layers": (
                    result.trial.trainable_last_n_layers
                ),
                "validation_loss": result.validation_loss,
                "training_time_s": result.training_time_s,
                "model_dir": str(result.model_dir),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be greater than zero."
        )

    if args.steps_per_epoch <= 0:
        raise ValueError(
            "--steps-per-epoch must be greater than zero."
        )

    trainable_values = _parse_trainable_values(
        args.trainable_last_n_layers
    )

    train_patch_size = _parse_patch_size(
        args.train_patch_size
    )

    paths = ProjectPaths.from_here(__file__)

    model_basedir = (
        args.model_basedir.resolve()
        if args.model_basedir is not None
        else paths.project_root / "models"
    )

    model_basedir.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_data = load_training_data_bundle(
        dataset_root=args.dataset_root.resolve(),
        split_manifest_path=args.split_manifest_path.resolve(),
        grayscale=args.grayscale,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    trials = iter_hparam_combinations(
        learning_rates=args.learning_rates,
        trainable_last_n_layers_values=trainable_values,
    )

    print("")
    print("Hyperparameter search")
    print("---------------------")
    print(f"Search name:        {args.search_name}")
    print(f"Source model:       {args.source_model_name}")
    print(f"Number of trials:   {len(trials)}")
    print(f"Training samples:   {len(training_data.X_train)}")
    print(f"Validation samples: {len(training_data.X_val)}")
    print("")

    results: list[HParamTrialResult] = []

    for trial_index, trial in enumerate(
        trials,
        start=1,
    ):
        print(
            f"[{trial_index}/{len(trials)}] "
            f"lr={trial.learning_rate}, "
            f"trainable_last_n_layers="
            f"{trial.trainable_last_n_layers}"
        )

        result = run_hparam_trial(
            trial=trial,
            trial_index=trial_index,
            search_name=args.search_name,
            training_data=training_data,
            model_basedir=model_basedir,
            source_model_name=args.source_model_name,
            train_patch_size=train_patch_size,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            training_mode=args.mode,
            augmentation_enabled=(
                not args.disable_augmentation
            ),
        )

        results.append(result)

        print(
            f"Validation loss: {result.validation_loss:.6f}"
        )

    best_result = select_best_trial(
        results
    )

    results_df = _results_to_dataframe(
        results
    )

    output_csv = (
        args.output_csv.resolve()
        if args.output_csv is not None
        else paths.misc_dir(
            "hparam_search"
        )
        / f"{args.search_name}_results.csv"
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_csv,
        index=False,
    )
    manifest_path = ( output_csv.parent / f"{args.search_name}_manifest.json" )
    manifest_path = _save_hparam_search_manifest(
        output_path=manifest_path,
        search_name=args.search_name,
        source_model_name=args.source_model_name,
        dataset_root=args.dataset_root.resolve(),
        split_manifest_path=args.split_manifest_path.resolve(),
        training_data=training_data,
        learning_rates=list(args.learning_rates),
        trainable_values=trainable_values,
        train_patch_size=train_patch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        training_mode=args.mode,
        augmentation_enabled=(
            not args.disable_augmentation
        ),
        results=results,
        best_result=best_result,
    )

    print("")
    print("Hyperparameter search completed")
    print("-------------------------------")
    print(f"Results CSV:        {output_csv}")
    print(f"Best model:         {best_result.model_name}")
    print(
        "Best learning rate: "
        f"{best_result.trial.learning_rate}"
    )
    print(
        "Best trainable last layers: "
        f"{best_result.trial.trainable_last_n_layers}"
    )
    print(
        f"Best validation loss: "
        f"{best_result.validation_loss:.6f}"
    )
    print(f"Search manifest:     {manifest_path}")


if __name__ == "__main__":
    main()