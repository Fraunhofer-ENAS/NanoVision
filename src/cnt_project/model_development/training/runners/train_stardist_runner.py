"""
Canonical StarDist training runner.

This runner orchestrates training or fine-tuning of a patched StarDist model
using train and validation subsets defined by a canonical split manifest.

Current scope
-------------
- Load canonical manifest-defined training and validation data.
- Initialize a fresh patched StarDist model.
- Resume an existing local patched StarDist model.
- Initialize a new local model from official pretrained StarDist weights.
- Initialize a new local model from another local pretrained model.
- Apply training-time model configuration and fine-tuning policy.
- Validate model/data channel compatibility.
- Apply the canonical StarDist augmentation function.
- Create a detailed run identifier.
- Save training status, history, manifests, and metric plots.

The runner intentionally does not:
- generate dataset metadata or split manifests;
- create COCO ground-truth files;
- load or evaluate the test subset;
- run inference or evaluation.
"""

from __future__ import annotations

import random
from datetime import datetime

import numpy as np
from stardist import gputools_available
from dataclasses import dataclass
from pathlib import Path
import tensorflow as tf
from typing import Any

from cnt_project.io.paths import ProjectPaths
from cnt_project.io.run_manager import (
    build_run_name,
    create_run,
    create_run_from_outputs_root,
    make_run_id,
)

from cnt_project.model_development.training.dataset_augmentation import augmenter
from cnt_project.model_development.training.training_artifacts import (
    save_training_history,
    save_training_manifest,
    save_training_metric_plots,
    save_training_status,
)

from cnt_project.model_development.training.training_data import (
    load_training_data_bundle,
)
from cnt_project.model_development.training.train_stardist_cli import (
    build_arg_parser,
)
from cnt_project.model_development.threshold_optimization.optimizer import (
    metric_for_evaluation,
    optimize_probability_threshold,
    save_probability_threshold,
)
from cnt_project.model_development.model_compat import check_model_compatibility
from cnt_project.model_development.training.stardist_training import (
    initialize_stardist_model,
    prepare_stardist_model_for_training,
    validate_model_directory_for_initialization,
    validate_stardist_training_arguments,
)


@dataclass(frozen=True)
class TrainingRunResult:
    run_id: str
    run_dir: Path
    train_dir: Path
    model_name: str
    model_dir: Path
    history_path: Path
    training_manifest_path: Path
    status_path: Path
    metric_paths: tuple[Path, ...]
    history: Any

def _resolve_run_id(
        *,
        explicit_run_name: str | None,
        model_name: str,
        initialization_mode: str,
        mode: str,
        n_rays: int,
        grid: tuple[int, int],
        epochs: int,
        steps_per_epoch: int,
        seed: int,
    ) -> str:
        """
        Resolve either an explicit run name or a detailed generated training ID.
        """
        if explicit_run_name is not None:
            return build_run_name( model_name=model_name, run_name=explicit_run_name, )

        grid_tag = "x".join( str(value) for value in grid )

        return make_run_id(
            model_name,
            f"init-{initialization_mode}",
            f"mode-{mode}",
            f"nrays-{n_rays}",
            f"grid-{grid_tag}",
            f"epochs-{epochs}",
            f"steps-{steps_per_epoch}",
            f"seed-{seed}",
        )



def run_stardist_training(args) -> TrainingRunResult:
    """
    Run the canonical StarDist training workflow.

    Parameters
    ----------
    args
        Parsed training arguments compatible with
        ``train_stardist_cli.build_arg_parser()``.

    Returns
    -------
    TrainingRunResult
        Paths and artifacts produced by the completed training run.

    Notes
    -----
    This function contains the reusable training workflow.

    CLI runners should parse their arguments separately and delegate the
    actual training execution to this function.
    """
    validate_stardist_training_arguments(
        initialization_mode=args.initialization_mode,
        pretrained_model_name=args.pretrained_model_name,
        source_model_name=args.source_model_name,
        unet_depth=args.unet_depth,
        trainable_last_n_layers=args.trainable_last_n_layers,
    )

    if args.n_rays <= 0:
        raise ValueError(
            f"n_rays must be greater than zero, got {args.n_rays}."
        )

    if args.epochs <= 0:
        raise ValueError(
            f"epochs must be greater than zero, got {args.epochs}."
        )

    if args.steps_per_epoch <= 0:
        raise ValueError(
            "steps_per_epoch must be greater than zero, "
            f"got {args.steps_per_epoch}."
        )

    for threshold in args.probability_thresholds:
        if not 0.0 < float(threshold) < 1.0:
            raise ValueError(
                "All probability-threshold candidates must be strictly "
                f"between 0 and 1. Got: {threshold}"
            )

    if args.use_gpu and not gputools_available():
        raise RuntimeError(
            "--use-gpu was requested, but StarDist gputools support "
            "is not available in the current environment."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Resolve repository defaults when needed
    project_paths: ProjectPaths | None = None

    if args.global_outputs_root is None:
        project_paths = ProjectPaths.from_here(
            __file__
        )
        project_paths.ensure_outputs()

    # Resolve dataset inputs
    dataset_root = args.dataset_root.resolve()
    split_manifest_path = args.split_manifest_path.resolve()
    split_name = split_manifest_path.stem

    val_gt_json_path = ( dataset_root / "COCO_mask" / split_name / "val" / "annotations_chain_approx_none.json" )

    if not val_gt_json_path.exists():
        raise FileNotFoundError(
            "Validation COCO ground-truth JSON not found: "
            f"{val_gt_json_path}"
        )

    if args.model_basedir is not None:
        model_basedir = args.model_basedir.resolve()

    elif project_paths is not None:
        model_basedir = project_paths.models_root

    else:
        raise ValueError(
            "--model-basedir is required when "
            "--global-outputs-root is supplied."
        )

    model_basedir.mkdir(
        parents=True,
        exist_ok=True,
    )
    model_basedir.mkdir( parents=True, exist_ok=True, )

    model_dir = model_basedir / args.model_name

    validate_model_directory_for_initialization( model_dir, initialization_mode=args.initialization_mode, )

    training_data = load_training_data_bundle(
        dataset_root=dataset_root,
        split_manifest_path=split_manifest_path,
        grayscale=args.grayscale,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    X_train = training_data.X_train
    Y_train = training_data.Y_train
    X_val = training_data.X_val
    Y_val = training_data.Y_val

    train_filenames = training_data.train_filenames
    val_filenames = training_data.val_filenames

    train_n_channels = training_data.n_channels

    initialized_model = initialize_stardist_model(
        initialization_mode=args.initialization_mode,
        model_name=args.model_name,
        model_basedir=model_basedir,
        train_n_channels=train_n_channels,
        n_rays=args.n_rays,
        grid=args.grid,
        use_gpu=args.use_gpu,
        distance_loss=args.distance_loss,
        train_loss_weights=args.train_loss_weights,
        batch_size=args.batch_size,
        reduce_lr_factor=args.reduce_lr_factor,
        reduce_lr_patience=args.reduce_lr_patience,
        reduce_lr_min_delta=args.reduce_lr_min_delta,
        epochs=args.epochs,
        tensorboard=args.tensorboard,
        train_patch_size=args.train_patch_size,
        learning_rate=args.learning_rate,
        unet_depth=args.unet_depth,
        pretrained_model_name=args.pretrained_model_name,
        source_model_name=args.source_model_name,
    )

    prepared_model = prepare_stardist_model_for_training(
        initialized_model=initialized_model,
        learning_rate=args.learning_rate,
        train_loss_weights=args.train_loss_weights,
        batch_size=args.batch_size,
        reduce_lr_factor=args.reduce_lr_factor,
        reduce_lr_patience=args.reduce_lr_patience,
        reduce_lr_min_delta=args.reduce_lr_min_delta,
        trainable_last_n_layers=args.trainable_last_n_layers,
    )

    model = prepared_model.model

    official_source_model = ( prepared_model.official_source_model )

    local_source_model = ( prepared_model.local_source_model )

    effective_learning_rate = ( prepared_model.effective_learning_rate )

    effective_unet_depth = ( prepared_model.effective_unet_depth )

    model_n_channels = ( prepared_model.model_n_channels )

    trainability_summary = ( prepared_model.trainability )

    check_model_compatibility(
        grayscale=args.grayscale,
        model_name=args.model_name,
        dataset_n_channels=train_n_channels,
        model_n_channels=model_n_channels,
    )
    
    if args.initialization_mode == "resume_local":
        print("")
        print("Loaded existing model configuration")
        print("-----------------------------------")
        print(f"Stored n_rays:         {model.config.n_rays}")
        print(f"Stored grid:           {tuple(model.config.grid)}")
        print(f"Stored U-Net depth:    {model.config.unet_n_depth}")
        print(f"Stored distance loss:  {model.config.train_dist_loss}")
        print(f"Stored input channels: {model.config.n_channel_in}")
        print("")

    if args.initialization_mode == "official_pretrained":
        assert official_source_model is not None

        print("")
        print("Loaded official pretrained source")
        print("---------------------------------")
        print(f"Source model:          {args.pretrained_model_name}")
        print(f"Source n_rays:         {official_source_model.config.n_rays}")
        print(f"Source grid:           {tuple(official_source_model.config.grid)}")
        print(
            f"Source distance loss:  "
            f"{official_source_model.config.train_dist_loss}"
        )
        print(
            f"Source input channels: "
            f"{official_source_model.config.n_channel_in}"
        )
        print(f"Local model name:      {args.model_name}")
        print( f"Source U-Net depth:   {official_source_model.config.unet_n_depth}" )
        print("")

    if args.initialization_mode == "local_pretrained":
        assert local_source_model is not None

        print("")
        print("Loaded local pretrained source")
        print("------------------------------")
        print(f"Source model:          {args.source_model_name}")
        print(f"Source n_rays:         {local_source_model.config.n_rays}")
        print(f"Source grid:           {tuple(local_source_model.config.grid)}")
        print(
            f"Source U-Net depth:    "
            f"{local_source_model.config.unet_n_depth}"
        )
        print(
            f"Source distance loss:  "
            f"{local_source_model.config.train_dist_loss}"
        )
        print(
            f"Source input channels: "
            f"{local_source_model.config.n_channel_in}"
        )
        print(f"Local model name:      {args.model_name}")
        print("")

    run_id = _resolve_run_id(
        explicit_run_name=args.run_name,
        model_name=args.model_name,
        initialization_mode=args.initialization_mode,
        mode=args.mode,
        n_rays=int(model.config.n_rays),
        grid=tuple(int(value) for value in model.config.grid),
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        seed=args.seed,
    )

    if args.global_outputs_root is not None:
        run = create_run_from_outputs_root(
            outputs_root=args.global_outputs_root,
            run_id=run_id,
        )

    else:
        assert project_paths is not None

        run = create_run(
            project_root=project_paths.project_root,
            run_id=run_id,
        )
    training_started_at = ( datetime.now() .astimezone() .isoformat(timespec="seconds") )

    current_phase = "saving_training_manifest"

    status_path = save_training_status(
        run.train_dir,
        run_id=run.run_id,
        model_name=args.model_name,
        initialization_mode=args.initialization_mode,
        status="running",
        phase=current_phase,
        started_at=training_started_at,
    )

    try:
        training_manifest_path = save_training_manifest(
            run.train_dir,
            dataset_root=dataset_root,
            split_manifest_path=split_manifest_path,
            model_name=args.model_name,
            model_basedir=model_basedir,
            initialization_mode=args.initialization_mode,
            pretrained_model_name=args.pretrained_model_name,
            source_model_name=args.source_model_name,
            train_filenames=train_filenames,
            val_filenames=val_filenames,
            max_train_samples=args.max_train_samples,
            train_loss_weights=tuple( float(value) for value in model.config.train_loss_weights ),
            batch_size=int( model.config.train_batch_size ),
            reduce_lr=dict( model.config.train_reduce_lr ),
            max_val_samples=args.max_val_samples,
            grayscale=args.grayscale,
            dataset_n_channels=train_n_channels,
            model_n_channels=model_n_channels,
            n_rays=int(model.config.n_rays),
            train_patch_size=tuple( int(value) for value in model.config.train_patch_size ),
            grid=tuple( int(value) for value in model.config.grid ),
            distance_loss=str(model.config.train_dist_loss),
            training_mode=args.mode,
            learning_rate=effective_learning_rate,
            unet_depth=effective_unet_depth,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            seed=args.seed,
            trainable_last_n_layers=( trainability_summary.trainable_last_n_layers ),
            total_model_layers=trainability_summary.total_layers,
            trainable_model_layers=trainability_summary.trainable_layers,
            frozen_model_layers=trainability_summary.frozen_layers,
            augmentation_enabled=not args.disable_augmentation,
            threshold_optimization_enabled=(
                not args.skip_threshold_optimization
            ),
        )

        print("")
        print("Training configuration")
        print("----------------------")
        print(f"Run ID:               {run.run_id}")
        print( f"Global outputs root:  " f"{run.run_dir.parent.parent}" )
        print(f"Run directory:        {run.run_dir}")
        print(f"Training artifacts:   {run.train_dir}")
        print(f"Model directory:      {model_dir}")
        print(f"Initialization mode:  {args.initialization_mode}")
        print(f"Dataset root:         {dataset_root}")
        print(f"Split manifest:       {split_manifest_path}")
        print(f"Training manifest:    {training_manifest_path}")
        print(f"Status file:          {status_path}")
        print(f"Training samples:     {len(X_train)}")
        print(f"Validation samples:   {len(X_val)}")
        print(f"Dataset channels:     {train_n_channels}")
        print(
            f"Training patch size:  "
            f"{tuple(model.config.train_patch_size)}"
        )
        print(f"Model channels:       {model_n_channels}")
        print(f"Grayscale:            {args.grayscale}")
        print(f"Training mode:        {args.mode}")
        print(f"n_rays:               {model.config.n_rays}")
        print(f"Grid:                 {tuple(model.config.grid)}")
        print(f"U-Net depth:          {effective_unet_depth}")
        print(f"Distance loss:        {model.config.train_dist_loss}")
        print(f"Learning rate:        {effective_learning_rate}")
        print(f"Epochs this run:      {args.epochs}")
        print(f"Steps per epoch:      {args.steps_per_epoch}")
        print( f"Loss weights:         " f"{tuple(model.config.train_loss_weights)}" )
        print( f"Batch size:           " f"{model.config.train_batch_size}" )
        print( f"Reduce LR:            " f"{model.config.train_reduce_lr}" )
        print(f"Trainable last layers:{trainability_summary.trainable_last_n_layers}" )
        print(f"Trainable layers:     {trainability_summary.trainable_layers}/" f"{trainability_summary.total_layers}" )
        print(f"Frozen layers:        {trainability_summary.frozen_layers}" )
        print(f"Augmentation:         {not args.disable_augmentation}")
        print(f"Debug plots:          {args.debug_plots}")
        print("")

        if args.print_filenames:
            print("Training filenames:")
            for filename in train_filenames:
                print(f"  {filename}")

            print("Validation filenames:")
            for filename in val_filenames:
                print(f"  {filename}")

        train_kwargs: dict[str, object] = {
            "validation_data": (
                X_val,
                Y_val,
            ),
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "mode": args.mode,
            "debug_plots": args.debug_plots,
            "debug_run_name": run.run_id,
        }

        if not args.disable_augmentation:
            train_kwargs["augmenter"] = augmenter

        current_phase = "training"

        save_training_status(
            run.train_dir,
            run_id=run.run_id,
            model_name=args.model_name,
            initialization_mode=args.initialization_mode,
            status="running",
            phase=current_phase,
            started_at=training_started_at,
        )

        print(
            "Trainable weight tensors: "
            f"{len(model.keras_model.trainable_weights)}"
        )
        print(
            "Non-trainable weight tensors: "
            f"{len(model.keras_model.non_trainable_weights)}"
        )

        history = model.train(
            X_train,
            Y_train,
            **train_kwargs,
        )


        if not args.skip_threshold_optimization:
            current_phase = "optimizing_probability_threshold"

            save_training_status(
                run.train_dir,
                run_id=run.run_id,
                model_name=args.model_name,
                initialization_mode=args.initialization_mode,
                status="running",
                phase=current_phase,
                started_at=training_started_at,
            )

            print("")
            print("Optimizing CNT probability threshold")
            print("------------------------------------")
            print( "Validation samples: " f"{len(X_val)}" )
            print( "Objective: " f"{args.threshold_objective}" )
            print( "Candidate thresholds: " f"{args.probability_thresholds}" )

            threshold_predictions_dir = ( run.inference_dir / "threshold_optimization" )

            threshold_results_dir = ( run.eval_dir / "threshold_optimization" )

            threshold_result = optimize_probability_threshold(
                thresholds=args.probability_thresholds,
                model=model,
                images=list(X_val),
                filenames=list(val_filenames),
                gt_json_path=val_gt_json_path,
                predictions_dir=threshold_predictions_dir,
                results_dir=threshold_results_dir,
                objective=args.threshold_objective,
                max_prediction_annotations=args.max_prediction_annotations,
                apply_smoothing=args.threshold_apply_smoothing,
            )

            print("")
            print("Threshold candidate summary")
            print("---------------------------")
            print(
                f"Requested candidates: {threshold_result.n_requested}"
            )
            print(
                f"Evaluated candidates: {threshold_result.n_evaluated}"
            )
            print(
                f"Skipped candidates:   {threshold_result.n_skipped}"
            )

            evaluated_metrics = [
                metric_for_evaluation(
                    evaluation,
                    objective=threshold_result.objective,
                )
                for evaluation in threshold_result.evaluations
                if evaluation.status == "evaluated"
            ]

            unique_metrics = {
                round(float(metric), 12)
                for metric in evaluated_metrics
            }

            thresholds_path: Path | None = None

            if threshold_result.n_evaluated < 2:
                print("")
                print("Threshold optimization inconclusive")
                print("-----------------------------------")
                print(
                    "Fewer than two threshold candidates completed metric evaluation."
                )
                print(
                    "The model thresholds.json will not be modified."
                )

            elif len(unique_metrics) == 1:
                print("")
                print("Threshold optimization inconclusive")
                print("-----------------------------------")
                print(
                    "All evaluated threshold candidates produced the same objective value."
                )
                print(
                    "The model thresholds.json will not be modified."
                )

            else:
                thresholds_path = save_probability_threshold(
                    model=model,
                    model_dir=model_dir,
                    probability_threshold=(
                        threshold_result.best_threshold
                    ),
                )

            print("")
            print("Threshold optimization completed")
            print("--------------------------------")
            print(
                "Best probability threshold: "
                f"{threshold_result.best_threshold:.6f}"
            )
            print(
                "Best objective metric:      "
                f"{threshold_result.best_metric:.6f}"
            )
            print(
                "Objective:                  "
                f"{threshold_result.objective}"
            )
            print(
                "Threshold results:          "
                f"{threshold_result.results_csv_path}"
            )

            if thresholds_path is None:
                print(
                    "Saved model thresholds:     not modified"
                )
            else:
                print(
                    "Saved model thresholds:     "
                    f"{thresholds_path}"
                )

        else:
            print(
                "Probability-threshold optimization skipped; "
                "the model's existing/default probability threshold "
                "will be preserved."
            )
        
        current_phase = "saving_training_artifacts"

        save_training_status(
            run.train_dir,
            run_id=run.run_id,
            model_name=args.model_name,
            initialization_mode=args.initialization_mode,
            status="running",
            phase=current_phase,
            started_at=training_started_at,
        )

        history_path = save_training_history(
            run.train_dir,
            history,
            overwrite=args.history_overwrite,
        )

        metric_paths = save_training_metric_plots(
            history,
            run.train_dir / "metrics",
            dpi=args.metric_plot_dpi,
        )

        current_phase = "completed"

        status_path = save_training_status(
            run.train_dir,
            run_id=run.run_id,
            model_name=args.model_name,
            initialization_mode=args.initialization_mode,
            status="completed",
            phase=current_phase,
            started_at=training_started_at,
        )

    except Exception as error:
        save_training_status(
            run.train_dir,
            run_id=run.run_id,
            model_name=args.model_name,
            initialization_mode=args.initialization_mode,
            status="failed",
            phase=current_phase,
            started_at=training_started_at,
            error=error,
        )

        raise
    print("")
    print("Training completed")
    print("------------------")
    print(f"Model directory:     {model_dir}")
    print(f"Training history:    {history_path}")
    print(f"Metric plots saved:  {len(metric_paths)}")
    print(f"Training manifest:   {training_manifest_path}")
    print(f"Status file:         {status_path}")
    print(f"Run directory:       {run.run_dir}")

    return TrainingRunResult(
        run_id=run.run_id,
        run_dir=run.run_dir,
        train_dir=run.train_dir,
        model_name=args.model_name,
        model_dir=model_dir,
        history_path=history_path,
        training_manifest_path=training_manifest_path,
        status_path=status_path,
        metric_paths=tuple(metric_paths),
        history=history,
    )

def main() -> None:
    """
    CLI entry point for canonical StarDist training.
    """
    args = build_arg_parser().parse_args()
    run_stardist_training(args)

if __name__ == "__main__":
    main()