# Cluster Training Scripts

This folder contains thin SLURM launch wrappers for the canonical StarDist
training and hyperparameter-search entrypoints under `src/cnt_project`.

The Python modules under `cnt_project.model_development.training` own the
training logic. The scripts in this folder only provide cluster scheduling
configuration and forward command-line arguments to those modules.

## Prerequisites

- Run commands from the repository root.
- The project should be installed in the active Python environment, preferably
  with:

  ```bash
  pip install -e .
  ```

- If the project is not installed on the cluster, expose the source package
  before submitting jobs:

  ```bash
  export PYTHONPATH=./src
  ```

- GPU-related SLURM directives in these wrappers may need to be adapted to the
  target cluster configuration.

## Scripts

### 1. StarDist training

Script:

```text
src/cnt_project/cluster_training/run_train.sh
```

This wrapper calls:

```text
cnt_project.model_development.training.runners.train_stardist_runner
```

The canonical training runner supports:

- fresh model construction;
- resuming an existing local model;
- initialization from an official StarDist pretrained model;
- initialization from another local pretrained model;
- configurable learning rate;
- configurable U-Net depth for fresh model construction;
- selective fine-tuning of the final N layers;
- canonical manifest-defined train and validation subsets;
- training manifests, status files, history files, and metric plots.

Example:

```bash
sbatch src/cnt_project/cluster_training/run_train.sh \
  --dataset-root data/cnt_segmentation \
  --split-manifest-path data/cnt_segmentation/splits/default_split.csv \
  --model-name stardist_cluster_train \
  --initialization-mode local_pretrained \
  --source-model-name stardist_fresh_smoke_test \
  --learning-rate 0.0001 \
  --trainable-last-n-layers 7 \
  --train-patch-size 256,256 \
  --epochs 100 \
  --steps-per-epoch 100 \
  --mode standard
```

See the full CLI contract with:

```bash
python -m cnt_project.model_development.training.runners.train_stardist_runner --help
```

---

### 2. StarDist hyperparameter search

Script:

```text
src/cnt_project/cluster_training/run_hparam_search.sh
```

This wrapper calls:

```text
cnt_project.model_development.training.runners.hparam_search_runner
```

The current canonical hyperparameter-search implementation performs
fine-tuning searches using an existing local pretrained model.

The currently supported search dimensions are:

- learning rate;
- number of trainable final layers;
- all layers trainable by using `all`.

Each trial creates a separate local model initialized from the specified source
model.

Example:

```bash
sbatch src/cnt_project/cluster_training/run_hparam_search.sh \
  --dataset-root data/cnt_segmentation \
  --split-manifest-path data/cnt_segmentation/splits/default_split.csv \
  --source-model-name stardist_fresh_smoke_test \
  --search-name stardist_cluster_hparam_search \
  --learning-rates 0.0001 0.0002 \
  --trainable-last-n-layers 7 all \
  --train-patch-size 256,256 \
  --epochs 20 \
  --steps-per-epoch 100 \
  --mode standard
```

The search writes a tabular result summary and a reproducibility manifest under:

```text
global_outputs/misc/hparam_search/
```

See the full CLI contract with:

```bash
python -m cnt_project.model_development.training.runners.hparam_search_runner --help
```

---

### 3. W&B StarDist hyperparameter search

Script:

```text
src/cnt_project/cluster_training/run_train_wandb.sh
```

This wrapper calls:

```text
cnt_project.model_development.training.runners.wandb_hparam_search_runner
```

The W&B runner uses the same canonical StarDist fine-tuning implementation as
the local hyperparameter-search runner. W&B is responsible only for sweep
orchestration, experiment tracking, visualization, and optional model artifact
upload.

The current W&B search supports:

- learning-rate sweeps;
- selective fine-tuning of the final N layers;
- all-layer fine-tuning;
- canonical manifest-defined train and validation subsets;
- per-epoch training and validation metric logging;
- W&B learning curves for all metrics returned by StarDist;
- optional training-image and ground-truth-mask logging;
- optional trained-model artifact upload;
- local CSV result summaries;
- local search manifests containing W&B sweep and run identifiers.

Example:

```bash
sbatch src/cnt_project/cluster_training/run_train_wandb.sh \
  --dataset-root data/cnt_segmentation \
  --split-manifest-path data/cnt_segmentation/splits/default_split.csv \
  --source-model-name stardist_fresh_smoke_test \
  --search-name stardist_wandb_cluster_search \
  --learning-rates 0.0001 0.0002 \
  --trainable-last-n-layers 7 all \
  --train-patch-size 256,256 \
  --epochs 20 \
  --steps-per-epoch 100 \
  --wandb-project stardist-training \
  --wandb-mode online
```

Optional W&B features include:

```text
--log-sample-images
--sample-image-index 0
--log-model-artifacts
```

By default, local W&B runtime files are stored under:

```text
global_outputs/misc/wandb_cache/
```

A different location can be supplied explicitly with:

```text
--wandb-dir <PATH>
```

Local project-owned W&B search artifacts are stored separately under:

```text
global_outputs/misc/wandb_hparam_search/<SEARCH_NAME>/
```

Each search directory contains:

```text
results.csv
manifest.json
```

See the full CLI contract with:

```bash
python -m cnt_project.model_development.training.runners.wandb_hparam_search_runner --help
```

## Output Contracts

### Standard training

Canonical training invocation artifacts are written under:

```text
global_outputs/runs/<RUN_ID>/train/
```

Typical contents include:

```text
training_manifest.json
status.json
history.json
metrics/
```

Trained StarDist model directories are stored under:

```text
models/<MODEL_NAME>/
```

### Local hyperparameter search

Local fine-tuning HPO summaries are written under:

```text
global_outputs/misc/hparam_search/
```

Each HPO trial creates its own model under:

```text
models/<TRIAL_MODEL_NAME>/
```

### W&B hyperparameter search

Local W&B runtime/cache files are written under:

```text
global_outputs/misc/wandb_cache/
```

Project-owned W&B HPO summaries are written under:

```text
global_outputs/misc/wandb_hparam_search/<SEARCH_NAME>/
```

Each search contains:

```text
results.csv
manifest.json
```

The manifest records the dataset and split used, the search space, training
configuration, W&B sweep and run identifiers, and the selected best trial.

## Scope

The current hyperparameter-search runners focus on fine-tuning parameters that
do not modify the pretrained model architecture.

Architecture-changing searches such as sweeping `n_rays`, U-Net depth, kernel
size, dropout, or other structural StarDist parameters are not part of the
current fine-tuning HPO contract and should be handled separately through a
fresh-model architecture-search workflow.