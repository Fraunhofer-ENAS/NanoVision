# Dataset Storage and Split Standard

This document defines the canonical dataset layout and the standard procedure used to generate reproducible training, validation, and test splits.

The dataset is stored once in an unsplit form. Split membership is represented separately through a versioned split manifest rather than through physical `train/`, `val/`, and `test/` directories (like we did before).

---

## Canonical Dataset Layout

```text
data/
└── dataset/ (this is a placeholder for now, later we move the children of this folder under data/ directly)
    ├── images/
    │   ├── sample_001.tif
    │   ├── sample_001.jpg
    │   ├── sample_002.tif
    │   └── sample_002.jpg
    │
    ├── masks/
    │   ├── sample_001.tif
    │   ├── sample_002.tif
    │   └── ...
    │
    ├── metadata/
    │   ├── density_classified_filenames.csv
    │   ├── noise_classification.csv
    │   └── ...
    │
    ├── splits/
    │   ├── default_split.csv
    │   ├── default_split.yaml
    │   └── ...
    │
    └── COCO_mask/
        └── <split_name>/
            ├── train/
            │   └── annotations.json
            ├── val/
            │   └── annotations.json
            └── test/
                └── annotations.json
```

---

## Canonical Dataset Rules

* All dataset samples are stored once under `images/` and `masks/`.
* Physical `train/`, `val/`, and `test/` image or mask directories are not part of the canonical dataset.
* Only `.tif` files under `images/` are considered dataset samples.
* `.jpg` files may be retained for visualization or convenience but are ignored during split generation.
* The full filename, including the `.tif` extension, uniquely identifies a sample.
* Each mask must have exactly the same filename as its corresponding `.tif` image.

Example:

```text
images/sample_001.tif
masks/sample_001.tif
```

* Each `.tif` image must have exactly one matching mask.
* Images or masks without a valid corresponding pair must cause dataset validation to fail.
* Split membership must never be inferred from directory contents.

---

## Metadata Standard

Density and noise metadata describe intrinsic properties of the samples and are therefore generated once for the complete dataset.

Examples:

```text
metadata/density_classified_filenames.csv
metadata/noise_classification.csv
```

These metadata files are not regenerated for every split.

When a workflow requires metadata for a particular subset, the full-dataset metadata are filtered using the split manifest.

Conceptually:

```text
full metadata + split manifest + requested subset
    → subset metadata
```

Subset-specific metadata files should only be written when required for compatibility with an existing workflow. The full-dataset metadata files remain the authoritative source.

---

## Required Processing Order

The dataset preparation workflow follows this order:

```text
1. Store all images and masks.
2. Validate image-mask pairing.
3. Generate density metadata for the complete dataset.
4. Generate noise metadata for the complete dataset.
5. Generate a seeded train/val/test split manifest.
6. Generate split-specific COCO JSON files.
7. Train using the train subset.
8. Evaluate using the test subset.
```

Density and noise metadata must be available before split generation because the split should preserve their distributions across the subsets.

---

## Split Membership Standard

Split membership is stored in a CSV manifest under:

```text
data/dataset/splits/
```

Example:

```text
data/dataset/splits/default_split.csv
```

The manifest contains one row per `.tif` sample.

```csv
filename,subset
sample_001.tif,train
sample_002.tif,train
sample_003.tif,val
sample_004.tif,test
```

The supported subset names are standardized as:

```text
train
val
test
```

Alternative names such as `validation`, `valid`, or `testing` should not be used.

The split manifest is the authoritative source of train, validation, and test membership.

---

## Split Reproducibility Metadata

Each split CSV should have a companion YAML file containing the generation parameters.

Example:

```text
data/dataset/splits/default_split.yaml
```

Example content:

```yaml
split_name: default_split
seed: 42

fractions:
  train: 0.70
  val: 0.15
  test: 0.15

sample_identifier: filename
image_extension: ".tif"
mask_pairing: exact_filename

stratification:
  columns:
    - density_class
    - noise_class
  combined_labels: true

sample_counts:
  total: 100
  train: 70
  val: 15
  test: 15
```

The CSV records the exact membership assignments.

The YAML records how those assignments were generated.

The CSV remains authoritative even if the same seed and parameters are later used to reproduce the split.

---

## Stratification Standard

The split should preserve both:

* density-class proportions;
* clean/noisy proportions.

The preferred approach is to create a combined stratification label from density and noise.

Example combined groups:

```text
Low + Clean
Low + Noisy
Mid + Clean
Mid + Noisy
High + Clean
High + Noisy
```

The split algorithm should distribute samples from each combined group across `train`, `val`, and `test` according to the requested proportions.

This preserves the joint density/noise distribution more accurately than stratifying the two variables independently.

If a combined group contains too few samples to be represented in all three subsets, the splitter must not silently ignore the problem. It should either:

* apply a documented fallback strategy; or
* stop and report that exact stratification is not possible.

The generated split report should show the density and noise distributions for the full dataset and for every subset.

---

## Split Generation Rules

The split must be generated as an explicit preprocessing step before training or evaluation.

Training code must not silently generate a new split.

The split generator must accept at least:

* dataset root;
* split name;
* train fraction;
* validation fraction;
* test fraction;
* random seed;
* density metadata path;
* noise metadata path.

The generator must validate that:

* the fractions sum to `1.0`;
* all dataset samples are `.tif` files;
* every image has a mask with the same filename;
* filenames are unique;
* every sample appears in the required metadata;
* every sample receives exactly one subset;
* only `train`, `val`, and `test` labels are used;
* subsets are mutually exclusive;
* all samples are assigned;
* the requested stratification is possible.

The generator should not overwrite an existing split unless an explicit overwrite option is provided.

---

## Training, Validation, and Evaluation Access

Training, validation, inference, and evaluation must request samples using:

```text
dataset root
+ split manifest
+ subset name
```

Example conceptual interface:

```python
train_dataset = CNTDataset(
    dataset_root=paths.dataset_root,
    split_manifest=paths.split_file("default_split.csv"),
    subset="train",
)
```

Validation uses:

```python
subset="val"
```

Evaluation uses:

```python
subset="test"
```

The dataset loader must filter the manifest by the requested subset and resolve the corresponding files from the unified `images/` and `masks/` directories.

The loader must not use directory scanning to determine split membership.

---

## Inference Access

Inference should support two modes.

### Split-based inference

Used for validation or evaluation runs:

```text
dataset root
+ split manifest
+ subset
```

### Explicit-file inference

Used for selected files or external inputs:

```text
explicit list of filenames
```

Inference should not assume that every inference run uses the test subset.

---

## COCO Annotation Standard

COCO annotation JSON files are split-specific artifacts and must be generated after the split manifest exists.

Recommended structure:

```text
COCO_mask/
└── <split_name>/
    ├── train/
    │   └── annotations.json
    ├── val/
    │   └── annotations.json
    └── test/
        └── annotations.json
```

For each subset, the generated JSON must contain:

* only the `images` entries assigned to the subset;
* only annotations associated with those image entries;
* the required category definitions;
* valid image and annotation references.

The split manifest determines which samples are included in each JSON file.

COCO generation must not independently create or modify split membership.

---

## Responsibility Separation

### Dataset storage

Responsible for storing the complete set of images and masks.

```text
images/
masks/
```

### Metadata generation

Responsible for calculating full-dataset sample properties such as density and noise classification.

```text
metadata/
```

### Split generation

Responsible for assigning each filename to `train`, `val`, or `test`.

```text
splits/
```

### Data loading

Responsible for loading samples assigned to a requested subset.

The loader consumes split membership but does not create it.

### COCO generation

Responsible for generating subset-specific COCO annotation files from the masks and split manifest.

---

## Recommended Code Location

Dataset split generation is a preprocessing responsibility and should be implemented under:

```text
src/cnt_project/preprocessing/dataset_splitting/
```

Recommended structure:

```text
src/cnt_project/preprocessing/
├── data_loader.py
│
├── dataset_splitting/
│   ├── __init__.py
│   ├── generate_split.py
│   ├── split_manifest.py
│   ├── stratification.py
│   └── validation.py
│
└── runners/
    └── generate_dataset_split_runner.py
```

Suggested responsibilities:

| File                               | Responsibility                                                      |
| ---------------------------------- | ------------------------------------------------------------------- |
| `generate_split.py`                | Orchestrates seeded train/val/test assignment                       |
| `split_manifest.py`                | Reads, writes, and validates split CSV files                        |
| `stratification.py`                | Builds density/noise stratification groups and performs allocation  |
| `validation.py`                    | Validates images, masks, metadata, fractions, and final assignments |
| `generate_dataset_split_runner.py` | User-facing runner or command-line entry point                      |

The existing data loader should remain responsible only for loading samples. It should be updated later to consume an existing split manifest and requested subset.


## Default Split

The canonical `default_split` reproduces the existing manually defined dataset partition exactly. Its manifest is generated by recording the current train, validation, and test filename assignments inside annotation_uniques.

The train, validation, and test proportions are derived from the recorded assignments. Because the original split was manually defined, its reproducibility is guaranteed by the saved manifest rather than by a random seed.

Future automatically generated splits must use an explicit random seed and must be saved under a separate split name.