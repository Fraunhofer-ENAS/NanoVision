# Dataset Preprocessing Runners

This directory contains command-line runners for generating canonical metadata from the complete unsplit dataset. as well as runners to generate individual metadata files.

The necessary metadata must be generated before creating train, validation, and test split manifests because the splitting process uses density and noise classifications to preserve their distributions across the subsets.

---
## Dataset Layout

The runners expect the following canonical dataset structure:

```text
data/
└── dataset/
    ├── images/
    │   ├── sample_001.tif
    │   ├── sample_001.jpg
    │   ├── sample_002.tif
    │   └── ...
    │
    ├── masks/
    │   ├── sample_001.tif
    │   ├── sample_002.tif
    │   └── ...
    │
    ├── metadata/
    ├── splits/
    └── COCO_mask/
```

Only `.tif` files under `images/` are considered dataset samples.

Each `.tif` image must have a mask with exactly the same filename:

```text
images/sample_001.tif
masks/sample_001.tif
```

Files with other extensions, such as `.jpg`, are ignored by the metadata generators.

---

# Density Metadata Runner

## File

```text
generate_density_metadata.py
```

## Purpose

Generates per-image ground-truth object counts and assigns each sample to a density class:

```text
Low
Mid
High
```

The density metadata are generated from the complete unsplit dataset and written by default to:

```text
<data-set-root>/metadata/density_classified_filenames.csv
```

## Default Output Schema

When comparison columns are enabled:

```text
filename
gt_object_count
density_class_tertile
density_class_kmeans
density_class
```

The `density_class` column contains the canonical classification selected through the runner configuration.

When comparison columns are disabled:

```text
filename
gt_object_count
density_class
```

## Classification Methods

The supported density-classification methods are:

### `kmeans`

Fits three KMeans clusters to the ground-truth object counts and maps the clusters to `Low`, `Mid`, and `High` according to their mean object counts.

This is the default method.

### `tertile`

Sorts the object counts and uses one-third and two-thirds positions as thresholds for assigning `Low`, `Mid`, and `High`.

---

## Usage

From the repository root:

```bash
python -m cnt_project.preprocessing.dataset.runners.generate_density_metadata
```

Specify a dataset root:

```bash
python -m cnt_project.preprocessing.dataset.runners.generate_density_metadata \
    --dataset-root data/dataset
```

Use tertile classification:

```bash
python -m cnt_project.preprocessing.dataset.runners.generate_density_metadata \
    --density-method tertile
```

Specify the KMeans random seed:

```bash
python -m cnt_project.preprocessing.dataset.runners.generate_density_metadata \
    --seed 42
```

Ignore connected components smaller than a selected area:

```bash
python -m cnt_project.preprocessing.dataset.runners.generate_density_metadata \
    --min-mask-area 2
```

Write only the canonical columns:

```bash
python -m cnt_project.preprocessing.dataset.runners.generate_density_metadata \
    --no-comparison-columns
```

Specify an output file:

```bash
python -m cnt_project.preprocessing.dataset.runners.generate_density_metadata \
    --output data/dataset/metadata/custom_density.csv
```

---

## Arguments

| Argument                  |                                                    Default | Description                                                     |
| ------------------------- | ---------------------------------------------------------: | --------------------------------------------------------------- |
| `--dataset-root`          |                                  Repository `data/dataset` | Dataset root containing `images/`, `masks/`, and `metadata/`.   |
| `--output`                | `<dataset-root>/metadata/density_classified_filenames.csv` | Output CSV path.                                                |
| `--density-method`        |                                                   `kmeans` | Canonical density-classification method: `kmeans` or `tertile`. |
| `--seed`                  |                                                       `42` | Random seed used by KMeans.                                     |
| `--min-mask-area`         |                                                        `1` | Minimum connected-component area in pixels.                     |
| `--no-comparison-columns` |                                                   Disabled | Drops the noncanonical comparison columns.                      |

---

# Noise Metadata Runner

## File

```text
generate_noise_metadata.py
```

## Purpose

Estimates image noise using wavelet decomposition and assigns each sample to one of the following classes:

```text
Clean
Noisy
```

The noise metadata are generated from all `.tif` images in the complete unsplit dataset.

The default output path is:

```text
<dataset-root>/metadata/noise_classification.csv
```

## Default Output Schema

When comparison columns are enabled:

```text
filename
noise_sigma
noise_class_otsu
noise_class_kmeans
noise_class
```

The `noise_class` column contains the canonical classification selected through the runner configuration.

When comparison columns are disabled:

```text
filename
noise_sigma
noise_class
```

## Noise Estimation

Noise is estimated from the diagonal detail coefficients of a two-dimensional wavelet decomposition.

The estimated noise value is:

```text
median(abs(diagonal coefficients)) / 0.6745
```

The default wavelet is:

```text
db1
```

## Classification Methods

### `otsu`

Applies Otsu thresholding to the full collection of estimated noise values.

Samples below or equal to the threshold are classified as `Clean`, and samples above the threshold are classified as `Noisy`.

This is the default method.

### `kmeans`

Fits two KMeans clusters to the estimated noise values.

The cluster with the higher mean noise value is assigned `Noisy`, while the other cluster is assigned `Clean`.

---

## Usage

From the repository root:

```bash
python -m cnt_project.preprocessing.dataset.runners.generate_noise_metadata
```

Specify a dataset root:

```bash
python -m cnt_project.preprocessing.dataset.runners.generate_noise_metadata \
    --dataset-root data/dataset
```

Use KMeans classification:

```bash
python -m cnt_project.preprocessing.dataset.runners.generate_noise_metadata \
    --noise-method kmeans
```

Specify the random seed:

```bash
python -m cnt_project.preprocessing.dataset.runners.generate_noise_metadata \
    --seed 42
```

Specify the wavelet:

```bash
python -m cnt_project.preprocessing.dataset.runners.generate_noise_metadata \
    --wavelet db1
```

Require a fixed image shape:

```bash
python -m cnt_project.preprocessing.dataset.runners.generate_noise_metadata \
    --expected-shape 256,256
```

Write only canonical columns:

```bash
python -m cnt_project.preprocessing.dataset.runners.generate_noise_metadata \
    --no-comparison-columns
```

Specify an output file:

```bash
python -m cnt_project.preprocessing.dataset.runners.generate_noise_metadata \
    --output data/dataset/metadata/custom_noise.csv
```

---

## Arguments

| Argument                  |                                            Default | Description                                                   |
| ------------------------- | -------------------------------------------------: | ------------------------------------------------------------- |
| `--dataset-root`          |                          Repository `data/dataset` | Dataset root containing `images/`, `masks/`, and `metadata/`. |
| `--output`                | `<dataset-root>/metadata/noise_classification.csv` | Output CSV path.                                              |
| `--noise-method`          |                                             `otsu` | Canonical noise-classification method: `otsu` or `kmeans`.    |
| `--seed`                  |                                               `42` | Random seed used by KMeans.                                   |
| `--wavelet`               |                                              `db1` | Wavelet used for noise estimation.                            |
| `--expected-shape`        |                                               None | Optional required image shape formatted as `H,W`.             |
| `--no-comparison-columns` |                                           Disabled | Drops the noncanonical comparison columns.                    |

---

# Compatibility Metadata Runner

A combined compatibility runner may also be retained:

```text
generate_dataset_metadata_runner.py
```

It supports the following modes:

```text
density
noise
both
```

Example:

```bash
python -m cnt_project.preprocessing.dataset.runners.generate_dataset_metadata_runner \
    --mode both
```

This runner is provided for convenience and backward compatibility. The dedicated density and noise runners are preferred because they expose clearer responsibilities and allow either metadata file to be regenerated independently.

---

# Validation

Before writing an output CSV, the metadata generators validate that:

* the `images/` directory exists;
* the `masks/` directory exists;
* at least one `.tif` image exists;
* every `.tif` image has a mask with the same filename;
* every `.tif` mask has a corresponding image;
* every dataset sample appears exactly once in the generated metadata;
* no metadata row contains an unknown filename;
* no duplicate metadata rows exist;
* unreadable images or masks cause the process to fail.

Validation errors raise:

```python
DatasetValidationError
```

Metadata files are written only after the generated rows pass dataset coverage validation.

---

# Canonical Columns

Downstream split generation must use only the canonical columns:

```text
density_class
noise_class
```

Comparison columns such as:

```text
density_class_tertile
density_class_kmeans
noise_class_otsu
noise_class_kmeans
```

are diagnostic outputs and must not be used implicitly by the split generator.

The selected canonical method is determined by the runner arguments.

---

# Processing Order

The expected preparation workflow is:

```text
1. Populate images/ and masks/.
2. Validate image-mask pairing.
3. Generate density metadata.
4. Generate noise metadata.
5. Generate the split manifest.
6. Generate subset-specific COCO JSON files.
7. Train on the train subset.
8. Validate on the val subset.
9. Evaluate on the test subset.
```

The metadata generators are split-independent. They always process the complete canonical dataset and do not read train, validation, or test assignments.

---

## objective:
                 Dataset Preparation
             (one-time preprocessing)

                    prepare_dataset()
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
generate density     generate noise     generate split
metadata             metadata           manifest
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                 generate COCO JSONs
                           │
                           ▼
               Prepared dataset artifacts
        ┌────────────────────────────────────┐
        │ metadata/*.csv                     │
        │ splits/*.csv                       │
        │ splits/*.yaml                      │
        │ COCO_mask/train/annotations.json   │
        │ COCO_mask/val/annotations.json     │
        │ COCO_mask/test/annotations.json    │
        └────────────────────────────────────┘
                           │
                           ▼
                     DataLoader
                           │
                           ▼
              Training / Evaluation / Inference



---

# Length Metadata Runner

## File

```text
generate_length_metadata_runner.py
```

## Purpose

Generates canonical CNT length metadata from the instance-indexed TIFF masks.

Unlike the density and noise runners, the length runner can generate two different metadata tables.

### Object-level metadata

One row per CNT instance.

Default output:

```text
<dataset-root>/metadata/object_lengths.csv
```

### Image-level metadata

One row per image generated by aggregating the object-level measurements.

Default output:

```text
<dataset-root>/metadata/image_lengths.csv
```

Supported output modes:

```text
object
image
both
```

The default mode is:

```text
both
```

---

## Length Measurement

Each instance-indexed object is measured independently.

For every object the runner computes:

- object area;
- connected-component count;
- skeleton pixel count;
- geodesic length (pixels);
- geodesic length (micrometres).

When an annotated CNT contains multiple disconnected connected components, each component is measured independently and the resulting geodesic lengths are summed.
