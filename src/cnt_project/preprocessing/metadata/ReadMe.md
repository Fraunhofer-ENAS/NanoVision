# Dataset Metadata Generation

This package generates full-dataset metadata used for dataset characterization and stratified train/validation/test splitting.

---

## Package Structure

```text
src/cnt_project/preprocessing/
├── metadata/
│   ├── __init__.py
│   ├── density.py
│   ├── noise.py
│   ├── length.py
│   ├── schemas.py
│   ├── validation.py
│   └── ReadMe.md
│
└── runners/
    ├── generate_dataset_metadata_runner.py
    ├── generate_density_metadata_runner.py
    └── generate_noise_metadata_runner.py
    └── generate_length_metadata_runner.py

```

---

## Expected Dataset Layout

The metadata generators expect an unsplit dataset with the following structure:

```text
data/
└── cnt_segmentation/
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

Only `.tif` files are considered dataset samples.

Each image must have a mask with exactly the same filename:

```text
images/sample_001.tif
masks/sample_001.tif
```

Files with other extensions, such as `.jpg`, are ignored during metadata generation.

---

# Processing Order

The expected dataset-preparation workflow is:

```text
1. Populate images/ and masks/.
2. Validate image-mask pairing.
3. Generate density metadata.
4. Generate noise metadata.
5. Generate the split manifest.
6. Generate split-specific COCO JSON files.
7. Train using the train subset.
8. Validate using the val subset.
9. Evaluate using the test subset.
```

Density and noise metadata must be generated before a stratified split because the split generator uses these classifications to preserve their distributions across `train`, `val`, and `test`.

---

# Shared Dataset Rules

The metadata generators enforce the following rules:

* Only `.tif` image files are processed.
* Every `.tif` image must have exactly one corresponding mask.
* Every mask must have a corresponding image.
* The full filename, including `.tif`, is the sample identifier.
* Every generated metadata table must contain exactly one row per dataset sample.
* Duplicate filename rows are rejected.
* Missing samples are rejected.
* Unknown filenames are rejected.
* Unreadable images or masks cause metadata generation to fail.
* Metadata are generated for the complete dataset and are independent of split membership.

---

# Density Metadata

## Implementation

```text
src/cnt_project/preprocessing/metadata/density.py
```

## Purpose

Density metadata describe the number of ground-truth CNT instances in each image and assign each sample to one of the following density classes:

```text
Low
Mid
High
```

The default output file is:

```text
<dataset-root>/metadata/density_classified_filenames.csv
```

---

## Ground-Truth Object Count

Ground-truth object counts are computed from the instance IDs encoded in each mask.

The mask convention is:

```text
0             = background
non-zero ID   = one CNT instance
```

Each unique non-zero mask value is counted as one object.

An optional minimum area threshold may be applied. Instance IDs occupying fewer than `min_area_px` pixels are excluded from the count.

Conceptually:

```text
ground-truth object count
    =
number of unique non-zero instance IDs
whose pixel area is at least min_area_px
```

The current implementation expects a single-channel instance-indexed TIFF mask.

---

## Density Classification Methods

Two classification methods are computed.

### Tertile classification

Samples are sorted by their ground-truth object count.

The values around one-third and two-thirds of the ordered counts are used as thresholds for assigning:

```text
Low
Mid
High
```

The resulting column is:

```text
density_class_tertile
```

### KMeans classification

A three-cluster KMeans model is fitted to the ground-truth object counts.

The clusters are ordered by their mean count and mapped to:

```text
lowest mean count   → Low
middle mean count   → Mid
highest mean count  → High
```

The resulting column is:

```text
density_class_kmeans
```

KMeans uses a configurable random seed.

---

## Canonical Density Column

The selected classification method is copied into:

```text
density_class
```

For example:

* `method="tertile"` uses `density_class_tertile`.
* `method="kmeans"` uses `density_class_kmeans`.

The split generator may also explicitly select one of the comparison columns, such as:

```text
density_class_tertile
```

The selected column must contain only:

```text
Low
Mid
High
```

---

## Density Output Schema

With comparison columns enabled:

```csv
filename,gt_object_count,density_class_tertile,density_class_kmeans,density_class
sample_001.tif,24,Low,Low,Low
sample_002.tif,67,Mid,Mid,Mid
sample_003.tif,103,High,High,High
```

With comparison columns disabled:

```csv
filename,gt_object_count,density_class
sample_001.tif,24,Low
sample_002.tif,67,Mid
sample_003.tif,103,High
```

---

## Density Runner

```text
src/cnt_project/preprocessing/runners/generate_density_metadata_runner.py
```

Run from the repository root:

```bash
python -m cnt_project.preprocessing.runners.generate_density_metadata_runner
```

Example with explicit arguments:

```bash
python -m cnt_project.preprocessing.runners.generate_density_metadata_runner \
    --dataset-root data/cnt_segmentation \
    --density-method kmeans \
    --seed 42 \
    --min-mask-area 1
```

### Density runner arguments

| Argument                  |                                                    Default | Description                                                         |
| ------------------------- | ---------------------------------------------------------: | ------------------------------------------------------------------- |
| `--dataset-root`          |                                    Repository dataset path | Root containing `images/`, `masks/`, and `metadata/`.               |
| `--output`                | `<dataset-root>/metadata/density_classified_filenames.csv` | Output CSV path.                                                    |
| `--density-method`        |                                                   `kmeans` | Canonical method: `kmeans` or `tertile`.                            |
| `--seed`                  |                                                       `42` | Random seed used by KMeans.                                         |
| `--min-mask-area`         |                                                        `1` | Minimum number of pixels required for an instance ID to be counted. |
| `--no-comparison-columns` |                                                   Disabled | Writes only the canonical density columns.                          |

---

# Noise Metadata

## Implementation

```text
src/cnt_project/preprocessing/metadata/noise.py
```

## Purpose

Noise metadata estimate the image noise level and classify each sample as:

```text
Clean
Noisy
```

The default output file is:

```text
<dataset-root>/metadata/noise_classification.csv
```

---

## Wavelet Noise Estimation

Noise is estimated using a one-level two-dimensional wavelet decomposition.

If the input image has multiple channels, it is first converted to grayscale.

The estimator uses the diagonal detail coefficients:

```text
noise_sigma
    =
median(abs(diagonal detail coefficients)) / 0.6745
```

The default wavelet is:

```text
db1
```

The output `noise_sigma` value is computed once for every `.tif` image.

---

## Noise Classification Methods

Two classification methods are computed.

### Otsu classification

Otsu thresholding is applied to the complete set of noise sigma values.

Samples are classified as:

```text
noise_sigma <= threshold  → Clean
noise_sigma > threshold   → Noisy
```

The resulting column is:

```text
noise_class_otsu
```

### KMeans classification

A two-cluster KMeans model is fitted to the noise sigma values.

The cluster with the higher mean sigma is mapped to:

```text
Noisy
```

The remaining cluster is mapped to:

```text
Clean
```

The resulting column is:

```text
noise_class_kmeans
```

KMeans uses a configurable random seed.

---

## Canonical Noise Column

The selected classification method is copied into:

```text
noise_class
```

For example:

* `method="otsu"` uses `noise_class_otsu`.
* `method="kmeans"` uses `noise_class_kmeans`.

The split generator may also explicitly select one of the comparison columns, such as:

```text
noise_class_otsu
```

The selected column must contain only:

```text
Clean
Noisy
```

---

## Noise Output Schema

With comparison columns enabled:

```csv
filename,noise_sigma,noise_class_otsu,noise_class_kmeans,noise_class
sample_001.tif,0.031,Clean,Clean,Clean
sample_002.tif,0.097,Noisy,Noisy,Noisy
```

With comparison columns disabled:

```csv
filename,noise_sigma,noise_class
sample_001.tif,0.031,Clean
sample_002.tif,0.097,Noisy
```

---

## Image Shape Validation

The noise generator can optionally enforce an expected image shape.

For example:

```text
256 × 256
```

When an expected shape is configured, any image with a different height or width causes metadata generation to fail.

Image-shape validation is disabled when no expected shape is supplied.

---

## Noise Runner

```text
src/cnt_project/preprocessing/runners/generate_noise_metadata_runner.py
```

Run from the repository root:

```bash
python -m cnt_project.preprocessing.runners.generate_noise_metadata_runner
```

Example with explicit arguments:

```bash
python -m cnt_project.preprocessing.runners.generate_noise_metadata_runner \
    --dataset-root data/cnt_segmentation \
    --noise-method otsu \
    --seed 42 \
    --wavelet db1 \
    --expected-shape 256,256
```

### Noise runner arguments

| Argument                  |                                            Default | Description                                           |
| ------------------------- | -------------------------------------------------: | ----------------------------------------------------- |
| `--dataset-root`          |                            Repository dataset path | Root containing `images/`, `masks/`, and `metadata/`. |
| `--output`                | `<dataset-root>/metadata/noise_classification.csv` | Output CSV path.                                      |
| `--noise-method`          |                                             `otsu` | Canonical method: `otsu` or `kmeans`.                 |
| `--seed`                  |                                               `42` | Random seed used by KMeans.                           |
| `--wavelet`               |                                              `db1` | Wavelet used for noise estimation.                    |
| `--expected-shape`        |                                               None | Optional expected shape formatted as `H,W`.           |
| `--no-comparison-columns` |                                           Disabled | Writes only the canonical noise columns.              |


# Length Metadata

## Implementation

```text
src/cnt_project/preprocessing/metadata/length.py
```

## Purpose

Length metadata characterize the morphology of every ground-truth CNT instance contained in the instance-indexed TIFF masks.

Unlike density and noise metadata, length metadata are produced at two different levels:

- object-level metadata (one row per CNT instance);
- image-level metadata (one row per image).

The object-level metadata are the canonical measurements. The image-level metadata are derived by aggregating the complete object-level table.

The default output files are:

```text
<dataset-root>/metadata/object_lengths.csv
<dataset-root>/metadata/image_lengths.csv
```

---

## Object-Level Measurements

Each non-background instance ID in an instance-indexed TIFF mask is measured independently.

For every object the following measurements are recorded:

- instance identifier;
- object area;
- number of connected components;
- skeleton pixel count;
- geodesic length in pixels;
- geodesic length in micrometres.

---

## Geodesic Length

The canonical CNT length is computed from the object's skeleton.

For every connected skeleton component:

1. the component is skeletonized;
2. the weighted geodesic path is computed for that component;
3. the component geodesic length is measured.

If a single annotated CNT consists of multiple disconnected connected components, every component is measured independently and the resulting geodesic lengths are summed to obtain the final object length.

This approach is robust to fragmented annotations while remaining consistent with the canonical feature extraction pipeline.

---

## Object-Level Output Schema

```csv
filename,
instance_id,
area_px,
connected_component_count,
skeleton_pixel_count,
geodesic_length_px,
geodesic_length_um
```

---

## Image-Level Aggregation

Image-level metadata are generated by aggregating the object-level measurements for every image.

Each image contains one row describing:

- number of objects;
- area statistics;
- skeleton statistics;
- geodesic-length statistics;
- annotation fragmentation statistics.

---

## Image-Level Output Schema

```csv
filename,
gt_object_count,

total_area_px,
mean_area_px,
median_area_px,
max_area_px,

mean_skeleton_pixel_count,
median_skeleton_pixel_count,
max_skeleton_pixel_count,

mean_geodesic_length_px,
median_geodesic_length_px,
max_geodesic_length_px,
p95_geodesic_length_px,

mean_geodesic_length_um,
median_geodesic_length_um,
max_geodesic_length_um,
p95_geodesic_length_um,

multi_component_object_count,
multi_component_object_fraction,
max_connected_component_count
```

---

# Combined Metadata Runner

## Implementation

```text
src/cnt_project/preprocessing/runners/generate_dataset_metadata_runner.py
```

The combined runner can generate:

```text
density metadata only
noise metadata only
both metadata files
```

using:

```text
--mode density
--mode noise
--mode both
```

Example:

```bash
python -m cnt_project.preprocessing.runners.generate_dataset_metadata_runner \
    --dataset-root data/cnt_segmentation \
    --mode both \
    --density-method kmeans \
    --noise-method otsu \
    --seed 42 \
    --wavelet db1 \
    --min-mask-area 1 \
    --expected-shape 256,256
```

The dedicated density and noise runners are preferred when only one metadata file needs to be regenerated.

---

# Shared Metadata Schema

## Implementation

```text
src/cnt_project/preprocessing/metadata/schemas.py
```

The shared schema constants are:

```text
filename

gt_object_count
density_class
density_class_tertile
density_class_kmeans

noise_sigma
noise_class
noise_class_otsu
noise_class_kmeans
```

Allowed density values:

```text
Low
Mid
High
```

Allowed noise values:

```text
Clean
Noisy
```

These constants provide a stable interface between:

* metadata generation;
* split generation;
* metadata validation;
* downstream dataset reporting.

---

# Metadata Validation

## Implementation

```text
src/cnt_project/preprocessing/metadata/validation.py
```

The shared validation utilities verify:

* the required image directory exists;
* the required mask directory exists;
* at least one `.tif` image exists;
* every `.tif` image has a matching mask;
* every `.tif` mask has a matching image;
* filenames match exactly;
* generated metadata cover the complete dataset;
* metadata contain no unknown filenames;
* metadata contain no duplicate filename rows;
* dataset sample identifiers are unique.

Validation failures raise:

```python
DatasetValidationError
```

Metadata files are written only after coverage validation succeeds.

---

# Relationship to Split Generation

Density and noise metadata describe intrinsic properties of the complete dataset.

They are generated once and are not regenerated for each split.

The split generator reads the selected density and noise columns and constructs a combined stratification group, for example:

```text
Low + Clean
Low + Noisy
Mid + Clean
Mid + Noisy
High + Clean
High + Noisy
```

The generated split manifest then assigns each filename to:

```text
train
val
test
```

Metadata remain global. Subset-specific metadata can be obtained by filtering the full metadata tables using the split manifest.

---

# VS Code Launch Configurations

Example density configuration:

```json
{
  "name": "Generate density metadata",
  "type": "debugpy",
  "request": "launch",
  "module": "cnt_project.preprocessing.runners.generate_density_metadata_runner",
  "cwd": "${workspaceFolder}",
  "env": {
    "PYTHONPATH": "${workspaceFolder}/src"
  },
  "console": "integratedTerminal",
  "justMyCode": true,
  "args": [
    "--dataset-root", "data/cnt_segmentation",
    "--density-method", "kmeans",
    "--seed", "42",
    "--min-mask-area", "1"
  ]
}
```

Example noise configuration:

```json
{
  "name": "Generate noise metadata",
  "type": "debugpy",
  "request": "launch",
  "module": "cnt_project.preprocessing.runners.generate_noise_metadata_runner",
  "cwd": "${workspaceFolder}",
  "env": {
    "PYTHONPATH": "${workspaceFolder}/src"
  },
  "console": "integratedTerminal",
  "justMyCode": true,
  "args": [
    "--dataset-root", "data/cnt_segmentation",
    "--noise-method", "otsu",
    "--seed", "42",
    "--wavelet", "db1",
    "--expected-shape", "256,256"
  ]
}
```

---
