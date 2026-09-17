# CNT Features

## Scope

## Package Structure

## Core

The `core/` package contains the canonical CNT morphology calculations and a small number of specialized geometry implementations that are reused by higher-level feature pipelines, preprocessing metadata generation, postprocessing, and paper-analysis code.

The main design rule is:

> Feature definitions belong in `core/`. Pipelines and runners may orchestrate them, but should not redefine morphology formulas.

### `cnt_feature_extraction.py`

Canonical CNT feature extraction module.

This is the primary implementation used for feature extraction from standard COCO polygon annotations representing filled CNT instances.

It provides:

- COCO polygon annotation decoding into binary masks.
- Object-level morphology extraction.
- Image-level aggregation.
- Row-wise line density.
- CNT density per physical image area.
- Orientation-distribution statistics.

The canonical object-level feature definitions are:

- **Area**
  - Foreground pixel count of the decoded annotation mask.
  - Exported in px² and µm².

- **Perimeter**
  - Crofton perimeter calculated on the filled binary mask.
  - Exported in pixels and µm.

- **Length**
  - Geodesic centerline length.
  - The centerline path is selected using the weighted pixel-graph implementation in `geodesic_length.py`.
  - The reported length is the geometric length of the selected path.

- **Width**
  - Twice the mean Euclidean Distance Transform value sampled along the geodesic centerline.

- **Aspect ratio**
  - `length / width`.

- **Orientation**
  - PCA on the original COCO polygon coordinates.
  - Image-space coordinates are converted to the project orientation convention:
    - `0°` = horizontal/right
    - `90°` = vertical/up
    - positive angles = counterclockwise
    - normalized to `[-90°, 90°]`.

- **Connected components**
  - Number of connected components in the decoded annotation mask.

Image-level outputs additionally include:

- number of CNT objects,
- CNT density per µm²,
- full row-wise line-density profile,
- mean/std/max line density,
- mean values for object-level morphology,
- nematic order parameter,
- nematic director angle,
- von Mises orientation-distribution statistics.

This file is the source of truth for standard filled-polygon CNT morphology.

Higher-level modules should import these calculations rather than implementing alternative versions.

---

### `geodesic_length.py`

Implements the canonical mask-based CNT centerline length.

The method operates on a filled binary CNT mask and constructs an 8-connected pixel graph.

The workflow is:

1. Crop the mask to its bounding box.
2. Build an 8-connected graph over foreground pixels.
3. Weight graph edges using the local Euclidean Distance Transform so that paths closer to the object center are preferred over boundary paths.
4. Select allowed start pixels using the historical notebook shell convention.
5. Search for the longest shortest path under the weighted graph cost.
6. Reconstruct the selected path.
7. Report the geometric path length rather than the weighted graph cost.

The final reported length uses:

- orthogonal step = `1 px`
- diagonal step = `sqrt(2) px`

The weighted cost therefore determines **which centerline path is selected**, while the returned CNT length remains a geometric pixel length.

The module can optionally return:

- the selected centerline coordinates,
- weighted graph cost,
- start and goal coordinates,
- step statistics,
- debug information.

Primary users include:

- `cnt_feature_extraction.py`
- preprocessing length-metadata generation.

---

### `coco_centerline_morphology.py`

Specialized morphology extractor for centerline-style COCO predictions, primarily Nano1D-style outputs.

This file is intentionally separate from `cnt_feature_extraction.py` because its input representation is geometrically different.

For normal CNT polygon predictions, a COCO segmentation represents a filled object boundary.

For Nano1D-style predictions, the segmentation coordinates may instead represent an ordered centerline/polyline.

Because of this difference, this module uses centerline-specific definitions:

- **Length**
  - Sum of traced polyline segment lengths.
  - Undirected edges are counted at most once to avoid artificial `A -> B -> A` retracing inflation.

- **Orientation**
  - PCA directly on the segmentation coordinates.

- **Area**
  - Mask-derived only when decoding is possible and retained mainly for compatibility.

- **Width, perimeter and aspect ratio**
  - Not considered valid centerline-native features and therefore returned as `NaN`.

This module should not be used as a replacement for the canonical filled-polygon morphology pipeline.

Use it only when the annotation representation itself is centerline-based.

---

### `shapely_polygon_pca_orientation.py`

Preserves the historical Shapely-polygon PCA orientation implementation used by the polygon-merging postprocessing code.

This module operates directly on:

- `shapely.geometry.Polygon`
- `shapely.geometry.MultiPolygon`

It does not decode COCO annotations or use raster masks.

The historical calculation is intentionally preserved for reproducibility:

1. Read polygon exterior coordinates.
2. Keep the repeated closing coordinate.
3. Center the vertices.
4. Fit `sklearn.decomposition.PCA`.
5. Use the first principal component.
6. Convert its direction to degrees.
7. Normalize to `[-90°, 90°]`.

For `MultiPolygon`, each component angle is calculated independently and the arithmetic mean is returned.

Important: this is a **legacy/reproducibility orientation implementation**.

The canonical standard CNT feature orientation is defined in `cnt_feature_extraction.py`.

Primary external use:

- polygon-merging conditions in model postprocessing.

This implementation should not be modified unless changing historical postprocessing results is explicitly accepted.

---

### `polygon_geometry_paper_figures.py`

Paper-analysis helper containing the historical minimum-rotated-rectangle geometry calculation used by some correlation and operational-limit figures.

It computes:

- rectangle-based length,
- rectangle-based width,
- elongation ratio,

from COCO polygon coordinates and can aggregate those values per image.

This is **not the canonical CNT length/width definition**.

The current canonical definitions are:

- length = geodesic centerline length,
- width = mean EDT diameter along the geodesic centerline.

This module remains because existing paper-analysis notebooks and scripts may still depend on the historical rectangle-based measurements.

It should be treated as analysis-specific compatibility code rather than a general morphology API.

A future cleanup may migrate the remaining paper figures to the canonical feature definitions and remove this module.

---

### `documentation.md`

Detailed technical documentation for morphology feature definitions.

This file complements the package-level `ReadMe.md`.

The top-level README documents package architecture and module responsibilities, while `documentation.md` should remain focused on the mathematical and implementation definitions of individual CNT morphology features.

## Comparison

The `comparison/` package contains the logic used to compare CNT feature distributions between ground truth and one or more prediction models.

Its scope is deliberately separate from `core/`:

> `core/` defines how CNT features are measured.  
> `comparison/` defines how those extracted feature distributions are compared.

The comparison package therefore depends on the canonical feature definitions from:

`features/core/cnt_feature_extraction.py`
and should not implement independent morphology calculations.
The typical flow is:

```text
GT COCO JSON
Prediction COCO JSON(s)
        |
        v
Canonical feature extraction
        |
        v
Object / image feature distributions
        |
        v
Distribution comparison metrics
        |
        v
Per-image model comparison
        |
        v
Aggregated model statistics / CSV outputs
```
### comparison_pipeline.py

High-level orchestration for comparing ground-truth CNT features against predictions from one or more models.

The pipeline uses the canonical feature extractor to construct comparable feature representations for GT and prediction annotations. In particular, build_image_result() calculates both row-wise line density and the standard CNT morphology metrics through cnt_feature_extraction.py.

The default compared feature distributions are:

- line density,
- length in µm,
- width in µm,
- aspect ratio,
- orientation angle.

For each image, the pipeline:

- Retrieves the GT annotations.
- Extracts canonical GT features.
- Retrieves the corresponding prediction annotations for each model.
- Extracts canonical prediction features.
- Compares the selected GT and prediction distributions.
- Records whether each model produced predictions.
- Applies explicit penalty values when predictions or valid feature distributions are unavailable.

The per-image comparison is handled by compare_one_image_against_models(). Depending on configuration, it uses either the basic or enhanced distribution-comparison API.

The nested per-image results are then converted into tabular outputs and completed so that every expected:

(image, model, feature)

combination exists. Missing combinations receive penalty rows rather than silently disappearing from aggregate statistics.

The pipeline also calculates:

- mean and standard deviation of Wasserstein distance per model/feature,
- mean and standard deviation of Jensen-Shannon distance,
- percentage of missing predictions,
- pivoted wide-format performance tables,
- model-level summary statistics,
- prediction coverage per model.

The main entry point is:

run_model_comparison_pipeline(...)

which performs the complete workflow:
```text
load GT
    |
load model prediction JSONs
    |
compare every image against every model
    |
flatten comparison records
    |
insert missing penalty records
    |
aggregate model/feature statistics
    |
write detailed and summary outputs
```

The generated outputs include:

- detailed per-image/per-model/per-feature comparison CSV,
- aggregated performance statistics CSV,
- pivoted performance CSV,
- model-level summary CSV,
- human-readable comparison summary text.

The module also contains a smaller helper for preparing one GT image and its model predictions for downstream feature-distribution visualization.

### feature_distribution_comparison.py

Mid-level helpers for preparing and summarizing feature-distribution comparisons.

This module sits between canonical feature extraction and the lower-level statistical distance functions.

Its responsibilities include:

- retrieving GT and prediction annotations for a specific image,
- calculating canonical CNT features for both annotation sets,
- calculating row-wise line-density distributions,
- packaging those results into a shared comparison structure,
- converting individual distribution comparisons into DataFrames,
- stacking several feature comparisons into one summary table.

The main input-building flow is:
```text
GT JSON + prediction JSON + image_id
                |
                v
get annotations for image
                |
                v
canonical CNT feature extraction
                |
                +---- line-density distribution
                |
                +---- morphology feature distributions
                |
                v
shared GT/pred comparison dictionary
```

build_distribution_inputs_for_image() returns both the original annotations and their derived canonical features:

- gt_annotations
- pred_annotations
- line_density_gt
- line_density_pred
- metrics_gt
- metrics_pred

This makes the helper useful for notebooks or analysis workflows that need both the raw annotations and their derived feature distributions.

build_distribution_comparison_table() applies the canonical distribution-statistics calculation to one GT/prediction feature pair and returns a one-row DataFrame.

build_multi_distribution_summary() repeats that operation for several named features and concatenates the results into a single comparison table.

safe_get_metric_values() is a small defensive helper for reading list-valued feature arrays from the canonical feature dictionaries.

This module should remain focused on comparison input preparation and lightweight orchestration. Statistical definitions themselves belong in feature_distribution_metrics.py.

### feature_distribution_metrics.py

Canonical statistical comparison functions for one-dimensional GT and prediction feature distributions.

This file defines the statistical meaning of a "feature distribution comparison" used by the rest of the comparison/ package.

It does not extract CNT morphology and does not know about COCO annotations.

Its input is simply:

- reference numeric distribution
- predicted numeric distribution
- Basic comparison

compare_distributions() provides the compact comparison used by the main model-comparison pipeline.

It calculates:

- Wasserstein distance,
- Jensen-Shannon distance.

Before comparison, NaN values are removed.

The output is a dictionary identified by the feature name.

Enhanced comparison

compare_distributions_enhanced() extends the comparison with descriptive statistics and hypothesis tests.

It includes:

- GT and prediction sample counts,
- mean,
- standard deviation,
- median,
- Wasserstein distance,
- Jensen-Shannon distance,
- KL divergence,
- Kolmogorov-Smirnov statistic and p-value,
- KS significance flag,
- Mann-Whitney U statistic and p-value for sufficiently large samples,
- Mann-Whitney significance flag.

This API is useful for deeper exploratory analysis but is not required for the standard comparison pipeline.

DataFrame-oriented comparison

compute_distribution_stats() provides a compact DataFrame-oriented interface.

It returns one row containing:

- mean_true
- std_true
- mean_pred
- std_pred
- wasserstein_distance
jensen_shannon_distance

This is useful for notebook analyses and summary-table generation where a tabular result is more convenient than the dictionary returned by compare_distributions().

Missing-prediction penalties

The module also defines explicit penalty values:

MAX_WASSERSTEIN = 1000.0
MAX_JENSEN_SHANNON = 1.0

build_penalty_result() constructs the standardized comparison record used when:

a model produced no prediction for an image, or
a feature comparison cannot be calculated because one of the required distributions is empty.

This prevents missing predictions from being accidentally excluded from aggregate model statistics.

### Comparison architecture

The three files have intentionally different responsibilities:
```text
feature_distribution_metrics.py
        |
        | statistical definitions
        v
feature_distribution_comparison.py
        |
        | feature-pair preparation / tabular helpers
        v
comparison_pipeline.py
        |
        | multi-image + multi-model orchestration
        v
CSV / summary outputs
```

The dependency direction should remain:
```text
comparison_pipeline
        |
        +--> feature_distribution_metrics
        |
        +--> cnt_feature_extraction

feature_distribution_comparison
        |
        +--> feature_distribution_metrics
        |
        +--> cnt_feature_extraction
```

The reverse direction should not occur: core/ feature extraction must remain independent of model-comparison logic.


One thing I like about the current structure is that `comparison_pipeline.py` is now genuinely a pipeline rather than just a collection of metrics: it has the complete model-comparison lifecycle from loading predictions through enforcing complete `(image, model, metric)` coverage to writing aggregate outputs. 


## Pipelines

### `prediction_feature_extraction.py`

Reusable pipeline for extracting canonical CNT features from prediction COCO JSON files.

The pipeline:

- groups prediction annotations by image,
- delegates morphology calculation to `core/cnt_feature_extraction.py`,
- produces an object-level DataFrame with one row per predicted CNT,
- produces an image-level DataFrame containing image metrics and mean object features.

The module defines the feature columns exported by the prediction pipeline through:

- `OBJECT_FEATURE_KEYS`
- `IMAGE_FEATURE_KEYS`

It does **not** define morphology formulas. All standard CNT morphology calculations must remain in `core/cnt_feature_extraction.py`.

The pipeline is independent of experiment output paths, plotting, and CLI handling; those responsibilities belong to `runners/` and `plotting/`.



## Plotting
## Plotting

The `plotting/` package contains visualization helpers for extracted CNT features and feature-debug workflows.

Plotting modules may consume canonical feature outputs, but they should not define or modify CNT morphology calculations.

---

### `histograms.py`

Provides reusable histogram generation for tabular feature outputs.

The main helper:

```python
plot_feature_table_histograms(...)
```
creates one histogram per selected feature column and saves both PNG and SVG versions.

It is used by the prediction feature extraction runner for object-level and image-level feature distributions.

### prediction_feature_debug.py
Provides debug visualizations for validating prediction feature extraction.

The module reads prediction COCO annotations, reuses the canonical feature extractor, and generates two main diagnostics:

- object overlays showing predicted contours together with extracted area, length, width, and orientation,
- line-density row diagnostics showing which CNT objects contribute to a selected image row.

The debug workflow also handles image lookup across configured dataset roots and can fall back to a blank canvas when the source image cannot be found.

The public entry point is:
```text
build_prediction_feature_debug_visualizations(...)
```
which selects a limited number of prediction images, computes canonical features, and saves the debug figures under:

```text
debug_visualizations/
  object_overlays/
  line_density_rows/
```

This module is intended for inspection and validation only; feature definitions remain in core/cnt_feature_extraction.py.

### utils.py

Contains small plotting-specific utilities shared across visualization modules.

Currently this includes:
```text
slugify_plot_name(...)
```

which converts plot labels and image names into filesystem-safe output filename components.

Utilities in this module should remain generic to plotting and should not contain feature extraction logic.

## Runners

The `runners/` package contains executable entry points that connect feature-processing pipelines to the project run/output structure.

Runners may handle:

- CLI arguments,
- run and output path resolution,
- CSV/JSON writing,
- optional plotting and debug outputs,
- user-facing progress messages.

They should not define new CNT morphology calculations.

### `extract_pred_cnt_features_runner.py`

CLI runner for extracting canonical CNT features from a prediction run.

The runner:

- locates `predicted_annotations_poly.json` through `ProjectPaths`,
- calls `pipelines/prediction_feature_extraction.py`,
- writes object-level and image-level feature CSVs,
- generates object-level and image-level feature histograms,
- optionally generates feature-debug visualizations,
- writes a JSON summary describing the generated outputs.

The main dependency flow is:

```text
prediction run
    |
    v
predicted_annotations_poly.json
    |
    v
prediction_feature_extraction.py
    |
    +--> object_level_features.csv
    |
    +--> image_level_features.csv
    |
    +--> plotting/histograms.py
    |
    +--> plotting/prediction_feature_debug.py
    |
    v
feature_extraction_summary.json
```


## Legacy
### mask_morphology_legacy.py
currently kept to support legacy evaluator eval_loader_masks.py should be delted or updated once the runner is migrated.

