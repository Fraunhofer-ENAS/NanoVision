# Postprocessing Visualization

This folder contains visualization and debugging utilities for the CNT polygon postprocessing workflow.

Current structure:

```text
visualization/
├── polygon_plot_util.py
├── postprocessing_debug_plots.py
└── __init__.py
```

The responsibility of this folder is to render intermediate and final polygon states without defining the underlying postprocessing algorithm.

The visualization code should therefore remain observational: it should inspect and display pipeline state without changing inference results.

---

# `polygon_plot_util.py`

This module contains the general polygon plotting utilities used throughout postprocessing, inference, and some evaluation workflows.

Its main responsibility is visualization of Shapely polygon predictions together with optional metadata such as:

```text
scores
candidate center points
StarDist rays
orientation vectors
```

The main functions are:

```python
generate_gradient_colors(...)
plot_polygon(...)
plot_orientation_vectors(...)
plot_polygons(...)
plot_merged_polygon(...)
```

---

## General polygon plotting flow

The main plotting path is:

```text
list of Shapely polygons
        │
        ▼
generate_gradient_colors(...)
        │
        ▼
plot_polygon(...)
        │
        ├── polygon exterior
        ├── polygon holes
        └── MultiPolygon components
        │
        ▼
optional annotations
        ├── center points
        ├── confidence scores
        ├── StarDist rays
        └── orientation vectors
        │
        ▼
plot_polygons(...)
        │
        ▼
PNG + SVG output
```

---

## `generate_gradient_colors(...)`

Despite the historical function name, this helper currently obtains colors from Matplotlib's:

```python
tab10
```

colormap and cycles through those colors when more than ten polygons are plotted.

Its purpose is simply to make neighboring polygon instances visually distinguishable.

It is an internal helper used by:

```python
plot_polygons(...)
```

---

## `plot_polygon(...)`

This helper renders a single:

```text
Polygon
```

or:

```text
MultiPolygon
```

onto an existing Matplotlib axis.

The visualization includes:

```text
filled polygon exterior
black polygon boundary
white polygon holes
```

The polygon is simplified with:

```python
polygon.simplify(
    0.1,
    preserve_topology=True,
)
```

before rendering.

This simplification is local to visualization. It does not replace or modify the polygon objects used by the actual postprocessing pipeline.

That distinction is important:

```text
pipeline simplification
    changes the polygon used for merging

plot_polygon simplification
    changes only the geometry used for rendering
```

---

## `plot_orientation_vectors(...)`

This helper renders the orientation of a polygon as a short line originating from the polygon centroid.

Input:

```text
polygon
orientation angle in degrees
Matplotlib axis
```

The orientation angle is expected to have already been calculated elsewhere.

The plotting module therefore does not calculate CNT orientation itself.

The dependency direction remains:

```text
features/core/
    calculates orientation
        │
        ▼
postprocessing pipeline
    carries orientation metadata
        │
        ▼
visualization/
    renders orientation
```

---

# `plot_polygons(...)`

This is the main general-purpose polygon visualization function.

It supports plotting:

```text
polygons
polygon center points
ray distances
confidence scores
orientation angles
```

and can optionally flip the vertical axis to match image-coordinate conventions.

The function is used in several different contexts, including:

```text
postprocessing pipeline diagnostics
iterative polygon-merging diagnostics
inference result visualization
evaluation runners
interactive notebooks
```

It therefore acts as the common polygon visualization utility for the current model-development workflow.

---

## Polygon rendering

Each polygon receives a distinguishable color and is rendered using:

```python
plot_polygon(...)
```

The function also reports basic polygon-boundary statistics such as:

```text
average polygon length
maximum polygon length
```

These values are diagnostic output only.

---

## Candidate center points

When both:

```python
points_arr
```

and:

```python
dist
```

are supplied, the supplied StarDist candidate point is used as the displayed center.

The stored StarDist point representation is:

```text
(y, x)
```

while plotting uses:

```text
(x, y)
```

so the coordinate order is explicitly converted before rendering.

When explicit candidate points are not supplied, the Shapely polygon centroid is used instead.

---

## Confidence scores

When:

```python
scores
```

are supplied, each polygon can be annotated with its confidence score near its center.

This is primarily useful for investigating:

```text
candidate ordering
NMS behavior
merging decisions
high-confidence prediction regions
```

---

## StarDist ray visualization

When all of the following are available:

```text
points_arr
dist
plot_rays=True
```

the original radial StarDist representation can be rendered.

Conceptually:

```text
candidate center
      │
      ├── ray 1
      ├── ray 2
      ├── ray 3
      └── ...
```

This makes it possible to inspect how the original ray-based candidate representation produced the reconstructed polygon.

---

## Orientation visualization

When:

```python
orientation_angles
```

is supplied, the function invokes:

```python
plot_orientation_vectors(...)
```

for each polygon.

This is particularly useful when debugging orientation-sensitive polygon-merging criteria.

---

## Vertical-axis handling

The argument:

```python
flip_vertical=True
```

inverts the Matplotlib Y axis.

This is used when the plot should follow image coordinate conventions:

```text
origin near upper-left
Y increasing downward
```

rather than the standard Cartesian plotting convention.

---

## Saved outputs

When a save location is supplied, the function currently writes both:

```text
PNG
SVG
```

versions of the plot.

Conceptually:

```text
plot title
    │
    ├── <title>.png
    └── <title>.svg
```

The PNG output is useful for quick inspection, while SVG provides a vector representation for higher-quality figures or detailed debugging.

---

# `plot_merged_polygon(...)`

This is a small historical debugging helper that plots one merged polygon in isolation.

It produces a simple visualization of:

```text
one merged polygon
```

and saves it using an identifier such as:

```text
merged_polygon_<id>.png
```

Its functionality overlaps partially with:

```python
plot_polygons(...)
```

because the same polygon can generally be rendered with:

```python
plot_polygons(
    [merged_polygon],
    ...
)
```

However, `plot_merged_polygon(...)` is currently retained as a dedicated legacy/debug helper.

Because the current repository-cleanup objective is reproducibility rather than redesign, it is reasonable to keep this helper until all historical debugging workflows that may depend on it have been reviewed.

A future cleanup can decide whether it should:

```text
remain as a convenience wrapper
be implemented internally using plot_polygons(...)
or be removed
```

without affecting the current postprocessing algorithm.

---

# `postprocessing_debug_plots.py`

This module contains postprocessing-specific diagnostic visualizations.

Unlike:

```text
polygon_plot_util.py
```

which is a general polygon plotting module, this file focuses on diagnostics tied directly to specific postprocessing stages.

Its responsibilities include:

```text
debug output path resolution
safe debug filename generation
per-image histograms
cross-image accumulated histograms
score-map visualization
local score-peak visualization
3D score-surface visualization
```

The current module also owns the output-path contract for postprocessing debug plots. :contentReference[oaicite:0]{index=0}

---

# Separation Between the Two Visualization Modules

The intended distinction is:

```text
polygon_plot_util.py
    generic polygon rendering

postprocessing_debug_plots.py
    pipeline-stage diagnostics
```

For example:

```python
plot_polygons(...)
```

can be used independently from the postprocessing pipeline.

In contrast:

```python
save_postprocessing_stage_histograms(...)
save_before_nms_score_peak_plot(...)
```

exist specifically to inspect the internal state of CNT postprocessing.

This separation should be preserved.

---

# Debug Filename Sanitization

The helper:

```python
_sanitize_tag(...)
```

converts image names or run identifiers into filesystem-safe debug tags.

It preserves:

```text
letters
numbers
-
_
```

and replaces other characters with underscores.

The resulting string is capped to a fixed length. :contentReference[oaicite:1]{index=1}

This helper is used by the postprocessing pipeline when generating filenames for per-image debug outputs.

---

# Histogram Diagnostics

The module contains internal utilities for collecting and plotting distributions of postprocessing quantities.

The main public entrypoint is:

```python
save_postprocessing_stage_histograms(...)
```

It currently calculates distributions for:

```text
polygon area
raw StarDist ray distance
```

from the pipeline state. :contentReference[oaicite:2]{index=2}

---

## Supported stages

The pipeline currently uses histogram diagnostics for stages such as:

```text
before_nms
after_nms
after_merging
```

Ray-distance histograms are generated only for stages where the original ray representation is still relevant:

```text
before_nms
after_nms
```

:contentReference[oaicite:3]{index=3}

After polygon merging, the final geometry no longer has a direct one-to-one raw-ray representation, so the post-merging diagnostics focus on polygon geometry.

---

## Per-image histograms

For each processed image, the module can save distributions such as:

```text
polygon area distribution
raw ray-distance distribution
```

under stage-specific directories.

The output hierarchy is approximately:

```text
<debug_root>/
└── histograms/
    └── <stage>/
        └── per_image/
            ├── all_polygon_area_histograms/
            └── all_pixel_distance_histograms/
```

This allows individual problematic images to be inspected independently.

---

## Across-image histograms

The same module also maintains process-local accumulated distributions.

Conceptually:

```text
image 1 values
        +
image 2 values
        +
image 3 values
        +
...
        │
        ▼
global histogram for the current process
```

The accumulation is stored in:

```python
_GLOBAL_HIST_CACHE
```

which is explicitly documented as a process-local cache retained for reproducibility. :contentReference[oaicite:4]{index=4}

The cache is keyed by:

```text
debug output root
postprocessing stage
metric
```

so different debug runs and stages remain separated. :contentReference[oaicite:5]{index=5}

The accumulated output hierarchy is approximately:

```text
<debug_root>/
└── histograms/
    └── <stage>/
        └── across_all_images/
            ├── all_polygon_area_histograms/
            └── all_pixel_distance_histograms/
```

---

# Histogram Styling

The module optionally uses:

```python
scienceplots
```

when it is installed.

If unavailable, it falls back to the standard Matplotlib style. :contentReference[oaicite:6]{index=6}

This means postprocessing debugging does not require `scienceplots` to function.

The plotting style is therefore an optional visualization enhancement rather than a runtime dependency of the core postprocessing algorithm.

---

# Before-NMS Score Peak Diagnostics

The function:

```python
save_before_nms_score_peak_plot(...)
```

creates spatial diagnostics of StarDist candidate confidence before NMS. :contentReference[oaicite:7]{index=7}

Its inputs include:

```text
candidate points
candidate scores
candidate polygons
image shape
```

The diagnostic builds a sparse score map where each candidate center contributes its confidence score.

If multiple candidate centers map to the same image pixel, the highest score is retained. :contentReference[oaicite:8]{index=8}

---

## Polygon mask restriction

The function rasterizes the candidate polygons into one union mask.

Score values outside that union are then removed from the masked score representation. :contentReference[oaicite:9]{index=9}

Conceptually:

```text
candidate score map
        +
union of candidate polygons
        │
        ▼
scores restricted to predicted polygon regions
```

This helps inspect where strong StarDist candidate responses occur relative to reconstructed polygon geometry.

---

## Local peak detection

The diagnostic identifies local score maxima using a neighborhood maximum filter.

The current implementation uses:

```text
peak radius = 10 pixels
```

and selects score pixels equal to the local maximum within the corresponding neighborhood. :contentReference[oaicite:10]{index=10}

It also tracks the overall maximum score location.

These diagnostics are intended to help investigate:

```text
candidate clustering
multiple detections of the same CNT
NMS behavior
score concentration
strongest local responses
```

---

# 2D Score Map

The first output from:

```python
save_before_nms_score_peak_plot(...)
```

is a 2D score map.

It shows:

```text
StarDist candidate score field
local maximum candidates
overall maximum score
```

The output is stored under:

```text
score_peak_maps/
└── before_nms/
```

:contentReference[oaicite:11]{index=11}

The score map is therefore directly associated with the pipeline state before NMS has removed duplicate candidates.

---

# 3D Score Surface

The same function also creates a 3D score-surface visualization.

Conceptually:

```text
X coordinate
Y coordinate
candidate score
```

becomes:

```text
3D surface height = prediction confidence
```

The surface uses the masked score field and includes optional:

```text
Gaussian smoothing
contour projection
local peak markers
global maximum marker
```

:contentReference[oaicite:12]{index=12}

This view is particularly useful when visually inspecting whether one CNT produces:

```text
one dominant probability peak
multiple nearby peaks
broad plateaus
isolated noisy peaks
```

before NMS.

---

# Debug Output Path Contract

Postprocessing debug locations are resolved by:

```python
resolve_postprocessing_debug_plot_dir(...)
```

The function uses the central:

```python
ProjectPaths
```

contract. :contentReference[oaicite:13]{index=13}

If an explicit directory is supplied:

```python
debug_plot_dir=...
```

that directory is used directly.

Otherwise, when a run name is available, outputs are stored under:

```text
global_outputs/
└── runs/
    └── <RUN>/
        └── viz/
            └── postprocessing/
                └── merging_debug/
                    └── <approach>/
```

Without a run name, the fallback location is:

```text
global_outputs/
└── misc/
    └── postprocessing/
        └── merging_debug/
            └── <approach>/
```

The current approach tags include:

```text
no_overlap
with_overlap
```

This keeps postprocessing diagnostics under the centralized output contract rather than creating arbitrary folders relative to the current working directory.

---

# Relationship to `cnt_postprocess_pipeline.py`

The postprocessing pipeline imports visualization helpers such as:

```python
plot_polygons(...)
resolve_postprocessing_debug_plot_dir(...)
save_before_nms_score_peak_plot(...)
save_postprocessing_stage_histograms(...)
```

The intended control flow is:

```text
cnt_postprocess_pipeline.py
        │
        ├── performs algorithm stage
        │
        ├── checks debug_plots
        │
        └── requests diagnostic visualization
                │
                ▼
visualization/
        └── renders and saves diagnostic
```

The important architectural boundary is:

```text
pipeline decides WHEN
visualization decides HOW
```

This prevents plotting implementation from becoming mixed with geometric merging logic.

---

# Relationship to `polygon_merging_algorithm.py`

The iterative polygon-merging algorithm also uses:

```python
plot_polygons(...)
```

for optional per-iteration debugging.

Conceptually:

```text
iteration 0
    │
    ├── candidate polygons
    └── candidate polygons + scores

iteration 1
    │
    ├── merged polygons
    └── merged polygons + scores

iteration 2
    │
    └── ...
```

These figures allow merge progression to be inspected visually.

The geometry module still owns the iterative merge logic; the visualization module only renders its state.

---

# Relationship to Inference

The inference runner also uses:

```python
plot_polygons(...)
```

to optionally create final prediction figures.

The flow is:

```text
predict_dataset(...)
        │
        ▼
PredictionBatch
        │
        ├── polygons
        ├── scores
        └── coords
        │
        ▼
infer_runner.py
        │
        ▼
plot_polygons(...)
```

This keeps plotting outside the low-level predictor.

That separation is intentional:

```text
predictor.py
    computes predictions

infer_runner.py
    decides whether final predictions should be visualized

visualization/
    renders them
```

---

# Relationship to Evaluation

Some evaluation runners also import:

```python
plot_polygons(...)
```

when intermediate or final polygon states need to be inspected.

Therefore this visualization package currently serves both:

```text
model-development debugging
evaluation diagnostics
```

while still remaining primarily associated with the custom CNT postprocessing implementation.

---

# General vs Postprocessing-Specific Visualization

The current architecture can be summarized as:

```text
polygon_plot_util.py
    generic Shapely polygon visualization
        │
        ├── inference
        ├── postprocessing pipeline
        ├── polygon-merging algorithm
        └── evaluation

postprocessing_debug_plots.py
    postprocessing-stage diagnostics
        │
        └── cnt_postprocess_pipeline.py
```

This distinction is useful and should remain.

---

# Reproducibility Constraints

Visualization code should not influence model predictions.

Changes that are generally safe include:

```text
moving visualization helpers between visualization modules
renaming private plotting helpers
changing figure titles
changing output folder organization
changing DPI
changing visualization styles
changing plot colors
adding new diagnostic figures
```

provided none of those changes alter the objects passed back into the postprocessing algorithm.

More care is required around code that computes intermediate representations for plotting.

For example:

```python
plot_polygon(...)
```

simplifies polygon geometry before rendering.

That is safe only because the simplified geometry is local to the plotting function and the original polygon object is not replaced in the pipeline.

The same rule should be maintained for all future diagnostic utilities.

---

# Current Design Summary

The visualization folder currently follows this responsibility structure:

```text
postprocessing state
        │
        ├───────────────┐
        ▼               ▼
polygon_plot_util.py    postprocessing_debug_plots.py
        │               │
        │               ├── stage histograms
        │               ├── accumulated histograms
        │               ├── score maps
        │               ├── peak detection views
        │               └── debug path resolution
        │
        ├── polygons
        ├── scores
        ├── rays
        ├── centroids
        └── orientations
        │
        └───────────────┬───────────────
                        ▼
              global_outputs/.../viz/
```

The intended architectural contract is:

```text
visualization = diagnostic rendering only
```

while:

```text
pipeline = algorithm orchestration
geometry = polygon mechanics
filtering = suppression and shape decisions
features = reusable measurements
adapters = representation conversion
```

This boundary should be preserved as the postprocessing package continues to be cleaned and documented.