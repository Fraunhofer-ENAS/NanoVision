# Postprocessing Filtering

This folder contains filtering and shape-classification utilities used by the CNT postprocessing pipeline before and during polygon merging.

The current structure is:

```text
filtering/
├── nms.py
├── shape_heuristics.py
└── __init__.py
```

This folder contains operations that answer questions such as:

```text
Which polygon candidates should be suppressed?
Which polygons have an elongated shape?
Would a merged polygon form an undesirable Y/T-like structure?
How many endpoints does a polygon skeleton contain?
```

The orchestration of these operations remains in:

```text
postprocessing/pipeline/cnt_postprocess_pipeline.py
```

while the actual iterative polygon-merging mechanics remain in:

```text
postprocessing/geometry/polygon_merging_algorithm.py
```

## `nms.py`

This module contains polygon-level non-maximum suppression.

Its main entry point is:

```python
nms_polygons(...)
```

The function operates on aligned candidate collections:

```text
polygons
scores
distances
points
```

and removes lower-scoring polygon candidates when they overlap sufficiently with higher-scoring candidates.

The associated metadata is filtered together with the polygon so that correspondence between:

```text
polygon ↔ score ↔ ray distances ↔ center point
```

is preserved.

### NMS flow

The current implementation approximately follows:

```text
polygon candidates
        │
        ▼
compute polygon areas
        │
        ▼
compute bounding boxes
        │
        ▼
compute centroids
        │
        ▼
build centroid KDTree
        │
        ▼
find nearby candidate polygons
        │
        ▼
bounding-box intersection check
        │
        ▼
exact Shapely intersection
        │
        ▼
overlap ratio
    intersection area
    -----------------
    minimum polygon area
        │
        ▼
compare with overlap threshold
        │
        ▼
suppress lower-scoring polygon
        │
        ▼
remaining polygons + aligned metadata
```

The default overlap threshold is currently:

```python
0.5
```

### Current usage

`nms_polygons(...)` is used by:

```text
postprocessing/pipeline/cnt_postprocess_pipeline.py
```

in both postprocessing approaches:

```text
no-overlap-region pipeline
with-overlap-region pipeline
```

It therefore represents the common candidate-suppression stage before the more specialized polygon-merging logic.

### Reproducibility note

The current NMS implementation contains some legacy comments and debugging code that are intentionally being left unchanged during the structural cleanup.

Potential algorithmic improvements, including changes to KD-tree search radius, bounding-box filtering, overlap handling, or the suppression strategy, should be treated separately because they may change inference results.

---

## `shape_heuristics.py`

This module contains polygon-shape heuristics used by the postprocessing algorithms.

The functions here are not general CNT feature-extraction utilities. They are specifically tied to decisions made during postprocessing, such as determining whether a merged polygon has an undesirable topology or whether a polygon should be treated as elongated.

The module currently contains three main groups of functionality:

```text
polygon rasterization / skeletonization
        +
skeleton endpoint analysis
        +
shape classification
```

## Polygon preparation and skeletonization

The following helpers form one internal processing chain:

```python
smooth_polygon(...)
rasterize_polygon_to_binary_image(...)
crop_mask_to_roi(...)
simple_skeletonize(...)
skeletonize_polygon(...)
```

Their relationship is approximately:

```text
Shapely polygon
      │
      ▼
smooth_polygon(...)
      │
      ▼
rasterize_polygon_to_binary_image(...)
      │
      ▼
binary mask
      │
      ▼
crop_mask_to_roi(...)
      │
      ▼
cropped binary mask
      │
      ▼
simple_skeletonize(...)
      │
      ▼
skeleton
```

These helpers support shape-based postprocessing decisions rather than the main morphological feature-extraction pipeline under:

```text
src/cnt_project/features/
```

### `smooth_polygon(...)`

Applies the current Chaikin-style polygon smoothing procedure used by the rasterization path.

This should not be confused with the separate final polygon smoothing behavior that may occur elsewhere in the postprocessing pipeline.

### `rasterize_polygon_to_binary_image(...)`

Converts a polygon into a binary raster representation.

The current implementation smooths the polygon before rasterization.

### `crop_mask_to_roi(...)`

Restricts a binary polygon mask to a padded region around the polygon bounds.

### `simple_skeletonize(...)`

Provides the local wrapper around:

```python
skimage.morphology.skeletonize
```

used by the polygon-skeletonization path.

### `skeletonize_polygon(...)`

Combines the previous helpers into a convenience operation:

```text
polygon
  → rasterized mask
  → cropped ROI
  → skeleton
```

---

## Y/T-shape detection

The function:

```python
check_y_or_t_shape(...)
```

uses polygon skeleton topology to determine whether a merged polygon appears to form a branching structure.

The current logic is:

```text
merged polygon
      │
      ▼
skeletonize_polygon(...)
      │
      ▼
count_endpoints(...)
      │
      ▼
more than two endpoints?
      │
   yes│      no
      ▼       ▼
   reject   accept
   as Y/T   this criterion
```

This heuristic exists because merging two candidate polygons can sometimes produce branched structures that are unlikely to represent a single CNT.

### Current usage

`check_y_or_t_shape(...)` is used by:

```text
postprocessing/geometry/polygon_merging_algorithm.py
```

in the overlap-region merging path.

Some references to this heuristic in the no-overlap path are currently commented legacy/debug code and are being retained for reproducibility/history rather than treated as active logic.

---

## Skeleton endpoint counting

Several endpoint-related helpers currently exist because different historical or experimental postprocessing paths used slightly different skeleton representations.

### `count_endpoints(...)`

Counts skeleton pixels having exactly one connected neighbor.

It is used by:

```python
check_y_or_t_shape(...)
count_polygon_skeleton_endpoints(...)
```

and therefore supports active postprocessing decisions.

### `count_polygon_skeleton_endpoints(...)`

This function was previously named:

```text
plot_skeleton
```

despite primarily computing an endpoint count.

It now has a name that reflects its actual behavior.

Its processing is approximately:

```text
merged Shapely polygon
        │
        ▼
create local binary mask
        │
        ▼
skeletonize
        │
        ▼
count_endpoints(...)
        │
        ▼
number of endpoints
```

The returned endpoint count is used by polygon-merging criteria in the no-overlap-region path.

The function currently accepts `merged_id` for compatibility with the historical call path, although its primary responsibility is computation rather than visualization.

---

## Elongated-polygon classification

The overlap-region postprocessing approach separates elongated polygons from more complex or star-like shapes before merging.

The main function for this is:

```python
is_elongated_polygon(...)
```

The current heuristic combines several checks:

```text
polygon
   │
   ▼
skeleton-based endpoint analysis
   │
   ▼
PCA-based shape ratio
   │
   ▼
convex-hull criterion
   │
   ▼
elongated / not elongated
```

The supporting endpoint helper is:

```python
count_endpoints_for_elongated_polygons(...)
```

### Current usage

`is_elongated_polygon(...)` is used by:

```text
postprocessing/pipeline/cnt_postprocess_pipeline.py
```

specifically in the:

```python
merge_instances_with_overlap_regions(...)
```

pipeline.

The purpose is to separate candidate polygons into approximately:

```text
elongated polygons
    → eligible for iterative CNT merging

non-elongated / star-like polygons
    → retained separately
```

This path corresponds to the alternative labeling/postprocessing approach where explicit overlap regions are present.

---

## Relationship to the postprocessing pipeline

The main integration looks like:

```text
StarDist polygon candidates
        │
        ▼
cnt_postprocess_pipeline.py
        │
        ├───────────────► nms.py
        │                  │
        │                  └── nms_polygons(...)
        │
        ├───────────────► shape_heuristics.py
        │                  │
        │                  ├── is_elongated_polygon(...)
        │                  ├── check_y_or_t_shape(...)
        │                  └── count_polygon_skeleton_endpoints(...)
        │
        ▼
polygon_merging_algorithm.py
        │
        ▼
final merged polygons
```

The exact functions used depend on which postprocessing approach is active.

### No-overlap-region approach

The active production path used by the patched StarDist model currently follows approximately:

```text
polygon construction
        ↓
score sorting
        ↓
nms_polygons(...)
        ↓
iterative polygon merging
        ↓
endpoint / shape checks
        ↓
optional smoothing
        ↓
final polygons
```

### With-overlap-region approach

The alternative overlap-aware path follows approximately:

```text
polygon construction
        ↓
score sorting
        ↓
nms_polygons(...)
        ↓
polygon simplification
        ↓
is_elongated_polygon(...)
        ↓
split elongated / star-like polygons
        ↓
merge elongated polygons
        ↓
recombine results
```

This path is currently retained because it corresponds to a different labeling strategy and may be useful again even though it is not the main active inference path.

---

## Design Boundary

The intended separation between postprocessing modules is:

```text
filtering/
    candidate suppression and shape-based filtering decisions

geometry/
    polygon construction, merge criteria, and iterative merge mechanics

pipeline/
    orchestration of the complete postprocessing sequence

adapters/
    conversion between CNT polygon representations and StarDist outputs

visualization/
    postprocessing diagnostics and plotting
```

This means functions in `filtering/` should primarily answer questions such as:

```text
"Should this candidate remain?"
"Does this shape satisfy a structural condition?"
"How many branches/endpoints does this polygon have?"
```

rather than implementing the entire merging process.

## Relationship to `features/`

Some operations in this folder, such as PCA, skeletonization, polygon smoothing, or shape analysis, may look similar to functionality under:

```text
src/cnt_project/features/
```

The distinction is intentional.

The `features/` package contains canonical measurement and morphology extraction functionality intended to describe CNTs.

The functions in:

```text
model_development/postprocessing/filtering/
```

exist to make intermediate model/postprocessing decisions.

##### They should therefore not automatically be replaced by feature-extraction implementations unless numerical equivalence has first been established.

## Reproducibility Notes

The filtering code contains historical and experimental heuristics that may not represent the final preferred algorithms.

During the current repository cleanup, the priority is structural clarification without changing numerical behavior.

In particular, changes to the following can alter final predicted polygons:

```text
NMS overlap calculations
candidate suppression ordering
polygon smoothing
rasterization behavior
skeletonization
endpoint counting
elongation criteria
Y/T-shape criteria
```

Such changes should therefore be handled as separate algorithmic work rather than as part of repository cleanup.