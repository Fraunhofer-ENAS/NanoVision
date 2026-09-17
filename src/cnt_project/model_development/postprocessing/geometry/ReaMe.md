# Postprocessing Geometry

This folder contains the geometric construction and iterative polygon-merging logic used by the CNT StarDist postprocessing pipeline.

The current structure is:

```text
geometry/
├── polygon_from_rays.py
├── polygon_merging_algorithm.py
├── polygon_merging_criteria.py
└── __init__.py
```

The responsibility of this folder is to answer geometric questions such as:

```text
How are StarDist ray predictions converted into Shapely polygons?
Should two polygon candidates be merged?
How are neighboring polygons discovered?
How are accepted polygon pairs iteratively combined?
How are scores and representative points propagated through merging?
```

The geometry package does **not** orchestrate the complete postprocessing workflow.

That responsibility remains in:

```text
postprocessing/pipeline/cnt_postprocess_pipeline.py
```

The intended separation is:

```text
pipeline/
    controls the sequence of postprocessing stages

geometry/
    constructs polygons and performs geometric merging

filtering/
    suppresses candidates and evaluates shape-based heuristics

adapters/
    converts final polygon outputs back into StarDist-compatible structures

visualization/
    produces optional debugging and diagnostic plots
```

---

## `polygon_from_rays.py`

This module converts raw StarDist ray predictions into Shapely polygon candidates.

Its main entry point is:

```python
construct_polygon_opp(...)
```

StarDist provides each candidate using:

```text
center point
+
radial distances
+
confidence score
```

The function converts this representation into:

```text
Shapely Polygon
+
original ray distances
+
candidate center point
+
candidate score
```

### Conversion flow

The current transformation is approximately:

```text
candidate index
      │
      ├── center point (y, x)
      │
      ├── N radial distances
      │
      └── confidence score
      │
      ▼
divide 360° into N ray directions
      │
      ▼
for every ray:
    x = center_x + distance * cos(angle)
    y = center_y + distance * sin(angle)
      │
      ▼
ordered polygon vertices
      │
      ▼
Shapely Polygon
```

The function returns:

```python
(
    polygon,
    distances,
    center_point,
    score,
)
```

Keeping the associated metadata together is important because the later postprocessing stages need to maintain correspondence between:

```text
polygon ↔ ray distances ↔ center point ↔ confidence score
```

### Current usage

`construct_polygon_opp(...)` is used by:

```text
postprocessing/pipeline/cnt_postprocess_pipeline.py
```

for construction of the initial candidate polygons.

It is also used independently by evaluation code such as:

```text
evaluation/runners/eval_pre_nms_polygon_masks_by_density_runner.py
```

This allows evaluation of the polygon representation before later NMS and merging stages.

### Reproducibility note

The current conversion is part of the historical postprocessing behavior.

Changes to:

```text
ray angle convention
coordinate order
distance interpretation
polygon vertex ordering
```

can modify the resulting polygons and therefore final inference results.

Such changes should be treated as algorithm changes rather than repository cleanup.

---

## `polygon_merging_criteria.py`

This module contains primitive geometric criteria and helpers associated with deciding whether polygon candidates should be merged.

The current active helper is:

```python
calculate_overlap(...)
```

A legacy alignment helper is also retained:

```python
calculate_perpendicular_offset(...)
```

The module intentionally remains separate from:

```text
polygon_merging_algorithm.py
```

because the conceptual responsibilities are different:

```text
polygon_merging_criteria.py
    primitive geometric measurements / criteria

polygon_merging_algorithm.py
    complete merge decisions and iterative merge mechanics
```

This separation remains useful even though the criteria module is currently small.

Over time, additional reusable primitives could naturally live here, for example:

```python
calculate_angle_difference(...)
calculate_centroid_alignment(...)
calculate_projected_centroid_distance(...)
```

provided moving or extracting them does not alter the existing numerical behavior.

---

## Polygon overlap

The function:

```python
calculate_overlap(...)
```

computes the intersection of two Shapely polygons and normalizes the intersection area by the area of the smaller polygon.

Conceptually:

```text
                 intersection area
overlap ratio = -------------------
                 min(area1, area2)
```

This makes the overlap measure relative to the smaller candidate.

The result is used by the no-overlap-region merge criterion in:

```text
polygon_merging_algorithm.py
```

---

## Legacy perpendicular-offset criterion

The function:

```python
calculate_perpendicular_offset(...)
```

computes the perpendicular distance between a point and an oriented line.

It belongs to a previous centroid/alignment criterion.

The active no-overlap merging implementation now uses a different projected-centroid alignment test, so this function is currently not part of the active inference path.

It is intentionally retained because:

```text
- it documents a previously investigated merge criterion;
- it can be useful for future ablation/comparison work;
- removing it is unnecessary for the current structural cleanup.
```

It should not be silently reintroduced into the active merge logic because doing so could change inference results.

---

## Endpoint helper

`polygon_merging_criteria.py` also currently contains:

```python
count_endpoints(...)
```

which counts skeleton endpoints using a convolution-based neighborhood test.

Most current shape and skeleton behavior has been consolidated under:

```text
postprocessing/filtering/shape_heuristics.py
```

so this helper should be considered carefully during future cleanup.

For the current repository-cleaning phase, it is acceptable to keep it unless it is confirmed to be completely unused and removing it cannot affect reproducibility.

---

# `polygon_merging_algorithm.py`

This module contains the actual polygon merge decisions and the iterative algorithms that repeatedly combine compatible polygon candidates.

It supports two historical labeling/postprocessing approaches:

```text
1. no explicit overlap regions
2. explicit overlap regions
```

The corresponding merge predicates are:

```python
should_merge_candidates_no_overlap_regions(...)
should_merge_candidates_with_overlap_regions(...)
```

and the iterative algorithms are:

```python
merge_polygons_iterative_no_overlap_regions(...)
merge_polygons_iterative_with_overlap_regions(...)
```

The **no-overlap-region path is the main path used by the current patched StarDist inference pipeline**.

The overlap-region implementation is retained because it corresponds to an alternative annotation strategy and may still be useful for experiments or future work.

---

# No-overlap-region merging

The main merge predicate is:

```python
should_merge_candidates_no_overlap_regions(...)
```

It determines whether two candidate polygons should represent the same CNT.

The current decision combines several geometric conditions.

Approximately:

```text
poly1 + poly2
    │
    ▼
overlap criterion
    │
    ├── fail → do not merge
    │
    ▼
PCA orientation comparison
    │
    ├── fail → do not merge
    │
    ▼
construct temporary union
    │
    ▼
skeleton endpoint count
    │
    ▼
centroid alignment along primary axis
    │
    ├── fail → do not merge
    │
    ▼
require suitable skeleton topology
    │
    ├── fail → do not merge
    │
    ▼
merge accepted
```

## Orientation calculation

The merge criterion intentionally uses the historical polygon-PCA orientation implementation from:

```text
features/core/shapely_polygon_pca_orientation.py
```

through:

```python
calculate_shapely_polygon_pca_orientation_legacy(...)
```

This is intentional because postprocessing reproducibility depends on preserving the orientation calculation that was historically used.

The orientation implementation therefore belongs to the canonical `features/` package, while the decision of **how orientation is used as a merge condition** remains here in postprocessing.

---

## Overlap criterion

The first check uses:

```python
calculate_overlap(...)
```

from:

```text
polygon_merging_criteria.py
```

If the overlap does not exceed the configured threshold, the pair is rejected.

---

## Orientation criterion

The PCA orientations of both polygons are calculated and compared.

The angular difference is normalized before comparison with:

```text
angle_threshold
```

If the polygons are not sufficiently aligned in orientation, merging is rejected.

---

## Centroid alignment

The active alignment criterion projects the centroid displacement of the second polygon onto the primary axis of the first polygon.

Conceptually:

```text
poly1 primary axis
──────────────────────────────>

        poly1 centroid ●
                       \
                        \ centroid displacement
                         \
                          ● poly2 centroid
```

The centroid separation along the primary axis is normalized using the extent of `poly1` along the same axis.

This prevents candidates that overlap but occupy incompatible positions relative to the CNT direction from being merged.

The historical perpendicular-offset approach remains visible in legacy comments and in:

```python
calculate_perpendicular_offset(...)
```

but is not currently active.

---

## Skeleton topology criterion

The temporary union:

```python
unary_union([poly1, poly2])
```

is analyzed using:

```python
count_polygon_skeleton_endpoints(...)
```

from:

```text
postprocessing/filtering/shape_heuristics.py
```

The current active no-overlap criterion requires the merged shape to have:

```text
2 skeleton endpoints
```

This acts as a guard against merges that create branched Y/T-like shapes.

---

# `merge_polygons_iterative_no_overlap_regions(...)`

This function performs the complete iterative polygon-merging procedure for datasets without explicit overlap-region labels.

It is the main polygon-merging implementation used by the current inference path.

Its processing is approximately:

```text
input polygons + scores + points
             │
             ▼
deterministic spatial sorting
             │
             ▼
build centroid KDTree
             │
             ▼
for each polygon:
    find nearby candidates
             │
             ▼
sort local candidates spatially
             │
             ▼
evaluate candidates one by one
             │
             ▼
should_merge_candidates_no_overlap_regions(...)
             │
         ┌───┴────┐
         │        │
       false     true
         │        │
         │        ▼
         │    unary_union(...)
         │        │
         │        ▼
         │    propagate highest score
         │    and corresponding point
         │
         └────────┘
             │
             ▼
construct next polygon generation
             │
             ▼
were any polygons merged?
        │             │
       yes            no
        │             │
        ▼             ▼
repeat iteration     stop
```

The iterative process stops when either:

```text
no additional polygons are merged
```

or:

```text
iteration_nbr
```

is reached.

---

## Spatial ordering

The no-overlap implementation intentionally sorts polygons approximately in image scan order using their centroids.

The current ordering is based on:

```text
descending centroid y
then ascending centroid x
```

Local KD-tree candidate neighbors are also sorted before merge evaluation.

This ordering is significant because iterative merging can be order-dependent.

Therefore this behavior should **not** be casually rewritten as part of structural cleanup.

Changing candidate ordering could change which polygons are merged first and consequently change the final result.

---

## KD-tree neighbor discovery

A KD-tree is rebuilt during each merge iteration using polygon centroids.

For every current polygon, nearby candidates are selected using a search radius derived from polygon size.

This prevents every polygon from being compared with every other polygon.

Conceptually:

```text
all polygons
     │
     ▼
polygon centroids
     │
     ▼
KDTree
     │
     ▼
local candidate neighborhood
     │
     ▼
merge criteria
```

The KD-tree neighborhood logic is part of the active algorithm and should remain unchanged during reproducibility-preserving cleanup.

---

## Score and point propagation

When polygons are merged, the no-overlap algorithm preserves the score and representative point associated with the highest-scoring contributing polygon.

Therefore a merged instance carries:

```text
merged geometry
+
highest contributor score
+
point belonging to that score
```

The function returns:

```python
(
    current_polygons,
    current_scores,
    current_points,
)
```

These values are consumed by the higher-level postprocessing pipeline and eventually converted into StarDist-compatible outputs by the adapter layer.

---

# With-overlap-region merging

The alternative merge predicate is:

```python
should_merge_candidates_with_overlap_regions(...)
```

This path corresponds to datasets or annotation strategies containing explicit overlap regions.

The current decision approximately checks:

```text
orientation similarity
        │
        ▼
centroid alignment
        │
        ▼
minimum polygon intersection area
        │
        ▼
temporary merged geometry
        │
        ▼
Y/T-shape rejection
        │
        ▼
merge accepted
```

Unlike the no-overlap criterion, this path uses a minimum absolute intersection area:

```text
min_overlap
```

rather than the normalized overlap ratio used by:

```python
calculate_overlap(...)
```

It also uses:

```python
check_y_or_t_shape(...)
```

from:

```text
postprocessing/filtering/shape_heuristics.py
```

to reject complex merged shapes.

---

# `merge_polygons_iterative_with_overlap_regions(...)`

This function performs iterative merging for the explicit-overlap labeling approach.

Its general structure is similar to the no-overlap implementation:

```text
polygons
    │
    ▼
build centroid KDTree
    │
    ▼
find local candidate polygons
    │
    ▼
should_merge_candidates_with_overlap_regions(...)
    │
    ▼
unary_union(...) accepted candidates
    │
    ▼
build next polygon generation
    │
    ▼
repeat until stable or iteration limit
```

The resulting merged polygons are returned to the higher-level overlap-region postprocessing pipeline.

This implementation is currently retained even though the no-overlap path is the main active inference route.

---

# Relationship to `cnt_postprocess_pipeline.py`

The geometry modules do not independently perform complete inference postprocessing.

The orchestration resides in:

```text
postprocessing/pipeline/cnt_postprocess_pipeline.py
```

The no-overlap path approximately follows:

```text
StarDist distances + points + scores
                │
                ▼
polygon_from_rays.py
    construct_polygon_opp(...)
                │
                ▼
candidate polygons
                │
                ▼
filtering/nms.py
    nms_polygons(...)
                │
                ▼
polygon_merging_algorithm.py
    merge_polygons_iterative_no_overlap_regions(...)
                │
                ▼
    should_merge_candidates_no_overlap_regions(...)
                │
                ├── calculate_overlap(...)
                ├── legacy PCA orientation
                ├── centroid alignment
                └── skeleton endpoint check
                │
                ▼
final merged polygons
```

The alternative overlap-region path follows approximately:

```text
StarDist candidates
        │
        ▼
construct_polygon_opp(...)
        │
        ▼
NMS
        │
        ▼
shape classification
        │
        ▼
merge_polygons_iterative_with_overlap_regions(...)
        │
        ▼
should_merge_candidates_with_overlap_regions(...)
        │
        ▼
merged polygons
```

---

# Relationship to other packages

## `features/`

Canonical reusable morphology measurements belong under:

```text
src/cnt_project/features/
```

For example, polygon PCA orientation is maintained there.

Geometry code in this folder may **consume** those canonical measurements when they are needed for a postprocessing decision.

The intended dependency is therefore:

```text
features/
    computes reusable geometric/morphological quantities
            │
            ▼
postprocessing/geometry/
    uses those quantities to decide how predictions should be merged
```

Postprocessing-specific decision logic should not be moved into `features/`.

---

## `filtering/`

The geometry algorithms depend on shape heuristics from:

```text
postprocessing/filtering/shape_heuristics.py
```

Examples include:

```python
check_y_or_t_shape(...)
count_polygon_skeleton_endpoints(...)
```

The distinction is:

```text
geometry/
    spatial relationships and polygon merging

filtering/
    shape validity / structural acceptance tests
```

---

## `visualization/`

Debug visualization is provided by:

```text
postprocessing/visualization/
```

The geometry algorithms may invoke plotting utilities when:

```python
debug_plots=True
```

but visualization is not part of the core geometric responsibility.

The generated plots are diagnostic artifacts and should not influence the geometry calculations themselves.

---

# Current Active Call Chain

For the main no-overlap inference path, the relevant call chain is approximately:

```text
MyStarDist2D.predict_instances(...)
        │
        ▼
postprocess_instances_merge_polygons(...)
        │
        ▼
instances_merge_adapter(...)
        │
        ▼
build_stardist_outputs_from_merged_polygons(...)
        │
        ▼
merge_instances_no_overlap_regions(...)
        │
        ├── construct_polygon_opp(...)
        │
        ├── nms_polygons(...)
        │
        └── merge_polygons_iterative_no_overlap_regions(...)
        │               │
        │               ▼
        │     should_merge_candidates_no_overlap_regions(...)
        │               │
        │               ├── calculate_overlap(...)
        │               ├── PCA orientation
        │               ├── centroid alignment
        │               └── skeleton endpoint test
        │
        ▼
final merged Shapely polygons
```

This call chain is important when determining whether a function is part of active inference or merely experimental/legacy code.

---

# Reproducibility Notes

Polygon merging is one of the most numerically and logically sensitive parts of the CNT inference pipeline.

Small changes can propagate into substantially different final instance predictions.

In particular, the following should be considered algorithm changes:

```text
changing polygon construction from rays
changing coordinate conventions
changing polygon sorting order
changing KD-tree search radius
changing candidate ordering
changing overlap calculation
changing orientation calculation
changing angle normalization
changing centroid-alignment conditions
changing skeleton endpoint criteria
changing merge thresholds
changing unary-union behavior
changing score propagation
changing iteration stopping conditions
```

During the current repository-cleanup phase, these behaviors should therefore remain unchanged unless numerical equivalence has been explicitly demonstrated.

Structural changes such as:

```text
removing confirmed dead imports
moving visualization helpers
improving module boundaries
clarifying names
adding documentation
removing confirmed unreachable duplicate helpers
```

are appropriate provided the inference outputs remain reproducible.

The existing inference and COCO-output comparisons should continue to be used as regression checks after structural modifications.