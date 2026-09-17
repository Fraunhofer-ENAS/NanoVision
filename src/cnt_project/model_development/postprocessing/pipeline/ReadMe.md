# Postprocessing Pipeline

This folder contains the high-level orchestration of the CNT polygon postprocessing workflow.

Current structure:

```text
pipeline/
├── cnt_postprocess_pipeline.py
└── __init__.py
```

The pipeline module connects the lower-level postprocessing components into complete processing sequences.

Its responsibility is not to implement every geometric operation itself. Instead, it coordinates:

```text
raw StarDist candidates
        │
        ▼
polygon construction
        │
        ▼
candidate sorting
        │
        ▼
non-maximum suppression
        │
        ▼
optional smoothing / shape filtering
        │
        ▼
iterative polygon merging
        │
        ▼
orientation calculation
        │
        ▼
final merged CNT polygons
```

The module currently supports two historical postprocessing variants:

```text
Approach A:
    labels without explicit overlap regions

Approach B:
    labels with explicit overlap regions
```

The main functions are:

```python
merge_instances_no_overlap_regions(...)
merge_instances_with_overlap_regions(...)
```

The no-overlap path is the main path used by the current patched StarDist inference implementation.

The overlap-region path is retained for the alternative historical labeling approach and possible future experiments.

---

# Role of the Pipeline Layer

The pipeline layer should be understood as orchestration code.

It combines functionality from:

```text
geometry/
filtering/
features/
visualization/
```

without replacing those modules.

The current dependency structure is approximately:

```text
pipeline/cnt_postprocess_pipeline.py
        │
        ├── geometry/polygon_from_rays.py
        │       └── construct_polygon_opp(...)
        │
        ├── filtering/nms.py
        │       └── nms_polygons(...)
        │
        ├── filtering/shape_heuristics.py
        │       └── is_elongated_polygon(...)
        │
        ├── geometry/polygon_merging_algorithm.py
        │       ├── merge_polygons_iterative_no_overlap_regions(...)
        │       └── merge_polygons_iterative_with_overlap_regions(...)
        │
        ├── features/core/shapely_polygon_pca_orientation.py
        │       └── calculate_all_shapely_polygon_pca_orientations_legacy(...)
        │
        └── visualization/
                ├── polygon_plot_util.py
                └── postprocessing_debug_plots.py
```

The pipeline imports and coordinates these components directly. :contentReference[oaicite:0]{index=0}

---

# `cnt_postprocess_pipeline.py`

This module defines the complete polygon-level postprocessing sequences applied to StarDist candidate predictions.

It contains:

```python
_sort_polygons_by_scores(...)
merge_instances_with_overlap_regions(...)
merge_instances_no_overlap_regions(...)
```

The private sorting helper keeps polygons and their associated metadata synchronized while sorting candidates by confidence score.

---

# Shared Input Representation

Both pipeline variants operate on StarDist-style candidate data:

```text
scores
dist
points_arr
```

Conceptually:

```text
scores
    confidence score per candidate

dist
    N radial distances for each candidate

points_arr
    candidate center positions
```

The first important transformation is:

```text
StarDist rays
    │
    ▼
construct_polygon_opp(...)
    │
    ▼
Shapely polygons
```

The pipeline then carries four associated collections:

```text
polygons
scores
distances
points
```

These should remain aligned throughout candidate sorting, NMS, and merging.

---

# Candidate Score Sorting

The helper:

```python
_sort_polygons_by_scores(...)
```

sorts polygon candidates in descending score order while keeping:

```text
polygon
score
ray distances
center point
```

together.

Conceptually:

```text
[
    polygon_i,
    score_i,
    distances_i,
    point_i,
]
        │
        ▼
sort by score descending
        │
        ▼
aligned sorted collections
```

This sorting occurs before NMS and should not be modified casually because candidate ordering can influence later suppression and merging behavior.

---

# Approach A — No Explicit Overlap Regions

The main active postprocessing path is:

```python
merge_instances_no_overlap_regions(...)
```

This function constructs, suppresses, optionally smooths, merges, and characterizes StarDist polygon candidates. 

The processing sequence is approximately:

```text
scores + dist + points_arr
            │
            ▼
construct polygons from rays
            │
            ▼
sort by confidence score
            │
            ▼
optional debug outputs
            │
            ▼
polygon NMS
            │
            ▼
optional polygon simplification
            │
            ▼
iterative no-overlap merging
            │
            ▼
legacy PCA orientation calculation
            │
            ▼
final polygons + scores + representative points
```

---

## 1. Polygon construction

Candidate polygons are created with:

```python
construct_polygon_opp(...)
```

for every StarDist candidate.

This converts:

```text
candidate point + ray distances + score
```

into:

```text
Shapely polygon + distances + point + score
```

---

## 2. Score sorting

The resulting candidates are sorted using:

```python
_sort_polygons_by_scores(...)
```

so that higher-confidence polygons are processed first. 

This ordering is part of the historical algorithm behavior and should be treated as result-sensitive.

---

## 3. Optional diagnostic candidate analysis

Before NMS, the no-overlap pipeline also identifies examples such as:

```text
largest polygons by area
longest candidates by maximum ray distance
```

These values are primarily used by debugging visualization when:

```python
debug_plots=True
```

and especially when:

```python
debug_plot_level == "full"
```

The diagnostic branch can generate:

```text
original polygon plots
score overlays
largest candidate plots
longest candidate plots
score-peak plots
stage histograms
```

These diagnostics should remain observational only and should not alter the polygon computation.

---

## 4. Non-maximum suppression

Candidate suppression is performed using:

```python
nms_polygons(...)
```

with the current pipeline threshold:

```python
0.5
```

The call preserves aligned outputs:

```text
polygons
scores
distances
points
```

after suppression.

The NMS implementation itself lives under:

```text
postprocessing/filtering/nms.py
```

The pipeline only controls when it is applied.

---

## 5. Optional stage capture

The no-overlap path supports:

```python
return_stage_outputs=True
```

This allows callers to inspect actual intermediate pipeline objects.

Currently captured stages include:

```text
after_nms
after_smoothing
```

For example, the NMS state stores:

```text
polygons
scores
distances
points
```

directly from the active pipeline.

This is useful for:

```text
ablation studies
intermediate evaluation
debugging
postprocessing audits
```

because the stages are captured from the real inference path rather than reconstructed afterward.

---

## 6. Polygon simplification

When:

```python
apply_smoothing=True
```

the polygons are simplified using:

```python
poly.simplify(
    1.2,
    preserve_topology=True,
)
```

before merging. 
Although this operation is described historically as smoothing, the active implementation is Shapely polygon simplification.

Because this directly modifies polygon geometry before merging, changing:

```text
the tolerance
the simplification method
the stage where it is applied
the preserve_topology behavior
```

would be an algorithmic change and may alter final predictions.

---

## 7. Iterative polygon merging

The simplified candidates are forwarded to:

```python
merge_polygons_iterative_no_overlap_regions(...)
```

from:

```text
postprocessing/geometry/polygon_merging_algorithm.py
```

The pipeline forwards the configurable merge parameters:

```text
merge_overlap_threshold
merge_angle_threshold
merge_alignment_tolerance
merge_aspect_ratio_threshold
```

together with candidate geometry and metadata. 

The geometry module then performs the actual KD-tree search, merge predicate evaluation, and polygon unions.

---

## 8. Orientation calculation

After iterative merging, orientation is calculated using:

```python
calculate_all_shapely_polygon_pca_orientations_legacy(...)
```

from:

```text
features/core/shapely_polygon_pca_orientation.py
```

This preserves the historical polygon-PCA orientation behavior currently used by postprocessing. 

The pipeline returns orientation as metadata describing the final merged polygons.

---

## No-overlap outputs

The normal return value is:

```python
(
    merged_polygons,
    orientation_angles,
    merged_scores,
    merged_coords,
)
```

When:

```python
return_stage_outputs=True
```

an additional item is returned:

```python
stage_outputs
```

so the result becomes:

```python
(
    merged_polygons,
    orientation_angles,
    merged_scores,
    merged_coords,
    stage_outputs,
)
```

The output contract is implemented directly at the end of the pipeline.

---

# Approach B — Explicit Overlap Regions

The alternative pipeline is:

```python
merge_instances_with_overlap_regions(...)
```

This path corresponds to the historical annotation strategy where overlap regions are explicitly represented.

Its documented sequence is:

```text
1. construct polygons from StarDist rays
2. sort candidates by confidence
3. run polygon NMS
4. simplify polygon boundaries
5. classify polygons by shape
6. merge elongated polygons using overlap-aware logic
7. preserve non-elongated polygons
8. calculate final orientation
```

This sequence is defined directly in the function. 

Since this part is not currenly used, i will add more documentation here if needed to avoid making the readme too long.

---

# Debug Visualization

Both pipeline variants support optional postprocessing diagnostics.

The visualization responsibilities have been separated into:

```text
postprocessing/visualization/polygon_plot_util.py
postprocessing/visualization/postprocessing_debug_plots.py
```

The pipeline invokes these helpers when:

```python
debug_plots=True
```

The current helpers include:

```python
plot_polygons(...)
resolve_postprocessing_debug_plot_dir(...)
save_before_nms_score_peak_plot(...)
save_postprocessing_stage_histograms(...)
```


This keeps plotting implementation outside the core pipeline while allowing the pipeline to decide **which processing stages should be visualized**.

That distinction is intentional:

```text
pipeline/
    decides when a diagnostic should happen

visualization/
    implements how the diagnostic is rendered
```

---

# Debug Output Stages

The pipeline can currently generate diagnostics for stages such as:

```text
before_nms
after_nms
after_merging
```

and, depending on the pipeline and debug level:

```text
original polygons
original polygons with scores
largest candidates
longest candidates
elongated candidates
star-like candidates
final merged polygons
polygon-area histograms
ray-distance histograms
score-peak visualizations
```

These outputs are intended for:

```text
algorithm investigation
regression analysis
postprocessing development
debugging problematic CNT merges
```

and should not modify inference behavior.

---

# Relationship to the Adapter Layer

The main no-overlap pipeline is called from:

```text
postprocessing/adapters/stardist_postprocess_adapter.py
```

The relevant flow is:

```text
StarDist prediction tensors
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
        ▼
final merged Shapely polygons
        │
        ▼
adapter converts polygons back to:
    label image
    StarDist coordinate array
    center points
    probabilities
```

Therefore:

```text
pipeline/
    defines the CNT polygon postprocessing algorithm

adapters/
    translates between StarDist representations and the pipeline representation
```

---

# Relationship to Geometry

The pipeline delegates actual iterative merge mechanics to:

```text
postprocessing/geometry/polygon_merging_algorithm.py
```

The separation is:

```text
cnt_postprocess_pipeline.py
    decides the processing sequence

polygon_merging_algorithm.py
    decides how candidate neighbors are iteratively merged
```

For example:

```text
pipeline:
    polygon construction
    → NMS
    → simplification
    → call merging algorithm

geometry:
    build KDTree
    → find neighbors
    → evaluate merge criteria
    → unary_union polygons
    → repeat
```

This separation should be preserved.

---

# Relationship to Filtering

The pipeline uses filtering functionality from:

```text
postprocessing/filtering/
```

including:

```python
nms_polygons(...)
is_elongated_polygon(...)
```

The intended distinction is:

```text
filtering/
    candidate suppression and shape classification

pipeline/
    decides at which processing stage those filters are applied
```

---

# Relationship to Features

The pipeline consumes canonical reusable feature calculations from:

```text
src/cnt_project/features/
```

Currently this includes the historical polygon PCA orientation functions.

The intended dependency direction is:

```text
features/
    reusable geometric or morphological measurements
        │
        ▼
postprocessing/
    uses those measurements for model-specific processing
```

Feature implementations should not be duplicated inside the postprocessing pipeline.

---

# Main Active Inference Path

The current production-style inference path is approximately:

```text
MyStarDist2D.predict_instances(...)
        │
        ▼
stardist_postprocess_adapter.py
        │
        ▼
merge_instances_no_overlap_regions(...)
        │
        ├── construct_polygon_opp(...)
        │
        ├── _sort_polygons_by_scores(...)
        │
        ├── nms_polygons(...)
        │
        ├── optional simplify(...)
        │
        ├── merge_polygons_iterative_no_overlap_regions(...)
        │
        └── calculate_all_shapely_polygon_pca_orientations_legacy(...)
        │
        ▼
merged CNT polygons
        │
        ▼
StarDist-compatible adapter outputs
        │
        ▼
inference export / evaluation
```

The no-overlap pipeline is therefore part of the active model prediction path and should be treated as result-sensitive code.

---

# Pipeline vs Runner

This module is intentionally a **pipeline**, not a runner.

A runner is typically an executable boundary such as:

```python
if __name__ == "__main__":
    ...
```

and is responsible for concerns such as:

```text
CLI argument parsing
loading paths
loading datasets
loading models
creating output directories
starting an operation
printing execution summaries
```

A pipeline instead represents reusable processing logic:

```text
input data
    │
    ▼
ordered processing stages
    │
    ▼
output data
```

Therefore:

```text
infer_runner.py
training runners
evaluation runners
```

are user-facing execution entrypoints, while:

```text
cnt_postprocess_pipeline.py
```

is intentionally named as a pipeline because it implements a reusable sequence of postprocessing operations.

---

# Reproducibility Constraints

The pipeline contains several operations whose ordering directly affects inference results.

The following should therefore be treated as algorithmic behavior:

```text
polygon construction order
score sorting
NMS threshold
NMS position in the pipeline
polygon simplification tolerance
whether smoothing is enabled
shape classification
merge thresholds
merge algorithm selection
orientation calculation
score propagation
stage ordering
```

Structural cleanup should avoid modifying these behaviors.

Safe changes generally include:

```text
moving plotting implementation into visualization modules
removing confirmed unused imports
adding documentation
improving formatting
clarifying variable names when semantics remain identical
extracting output-path logic
removing confirmed dead code
```

Changes such as:

```text
reordering operations
changing sorting rules
changing NMS behavior
changing polygon simplification
changing merge thresholds
changing orientation implementation
changing overlap-region classification
```

should instead be handled as explicit algorithm changes with regression tests or dedicated issues.

---

# Current Design Summary

The pipeline package currently follows this responsibility boundary:

```text
StarDist raw candidate representation
              │
              ▼
        pipeline/
              │
              ├── polygon creation      → geometry/
              ├── suppression           → filtering/
              ├── shape classification  → filtering/
              ├── polygon merging       → geometry/
              ├── orientation           → features/
              └── diagnostics           → visualization/
              │
              ▼
      final CNT polygons
```

This is the intended architecture moving forward:

```text
pipeline = orchestration
geometry = polygon construction and merging
filtering = suppression and shape decisions
features = reusable measurements
visualization = diagnostics
adapters = external representation conversion
```

The current no-overlap implementation should remain reproducible while further repository cleanup is performed.