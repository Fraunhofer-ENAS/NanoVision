# Postprocessing Adapters

This folder contains the adapter layer between the CNT-specific polygon postprocessing pipeline and the StarDist-compatible representations expected by the patched StarDist model.

The files in this folder do not implement the core polygon-merging algorithm itself. Instead, they are responsible for:

- converting between Shapely polygon representations and StarDist-style arrays or label images;
- adapting raw StarDist candidate predictions to the CNT polygon postprocessing pipeline;
- reconstructing StarDist-compatible outputs after the CNT-specific merging stage.

The current structure is:

```text
adapters/
├── format_conversion.py
├── stardist_postprocess_adapter.py
└── __init__.py
```

## `format_conversion.py`

This module contains representation-conversion helpers used after polygon merging.

Its main responsibility is to translate between:

```text
Shapely polygons
    ↕
raster instance labels
    ↕
StarDist-style polygon coordinate arrays
    ↕
StarDist-style center/probability representations
```

The module currently provides helpers for:

- rasterizing final Shapely `Polygon` and `MultiPolygon` objects into an instance-label image;
- converting polygon boundaries into row/column coordinates suitable for `skimage.draw.polygon`;
- sampling a merged polygon into a fixed number of radial coordinates;
- constructing StarDist-compatible polygon coordinate arrays;
- assigning representative center coordinates and probabilities to merged polygons.

These functions mainly support reconstruction of the output variables expected by the patched StarDist inference code after CNT-specific polygon merging has taken place.

### Main downstream usage

The public conversion helpers are currently consumed by:

```text
stardist_postprocess_adapter.py
```

In particular:

```text
draw_polygons_on_image(...)
create_polygon_array(...)
extract_centers_probabilities(...)
```

are used when rebuilding the StarDist-compatible outputs from the final merged polygons.

The lower-level helpers:

```text
polygon_to_coords(...)
sample_polygon(...)
```

are internal dependencies of those conversion functions.

## `stardist_postprocess_adapter.py`

This module is the main integration layer between patched StarDist inference and the CNT-specific postprocessing implementation.

Its purpose is to preserve a StarDist-compatible interface while delegating the actual polygon processing to:

```text
postprocessing/pipeline/cnt_postprocess_pipeline.py
```

The adapter currently uses the no-overlap-region postprocessing path:

```python
merge_instances_no_overlap_regions(...)
```

The core pipeline therefore remains independent of the exact output tuple expected by the patched StarDist model.

## Adapter Flow

The current execution sequence is approximately:

```text
Patched StarDist inference
        │
        │ raw candidate outputs
        │
        │ dist
        │ prob
        │ points
        ▼
postprocess_instances_merge_polygons(...)
        │
        │ validates/converts arrays
        │ sorts candidates by probability
        ▼
instances_merge_adapter(...)
        │
        │ prepares contiguous NumPy arrays
        ▼
build_stardist_outputs_from_merged_polygons(...)
        │
        │ delegates CNT-specific processing
        ▼
merge_instances_no_overlap_regions(...)
        │
        │
        │  CNT postprocessing pipeline:
        │  polygon construction
        │  candidate filtering / NMS
        │  polygon merging
        │  optional smoothing
        │  final polygon construction
        │
        ▼
final Shapely polygons
        │
        ├──► draw_polygons_on_image(...)
        │       └── instance-label image
        │
        ├──► create_polygon_array(...)
        │       └── StarDist-compatible coordinate array
        │
        └──► extract_centers_probabilities(...)
                └── representative centers and probabilities
        │
        ▼
StarDist-compatible postprocessed outputs
```

## `postprocess_instances_merge_polygons(...)`

This is the main adapter entry point used by the patched StarDist model.

It accepts StarDist candidate predictions:

```text
dist
prob
points
```

and prepares them for the CNT-specific merging pipeline.

Before delegating the processing, candidates are sorted by probability in descending order.

The function ultimately returns the postprocessed results in the tuple structure expected by the patched StarDist inference implementation.

An optional `stage_outputs` value can also be propagated when:

```python
return_stage_outputs=True
```

This is primarily intended for diagnostics and postprocessing-stage inspection while keeping the default return contract compatible with the existing inference path.

## `instances_merge_adapter(...)`

This function is an intermediate compatibility layer.

It:

- validates the main array dimensions;
- converts inputs to contiguous NumPy arrays with the expected data types;
- delegates the actual construction of postprocessed outputs to `build_stardist_outputs_from_merged_polygons(...)`;
- preserves the surrounding StarDist-style return structure.

It intentionally does not implement the polygon-merging algorithm itself.

## `build_stardist_outputs_from_merged_polygons(...)`

This function forms the boundary between the CNT polygon pipeline and StarDist-compatible outputs.

It first delegates polygon processing to:

```python
merge_instances_no_overlap_regions(...)
```

which returns the final CNT polygon representations.

The resulting polygons are then converted back into representations needed by the patched StarDist code:

```text
final_cell_shapes
        │
        ├──► new_label
        │     raster instance-label image
        │
        ├──► new_coord
        │     fixed-size polygon coordinate representation
        │
        ├──► new_points
        │     representative merged-instance centers
        │
        └──► new_prob
              representative probabilities
```

The final Shapely polygons, scores, and coordinates are also preserved and returned.

## Integration with Patched StarDist

The primary external consumer of this adapter layer is:

```text
src/cnt_project/model_development/stardist_patched/
    stardist_model_configuration.py
```

The patched model calls:

```python
postprocess_instances_merge_polygons(...)
```

during instance prediction when StarDist candidate points are available.

The relationship is therefore:

```text
stardist_patched/
        │
        ▼
postprocessing/adapters/
        │
        ▼
postprocessing/pipeline/
        │
        ├──► filtering/
        ├──► geometry/
        └──► visualization/   # when debug plotting is enabled
```

## Design Boundary

The intended responsibility of the adapter folder is:

```text
StarDist representation
        ↕
CNT postprocessing representation
```

Algorithm-specific functionality should remain outside this folder.

For example:

```text
polygon construction
    → geometry/

NMS and shape filtering
    → filtering/

polygon merging
    → geometry/

complete postprocessing orchestration
    → pipeline/

debug and polygon visualization
    → visualization/

StarDist ↔ polygon representation conversion
    → adapters/
```

Keeping this boundary makes it possible to modify or inspect the CNT postprocessing pipeline without tightly coupling its internal representation to the patched StarDist model.

