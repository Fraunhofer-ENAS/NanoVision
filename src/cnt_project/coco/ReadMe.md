# COCO Generation

This package contains both the legacy COCO conversion helpers and the new split-aware COCO generation workflow for the unified dataset layout.

## Scope

The `cnt_project.coco` package now supports two generation styles:

1. Legacy conversion from a whole mask directory.
2. Split-aware generation from:
   - unified dataset root;
   - split manifest;
   - subset name.

The split-aware path is the canonical path for the new dataset structure.

---

## Legacy Conversion

Legacy TIFF-to-COCO conversion remains available in:

- `convert_tif_annotations_to_json(...)`

File:

- `src/cnt_project/coco/convert.py`

### Legacy behavior

When called with only:

- `mask_folder`
- `output_json_path`

it scans all `.tif` masks in the folder and writes one COCO JSON file.

This preserves backward compatibility with existing preprocessing and notebook flows.

### Extended behavior

The same function now also supports:

- `selected_filenames`
- `preserve_tif_filenames`
- `category_id`

It can therefore be used in both legacy and split-aware workflows.

### New guarantees for selected-filename mode

When `selected_filenames` is provided, the function:

- accepts an explicit list of `.tif` filenames;
- rejects duplicate selected filenames;
- validates that every selected mask exists;
- validates that every mask is two-dimensional;
- converts only the selected files;
- preserves `.tif` filenames in `images[*].file_name` when requested;
- computes annotation area as the sum of contour areas;
- returns the generated COCO dictionary in addition to writing JSON.

### Related helper

- `build_coco_from_instance_masks(...)`

This helper constructs the COCO dictionary directly from instance-indexed masks and is the reusable core behind TIFF-to-COCO generation.

---

## Split-Aware COCO Generation

Split-aware orchestration lives in:

- `src/cnt_project/coco/generate.py`

Validation lives in:

- `src/cnt_project/coco/validation.py`

Runner lives in:

- `src/cnt_project/coco/runners/generate_split_coco_runner.py`

### `generate_coco_for_subset(...)`

Responsibilities:

- validate `dataset_root/images` and `dataset_root/masks`;
- read and validate the split manifest;
- filter filenames for one subset;
- reject an empty subset;
- convert only selected masks;
- preserve `.tif` names in COCO image entries;
- write the subset JSON or JSON variants;
- validate generated COCO structure.

Default output path:

```text
data/<dataset_root>/COCO_mask/<split_name>/<subset>/annotations_chain_approx_simple.json
data/<dataset_root>/COCO_mask/<split_name>/<subset>/annotations_chain_approx_none.json
```

By default, split-aware generation emits both contour-approximation variants to preserve legacy GT workflows.

- `annotations_chain_approx_simple.json`
- `annotations_chain_approx_none.json`

If needed, the runner can be told to emit only a single `annotations.json` file.

### `generate_coco_for_split(...)`

Responsibilities:

- read the manifest once;
- determine which subsets exist;
- generate COCO JSON for `train`, `val`, and `test`;
- skip or reject missing subsets according to explicit policy;
- return generated output paths.

Supported missing-subset policies:

- `reject`
- `skip`

---

## Validation

COCO-specific validation is intentionally separate from generation logic.

### `validate_coco_dataset_structure(...)`

Validates the unified dataset structure and exact image-mask pairing.

### `validate_coco_json_dict(...)`

Validates generated COCO content:

- required top-level keys exist;
- `images`, `annotations`, `categories` are lists;
- image IDs are unique;
- annotation IDs are unique;
- annotation `image_id` references are valid;
- annotation `category_id` references are valid;
- annotation `bbox` values have valid shape.

### `validate_split_manifest_for_coco(...)`

Validates that the manifest contains the columns needed by COCO generation:

- `filename`
- `subset`

---

## Public API

The `cnt_project.coco` package exports:

- `build_coco_from_instance_masks`
- `convert_tif_annotations_to_json`
- `generate_coco_for_subset`
- `generate_coco_for_split`
- `validate_coco_dataset_structure`
- `validate_coco_json_dict`

along with the existing prediction/polygon conversion helpers.

---

## Runner Usage

### Generate COCO for one subset

```powershell
$env:PYTHONPATH='src'
python -m cnt_project.coco.runners.generate_split_coco_runner \
  --dataset-root data/cnt_segmentation \
  --split-manifest-path data/cnt_segmentation/splits/stratified_seed_42.csv \
  --subset train \
  --overwrite
```

### Generate COCO for all subsets in a split

```powershell
$env:PYTHONPATH='src'
python -m cnt_project.coco.runners.generate_split_coco_runner \
  --dataset-root data/cnt_segmentation \
  --split-manifest-path data/cnt_segmentation/splits/stratified_seed_42.csv \
  --missing-subset-policy skip \
  --overwrite
```

### Generate only a single default JSON per subset

```powershell
$env:PYTHONPATH='src'
python -m cnt_project.coco.runners.generate_split_coco_runner \
  --dataset-root data/cnt_segmentation \
  --split-manifest-path data/cnt_segmentation/splits/stratified_seed_42.csv \
  --subset train \
  --single-default-json \
  --overwrite
```

---

## Legacy vs Canonical Guidance

### Use legacy conversion when

- you want to convert an entire mask directory without a split manifest;
- you are maintaining older preprocessing or notebook flows;
- you do not need subset-aware generation.

### Use split-aware generation when

- you are working with the unified dataset layout;
- subset membership should come from a split manifest;
- COCO JSON files must be generated per subset;
- reproducibility and validation matter.

---

## Design Rule

Legacy conversion is kept for compatibility.

Split-aware COCO generation is the intended path for the new dataset architecture.
