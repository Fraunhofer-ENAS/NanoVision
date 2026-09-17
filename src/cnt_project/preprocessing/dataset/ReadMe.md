# Dataset Preprocessing

This package contains dataset-level preprocessing documentation and helpers for working with CNT image datasets.

## Purpose

The dataset preprocessing layer supports:

- dataset organization
- sample validation
- image and mask pairing checks
- metadata generation inputs
- split-ready dataset preparation

## Expected Dataset Structure

```text
data/
└── cnt_segmentation/
    ├── images/
    ├── masks/
    ├── metadata/
    ├── splits/
    └── COCO_mask/
```

## Dataset Rules

- Only `.tif` files are treated as dataset samples.
- Each image must have a matching mask with the same filename.
- Every mask must have a corresponding image.
- Filenames are used as stable sample identifiers.

## Related Modules

- `cnt_project.preprocessing.metadata`
- `cnt_project.preprocessing.runners`
- `cnt_project.preprocessing.validation`

## Workflow

1. Load dataset files.
2. Validate image-mask pairing.
3. Generate metadata.
4. Prepare split-ready outputs.