# Preprocessing

This package contains the data preparation code used before model training, evaluation, and reporting.

## Responsibilities

- dataset loading
- image and mask normalization
- dataset conversion and augmentation
- metadata generation
- validation utilities
- preprocessing runners

## Package Layout

```text
src/cnt_project/preprocessing/
├── dataset/
├── metadata/
├── runners/
└── ...
```

## Main Areas

### `dataset/`
Dataset-specific helpers and documentation for preprocessing workflows.

### `metadata/`
Generation and validation of dataset metadata such as density and noise labels.

### `runners/`
Command-line entry points for preprocessing tasks.

## Notes

- Keep reusable preprocessing logic in this package.
- Prefer absolute imports rooted at `cnt_project`.
- Preserve notebook and legacy compatibility unless explicitly refactoring behavior.