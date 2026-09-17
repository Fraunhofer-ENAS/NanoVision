# IO

The `io` package defines the repository-wide filesystem contract and experiment run management used throughout the project.

Its purpose is to centralize path construction and avoid scattered absolute paths, relative-path assumptions, and duplicated run-directory logic.

## Structure

```text
io/
├── file_io.py
├── paths.py
├── run_manager.py
└── __init__.py
``` 

### file_io.py

Contains small generic file I/O helpers currently used by project workflows.

Current responsibilities include:

- reading UTF-8 JSON files;
- writing UTF-8 text files;
- creating parent directories when necessary.

Workflow-specific serialization should remain in the corresponding package rather than being added here automatically.

### paths.py

Defines the canonical filesystem structure of the repository.

The main abstractions are:

- ProjectPaths: repository-level paths and generated-output paths;
- DatasetPaths: paths associated with one prepared unsplit dataset.

#### Repository layout
```text
repo/
├── data/
│   └── <dataset_name>/
├── models/
├── global_outputs/
│   ├── runs/
│   ├── reports/
│   └── misc/
└── src/
    └── cnt_project/
```

#### Dataset layout

Canonical prepared datasets use an unsplit physical layout:
```text
data/
└── <dataset_name>/
    ├── images/
    ├── masks/
    ├── metadata/
    ├── splits/
    │   ├── default_split.csv
    │   └── default_split.yaml
    └── COCO_mask/
        └── <split_name>/
            ├── train/
            ├── val/
            └── test/
```

Images and masks are stored only once.

Train/validation/test membership is defined by the split manifest rather than by physically separate image or mask folders.

Typical usage:
```code
from cnt_project.io.paths import ProjectPaths

paths = ProjectPaths.from_here(__file__)

dataset = paths.dataset("cnt_segmentation")

images_dir = dataset.images_root
masks_dir = dataset.masks_root
metadata_dir = dataset.metadata_root

split_csv = dataset.split_manifest_csv("default_split")
test_coco = dataset.coco_json(
    "default_split",
    "test",
)
```
DatasetPaths.from_root(...) can also be used independently of the repository layout, which allows library and CLI workflows to operate on external datasets.

#### Generated outputs

Generated workflow artifacts follow:
```text
global_outputs/
├── runs/
│   └── <run_name>/
│       ├── train/
│       ├── inference/
│       ├── eval/
│       └── viz/
├── reports/
└── misc/
```
ProjectPaths provides helpers such as:

- run_dir(...)
- train_dir(...)
- inference_dir(...)
- eval_dir(...)
- eval_subdir(...)
- viz_dir(...)
- report_dir(...)
- misc_dir(...)

Prediction JSON helpers are also available for the standard inference artifacts:

- predicted_poly_json(...)
- predicted_mask_json(...)
- predicted_rle_json(...)

#### Models

Repository-managed model assets are stored under:

models/

and exposed through:

paths.models_root

Training-run artifacts should normally remain under:

global_outputs/runs/<run_name>/train/

rather than being mixed with persistent model assets.

#### Legacy annotations_uniques compatibility

annotations_uniques is the historical manually split dataset.

It is not the canonical dataset layout.

It remains available because:

some older evaluation and visualization code still expects its physical train/test directory structure;
its historical train/test membership is currently used as the reference for reproducing the project's default split.

Legacy fields and helpers such as:

- test_root
- test_images_root
- test_coco_json
- images_root("test")
- metadata_file("test", ...)

are retained temporarily for compatibility.

New code should use DatasetPaths instead.

These compatibility paths can be removed after the remaining legacy evaluation and visualization callers are migrated.

### run_manager.py

Handles experiment run identity and run lifecycle.

Responsibilities include:

- generating filesystem-safe run names;
- generating timestamped run identifiers;
- creating canonical run directories;
- attaching to existing runs;
- returning a RunPaths bundle for one run.

Main API:

- safe_run_name(...)
- timestamp_now(...)
- make_run_id(...)
- build_run_name(...)
- create_run(...)
- attach_run(...)
- RunPaths

A run follows the canonical structure:
```text
global_outputs/
└── runs/
    └── <run_id>/
        ├── train/
        ├── inference/
        ├── eval/
        └── viz/
            └── figures/
```
ProjectPaths defines where run storage lives, while run_manager.py manages the lifecycle of an individual run.

#### Design rule

Core library functions should accept explicit paths or dataset objects whenever practical.

Repository-specific path discovery belongs primarily in runners, CLI entry points, and orchestration code.

This allows the project to support both:

reproducible workflows when cloned as a repository;
reusable library workflows operating on externally supplied datasets and output locations.

### Library Usage Status

The repository is being structured so that `cnt_project` can be used both as a cloned research repository and as an installable Python library.

The current path architecture follows this distinction:

- **Repository workflows and runners** may use `ProjectPaths` to discover the repository structure and provide convenient default locations for datasets, models, and generated outputs.
- **Reusable library APIs** should accept explicit paths, data objects, or configuration rather than depending on the repository layout.
- `DatasetPaths.from_root(...)` allows datasets located outside the repository to use the canonical CNT dataset structure without requiring `<repo>/data/...`.
- Legacy dataset properties such as `test_root`, `test_images_root`, and `test_coco_json` are retained temporarily for compatibility with code that has not yet been refactored, particularly evaluation workflows.

Remaining library-readiness work:

- Refactor remaining legacy/evaluation workflows away from assumptions about `annotations_uniques`.
- Ensure reusable APIs do not depend on `ProjectPaths.from_here(...)` or the current working directory.
- Verify editable installation with `pip install -e .`.
- Build and install the package wheel in a clean environment.
- Run smoke tests from outside the repository to detect hidden repository-path assumptions.

`ProjectPaths` is intentionally repository-oriented; external library users should normally use explicit paths and abstractions such as `DatasetPaths` rather than depending on the repository filesystem layout.