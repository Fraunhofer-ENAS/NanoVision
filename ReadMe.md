# CNTLib

CNTLib is a Python library and research repository for carbon nanotube (CNT) image analysis.

The project provides reusable functionality for CNT data preprocessing,
StarDist-based instance segmentation, inference, morphological feature
extraction, evaluation, and visualization. The repository also contains
notebooks and experiment workflows for developing and evaluating new methods.

The reusable Python source code lives under `src/`.

## Quick Start: Example Workflows

For a practical introduction to CNTLib, the recommended entry point is the
`notebooks/examples/` directory. These notebooks provide compact workflows that
use the reusable implementation under `src/cnt_project/` to demonstrate the
main stages of the CNT analysis pipeline.

The examples are organized in workflow order:

```text
notebooks/examples/
├── 01_dataset_preprocessing.ipynb       # Dataset preparation and COCO generationSON 
├── 02_training.ipynb                    # StarDist model training
├── 03_inference_and_evaluation.ipynb    # Model inference and segmentation evaluation
└── 04_feature_extraction.ipynb          # CNT morphological feature extraction
```

For reviewers and users who want to reproduce or inspect the main functionality, these notebooks provide the
simplest entry point.


## Main Capabilities

The repository currently provides functionality for:

- dataset splitting and metadata generation
- COCO annotation generation, conversion, validation, and manipulation
- StarDist model configuration, training, and inference
- prediction postprocessing and threshold optimization
- CNT morphological feature extraction
- COCO-, DSB-, density-, noise-, and morphology-based evaluation
- evaluation diagnostics and visualization
- experiment and notebook-based research workflows

More detailed documentation is provided by the README files inside the
corresponding source packages.


## Repository Structure

```text
cnt-lib/
├── src/
│   └── cnt_project/        # Reusable Python library
├── notebooks/              # Interactive analyses and research workflows
├── data/                   # Local datasets and annotations datasets should be located under here
├── models/                 # Local model files
├── global_outputs/         # Generated runs, reports, and other outputs
├── pyproject.toml          # Package metadata and dependency definitions
├── requirements.txt        # Convenience development installation
└── ReadMe.md
```

The main Python package is organized into several functional areas:

- `cnt_project.preprocessing` — preprocessing, dataset preparation, splitting,
  and metadata
- `cnt_project.coco` — COCO conversion, masks, filtering, validation, and
  related utilities
- `cnt_project.model_development` — training, inference, postprocessing, and
  threshold optimization (stardist based)
- `cnt_project.features` — CNT morphological feature extraction and analysis
- `cnt_project.evaluation` — evaluation metrics, pipelines, diagnostics, and
  runners
- `cnt_project.visualizations` — reusable visualization functionality
- `cnt_project.io` — project paths, file I/O, and run management

Consult the localized README files for detailed documentation of these
components.

## Python Version

CNTLib currently targets Python 3.12:

```text
Python >=3.12,<3.13
```

Python 3.12.10 is the currently tested development version.



## What Can I Use From CNTLib?

CNTLib is organized around several high-level capabilities. Depending on your
task, the following packages are the main entry points.

| Task | Package |
| --- | --- |
| Work with COCO annotations and masks | `cnt_project.coco` |
| Train segmentation models | `cnt_project.model_development.training` |
| Run model inference | `cnt_project.model_development.inference` |
| Postprocess model predictions | `cnt_project.model_development.postprocessing` |
| Optimize model thresholds | `cnt_project.model_development.threshold_optimization` |
| Extract CNT morphological features | `cnt_project.features` |
| Evaluate segmentation results | `cnt_project.evaluation` |
| Generate plots and visualizations | `cnt_project.visualizations` |

Each area contains more specialized functionality and, where applicable,
localized documentation describing its workflows and available entry points.

A typical workflow through the library is:

```text
preprocessing
   ↓
model_development.training
   ↓
model_development.inference
   ↓
model_development.postprocessing
   ↓
coco
   ↓
evaluation / features
   ↓
visualizations
```


## Installation

### Development Installation

Clone the repository and create a virtual environment:

```powershell
git clone <repository-url>
cd cnt-lib

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` is intentionally minimal and installs the project in
editable mode together with the standard development and notebook
dependencies:

```text
-e .[dev,notebooks]
```

The authoritative dependency definitions are maintained in
`pyproject.toml`.

Because the project is installed in editable mode, changes made under `src/`
are immediately available from the active development environment without
reinstalling the package.



## Notebooks and Experiments

The `notebooks/` directory contains interactive research, development, and
analysis workflows.


The repository includes notebook workflows for areas such as model training,
inference, evaluation, dataset analysis, annotation, and experimental
development.

Notebook-specific dependencies are available through the `notebooks`
optional dependency group.


## Data and Generated Outputs

Datasets and generated artifacts are kept outside the Python package.

The main repository-level locations are:

```text
data/             # datasets, masks, annotations, metadata, split manifests
models/           # local trained/imported models
global_outputs/   # generated runs, evaluations, reports, and visualizations
```

Run-specific artifacts generally follow:

```text
global_outputs/runs/<RUN_NAME>/
```

while cross-run reports generally follow:

```text
global_outputs/reports/<REPORT_NAME>/
```

Detailed input/output conventions are documented in the corresponding
localized package documentation.











