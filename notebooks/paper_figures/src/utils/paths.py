# src/utils/paths.py
from pathlib import Path

# ----------------------------------------------------------------------
# 1. Determine project folders
# ----------------------------------------------------------------------
# paths.py -> utils -> src -> paper_figures -> notebooks -> cnt_project_v2
PROJECT_ROOT = Path(__file__).resolve().parents[4]

# paper_figures folder
ROOT = PROJECT_ROOT / "notebooks" / "paper_figures"

# ----------------------------------------------------------------------
# 2. Common output folders
# ----------------------------------------------------------------------
OUTPUTS = {
    "density": ROOT / "outputs" / "density",
    "noise": ROOT / "outputs" / "noise",
    "scoring": ROOT / "outputs" / "scoring",
    "histograms": ROOT / "outputs" / "histograms",
}

for p in OUTPUTS.values():
    p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 3. Ground-truth annotation paths
# ----------------------------------------------------------------------
GT_ANNOTATIONS_PATH = (
    PROJECT_ROOT / "data" / "annotations_uniques" / "COCO_mask" / "annotations.json"
)

GT_TEST_ANNOTATIONS_PATH_LEGACY = (
    PROJECT_ROOT / "data" / "annotations_uniques" / "test" / "COCO_mask" / "annotations.json"
)
GT_TEST_ANNOTATIONS_PATH = (
    PROJECT_ROOT / "data" / "cnt_segmentation" / "COCO_mask" / "default_split" / "test" /"annotations_chain_approx_none.json"
)

# ----------------------------------------------------------------------
# 4. Model output locations
# ----------------------------------------------------------------------
MODEL_OUTPUTS_ROOT = PROJECT_ROOT / "global_outputs" / "runs"

Older_MODEL_DIRS = {
    "Nanovision": MODEL_OUTPUTS_ROOT / "fluo_new_edt" / "inference",
    "MaskRCNN":   MODEL_OUTPUTS_ROOT / "detectron2_trial_1" / "inference",
    "WormSwin":   MODEL_OUTPUTS_ROOT / "wormswin" / "inference",
    "Nano1D":     MODEL_OUTPUTS_ROOT / "nano1D" / "inference",
}

NERWER_MODEL_DIRS = {
    "Nanovision": MODEL_OUTPUTS_ROOT / "fluo_edt_edt_score_symmetry_score_without_smoothing_20260610_1004" / "inference",
    "MaskRCNN":   MODEL_OUTPUTS_ROOT / "detectron2_trial_1_LB_legacy_recheck" / "inference",
    "WormSwin":   MODEL_OUTPUTS_ROOT / "wormswin_20260610" / "inference",
    "Nano1D":     MODEL_OUTPUTS_ROOT / "nano1D_20260610" / "inference",
}