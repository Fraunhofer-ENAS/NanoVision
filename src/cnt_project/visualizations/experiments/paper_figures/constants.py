
# src/utils/constants.py

MICRONS_PER_PIXEL = 5.0 / 256.0
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# Default color palette used across figures
COLOR_MAP = {
    'NanoVision': '#4169e1',   # Royal Blue
    'MaskRCNN':   '#e34234',   # Vermilion
    'Nano1D':     '#50c878',   # Emerald
    'WormSwin':   '#9966cc',   # Amethyst,
    'GT':         '#f0e68c',   # Khaki for ground-truth
}

# Common figure sizes (inches) for single-column papers
FIG_SIZES = {
    'single_full': (6.0, 4.0),
    "hist_fig" : (4.3, 2.5),
    'single_max':  (7.0, 4.5),
    'square':      (6.0, 6.0),
    "density_dice_fig": (4.5, 2.8),
    'double_half': (3.3, 2.5),   # for 2 side-by-side
    'triple_third':(2.0, 1.8),   # for 3 side-by-side
}

# --- Scoring / model display names used across figures ---
SCORING_NAME_MAP = {
    'fluo_centerness': 'SYM',
    'fluo_edt_standard': 'EDT',
    'fluo_new_edt': 'EDT-SYM',
    'fluo_edt_LB': 'EDT-FULL',
}

# --- Palette for scoring figures (SYM / EDT variants) ---
SCORING_COLOR_MAP = {
    'SYM': '#a6cee3',       # light blue
    'EDT': '#1f78b4',       # medium blue
    'EDT-SYM': '#6baed6',   # soft steel blue
    'EDT-FULL': '#08306b',  # dark navy blue
}

# --- Consistent order on x-axis / legends for scoring plots ---
SCORING_EXPERIMENT_ORDER = ['EDT', 'EDT-SYM', 'EDT-FULL', 'SYM']


# ----------------------------------------------------------------------
# AP CURVE CONSTANTS (IoU vs AP)
# ----------------------------------------------------------------------

# Short human-readable scoring names
AP_NAME_MAP = {
    'fluo_centerness': 'SYM',
    'fluo_edt_standard': 'EDT',
    'fluo_new_edt': 'EDT-SYM',
    'fluo_edt_LB': 'EDT-FULL',
}

# Colors for AP curves
AP_COLOR_MAP = {
    'SYM': '#a6cee3',       # light blue
    'EDT': '#1f78b4',       # medium blue
    'EDT-SYM': '#6baed6',   # soft steel blue
    'EDT-FULL': '#08306b',  # deep navy
}

# Dash patterns for AP curve line styles
AP_DASH_MAP = {
    'EDT-SYM': (),          # solid
    'EDT': (4, 2),          # medium dashes
    'EDT-FULL': (6, 2),     # long dashes
    'SYM': (2, 2, 6, 2),    # complex dashed
}

# Legend order for consistent plotting
AP_LEGEND_ORDER = ["EDT", "EDT-SYM", "EDT-FULL", "SYM"]


# ----------------------------------------------------------------------
# F1 CURVE CONSTANTS (IoU vs F1)
# ----------------------------------------------------------------------

F1_NAME_MAP = {
    'fluo_new_edt': 'NanoVision',
    'wormswin': 'WormSwin',
    'detectron2_trial_1': 'MaskRCNN',
    'nano1D': 'Nano1D',
}

# Default order when plotting
F1_LEGEND_ORDER = ['NanoVision', 'MaskRCNN', 'WormSwin', 'Nano1D']


# ----------------------------------------------------------------------
# PRECISION/AP vs IoU curves (per model)
# ----------------------------------------------------------------------

AP2_NAME_MAP = {
    'fluo_new_edt': 'NanoVision',
    'detectron2_trial_1': 'MaskRCNN',
    'wormswin': 'WormSwin',
    'nano1D': 'Nano1D',
}

AP2_COLOR_MAP = {
    'NanoVision': '#4169e1',   # Royal Blue
    'MaskRCNN':   '#e34234',   # Vermilion
    'Nano1D':     '#50c878',   # Emerald
    'WormSwin':   '#9966cc',   # Purple
}

AP2_LEGEND_ORDER = ['NanoVision', 'MaskRCNN', 'WormSwin', 'Nano1D']
