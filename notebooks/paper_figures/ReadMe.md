
CNT Paper Figure Notebooks & Required Input Files

  

This document explains how to run the analysis notebooks, where to place the required input files, and how the folder structure must be organized so all notebooks run correctly.

  

Paths are determined by the project’s file src/utils/paths.py.

  

# 1. Project Structure Overview

  

Your project must contain the following directories:

  

paper_figures/ ← ROOT for notebooks

testing/ ← TESTING_PROJECT_ROOT

cnt_project_v2/ ← PROJECT_ROOT

cnt_project_v2/Global_Outputs/edt_comparison/ ← MODEL_OUTPUTS_ROOT

  
  

Notebooks expect input files in these folders.

  

# 2. Required Input Files (Place These Exactly)

A. Ground-Truth Annotation Files

1. Main GT annotations

testing/data/annotations_uniques/COCO_mask/annotations.json

  

2. GT annotations for the test set

testing/data/annotations_uniques/test/COCO_mask/annotations.json

  

B. Noise Estimation Inputs (performed on the training data to illustrate that in the figure)

1. Images used for noise computation

testing/data/annotations_uniques/images/*.png|jpg|tif

  

2. Noise CSV (computed or pre-generated)

testing/data/annotations_uniques/images/noise_evaluation_results.csv

  

3. Noise with classes (used for Precision vs Noise / Width figures)

testing/data/annotations_uniques/test/COCO_mask/noise_evaluation_with_classes.csv

  

C. Density Classification File

  

Required for “Density × DICE” plots.

  

TODO: i need to change the location of this file to make more sense

  

cnt_project_v2/GlobalDensity_classified_filenames.csv

  

D. Model Prediction Files

  

Each model must have:

  

cnt_project_v2/Global_Outputs/edt_comparison/<model_name>/predicted_annotations_poly.json

  
  

Required models:

  

fluo_new_edt

detectron2_trial_1

wormswin

nano1D

  

E. Model Metric Files

  

Required for F1 curves, AP curves, Precision vs Width, DICE boxplots.

  

Each model folder must contain:

  

ap_results_poly_ALL.csv

per_image_threshold_metrics.csv

per_image_threshold_metrics_calc_dsb_map.csv

  
  

Example:

  

cnt_project_v2/Global_Outputs/edt_comparison/fluo_new_edt/ap_results_poly_ALL.csv

cnt_project_v2/Global_Outputs/edt_comparison/fluo_new_edt/per_image_threshold_metrics.csv

cnt_project_v2/Global_Outputs/edt_comparison/fluo_new_edt/per_image_threshold_metrics_calc_dsb_map.csv

  

F. Test-Set Line Density File

  

Used for operational limit analysis (obtained through the GUI on the test set):

  

testing/data/interrim/CNT_analysis_tool_test_set_vonmises_fit/line_density_per_row.csv

  

G. High-Density Dataset Files

1. Martin Excel GT file

testing/data/interrim/high_density_images/martin/CNT_counts_summary.xlsx

  

2. KI-tool predicted density rows

testing/data/interrim/high_density_images/ki_tool/line_density_per_row.csv

  

# 3. Output Folders (Created Automatically)

  

The notebooks write outputs to:

  

paper_figures/outputs/density/

paper_figures/outputs/noise/

paper_figures/outputs/scoring/

paper_figures/outputs/histograms/

paper_figures/outputs/precision_width/

  
  

These are created automatically by paths.py.

  

# 4. How to Use the Notebooks (Step-by-Step)

Step 1 — Place all required input files

  

Ensure all CSV, Excel, JSON, and prediction files are in their correct paths (as listed above).

  

Step 2 — Install dependencies

  

Create your environment and install dependencies:

  

pip install -r requirements.txt

  
  

Or use conda:

  

conda create -n cnt_env python=3.10

conda activate cnt_env

pip install -r requirements.txt

  

Step 3 — Open paper_figures/ in VS Code or Jupyter

  

Notebooks rely on relative paths defined in src/utils/paths.py.

  

Step 4 — Run any notebook

  

Each notebook uses:

  

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))

sys.path.append(project_root)

  
  

So they automatically locate src/ and the input files.

  

Step 5 — Outputs appear under paper_figures/outputs/

  

Figures are saved as *.svg inside the correct subfolder.
