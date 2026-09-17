from __future__ import annotations

FILENAME_COL = "filename"

GT_OBJECT_COUNT_COL = "gt_object_count"
DENSITY_CLASS_COL = "density_class"
DENSITY_CLASS_TERTILE_COL = "density_class_tertile"
DENSITY_CLASS_KMEANS_COL = "density_class_kmeans"

NOISE_SIGMA_COL = "noise_sigma"
NOISE_CLASS_COL = "noise_class"
NOISE_CLASS_OTSU_COL = "noise_class_otsu"
NOISE_CLASS_KMEANS_COL = "noise_class_kmeans"

ALLOWED_DENSITY_CLASSES = ("Low", "Mid", "High")
ALLOWED_NOISE_CLASSES = ("Clean", "Noisy")
