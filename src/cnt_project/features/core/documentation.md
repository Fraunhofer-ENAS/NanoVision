# CNT Morphology Feature Extraction

This document defines the **current morphology feature extraction pipeline** used throughout the project. All features are extracted from COCO polygon annotations using a standardized extraction procedure implemented in `cnt_feature_extraction.py`.

Unless otherwise stated, morphology measurements are performed on a **decoded binary mask** obtained from the COCO polygon annotation, while orientation is computed directly from the original polygon coordinates.

---

# Object-Level Feature Definitions

| Feature | Source | Definition | Extraction Method | Unit | Notes |
|---------|--------|------------|-------------------|------|------|
| **Area** | Decoded binary mask | Total foreground area occupied by the CNT | Count the number of foreground pixels in the decoded COCO mask (`np.count_nonzero(mask)`) | px² | One value per annotation. |
| **Area (µm²)** | Derived | Physical CNT area | `Area × (µm/pixel)²` | µm² | Uses the configured pixel calibration. |
| **Perimeter** | Hole-filled binary mask | Total CNT boundary length | Crofton perimeter estimator (`perimeter_crofton`, four directions) applied to the hole-filled binary mask. The Crofton method estimates the object boundary length by averaging boundary intersections along multiple directions, providing a robust and nearly rotation-invariant perimeter estimate for digital objects. | px | Canonical perimeter definition. |
| **Perimeter (µm)** | Derived | Physical perimeter | `Perimeter × µm/pixel` | µm | Converted using the image calibration. |
| **Length** | Binary mask | Geodesic centerline length of the CNT | `polygon_mask_geodesic_length()` computes the longest shortest path through a weighted 8-connected pixel graph. The weighted graph is only used to determine the optimal centerline path; the reported length is the geometric length of that path using orthogonal steps = 1 px and diagonal steps = √2 px. | px / µm | Canonical length definition. |
| **Width** | Binary mask + EDT | Mean CNT thickness | Compute the Euclidean Distance Transform (EDT) of the binary mask and evaluate `2 × mean(EDT)` along the geodesic centerline path. If no valid path is available, use `2 × mean(EDT)` over all foreground pixels. | px / µm | Canonical width definition. |
| **Aspect Ratio** | Derived | CNT elongation | `Geodesic Length / Path-based Width` | Unitless | Derived from the canonical length and width definitions. |
| **Orientation Angle** | Original COCO polygon coordinates | Principal orientation of the CNT | Principal Component Analysis (PCA) performed directly on the original polygon coordinates. The resulting angle is normalized to the range [-90°, 90°]. | Degrees | Independent of rasterization resolution. |
| **Connected Components** | Binary mask | Number of disconnected foreground regions within a single annotation | Connected-component labeling (`skimage.measure.label`) | Count | Used as a morphology diagnostic feature. |

---

# Image-Level Features

After all object-level features have been extracted, the following image-level statistics are computed.

| Feature | Definition |
|----------|------------|
| `num_polygons` | Number of successfully extracted CNT objects. |
| `cnt_density_per_um2` | Number of successfully extracted CNT objects divided by the physical image area (CNTs/µm²). |
| `average_area_pixels2` | Mean object area. |
| `average_area_um2` | Mean physical object area. |
| `average_perimeter` | Mean object perimeter. |
| `average_perimeter_um` | Mean physical perimeter. |
| `average_length` | Mean geodesic CNT length. |
| `average_length_um` | Mean physical geodesic length. |
| `average_width` | Mean EDT-based width. |
| `average_width_um` | Mean physical width. |
| `average_aspect_ratio` | Mean aspect ratio of all CNTs. |
| `average_orientation_angle` | Mean orientation angle of all CNTs. |
| `line_density_mean` | Mean value of the line-density profile. |
| `line_density_std` | Standard deviation of the line-density profile. |
| `line_density_max` | Maximum number of CNTs intersecting any image row. |

---

# Line Density

Line density characterizes the spatial distribution of CNTs across the image.

For every annotation, all image rows intersecting the decoded binary mask are identified, and a counter is incremented for each intersected row. The resulting line-density profile contains one value for every image row.

The following outputs are exported:

| Feature | Definition |
|----------|------------|
| `line_density` | Number of CNTs intersecting each image row. |
| `line_density_mean` | Mean value of the line-density profile. |
| `line_density_std` | Standard deviation of the line-density profile. |
| `line_density_max` | Maximum number of CNTs intersecting any image row. |

---

# Orientation Statistics

Orientation statistics are computed from the set of object orientation angles extracted for each image.

## Nematic Order Parameter

The nematic order parameter is computed using the doubled-angle formulation, treating orientations θ and θ + 180° as equivalent.

The following quantities are exported:

- `nematic_order_parameter`
- `nematic_director_angle_deg`
- `nematic_calculation_success`

---

## Von Mises Orientation Distribution

A von Mises distribution is fitted to the doubled orientation angles to characterize the angular distribution of CNT orientations.

The following quantities are exported:

- `von_mises_mean_orientation_deg`
- `von_mises_concentration_kappa`
- `von_mises_circular_variance`
- `von_mises_confidence_95_deg`
- `von_mises_fit_success`
- `num_objects_for_fit`

---

# Implementation Notes

- Each COCO annotation corresponds to one extracted CNT object.
- Morphological measurements (area, perimeter, length, width, aspect ratio, connected components and line density) are computed from the decoded binary mask.
- Perimeter is computed on the **hole-filled** binary mask using the Crofton estimator.
- Length is defined as the geodesic centerline returned by `polygon_mask_geodesic_length()`.
- Width is computed from the Euclidean Distance Transform (EDT) sampled along the geodesic centerline.
- Aspect ratio is defined as the geodesic length divided by the path-based width.
- Orientation is computed directly from the original COCO polygon coordinates using PCA.
- Image-level statistics are computed only from successfully extracted objects.

---