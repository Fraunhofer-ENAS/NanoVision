from __future__ import annotations

from typing import Any

import numpy as np
from pycocotools import mask as maskUtils
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from scipy.special import iv
import cv2
from scipy.stats import vonmises
from skimage.measure import label, perimeter_crofton, regionprops
from shapely import make_valid
from shapely.geometry import LineString, Polygon
from cnt_project.coco.masks import (
    rle_segmentation_to_mask,
)
from cnt_project.features.core.tree_length import (
    branching_skeleton_length,
)

MICRONS_PER_PIXEL = 5.0 / 256.0
MICRONS_PER_PIXEL_SQUARED = MICRONS_PER_PIXEL ** 2
DEFAULT_IMAGE_AREA_UM2 = 25.0


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def _normalize_angle_minus90_90(angle: float) -> float:
    """Normalize orientation angle to [-90, 90] degrees."""
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    return float(angle)


def decode_annotation_mask(
    ann: dict[str, Any],
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """Decode a COCO polygon or RLE annotation into a binary mask."""
    segmentation = ann.get(
        "segmentation"
    )

    if isinstance(segmentation, dict):
        decoded = rle_segmentation_to_mask(
            segmentation,
            image_height,
            image_width,
        )
    elif isinstance(segmentation, list):
        rles = maskUtils.frPyObjects(
            segmentation,
            image_height,
            image_width,
        )

        merged_rle = maskUtils.merge(
            rles
        )

        decoded = maskUtils.decode(
            merged_rle
        )
    else:
        raise TypeError(
            "Annotation segmentation must be a polygon list "
            "or an RLE dictionary."
        )

    decoded = np.asarray(
        decoded
    )

    if decoded.ndim == 3:
        decoded = np.any(
            decoded,
            axis=2,
        )

    if decoded.shape != (
        image_height,
        image_width,
    ):
        raise ValueError(
            "Decoded annotation shape does not match image "
            f"dimensions: decoded={decoded.shape}, "
            f"expected={(image_height, image_width)}."
        )

    return decoded.astype(
        bool,
        copy=False,
    )


def _annotation_coords_xy(ann: dict[str, Any]) -> np.ndarray:
    """Return all COCO polygon coordinates as an (N, 2) array in x,y order."""
    seg = ann.get("segmentation", [])

    if not isinstance(seg, list):
        return np.empty((0, 2), dtype=float)

    coords_all = []

    for poly in seg:
        if not isinstance(poly, list) or len(poly) < 4:
            continue

        coords = np.asarray(poly, dtype=float).reshape(-1, 2)

        if coords.shape[0] >= 2:
            coords_all.append(coords)

    if not coords_all:
        return np.empty((0, 2), dtype=float)

    return np.vstack(coords_all)


# ---------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------

def polygon_pca_orientation_professional_from_coordinates(
    coordinates_xy: np.ndarray,
) -> float:
    """
    Calculate polygon-PCA orientation from (x, y) coordinates.

    Convention:
    - 0° = East / right / horizontal
    - 90° = North / up / vertical
    - positive angles are counterclockwise
    - range = [-90°, 90°]

    The input coordinates use image coordinates, where y increases
    downward.
    """
    coordinates_xy = np.asarray(
        coordinates_xy,
        dtype=float,
    )

    if (
        coordinates_xy.ndim != 2
        or coordinates_xy.shape[1] != 2
    ):
        raise ValueError(
            "coordinates_xy must have shape (N, 2); "
            f"received {coordinates_xy.shape}."
        )

    if coordinates_xy.shape[0] < 2:
        return np.nan

    if not np.isfinite(coordinates_xy).all():
        raise ValueError(
            "coordinates_xy must contain only finite values."
        )

    centered_coordinates = (
        coordinates_xy
        - coordinates_xy.mean(
            axis=0,
            keepdims=True,
        )
    )

    covariance_matrix = np.cov(
        centered_coordinates.T
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance_matrix
    )

    major_axis = eigenvectors[
        :,
        np.argmax(eigenvalues),
    ]

    dx = major_axis[0]
    dy_image = major_axis[1]

    image_angle = np.degrees(
        np.arctan2(
            dy_image,
            dx,
        )
    )

    # Image y increases downward, while the professional
    # coordinate convention uses y increasing upward.
    professional_angle = -image_angle

    return _normalize_angle_minus90_90(
        float(professional_angle)
    )

def polygon_pca_orientation_professional_from_mask(
    mask: np.ndarray,
) -> float:
    """Calculate polygon-PCA orientation from complete mask contours."""
    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError(
            "mask must be two-dimensional; "
            f"received shape {mask.shape}."
        )

    mask = np.ascontiguousarray(
        mask.astype(
            np.uint8,
            copy=False,
        )
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    coordinates = [
        contour.reshape(-1, 2)
        for contour in contours
        if contour.shape[0] >= 2
    ]

    if not coordinates:
        return float("nan")

    coordinates_xy = np.concatenate(
        coordinates,
        axis=0,
    )

    return (
        polygon_pca_orientation_professional_from_coordinates(
            coordinates_xy
        )
    )

def polygon_pca_orientation_professional_from_annotation(
    ann: dict[str, Any],
) -> float:
    """
    Calculate polygon-PCA orientation from a COCO polygon annotation.
    """
    coordinates_xy = _annotation_coords_xy(
        ann
    )

    return (
        polygon_pca_orientation_professional_from_coordinates(
            coordinates_xy
        )
    )
# ---------------------------------------------------------------------
# Object-level feature extraction
# ---------------------------------------------------------------------

def extract_annotation_features(
    ann: dict[str, Any],
    image_height: int,
    image_width: int,
    *,
    microns_per_pixel: float = MICRONS_PER_PIXEL,
) -> dict[str, float] | None:
    """
    Extract canonical CNT morphology features for one COCO annotation.

    Definitions:
    - Area: foreground pixel count from decoded mask.
    - Perimeter: Crofton perimeter from filled mask.
    - Length: existing weighted geodesic path method.
    - Width: 2 * mean EDT along geodesic path.
    - Orientation: PCA on complete decoded-mask contour coordinates,
        using the professional convention.
    """
    from cnt_project.features.core.geodesic_length import polygon_mask_geodesic_length

    if "segmentation" not in ann:
        return None

    mask = decode_annotation_mask(ann, image_height, image_width)

    if mask.sum() == 0:
        return None

    # Area: same as newer repo.
    area_px2 = float(np.count_nonzero(mask))
    area_um2 = area_px2 * (microns_per_pixel ** 2)

    # Perimeter: same as newer repo.
    mask_filled = binary_fill_holes(mask)
    perimeter_px = float(perimeter_crofton(mask_filled, directions=4))
    perimeter_um = perimeter_px * microns_per_pixel

    # A single COCO annotation may contain multiple segmentation polygons.
    # After decoding, disconnected polygons appear as separate mask components.
    # Measure every component independently and sum their lengths.
    #
    # connectivity=2 means 8-connectivity in 2D, matching the 8-connected
    # graph used by polygon_mask_geodesic_length().
    labeled_mask = label(mask, connectivity=2)
    n_connected_components = int(labeled_mask.max())

    component_lengths_px: list[float] = []
    component_paths: list[np.ndarray] = []

    for component_id in range(1, n_connected_components + 1):
        component_mask = labeled_mask == component_id

        (
            component_length_px,
            _,
            component_path_rc,
            _,
        ) = polygon_mask_geodesic_length(
            component_mask,
            microns_per_pixel=microns_per_pixel,
            alpha=2.0,
            eps=1e-3,
            return_path=True,
            return_debug=True,
        )

        component_lengths_px.append(float(component_length_px))

        if component_path_rc is not None and len(component_path_rc) > 0:
            component_paths.append(component_path_rc)

    # The total object length is the sum of all fragment lengths.
    length_px = float(np.sum(component_lengths_px))
    length_um = float(length_px * microns_per_pixel)

    # Branch-aware skeleton length. The graph contains every connected
    # component of this annotation, so graph edge weights are summed
    # across all fragments without connecting fragments artificially.
    tree_result = branching_skeleton_length(
        mask,
        min_branch_length=None,
    )

    tree_length_px = float(
        tree_result["tree_length"]
    )
    tree_length_um = float(
        tree_length_px
        * microns_per_pixel
    )

    # Combine the component paths only for the width calculation below.
    # This does not create artificial steps between disconnected fragments.
    path_rc = (
        np.concatenate(component_paths, axis=0)
        if component_paths
        else np.empty((0, 2), dtype=int)
    )

    # Width: 2 * mean EDT along the geodesic centerline path.
    distance_map = distance_transform_edt(mask)

    if path_rc is not None and len(path_rc) > 0:
        rr = path_rc[:, 0]
        cc = path_rc[:, 1]
        path_values = distance_map[rr, cc]
        width_px = float(2.0 * np.mean(path_values)) if path_values.size else np.nan
    else:
        fg_values = distance_map[mask]
        width_px = float(2.0 * np.mean(fg_values)) if fg_values.size else np.nan

    if width_px <= 0:
        width_px = np.nan

    width_um = width_px * microns_per_pixel if not np.isnan(width_px) else np.nan

    aspect_ratio = (
        float(length_px / width_px)
        if width_px and not np.isnan(width_px)
        else np.nan
    )

    # orientation_angle = polygon_pca_orientation_professional_from_annotation(ann)
    orientation_angle = ( polygon_pca_orientation_professional_from_mask( mask ) )


    return {
        "area_pixels2": area_px2,
        "area_um2": area_um2,
        "perimeter": perimeter_px,
        "perimeter_um": perimeter_um,
        "length": float(length_px),
        "length_um": float(length_um),
        "tree_length": tree_length_px,
        "tree_length_um": tree_length_um,
        "width": width_px,
        "width_um": width_um,
        "aspect_ratio": aspect_ratio,
        "orientation_angle": orientation_angle,
        "n_connected_components": n_connected_components,
    }


# ---------------------------------------------------------------------
# Line density
# ---------------------------------------------------------------------

def compute_line_densities_from_polygons_legacy_rasterized(
    annotations: list[dict[str, Any]],
    image_height: int,
    image_width: int,
    *,
    min_object_size: int = 2,
) -> list[int]:
    """
    Legacy row-wise CNT line-density implementation.

    Each COCO polygon is first rasterized into a binary mask. An annotation
    contributes to a row when at least one rasterized foreground pixel lies
    on that row.

    Kept for reproducibility of historical results.
    """
    line_densities = np.zeros(image_height, dtype=int)

    for ann in annotations:
        if "segmentation" not in ann:
            continue

        mask = decode_annotation_mask(
            ann,
            image_height,
            image_width,
        )

        if mask.sum() < min_object_size:
            continue

        rows_with_object = np.any(mask, axis=1)
        line_densities[rows_with_object] += 1

    return line_densities.tolist()

def compute_line_densities_from_annotations(
    annotations: list[dict[str, Any]],
    image_height: int,
    image_width: int,
    *,
    min_object_size: int = 1,
) -> list[int]:
    """Calculate row-wise object count from decoded annotation masks."""
    line_densities = np.zeros(
        image_height,
        dtype=int,
    )

    for ann in annotations:
        if "segmentation" not in ann:
            continue

        mask = decode_annotation_mask(
            ann,
            image_height,
            image_width,
        )

        if np.count_nonzero(mask) < min_object_size:
            continue

        rows_with_object = np.any(
            mask,
            axis=1,
        )

        line_densities[
            rows_with_object
        ] += 1

    return line_densities.tolist()

def compute_line_densities_from_polygons(
    annotations: list[dict[str, Any]],
    image_height: int,
    image_width: int,
) -> list[int]:
    """
    Compute row-wise CNT line density directly from COCO polygon geometry.

    line_density[row] = number of annotations whose polygon geometry
    intersects the horizontal line corresponding to that image row.

    we take the annotations from SON COCO file """
    line_densities = np.zeros(image_height, dtype=int)

    # for one annotation in all annotations in an image
    for ann in annotations:
        segmentation = ann.get("segmentation")

        if not isinstance(segmentation, list):
            continue

        geometries = []

        # for every polygon that belong to that annotation (can be one or can be more than one if CNT was fragmented) 
        for segment in segmentation:
            if not isinstance(segment, list) or len(segment) < 6:
                continue

            coords = np.asarray(
                segment,
                dtype=float,
            ).reshape(-1, 2)

            if coords.shape[0] < 3:
                continue

            geometry = Polygon(coords)

            if geometry.is_empty:
                continue

            if not geometry.is_valid:
                # if the coords create weird shapped polygon with self interesection for example make valid will create 
                # valid geometric representation of the same coordinate structure (if a polygon has self interesect it can create two valid polygons)
                # https://shapely.readthedocs.io/en/2.1.1/reference/shapely.make_valid.html
                geometry = make_valid(geometry)

            if geometry.is_empty:
                continue

            geometries.append(geometry)

        if not geometries:
            continue

        # Restrict intersection checks to rows within the annotation's
        # vertical extent. using bounding boxes .bounds
        min_y = min(
            geometry.bounds[1]
            for geometry in geometries
        )
        max_y = max(
            geometry.bounds[3]
            for geometry in geometries
        )

        first_row = max(
            0,
            int(np.ceil(min_y)),
        )
        last_row = min(
            image_height - 1,
            int(np.floor(max_y)),
        )

        # for every row in the valid range of y we construct a horizontal line and check if it coresses with the polygon 
        for row in range(first_row, last_row + 1):
            horizontal_line = LineString(
                [
                    (0.0, float(row)),
                    (float(image_width), float(row)),
                ]
            )

            if any(
                geometry.intersects(horizontal_line)
                for geometry in geometries
            ):
                line_densities[row] += 1

    return line_densities.tolist()


# ---------------------------------------------------------------------
# Orientation-distribution features
# ---------------------------------------------------------------------

def compute_nematic_order_from_angles(
    orientation_angles_deg: list[float],
) -> dict[str, float | bool]:
    """
    Compute nematic order parameter from orientation angles.

    Uses doubled-angle representation because CNT orientation has nematic
    symmetry: theta and theta + 180° are equivalent.
    """
    angles = np.asarray(orientation_angles_deg, dtype=float)
    angles = angles[np.isfinite(angles)]

    if angles.size < 2:
        return {
            "nematic_order_parameter": 0.0,
            "nematic_director_angle_deg": 0.0,
            "nematic_calculation_success": False,
        }

    angles_rad = np.radians(angles)
    doubled = 2.0 * angles_rad

    q_xx = float(np.mean(np.cos(doubled)))
    q_xy = float(np.mean(np.sin(doubled)))

    order = float(np.sqrt(q_xx ** 2 + q_xy ** 2))

    director_rad = 0.5 * np.arctan2(q_xy, q_xx)
    director_deg = _normalize_angle_minus90_90(float(np.degrees(director_rad)))

    return {
        "nematic_order_parameter": order,
        "nematic_director_angle_deg": director_deg,
        "nematic_calculation_success": True,
    }


def fit_von_mises_to_orientations(
    orientation_angles_deg: list[float],
) -> dict[str, float | int | bool]:
    """
    Fit a von Mises distribution to doubled orientation angles.

    The doubled-angle representation handles nematic symmetry.
    """
    angles = np.asarray(orientation_angles_deg, dtype=float)
    angles = angles[np.isfinite(angles)]

    if angles.size < 3:
        return {
            "von_mises_mean_orientation_deg": 0.0,
            "von_mises_concentration_kappa": 0.0,
            "von_mises_circular_variance": 1.0,
            "von_mises_confidence_95_deg": 180.0,
            "von_mises_fit_success": False,
            "num_objects_for_fit": int(angles.size),
        }

    try:
        angles_rad = np.radians(angles)
        doubled = (2.0 * angles_rad) % (2.0 * np.pi)

        kappa, loc, scale = vonmises.fit(doubled, fscale=1)

        mean_doubled_rad = loc % (2.0 * np.pi)
        mean_original_rad = mean_doubled_rad / 2.0
        mean_angle_deg = _normalize_angle_minus90_90(
            float(np.degrees(mean_original_rad))
        )

        if kappa > 0:
            R = float(iv(1, kappa) / iv(0, kappa))
            circular_variance = float(1.0 - R)
        else:
            R = 0.0
            circular_variance = 1.0

        n = int(angles.size)

        if kappa > 0 and n > 1 and R > 0:
            std_error = 1.0 / np.sqrt(n * kappa * R)
            confidence_95_doubled_rad = 1.96 * std_error
            confidence_95_original_rad = confidence_95_doubled_rad / 2.0
            confidence_95_deg = min(float(np.degrees(confidence_95_original_rad)), 90.0)
        else:
            confidence_95_deg = 90.0

        return {
            "von_mises_mean_orientation_deg": mean_angle_deg,
            "von_mises_concentration_kappa": float(kappa),
            "von_mises_circular_variance": circular_variance,
            "von_mises_confidence_95_deg": confidence_95_deg,
            "von_mises_fit_success": True,
            "num_objects_for_fit": n,
        }

    except Exception:
        return {
            "von_mises_mean_orientation_deg": 0.0,
            "von_mises_concentration_kappa": 0.0,
            "von_mises_circular_variance": 1.0,
            "von_mises_confidence_95_deg": 180.0,
            "von_mises_fit_success": False,
            "num_objects_for_fit": int(angles.size),
        }


# ---------------------------------------------------------------------
# Main annotation-list extractor
# ---------------------------------------------------------------------

def calculate_cnt_features_from_annotations(
    annotations: list[dict[str, Any]],
    image_height: int,
    image_width: int,
    *,
    microns_per_pixel: float = MICRONS_PER_PIXEL,
    image_area_um2: float | None = None,
) -> dict[str, Any]:
    """
    Extract object-level and image-level CNT features from COCO annotations.

    Returns a dictionary containing:
    - list-valued object features
    - average_* image-level summaries
    - line-density summaries
    - CNT density
    - nematic and von Mises orientation statistics
    """
    object_props: list[dict[str, float]] = []
    object_annotation_ids: list[int | None] = []

    for ann in annotations:
        props = extract_annotation_features(
            ann,
            image_height,
            image_width,
            microns_per_pixel=microns_per_pixel,
        )

        if props is not None:
            object_props.append(
                props
            )

            annotation_id = ann.get(
                "id"
            )

            object_annotation_ids.append(
                int(annotation_id)
                if annotation_id is not None
                else None
            )

    if image_area_um2 is None:
        image_area_um2 = (
            image_height * microns_per_pixel
        ) * (
            image_width * microns_per_pixel
        )

    n_objects = len(object_props)
    cnt_density_per_um2 = (
        float(n_objects / image_area_um2)
        if image_area_um2 and image_area_um2 > 0
        else np.nan
    )

    line_density = compute_line_densities_from_annotations(
        annotations,
        image_height,
        image_width,
    )

    # line_density = compute_line_densities_from_polygons(
    #     annotations,
    #     image_height,
    #     image_width,
    # )
    line_density_np = np.asarray(line_density, dtype=float)

    if n_objects == 0:
        return {
            "num_polygons": 0,
            "annotation_ids": [],
            "cnt_density_per_um2": cnt_density_per_um2,
            "line_density": line_density,
            "line_density_mean": float(np.nanmean(line_density_np)) if line_density_np.size else np.nan,
            "line_density_std": float(np.nanstd(line_density_np)) if line_density_np.size else np.nan,
            "line_density_max": float(np.nanmax(line_density_np)) if line_density_np.size else np.nan,
            **compute_nematic_order_from_angles([]),
            **fit_von_mises_to_orientations([]),
        }

    keys = list(object_props[0].keys())

    output: dict[str, Any] = {
        key: [d[key] for d in object_props]
        for key in keys
    }

    output["num_polygons"] = n_objects
    output["annotation_ids"] = ( object_annotation_ids )
    output["cnt_density_per_um2"] = cnt_density_per_um2

    output["line_density"] = line_density
    output["line_density_mean"] = float(np.nanmean(line_density_np)) if line_density_np.size else np.nan
    output["line_density_std"] = float(np.nanstd(line_density_np)) if line_density_np.size else np.nan
    output["line_density_max"] = float(np.nanmax(line_density_np)) if line_density_np.size else np.nan

    for key in keys:
        values = np.asarray(output[key], dtype=float)
        output[f"average_{key}"] = (
            float(np.nanmean(values)) if values.size else np.nan
        )

    orientation_angles = output.get("orientation_angle", [])

    output.update(compute_nematic_order_from_angles(orientation_angles))
    output.update(fit_von_mises_to_orientations(orientation_angles))

    return output