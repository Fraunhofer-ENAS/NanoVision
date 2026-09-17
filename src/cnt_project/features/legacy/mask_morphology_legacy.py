import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import rotate

"""
used by runners in src/cnt_project/evaluation/
works on binary masks of polygons (CNTs) to extract properties like area, perimeter, length, width, aspect ratio, and orientation angle
uses skeletonize on the masks to compute length as the sum of skeleton pixels and width as 2x the mean distance from the skeleton to the edge of the mask
orientation angle is computed using the regionprops orientation normalized to 0-180 degrees
# TODO: this file uses the full image isntead of cococ json files which means this is legacy !!! this need to be edited to use the coco json file instead
"""


def calculate_polygon_properties(image):
    
    """Calculate properties of polygons (CNTs) in the image."""
    MICRONS_PER_PIXEL = 5 / 256  
    labeled_image, num_polygons = label(image, return_num=True)
    props = regionprops(labeled_image)

    areas = [prop.area for prop in props]
    perimeters = [prop.perimeter for prop in props]

    lengths = []
    widths = []

    for prop in props:
        # Extract binary mask for the current CNT
        cnt_mask = labeled_image == prop.label

        # Skeletonize only this CNT
        skeleton = skeletonize(cnt_mask)

        # CNT length: Sum of skeleton pixels
        cnt_length = np.sum(skeleton)
        lengths.append(cnt_length)

        # Compute distance transform *inside* the CNT only
        distance_map = distance_transform_edt(cnt_mask)  # Now computed inside each CNT
        cnt_width = 2 * np.mean(distance_map[skeleton])  # Width is 2x the mean radius from skeleton
        widths.append(cnt_width if cnt_width > 0 else np.nan)  # Avoid zero-width errors


    aspect_ratios = [length / width if width > 0 else np.nan for length, width in zip(lengths, widths)]
    orientation_angles = [np.degrees(prop.orientation) for prop in props]
    orientation_angles = [(angle + 360) % 360 for angle in orientation_angles]  # Normalize to 0-360 degrees
    orientation_angles = [angle % 180 for angle in orientation_angles]  # Restrict to 0-180 degrees

    # Convert pixel values to micrometers
    areas_um2 = [area * (MICRONS_PER_PIXEL ** 2) for area in areas]
    lengths_um = [length * MICRONS_PER_PIXEL for length in lengths]
    widths_um = [width * MICRONS_PER_PIXEL for width in widths]

    return {
        "num_polygons": num_polygons,
        "average_area": np.mean(areas) if areas else 0,
        "average_perimeter": np.mean(perimeters) if perimeters else 0,
        "average_length": np.mean(lengths) if lengths else 0,
        "average_width": np.mean(widths) if widths else 0,
        "average_aspect_ratio": np.mean(aspect_ratios) if aspect_ratios else 0,
        "average_area_um2": np.mean(areas_um2) if areas_um2 else 0,
        "average_length_um": np.mean(lengths_um) if lengths_um else 0,
        "average_width_um": np.mean(widths_um) if widths_um else 0,
        "areas": areas,
        "perimeters": perimeters,
        "lengths": lengths,
        "widths": widths,
        "areas_um2": areas_um2,
        "lengths_um": lengths_um,
        "widths_um": widths_um,
        "aspect_ratios": aspect_ratios,
        "orientation_angles": orientation_angles,
    }


def compute_density_vs_angle(image, angles, min_object_size=2):
    """Compute mean line density for multiple angles."""
    angle_densities = {}

    for angle in angles:
        mean_density = rotate_and_compute_density(image, angle, min_object_size)
        angle_densities[angle] = mean_density

    return angle_densities


def compute_line_densities(image, min_object_size=2):
    """Compute the line densities (number of CNTs per row) for the entire image."""
    line_densities = []
    for row in image:
        object_count = count_objects_in_line(row, min_object_size)  # Count objects in the row
        line_densities.append(object_count)
    return line_densities


def count_objects_in_line(row, min_object_size=2):
    """Count the number of objects (CNTs) in a single row of the image."""
    object_count = 0
    current_object = None
    current_size = 0

    for pixel in row:
        if pixel != 0:  # If the pixel is part of an object
            if pixel == current_object:
                current_size += 1  # Increment the size of the current object
            else:
                if current_size >= min_object_size:  # Check the size of the previous object
                    object_count += 1  # Count the previous object
                current_object = pixel  # Start a new object
                current_size = 1  # Reset size for the new object
        else:
            if current_size >= min_object_size:  # Check the size of the last object when we hit a background pixel
                object_count += 1
            current_object = None  # Reset for background
            current_size = 0  # Reset size for background

    # Check for any remaining object at the end of the row
    if current_size >= min_object_size:
        object_count += 1

    return object_count


def rotate_and_compute_density(image, angle, min_object_size=3):
    """Rotate image by given angle and compute line density."""
    rotated_image = rotate(image, angle, reshape=False, order=0)  # Nearest-neighbor interpolation
    return np.mean(compute_line_densities(rotated_image, min_object_size))  # Compute mean line density

