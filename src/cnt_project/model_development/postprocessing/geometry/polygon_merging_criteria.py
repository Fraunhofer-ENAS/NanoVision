from scipy.ndimage import convolve
import numpy as np

def calculate_overlap(polygon1, polygon2):
    intersection = polygon1.intersection(polygon2)
    overlap_area = intersection.area
    min_area = min(polygon1.area, polygon2.area)
    overlap_ratio = overlap_area / min_area if min_area > 0 else 0
    return overlap_ratio


def count_endpoints(skeleton):
    """ Count the number of endpoints in the skeleton. """
    # Define a 3x3 neighborhood kernel
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])

    # Convolve the skeleton with the kernel
    convolved = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)

    # Endpoints have a value of 11 (1 foreground pixel with exactly 1 neighbor)
    endpoints = np.argwhere(convolved == 11)

    return len(endpoints)


# function to check if we have parallel but not aligned polygons
# this is a legacy function previously used in merging criteria
def calculate_perpendicular_offset(point, orientation_vector, line_point):
    """
    Calculate the perpendicular distance from a point to an oriented line.

    Notes
    -----
    This helper belongs to a legacy centroid-alignment criterion and is not
    currently used by the active polygon-merging path. It is retained for
    reproducibility and possible future comparison with the current alignment
    criterion.
    """
    # Vector from line_point to the point
    vector_to_point = np.array([point.x - line_point.x, point.y - line_point.y])

    # Normalize orientation vector
    orientation_vector = np.array(orientation_vector)
    orientation_vector /= np.linalg.norm(orientation_vector)

    # Calculate perpendicular distance
    perpendicular_distance = np.linalg.norm(np.cross(vector_to_point, orientation_vector))

    return perpendicular_distance
