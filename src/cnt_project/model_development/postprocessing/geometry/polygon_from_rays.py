from shapely.geometry import Polygon
import numpy as np


# Constructs a polygon from a central point using radial distances
def construct_polygon_opp(center_idx, dist, points, scores):
    """
    Constructs a polygon from a central point using radial distances.

    Parameters:
        center_idx (int): Index of the central point in 'points'.
        dist (ndarray): Radial distances for each angle from the center.
        points (ndarray): Array of (y, x) coordinates of points.
        scores (ndarray): Scores associated with the points.

    Returns:
        tuple: (Polygon object, radial distances, center coordinates, center score).
    """
    py, px = points[center_idx]
    n_rays = dist.shape[-1]
    ANGLE_PI = 2 * np.pi / n_rays
    poly = []
    distances = []
    for k in range(n_rays):
        dx = dist[center_idx, k]
        dy = dist[center_idx, k]
        y = py + dy * np.sin(ANGLE_PI * k)
        x = px + dx * np.cos(ANGLE_PI * k)
        poly.append((x, y))
        distances.append(dist[center_idx, k])

    polygon = Polygon(poly)
    return polygon, distances, points[center_idx], scores[center_idx]
