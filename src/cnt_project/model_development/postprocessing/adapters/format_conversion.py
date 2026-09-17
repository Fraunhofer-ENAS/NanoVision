from skimage.draw import polygon as draw_polygon

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LinearRing, Point



# -------------------------------------
# Functions needed in other files
# -------------------------------------
# used in nms.py to rasterize the polygons
def draw_polygons_on_image(image, polygons, labels=None):
    """
    Draws polygons on the image array. Each polygon is filled with a label value,
    accounting for intricate shapes like holes and multipolygons.
    """
    if labels is None:
        labels = np.arange(len(polygons))

    for label, polygon in zip(labels, polygons):
        if polygon.is_empty:
            continue

        # Handle MultiPolygon
        if isinstance(polygon, MultiPolygon):
            for poly in polygon.geoms:
                # Draw exterior
                rr, cc = polygon_to_coords(poly)
                rr, cc = draw_polygon(rr, cc, image.shape)
                image[rr, cc] = label + 1

                # Draw interiors (holes)
                for interior in poly.interiors:
                    rr, cc = polygon_to_coords(interior)
                    rr, cc = draw_polygon(rr, cc, image.shape)
                    image[rr, cc] = 0  # Fill holes with background (0)
        else:
            # Handle Polygon
            # Draw exterior
            rr, cc = polygon_to_coords(polygon)
            rr, cc = draw_polygon(rr, cc, image.shape)
            image[rr, cc] = label + 1

            # Handle interiors (holes)
            for interior in polygon.interiors:
                rr, cc = polygon_to_coords(interior)
                rr, cc = draw_polygon(rr, cc, image.shape)
                image[rr, cc] = 0  # Fill holes with background (0)
# used in nms.py to create an array of the polygons
def create_polygon_array(polygons, n_rays):
    """
    Creates an ndarray of shape (number_of_polygons, 2, n_rays) to store the x and y coordinates of the polygon vertices.
    """
    number_of_polygons = len(polygons)
    polygon_array = np.zeros((number_of_polygons, 2, n_rays), dtype=np.float32)

    for i, polygon in enumerate(polygons):
        sampled_coords = sample_polygon(polygon, n_rays)
        polygon_array[i, 0, :] = sampled_coords[0]  # x coordinates
        polygon_array[i, 1, :] = sampled_coords[1]  # y coordinates

    return polygon_array

# trying to create the "LABEL" variable in model2d.py
def polygon_to_coords(polygon):
    """
    Converts a shapely Polygon or LinearRing to a list of coordinates suitable for skimage.draw.polygon.
    """
    if isinstance(polygon, MultiPolygon):
        coords = []
        for poly in polygon.geoms:
            coords.extend(np.array(poly.exterior.coords))
        return np.array(coords)[:, 1], np.array(coords)[:, 0]
    elif isinstance(polygon, Polygon):
        exterior_coords = np.array(polygon.exterior.coords)
        return exterior_coords[:, 1], exterior_coords[:, 0]
    elif isinstance(polygon, LinearRing):
        coords = np.array(polygon.coords)
        return coords[:, 1], coords[:, 0]
    else:
        raise ValueError("Input must be a Polygon, MultiPolygon, or LinearRing.")
# this is trying to create the "COORD" variable in model2d.py
def sample_polygon(polygon, n_rays):
    """
    Samples the coordinates of a polygon at n_rays angles around its centroid.
    """
    centroid = polygon.centroid
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    sampled_coords = []

    for angle in angles:
        ray = Point(centroid.x + np.cos(angle), centroid.y + np.sin(angle))
        intersection = polygon.boundary.intersection(ray)

        # Find the closest point of intersection to the centroid
        if intersection.is_empty:
            sampled_coords.append((centroid.x, centroid.y))
        elif 'Point' == intersection.geom_type:
            sampled_coords.append((intersection.x, intersection.y))
        else:
            # MultiPoint or LineString
            closest_point = min(intersection, key=lambda p: p.distance(centroid))
            sampled_coords.append((closest_point.x, closest_point.y))

    return np.array(sampled_coords).T

# this function is used to create the "PROB" varibable in model2d.py
def extract_centers_probabilities(merged_polygons, points_arr, prob):
    """
    Extracts center points and probabilities for each polygon after merging.
    """
    centers = []
    probabilities = []

    for polygon in merged_polygons:
        centroid = polygon.centroid
        centers.append((centroid.x, centroid.y))

        # Find the closest point in points_arr to centroid (assuming points_arr is (N, 2))
        closest_index = np.argmin(np.linalg.norm(points_arr - [centroid.x, centroid.y], axis=1))
        probabilities.append(prob[closest_index])

    return centers, probabilities