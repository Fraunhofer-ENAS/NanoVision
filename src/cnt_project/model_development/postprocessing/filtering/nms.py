from scipy.spatial import KDTree
import numpy as np

# removing the unnecessary polygons before starting merging
def nms_polygons(polygons, scores, distances, points_arr, overlap_threshold=0.5):
    """
    Perform non-maximum suppression on a list of polygons based on their scores, distances, and points.

    :param polygons: List of Shapely Polygons.
    :param scores: List of scores corresponding to each polygon.
    :param distances: List of distances corresponding to each polygon.
    :param points_arr: List of points corresponding to each polygon (polygon centers).
    :param overlap_threshold: Threshold for overlap ratio above which a polygon is suppressed.
    :return: Non-suppressed polygons, their corresponding scores, distances, and points.
    """
    n_polys = len(polygons)
    areas = np.array([polygon.area for polygon in polygons])
    suppressed = np.zeros(n_polys, dtype=bool)

    # Bounding boxes
    bbox_x1 = np.array([polygon.bounds[0] for polygon in polygons])
    bbox_x2 = np.array([polygon.bounds[2] for polygon in polygons])
    bbox_y1 = np.array([polygon.bounds[1] for polygon in polygons])
    bbox_y2 = np.array([polygon.bounds[3] for polygon in polygons])

    # Polygon centroids
    points = np.array([[polygon.centroid.x, polygon.centroid.y] for polygon in polygons])

    # Build KD-Tree
    kdtree = KDTree(points)

    for i in range(n_polys):
        if suppressed[i]:
            continue

        # Find neighbors using KD-Tree
        neighbors = kdtree.query_ball_point(points[i], r=(max(bbox_x2 - bbox_x1) + max(bbox_y2 - bbox_y1)) / 2)

        for j in neighbors:
            if suppressed[j] or j == i:
                continue

            # Check if bounding boxes intersect
            # todo remove the bounding box check if not needed
            if bbox_x1[i] > bbox_x2[j] or bbox_x1[j] > bbox_x2[i] or bbox_y1[i] > bbox_y2[j] or bbox_y1[j] > bbox_y2[i]:
                continue

            # Calculate intersection and overlap
            intersection_area = polygons[i].intersection(polygons[j]).area
            overlap = intersection_area / min(areas[i], areas[j])

            # # COMMENT FROM HERE TO "STOP DEBUG"  WHEN STOP DEBUGGING WITH IF ELSE !!
            # # DEBUG: SET THE POLYGONS VALUE BELOW TO INVESTIGATE OVERLAPPING ONES (check on Iteration 0 first!)
            # # CHANGE the Ps value and if/else conditions to demonstrate your point 
            # P3=0.37
            # P2=0.41 
            # P1=0.26
            # if (round(float(scores[j]), 2) == round(P1,2) and round(float(scores[i]), 2) == round(P2,2)) \
            #           or (round(float(scores[j]), 2) == round(P2,2) and round(float(scores[i]), 2) == round(P3,2)):
            #     print(f'For each poly, loop over the neighbors \n')
            #     print(f'poly: {scores[i]} --> rounded: {round(float(scores[i]), 2)}')
            #     print(f'neighbor: {scores[j]} --> rounded: {round(float(scores[j]), 2)}')
            # # STOP DEBUG 
            # #############


            # Suppress the polygon with the lower score if overlap is above the threshold
            if overlap > overlap_threshold:
                if scores[i] >= scores[j]:
                    suppressed[j] = True  # Suppress the lower-scoring polygon
                else:
                    suppressed[i] = True
                    break

    # Return the non-suppressed polygons, scores, distances, and points
    return (
        [polygons[i] for i in range(n_polys) if not suppressed[i]],
        [scores[i] for i in range(n_polys) if not suppressed[i]],
        [distances[i] for i in range(n_polys) if not suppressed[i]],
        [points_arr[i] for i in range(n_polys) if not suppressed[i]]
    )


