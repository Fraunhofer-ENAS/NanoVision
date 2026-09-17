from scipy.spatial import KDTree
from shapely.ops import unary_union
import numpy as np

from cnt_project.features.core.shapely_polygon_pca_orientation import (
    calculate_shapely_polygon_pca_orientation_legacy,
    get_shapely_polygon_pca_orientations_legacy,
)
from cnt_project.io.paths import ProjectPaths
from cnt_project.model_development.postprocessing.filtering.shape_heuristics import check_y_or_t_shape, count_polygon_skeleton_endpoints
from cnt_project.model_development.postprocessing.geometry.polygon_merging_criteria import calculate_overlap
from cnt_project.model_development.postprocessing.visualization.polygon_plot_util import plot_polygons 

# should merge function for no overlap labels
def should_merge_candidates_no_overlap_regions(
    poly1,
    poly2,
    overlap_threshold,
    angle_threshold,
    alignment_tolerance=1.3,
    aspect_ratio_threshold=5,
    normalized_centroid_threshold=0.5,
    merge_id=0,
):
    """
    Decide whether two polygons (no-overlap labeling) should be merged.

    The decision uses:
      - overlap ratio threshold,
      - orientation agreement (angle difference),
      - a shape-complexity guard via skeleton endpoints (reject Y/T-like merges),
      - centroid alignment along the primary axis.

    Parameters
    ----------
    poly1, poly2 : shapely.geometry.Polygon
        Candidate polygons to evaluate.
    overlap_threshold : float
        Minimum required overlap ratio.
    angle_threshold : float
        Maximum allowed orientation difference (degrees).
    alignment_tolerance : float
        Reserved/legacy parameter (not currently used in the active path).
    aspect_ratio_threshold : float
        Reserved/legacy parameter (not currently used in the active path).
    merge_id : int
        Identifier forwarded to debugging/plotting hooks.

    Returns
    -------
    bool
        True if polygons satisfy merge criteria, else False.
    """
    # Absolute Angle Difference of polygons
    overlapp = calculate_overlap(poly1, poly2)
    if overlapp <= overlap_threshold:
        return False


    # Sensitive to noise of the rays direction
    # TODO: add smoothing of vertices for more reliable results?
    # See histograms of orientation to assess
    angle1 = calculate_shapely_polygon_pca_orientation_legacy(poly1)
    angle2 = calculate_shapely_polygon_pca_orientation_legacy(poly2)
    angle_diff = abs(angle1 - angle2)

    # Normalize the angle difference to be within [0, 180] degrees
    angle_diff = min(angle_diff, 180 - angle_diff)

    if angle_diff >= angle_threshold:
        return False


    # Create the merged polygon
    merged_polygon = unary_union([poly1, poly2])

    # Check if the merged polygon would form a Y or T shape
    # is_y_or_t_shape = check_y_or_t_shape(merged_polygon)

    # Plotting the polygons and merged polygon
    # skeleton = skeletonize_polygon(merged_polygon)
    # plot_merged_polygon(merged_polygon, merge_id)
    # plot_skeleton_test(merged_polygon)
    end_points = count_polygon_skeleton_endpoints(merged_polygon, merge_id)



    # Centroid and Perpendicular Offset Check (previous approach)
    # centroid1 = poly1.centroid
    # centroid2 = poly2.centroid
    # orientation_vector1 = np.array([np.cos(np.radians(angle1)), np.sin(np.radians(angle1))])
    # offset = calculate_perpendicular_offset(centroid2, orientation_vector1, centroid1)

    # if width <= 1.1 or height <= 1.1:
    #     if offset >= alignment_tolerance:
    #         return False
    # get centroids


    # Relative Alignment of centroids
    centroid1 = np.array([poly1.centroid.x, poly1.centroid.y])
    centroid2 = np.array([poly2.centroid.x, poly2.centroid.y])

    # Get the unit vector representing the primary axis of poly1 (orientation direction)
    orientation_vector1 = np.array([np.cos(np.radians(angle1)), np.sin(np.radians(angle1))])

    # Project vertices of poly1 onto its primary axis to find edges
    vertices = np.array(poly1.exterior.coords)
    projections = np.dot(vertices - centroid1, orientation_vector1)
    min_edge, max_edge = projections.min(), projections.max()

    # Compute centroid-to-edge distance (distance from centroid1 to its farthest edge)
    centroid_to_edge_distance = max(abs(min_edge), abs(max_edge))
    if centroid_to_edge_distance <= 0.0:
      return False

    # Project centroid of poly2 onto the primary axis of poly1
    projection_distance = np.dot(centroid2 - centroid1, orientation_vector1)

    # Normalize the centroid-to-centroid distance
    normalized_centroid_distance = abs(projection_distance) / centroid_to_edge_distance

    if (
        normalized_centroid_distance
        < normalized_centroid_threshold
    ):
        return False

    if end_points != 2:
        # If the merged polygon has more than 2 endpoints, it likely forms a Y or T shape
        return False

    return True
# new should merge function for the overlap labels using centeroids to avoind non aligned poly
def should_merge_candidates_with_overlap_regions(poly1, poly2, angle_threshold=50, min_overlap=4.0, centroid_threshold_factor=0.5):
    """
    Decide whether two polygons (overlap-region labeling) should be merged.

    Checks:
      - orientation similarity,
      - centroid alignment along poly1's primary axis,
      - minimum intersection area,
      - reject complex merged shapes (Y/T-like) via a shape test.

    Parameters
    ----------
    poly1, poly2 : shapely.geometry.Polygon
        Candidate polygons to evaluate.
    angle_threshold : float
        Maximum allowed orientation difference (degrees).
    min_overlap : float
        Minimum required intersection area for merging (pixel area in image coordinates).
    centroid_threshold_factor : float
        Minimum normalized projected centroid distance along poly1's axis.

    Returns
    -------
    bool
        True if polygons satisfy merge criteria, else False.
    """

    # Calculate angle difference
    angle1 = calculate_shapely_polygon_pca_orientation_legacy(poly1)
    angle2 = calculate_shapely_polygon_pca_orientation_legacy(poly2)
    angle_diff = abs(angle1 - angle2)
    angle_diff = min(angle_diff, 360 - angle_diff)  # Normalize to [0, 180]

    if angle_diff >= angle_threshold:
        return False  # Discard if angle difference exceeds threshold

    # get centroids
    centroid1 = np.array([poly1.centroid.x, poly1.centroid.y])
    centroid2 = np.array([poly2.centroid.x, poly2.centroid.y])

    # Get the unit vector representing the primary axis of poly1 (orientation direction)
    orientation_vector1 = np.array([np.cos(np.radians(angle1)), np.sin(np.radians(angle1))])


    # Project vertices of poly1 onto its primary axis to find edges
    vertices = np.array(poly1.exterior.coords)
    projections = np.dot(vertices - centroid1, orientation_vector1)
    min_edge, max_edge = projections.min(), projections.max()

    # Compute centroid-to-edge distance (distance from centroid1 to its farthest edge)
    centroid_to_edge_distance = max(abs(min_edge), abs(max_edge))
    # Project centroid of poly2 onto the primary axis of poly1
    projection_distance = np.dot(centroid2 - centroid1, orientation_vector1)

    # Normalize the centroid-to-centroid distance
    normalized_centroid_distance = abs(projection_distance) / centroid_to_edge_distance

    if normalized_centroid_distance < centroid_threshold_factor:
        return False  # Discard if centroids are not aligned based on the threshold

    """# Ensure projection distance is not too close to the centroid of poly1
    distance_from_centroid = abs(projection_distance)
    centroid_distance_threshold = centroid_threshold_factor * (max_edge - min_edge)

    if distance_from_centroid < 0.7*centroid_distance_threshold:
        return False  # Discard if projected centroid of poly2 is too close to centroid of poly1"""

    # Check minimal overlap condition
    overlap_area = poly1.intersection(poly2).area
    if overlap_area < min_overlap:
        return False  # Discard if there is insufficient overlap

    # Check if the merged polygon would form a Y or T shape
    merged_polygon = unary_union([poly1, poly2])
    # plot_skeleton_test(merged_polygon)
    # plot_merged_polygon(merged_polygon, 0)
    if check_y_or_t_shape(merged_polygon):
        return False  # Discard if shape is complex (Y or T)

    return True  # Merge if all conditions are met


# Merge function each one is for a specfic labeling type
# Merge function for the labels with overlaps
def merge_polygons_iterative_with_overlap_regions(
    polygons,
    all_points,
    all_distances,
    all_scores,
    n_rays,
    overlap_threshold=4.0,
    angle_threshold=50,
    centroid_threshold_factor=0.5,
    iteration_nbr=5,
    distance_multiplier=1.5,
    debug_plots=False,
    debug_plot_dir=None,
):
    """
    Merges overlapping polygons based on geometric and alignment criteria.

    Parameters:
        polygons (list of Polygon): List of polygons to process.
        all_points (list): Corresponding points of each polygon for reference.
        all_distances (list): Distance arrays associated with each polygon.
        all_scores (list): Confidence scores of the polygons.
        n_rays (int): Number of rays for polygon visualization and orientation computation.
        overlap_threshold (float, optional): Minimum overlap area required for merging.
        angle_threshold (float, optional): Maximum orientation angle difference (degrees) allowed.
        centroid_threshold_factor (float, optional): Minimum normalized projected centroid distance.
        iteration_nbr (int, optional): Maximum number of merging iterations. 
        distance_multiplier (float, optional): Multiplier for the KD-Tree search radius. 

    Returns:
        list of Polygon: Merged polygons after applying the merging criteria.

    Key Steps:
        - Sort polygons by centroid for processing.
        - Use a KD-Tree for efficient spatial queries.
        - Iteratively merge polygons based on overlap, alignment, and orientation.
        - Stop merging if no further changes occur or maximum iterations are reached.
    """
    iteration = 0
    merge_counter = 0  # Counter to track unique merges

    while iteration < iteration_nbr:
        print(f"Iteration {iteration}")
        merged_polygons = []
        visited = set()

        # Rebuild KD-Tree for the current set of polygons
        r_tree_index = KDTree([polygon.centroid.coords[0] for polygon in polygons])

        for i, poly1 in enumerate(polygons):
            if i in visited:
                continue

            # Dynamically adjust radius for KD-Tree search based on polygon size
            radius = distance_multiplier * ((poly1.bounds[2] - poly1.bounds[0] + poly1.bounds[3] - poly1.bounds[1]) / 2)
            candidates = r_tree_index.query_ball_point(poly1.centroid.coords[0], r=radius)
            candidates = [j for j in candidates if j < len(polygons)]
            if not candidates:
                continue

            merged_poly = poly1

            for j in candidates:
                if j == i or j in visited:
                    continue

                poly2 = polygons[j]
                # title = f"Candidate Polygons"
                # plot_polygons([merged_poly, poly2], title, n_rays)
                merge_counter += 1
                should_merge_flag = should_merge_candidates_with_overlap_regions(
                    merged_poly,
                    poly2,
                    angle_threshold=angle_threshold,
                    min_overlap=overlap_threshold,
                    centroid_threshold_factor=centroid_threshold_factor,
                )

                if should_merge_flag:
                    merged_poly = unary_union([merged_poly, poly2])
                    visited.add(j)

            merged_polygons.append(merged_poly)
            visited.add(i)

        orientation_angles_merged_polygons = get_shapely_polygon_pca_orientations_legacy(merged_polygons)
        if debug_plots and len(merged_polygons) > 0:
            plot_polygons(
                merged_polygons,
                f"Merged Polygons - After Iteration {iteration}",
                n_rays,
                # orientation_angles=orientation_angles_merged_polygons,
                save_path=debug_plot_dir,
            )

        if len(merged_polygons) == len(polygons):
            break

        polygons = merged_polygons
        iteration += 1

    return polygons
# Merge function for the no overlap labels
def merge_polygons_iterative_no_overlap_regions(
    polygons,
    all_points,
    all_distances,
    all_scores,
    n_rays,
    overlap_threshold=0.0,
    angle_threshold=50,
    normalized_centroid_threshold=0.5,
    alignment_tolerance=1.3,
    aspect_ratio_threshold=5,
    iteration_nbr=5,
    fname=None,
    debug_plots=False,
    debug_plot_dir=None,
):
    """
    Iteratively merge polygons for datasets without explicit overlap regions.

    Builds a KD-tree over centroids, searches local neighbors within a size-scaled
    radius, applies a no-overlap merge predicate, and unions polygons until no
    further merges occur or max iterations is reached.

    Returns
    -------
    merged_polygons : list[Polygon]
    merged_scores : list[float]
        Score propagated from the highest-score contributor within each merged instance.
    merged_points : list
        Representative point carried along with the selected score.
    """
    # Build filename prefix (used only if fname is provided)
    fname_prefix = ""
    if fname is not None:
        fname_prefix = f"{fname}__"
    if debug_plot_dir is not None:
        merging_folder = debug_plot_dir
    else:
        merging_folder = ProjectPaths.from_here(__file__).misc_dir("debug_merging_algorithm")


    iteration = 0
    aspect_ratios_list = []  # Initialize aspect ratio list for the iteration
    radius_multiplier = 3.5  # Start with the original radius value
    merge_counter = 0  # Counter to track unique merges


    # Sort with coordinates in the image to "grow the CNTs from left to right, top to bottom"
    combined_polygons = list(zip(polygons, all_scores, all_points))
    combined_sorted = sorted(combined_polygons, key=lambda x: (-x[0].centroid.y, x[0].centroid.x))
    current_polygons, current_scores, current_points = map(list, zip(*combined_sorted)) #unzip in lists

    
    if debug_plots:
        plot_polygons(current_polygons, f"{fname_prefix}Merged Polygons - After Iteration 0", n_rays, save_path=merging_folder,flip_vertical=True, scores=None)
        plot_polygons(current_polygons, f"{fname_prefix}Merged Polygons with Scores - After Iteration 0", n_rays, save_path=merging_folder,flip_vertical=True, scores=current_scores)

    while iteration < iteration_nbr:
        merged_polygons = []
        merged_scores = []
        merged_points = []
        visited = set()

        # Rebuild KD-Tree for current polygons
        r_tree_index = KDTree([poly.centroid.coords[0] for poly in current_polygons])
        
        for i, poly1 in enumerate(current_polygons):
            if i in visited:
                continue

            radius = radius_multiplier * ((poly1.bounds[2] - poly1.bounds[0] + poly1.bounds[3] - poly1.bounds[1]) / 2)
            candidates = r_tree_index.query_ball_point(poly1.centroid.coords[0], r=radius)
            candidates = [j for j in candidates if j < len(current_polygons)]

            # Sort with coordinates in the image to "grow the CNTs from left to right, top to bottom"
            # 03062025 TODO: 
            # Experiment with new method to sort polygons
            # Sort by confidence score to supress the directional bias?
            # How does it affect the performances
                
            kd_candidates_polygons = [current_polygons[i] for i in candidates]
            ys = np.array([-p.centroid.y for p in kd_candidates_polygons])  # negative for descending
            xs = np.array([p.centroid.x for p in kd_candidates_polygons])
            indices = np.lexsort((xs, ys)) #for multi-key sorting (like -y then x).
            candidates = [candidates[i] for i in indices]

            merged_poly = poly1
            merged_score = current_scores[i]
            merged_point = current_points[i]

            for j in candidates:
                if j == i or j in visited:
                    continue

                poly2 = current_polygons[j]
                merge_counter += 1

                # # COMMENT FROM HERE TO "STOP DEBUG"  WHEN STOP DEBUGGING WITH IF ELSE !!
                # # DEBUG: SET THE POLYGONS VALUE BELOW TO INVESTIGATE OVERLAPPING ONES (check on Iteration 0 first!)
                # # CHANGE the Ps value and if/else conditions to demonstrate your point 
                # P3=0.41
                # P2=0.24
                # P1=0.28
                
                # if (round(float(current_scores[j]), 2) == round(P1,2) and round(float(current_scores[i]), 2) == round(P3,2)) \
                # or (round(float(current_scores[j]), 2) == round(P1,2) and round(float(current_scores[i]), 2) == round(P2,2))\
                # or (round(float(current_scores[j]), 2) == round(P2,2) and (round(float(current_scores[i]), 2) == round(P3,2))) \
                # or (round(float(current_scores[i]), 2) == round(P2,2) and (round(float(current_scores[j]), 2) == round(P3,2))) :
                # #or round(float(current_scores[i]), 2) == round(P1,2):
                #     print(f'For each poly, loop over the neighbors \n')
                #     print(f'poly: {current_scores[i]} --> rounded: {round(float(current_scores[i]), 2)}')
                #     print(f'candidate: {current_scores[j]} --> rounded: {round(float(current_scores[j]), 2)}')
                # # STOP DEBUG 
                ##############
                should_merge_flag = should_merge_candidates_no_overlap_regions(
                    merged_poly,
                    poly2,
                    overlap_threshold=overlap_threshold,
                    angle_threshold=angle_threshold,
                    alignment_tolerance=alignment_tolerance,
                    aspect_ratio_threshold=aspect_ratio_threshold,
                    normalized_centroid_threshold=( normalized_centroid_threshold ),
                    merge_id=merge_counter,
                )

                if should_merge_flag:
                    merged_poly = unary_union([merged_poly, poly2])
                    if current_scores[j] > merged_score:
                        merged_score = current_scores[j]
                        merged_point = current_points[j]
                    visited.add(j)

            merged_polygons.append(merged_poly)
            merged_scores.append(merged_score)
            merged_points.append(merged_point)
            visited.add(i)

        if len(merged_polygons) == len(current_polygons):
            break  # No more merges possible


        current_polygons = merged_polygons
        current_scores = merged_scores
        current_points = merged_points
        iteration += 1
        if debug_plots:
            plot_polygons(current_polygons, f"{fname_prefix} Merged Polygons - After Iteration {iteration}", n_rays, save_path=merging_folder, flip_vertical=True, scores=None)
            plot_polygons(current_polygons, f"{fname_prefix} Merged Polygons with Scores - After Iteration {iteration}", n_rays, save_path=merging_folder, flip_vertical=True, scores=current_scores )

    return current_polygons, current_scores, current_points

