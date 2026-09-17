import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from src.orientation_metrics import extract_orientations_from_profile

import matplotlib.pyplot as plt
from fil_finder import FilFinder2D
import astropy.units as u

import cv2
from skimage.morphology import thin, skeletonize, binary_opening, binary_closing, disk

def calculate_skeleton_length_accurate(polygon, skeleton):
    """
    More accurate skeleton length calculation using polygon-based approach.
    """

    if np.sum(skeleton) == 0:
        return 0

    # Method1: Euclidean distance between connected points (most accurate)
    from scipy import ndimage
    from scipy.spatial.distance import euclidean
    polygon = polygon
    labeled_skeleton, num_components = ndimage.label(skeleton)
    total_length = 0

    for comp_id in range(1, num_components + 1):
        component = (labeled_skeleton == comp_id)
        points = np.column_stack(np.where(component))

        if len(points) < 2:
            total_length += 1
            continue

        # Use minimum spanning tree for accurate length measurement
        try:
            from sklearn.neighbors import NearestNeighbors
            # Create path through all skeleton points
            nbrs = NearestNeighbors(n_neighbors=2).fit(points)
            distances, indices = nbrs.kneighbors(points)

            # Use maximum path length through the skeleton
            graph = {}
            for i, (dists, idxs) in enumerate(zip(distances, indices)):
                graph[i] = [(idxs[1], dists[1])]

            # Simple approximation: sum of Euclidean distances along sorted path
            points_sorted = sort_points_along_curve(points)
            comp_length = 0
            for i in range(1, len(points_sorted)):
                comp_length += euclidean(points_sorted[i - 1], points_sorted[i])

            total_length += comp_length
        except:
            # Fallback: pixel count with diagonal correction
            comp_length = np.sum(component)
            # Apply correction for diagonal connections
            total_length += comp_length * 1.2  # Empirical factor

    return total_length


def sort_points_along_curve(points):
    """
    Sort a skeleton curve using 8-connected adjacency graph.
    Finds endpoints, then runs BFS to produce the correct order.
    """
    import numpy as np
    from collections import defaultdict, deque
    if len(points) == 0:
        return points

    # Convert to tuples for hashing
    pts = [tuple(p) for p in points]

    # Build adjacency graph (8-connected)
    graph = defaultdict(list)

    pt_set = set(pts)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    for p in pts:
        y, x = p
        for dy, dx in directions:
            n = (y + dy, x + dx)
            if n in pt_set:
                graph[p].append(n)

    # Find endpoints = nodes with degree 1
    endpoints = [p for p in pts if len(graph[p]) == 1]

    if len(endpoints) == 0:
        # Closed loop? Start anywhere
        start = pts[0]
    else:
        # Start at one endpoint
        start = endpoints[0]

    # BFS to traverse the curve
    visited = set([start])
    order = [start]
    queue = deque([start])

    while queue:
        cur = queue.popleft()
        for nei in graph[cur]:
            if nei not in visited:
                visited.add(nei)
                queue.append(nei)
                order.append(nei)

    return np.array(order)


def get_orientation_from_polygon(polygon):
    """Extract orientation of a polygon using PCA on its boundary coordinates."""
    if polygon.is_empty or polygon.area <= 0:
        return None

    coords = np.array(polygon.exterior.coords)
    coords = np.unique(coords, axis=0)
    if len(coords) < 2:
        return None

    cov = np.cov(coords, rowvar=False)
    eigvals, eigvecs = np.linalg.eig(cov)
    major_axis = eigvecs[:, np.argmax(eigvals)]
    # 0 to 180 pointing downwards
    angle = np.degrees(np.arctan2(major_axis[1], major_axis[0])) 
    return angle % 180


def get_length_from_polygon(polygon, image_height=256, image_width=256):
    """Polygon-based skeletonization using FilFinder2D for accurate length measurement."""



    # Create mask
    poly_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    exterior_coords = np.array(polygon.exterior.coords, dtype=np.int32)
    cv2.fillPoly(poly_mask, [exterior_coords], 1)

    # Handle holes
    for interior in polygon.interiors:
        interior_coords = np.array(interior.coords, dtype=np.int32)
        cv2.fillPoly(poly_mask, [interior_coords], 0)
    cleaned_mask = binary_opening(poly_mask, disk(1))
    cleaned_mask = binary_closing(cleaned_mask, disk(1))

    # Skeletonization methods
    skeleton = thin(cleaned_mask)

    # Make sure 'skeleton' is numeric (0 and 1), not boolean
    skeleton_numeric = skeleton.astype(float)
    skeleton_length_pix = np.sum(skeleton_numeric)
    # Initialize FilFinder2D
    #fil = FilFinder2D(skeleton_numeric, distance=250 * u.pc, mask=skeleton_numeric)

    # # Preprocess image
    # fil.preprocess_image(flatten_percent=85, skip_flatten=True)

    # # Create mask
    # fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)

    # # Medial skeleton
    # fil.medskel(verbose=False)
    #fil.skeleton = skeleton_numeric

    # Analyze skeletons
    # fil.analyze_skeletons(branch_thresh=40 * u.pix, skel_thresh=10 * u.pix, prune_criteria='length')

    # # Get skeleton length (in pixels)
    # skeleton_length_pix = fil.lengths()[0]
    #text_str = f"Length: {skeleton_length_pix:.1f} px"
    return skeleton_length_pix





# def extract_main_orientations_from_profile(profile, angle_bins, n_peaks=5, sigma=1, min_distance_deg=20, wrap=True):
#     """Extract dominant angles from a smoothed circular histogram."""
    
#     # Circular convolution mode with "wrap"
#     # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html
#     smoothed = gaussian_filter1d(profile.astype(float), sigma=sigma, mode='wrap' if wrap else 'nearest')
    
#     bin_size_deg = 180 / len(angle_bins)
#     distance_bins = int(np.ceil(min_distance_deg / bin_size_deg))

#     if wrap:
#         extended = np.concatenate([smoothed, smoothed])
#         peaks, _ = find_peaks(extended, distance=distance_bins)
#         mapped_peaks = peaks % len(smoothed)
#         mapped_vals = extended[peaks]
#         unique_peaks, idx = np.unique(mapped_peaks, return_index=True)
#         peak_vals = np.zeros_like(unique_peaks, dtype=float)
#         for i, p in enumerate(unique_peaks):
#             mask = mapped_peaks == p
#             peak_vals[i] = mapped_vals[mask].max()
#         peaks = unique_peaks
#     else:
#         peaks, _ = find_peaks(smoothed, distance=distance_bins)
#         peak_vals = smoothed[peaks]

#     top_idx = np.argsort(peak_vals)[::-1][:n_peaks]
#     main_angles = angle_bins[peaks[top_idx]] % 180
#     return main_angles, smoothed, peaks


from tqdm import tqdm
import numpy as np
from shapely.geometry import Polygon

def add_features_to_coco(coco_dict):

    annotations = coco_dict.get("annotations", [])

    for ann in tqdm(annotations, desc="Processing annotations"):
        seg = ann.get("segmentation", [])
        if not seg or not seg[0]:
            ann["orientation"] = None
            continue

        # Convert segmentation list to polygon
        coords = np.array(seg[0]).reshape(-1, 2)
        polygon = Polygon(coords)

        # Compute features
        if polygon.is_empty or polygon.area < 1:
            continue
        length = get_length_from_polygon(polygon)
        angle = get_orientation_from_polygon(polygon)
        angle = angle/length if length>0 else None
        

        # Store rounded values
        ann["orientation"] = round(angle, 2) if angle is not None else None
        ann["length"] = round(length, 4) if length is not None else None

    return coco_dict



def coco_orientations_to_df(coco):
    """Return a DataFrame of image_id, annotation_id, and orientation (angle)."""

    import pandas as pd
    # Collect data
    records = []
    for ann in coco.get("annotations", []):
        records.append({
            "image_id": ann.get("image_id"),
            "annotation_id": ann.get("id"),
            "angle": ann.get("orientation")
        })

    # Build DataFrame
    df = pd.DataFrame(records)

    return df




def find_main_polygon_orientations(updated_coco, image_id, n_peaks=4, bin_size=5):
    """Find main orientations in circular (axial) data from polygon angles."""

    df_image_cnt_angle = coco_orientations_to_df(updated_coco)

    angles = df_image_cnt_angle[df_image_cnt_angle['image_id'] == image_id]['angle']
        
    bins = np.arange(0, 181, bin_size)
    hist, edges = np.histogram(angles, bins=bins)

    # Use the shared circular-aware peak extraction
    hist_smoothed, peaks = extract_orientations_from_profile(
        hist,
        edges[:-1],
        n_peaks=n_peaks,
        sigma=1,
        min_distance_deg=bin_size * 4,
        wrap=True
    )

    return hist_smoothed, peaks, edges