from PIL import ImageDraw
from PIL import Image
from scipy.ndimage import convolve
from shapely.geometry import LinearRing, Polygon
import numpy as np
import cv2
from sklearn.decomposition import PCA
from skimage.morphology import skeletonize, remove_small_objects
from skimage.draw import polygon as draw_polygon


def smooth_polygon(polygon, iterations=2):
    """
    Smooths a Shapely polygon using Chaikin's algorithm.

    :param polygon: Shapely Polygon object.
    :param iterations: Number of smoothing iterations to apply.
    :return: Smoothed Shapely Polygon object.
    """

    def chaikin_smooth(coords):
        smoothed_coords = []
        for i in range(len(coords) - 1):  # Iterate through pairs of points
            p1 = coords[i]
            p2 = coords[i + 1]
            # Create new points closer to p1 and p2
            smoothed_coords.append((0.75 * p1[0] + 0.25 * p2[0], 0.75 * p1[1] + 0.25 * p2[1]))
            smoothed_coords.append((0.25 * p1[0] + 0.75 * p2[0], 0.25 * p1[1] + 0.75 * p2[1]))
        return smoothed_coords + [smoothed_coords[0]]  # Close the ring

    coords = list(polygon.exterior.coords)
    for _ in range(iterations):
        coords = chaikin_smooth(coords)
    return Polygon(LinearRing(coords))

def rasterize_polygon_to_binary_image(polygon, image_size):
    """
    Converts a Shapely polygon to a binary mask (1 inside polygon, 0 outside).

    :param polygon: Shapely Polygon object.
    :param image_size: Tuple (width, height) representing the size of the binary mask.
    :return: A numpy array representing the binary mask.
    """
    polygon = smooth_polygon(polygon, iterations=2)

    # Create a blank white image (all zeros)
    mask_image = Image.new('1', image_size, 0)

    # Create a drawing context
    draw = ImageDraw.Draw(mask_image)

    # Use the polygon's exterior coordinates to draw on the image
    coords = list(polygon.exterior.coords)
    draw.polygon(coords, fill=1)  # fill=1 means inside the polygon is 1

    # Convert the mask image to a numpy array
    binary_mask = np.array(mask_image)

    return binary_mask

def crop_mask_to_roi(mask, polygon, padding=10):
    """
    Crops the binary mask to the region of interest (ROI) defined by the polygon's bounding box.
    Adds padding around the bounding box.

    :param mask: The original binary mask (NumPy array).
    :param polygon: Shapely Polygon object.
    :param padding: Amount of padding to add around the bounding box.
    :return: Cropped binary mask with padding.
    """
    # Get the bounding box of the polygon (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = polygon.bounds

    # Add padding around the bounding box
    minx = int(max(minx - padding, 0))  # Ensure the values are within bounds
    miny = int(max(miny - padding, 0))
    maxx = int(min(maxx + padding, mask.shape[1]))
    maxy = int(min(maxy + padding, mask.shape[0]))

    # Crop the mask to the bounding box with padding
    cropped_mask = mask[miny:maxy, minx:maxx]

    return cropped_mask

def simple_skeletonize(binary_mask):
    """
    Perform skeletonization using skimage's skeletonize function.

    :param binary_mask: The binary mask (NumPy array with 0s and 1s).
    :return: The skeletonized binary mask (NumPy array with skeleton as 1s).
    """
    # Ensure the mask is binary (0 and 1)
    binary_mask = (binary_mask > 0).astype(np.uint8)

    # Perform skeletonization
    skeleton = skeletonize(binary_mask)

    # Convert the skeleton to uint8 for consistency
    skeleton = skeleton.astype(np.uint8)

    return skeleton

def skeletonize_polygon(polygon):
    """
    Skeletonize a Shapely polygon.
    """
    binary_image = rasterize_polygon_to_binary_image(polygon, (256, 256))
    binary_image = crop_mask_to_roi(binary_image, polygon)

    skeleton = simple_skeletonize(binary_image)

    return skeleton

def check_y_or_t_shape(merged_polygon):
    skeleton = skeletonize_polygon(merged_polygon)
    endpoints = count_endpoints(skeleton)

    # If there are more than two distinct endpoints, it indicates a Y or T shape
    if endpoints > 2:
        return True  # It's a Y or T shape
    return False

# previously named plot_skeleton
def count_polygon_skeleton_endpoints(merged_polygon, merged_id):
    # Convert merged polygon to binary image for skeletonization
    min_x, min_y, max_x, max_y = merged_polygon.bounds
    image_shape = (int(max_y - min_y) + 1, int(max_x - min_x) + 1)

    # Create a filled binary mask from the merged polygon
    binary_image = np.zeros(image_shape, dtype=np.uint8)

    # Get the coordinates of the polygon and fill the image
    coords = np.array(merged_polygon.exterior.coords)
    rr, cc = draw_polygon(coords[:, 1] - min_y, coords[:, 0] - min_x, binary_image.shape)
    binary_image[rr, cc] = 1  # Fill the polygon area

    # Skeletonize the filled binary image
    skeleton = skeletonize(binary_image)

    # Count endpoints in the skeleton
    num_endpoints = count_endpoints(skeleton)
    return num_endpoints


# attempt to filter non elongated polygons for future post processing (for labeling approach B: with overlap)
def count_endpoints(skeleton):
    """
    Counts the number of endpoints in a skeletonized image.

    Parameters:
        skeleton (ndarray): Binary image where the skeleton is represented by 1s.

    Returns:
        int: The number of endpoints in the skeleton.
    """
    # Create a kernel to check the 8-connected neighbors of each pixel
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    # Find the coordinates of the skeleton pixels (where value is 1)
    skeleton_coords = np.argwhere(skeleton == 1)

    # Initialize the count for endpoints
    endpoints = 0

    # Iterate over the skeleton coordinates and check the number of neighbors
    for coord in skeleton_coords:
        x, y = coord
        # Extract a 3x3 patch around the pixel
        patch = skeleton[max(0, x - 1):x + 2, max(0, y - 1):y + 2]

        # Count the number of neighbors by summing the patch
        num_neighbors = np.sum(patch) - 1  # Subtract 1 to exclude the central pixel

        # If a skeleton pixel has exactly 1 neighbor, it is an endpoint
        if num_neighbors == 1:
            endpoints += 1

    return endpoints

# filter the elongated from the overlap polygons usefull only for labels with overlap section
def count_endpoints_for_elongated_polygons(skeleton):
    """
    Count endpoints in a skeletonized image.
    An endpoint is a pixel with only one neighboring pixel.
    """

    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])

    filtered = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
    endpoints = np.sum(filtered == 11)

    return endpoints

# check if a polygon is elongated or if it is an overlap region (for labeling approach B: with overlap)
def is_elongated_polygon(polygon, min_aspect_ratio=4, prune_threshold=2):
    """
    Determine if a polygon is elongated using multiple criteria:
    skeleton pruning, principal axis, and convex hull length.
    """
    if polygon.is_empty:
        return False

    # 1. Prune Skeleton: Skeletonize and prune short branches
    skeleton = skeletonize(np.array(polygon.exterior.coords.xy).astype(bool))
    pruned_skeleton = remove_small_objects(skeleton, prune_threshold)

    # Count endpoints in the pruned skeleton
    endpoints = count_endpoints_for_elongated_polygons(pruned_skeleton)
    if endpoints > 2:
        return False

    # 2. Principal Axis Check (PCA)
    coords = np.array(polygon.exterior.coords)
    pca = PCA(n_components=2)
    pca.fit(coords)
    aspect_ratio = pca.explained_variance_ratio_[0] / pca.explained_variance_ratio_[1]

    if aspect_ratio < min_aspect_ratio:
        return False

    # 3. Convex Hull Check
    hull_length = polygon.convex_hull.length
    if hull_length < min_aspect_ratio:
        return False

    return True

def skeletonize_mask_with_polygon_smoothing(
    binary_mask,
    *,
    smoothing_iterations=2,
):
    """
    Smooth a binary mask's exterior contours using Chaikin smoothing
    and then skeletonize the resulting mask.

    This is the mask equivalent of the preprocessing performed by
    check_y_or_t_shape().
    """
    binary_mask = np.ascontiguousarray(
        np.asarray(binary_mask).astype(bool),
        dtype=np.uint8,
    )

    if binary_mask.ndim != 2:
        raise ValueError(
            "binary_mask must be two-dimensional; "
            f"received shape {binary_mask.shape}."
        )

    if not binary_mask.any():
        raise ValueError(
            "binary_mask must contain foreground pixels."
        )

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    smoothed_mask = np.zeros(
        binary_mask.shape,
        dtype=np.uint8,
        order="C",
    )

    rasterized_component_count = 0

    for contour in contours:
        coordinates_xy = contour.reshape(
            -1,
            2,
        )

        if coordinates_xy.shape[0] < 3:
            continue

        polygon = Polygon(
            coordinates_xy
        )

        if (
            polygon.is_empty
            or polygon.area <= 0.0
        ):
            continue

        smoothed_polygon = smooth_polygon(
            polygon,
            iterations=smoothing_iterations,
        )

        if (
            smoothed_polygon.is_empty
            or smoothed_polygon.area <= 0.0
        ):
            continue

        smoothed_coordinates = np.ascontiguousarray(
            np.rint(
                np.asarray(
                    smoothed_polygon.exterior.coords,
                    dtype=np.float64,
                )
            ).astype(np.int32)
        ).reshape(-1, 1, 2)

        cv2.fillPoly(
            smoothed_mask,
            [smoothed_coordinates],
            1,
        )

        rasterized_component_count += 1

    # Very small or degenerate objects may not produce a valid
    # polygon. Preserve the original mask in that case.
    if rasterized_component_count == 0:
        smoothed_mask = binary_mask

    return simple_skeletonize(
        smoothed_mask
    )