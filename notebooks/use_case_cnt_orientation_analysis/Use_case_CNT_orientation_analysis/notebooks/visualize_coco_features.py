import os
import sys
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon
import importlib

# ---------------------------------------------------------
# 1. Determine the absolute path to the project
# ---------------------------------------------------------
# Visual Studio Code is launched from:
#   C:\Users\...\PBC\MSD\testing
#
# Your notebooks and src live in:
#   testing\Notebooks\Analysis\CNT_orientation_analysis_template\notebooks\src

testing_root = r"C:\Users\lud44009\Documents\Projects\20240513 - A.Cherchari\PBC\MSD\testing"

# Path where src/ lives
src_path = os.path.join(
    testing_root,
    "Notebooks",
    "Analysis",
    "CNT_orientation_analysis_template"
)

# Add src/ to Python path
if src_path not in sys.path:
    sys.path.append(src_path)

print(">> Added to PYTHONPATH:", src_path)

# ---------------------------------------------------------
# 2. Now imports from src will work properly
# ---------------------------------------------------------
from src.orientation_metrics import extract_orientations_from_profile
from src.mask_utils import (
    add_features_to_coco,
    find_main_polygon_orientations,
    get_orientation_from_polygon,
    get_length_from_polygon,
)
from src.visualization import show_angle_orientation

# Reload in case files changed
importlib.reload(sys.modules['src.orientation_metrics'])
importlib.reload(sys.modules['src.mask_utils'])
importlib.reload(sys.modules['src.visualization'])





# ----------------------------------------------------------------------
# --- Utility: draw polygon mask --------------------------------------
# ----------------------------------------------------------------------
def polygon_to_mask(polygon, H=256, W=256):
    mask = np.zeros((H, W), dtype=np.uint8)
    coords = np.array(polygon.exterior.coords, dtype=np.int32)
    cv2.fillPoly(mask, [coords], 1)
    return mask.astype(bool)


# ----------------------------------------------------------------------
# --- Utility: compute skeleton (for visualization only)
# ----------------------------------------------------------------------
def get_skeleton(polygon, H=256, W=256):
    from skimage.morphology import thin

    mask = polygon_to_mask(polygon, H, W)
    skeleton = thin(mask)
    return skeleton


# ----------------------------------------------------------------------
# --- Utility: get zoomed bounding box ---------------------------------
# ----------------------------------------------------------------------
def get_zoom_bounds(polygon, margin=10, image_size=256):
    xs, ys = polygon.exterior.xy
    xmin, xmax = int(max(0, min(xs) - margin)), int(min(image_size, max(xs) + margin))
    ymin, ymax = int(max(0, min(ys) - margin)), int(min(image_size, max(ys) + margin))
    return xmin, xmax, ymin, ymax


# ----------------------------------------------------------------------
# --- Visualization per annotation -------------------------------------
# ----------------------------------------------------------------------
def visualize_annotation(image, polygon, orientation, length, image_name, figsize=(15, 8)):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    
    H, W = image.shape[:2]

    mask = polygon_to_mask(polygon, H, W)
    skeleton = get_skeleton(polygon, H, W)

    xmin, xmax, ymin, ymax = get_zoom_bounds(polygon, margin=15, image_size=H)

    poly_coords = np.array(polygon.exterior.coords)
    center = np.array(polygon.centroid.coords[0])

    # --- Polygon bounds used to scale arrows dynamically ---
    minx, miny, maxx, maxy = polygon.bounds
    bbox_diag = np.sqrt((maxx - minx)**2 + (maxy - miny)**2)

    # Arrow lengths (zoom is smaller)
    arrow_len_full = bbox_diag * 0.6 or 2
    arrow_len_zoom = bbox_diag * 0.3 or 1

    # Angle → vector
    angle_rad = np.radians(orientation)
    dx_full = np.cos(angle_rad) * arrow_len_full
    dy_full = np.sin(angle_rad) * arrow_len_full

    dx_zoom = np.cos(angle_rad) * arrow_len_zoom
    dy_zoom = np.sin(angle_rad) * arrow_len_zoom

    # Arrow head sizes
    head_w_full = arrow_len_full * 0.08
    head_l_full = arrow_len_full * 0.12
    head_w_zoom = arrow_len_zoom * 0.08
    head_l_zoom = arrow_len_zoom * 0.12

    # ---------------------------------------------------------
    # FIGURE with 4 subplots
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # ---------------------------------------------------------
    # 1. FULL IMAGE
    # ---------------------------------------------------------
    ax1.imshow(image, cmap="gray")
    ax1.add_patch(MplPolygon(poly_coords, fill=True, alpha=0.30, color='red'))

    y_skel, x_skel = np.where(skeleton)
    ax1.scatter(x_skel, y_skel, s=5, c='yellow')

    ax1.set_title(f"{image_name}")
    ax1.set_axis_off()

    # ---------------------------------------------------------
    # 2. ZOOMED VIEW
    # ---------------------------------------------------------
    ax2.imshow(image[ymin:ymax, xmin:xmax], cmap="gray")

    shifted = poly_coords.copy()
    shifted[:, 0] -= xmin
    shifted[:, 1] -= ymin
    ax2.add_patch(MplPolygon(shifted, fill=True, alpha=0.30, color='red'))

    idx_zoom = (
        (y_skel >= ymin) & (y_skel <= ymax) &
        (x_skel >= xmin) & (x_skel <= xmax)
    )
    ax2.scatter(
        x_skel[idx_zoom] - xmin,
        y_skel[idx_zoom] - ymin,
        s=5, c='yellow'
    )

    ax2.set_title("Zoomed Region")
    

    # ---------------------------------------------------------
    # 3. ORIENTATION FULL VIEW (KEEP AXIS)
    # ---------------------------------------------------------
    ax3.imshow(image, cmap='gray')
    ax3.arrow(center[0], center[1], dx_full, dy_full,
              width=1.2,
              head_width=head_w_full,
              head_length=head_l_full,
              color='cyan',
              length_includes_head=True)

    ax3.set_title("Orientation (Full View)")
    ax3.set_axis_off()
    # ---------------------------------------------------------
    # 4. ORIENTATION ZOOM VIEW (KEEP AXIS)
    # ---------------------------------------------------------
    ax4.imshow(image[ymin:ymax, xmin:xmax], cmap='gray')
    ax4.arrow(center[0] - xmin, center[1] - ymin,
              dx_zoom, dy_zoom,
              width=0.3,
              head_width=head_w_zoom,
              head_length=head_l_zoom,
              color='cyan',
              length_includes_head=True)

    ax4.set_title("Orientation (Zoomed)")

    # ---------------------------------------------------------
    # TEXTBOX (moved to avoid overlap with right plots)
    # ---------------------------------------------------------
    txt = f"Orientation: {orientation:.2f}°\nLength: {length:.4f}"

    fig.text(
        0.80, 0.50, txt,
        fontsize=14,
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.85)
    )

    # Improve spacing
    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    plt.tight_layout()
    plt.show()




# ----------------------------------------------------------------------
# --- Main Runner -------------------------------------------------------
# ----------------------------------------------------------------------
def visualize_coco_with_features(coco_json_path, images_dir):
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}

    for ann in coco["annotations"]:
        seg = ann.get("segmentation", [])
        if not seg or not seg[0]:
            print("Skipping empty segmentation.")
            continue

        image_info = images_by_id[ann["image_id"]]
        image_path = f"{images_dir}/{image_info['file_name']}.jpg"

        image = plt.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            continue

        coords = np.array(seg[0]).reshape(-1, 2)
        polygon = Polygon(coords)

        orientation = get_orientation_from_polygon(polygon)
        length = get_length_from_polygon(polygon)

        print(f"Annotation {ann['id']} >> Orientation={orientation}, Length={length}")

        visualize_annotation(image, polygon, orientation, length, image_info['file_name'])


# ----------------------------------------------------------------------
# --- Example usage -----------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
   # ---------------------------------------------------------
# 3. Prepare dataset paths
# ---------------------------------------------------------
    import os
    from visualize_coco_features import visualize_coco_with_features
    base_path = r"C:\Users\lud44009\Documents\Projects\20240513 - A.Cherchari\PBC\data\annotations_filtered_artifacts\test"
    json_path = os.path.join(base_path, "COCO_mask", "predicted_annotations.json")
    image_folder = os.path.join(base_path, "images")

    print(">> JSON path:", json_path)
    print(">> Image folder:", image_folder)
    visualize_coco_with_features(
        coco_json_path=json_path,
        images_dir=image_folder
    )
