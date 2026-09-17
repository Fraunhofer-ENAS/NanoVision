import random
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image

def plot_polygons_from_json(
    coco_json_path: str,
    image_dir: str,
    output_dir: str,
    image_key: str = "file_name",
    file_ext: str = ".tif",
    seg_key: str = "segmentation",
    annotation_key: str = "annotations"
):
    """
    Reads a COCO-style JSON with polygon segmentations, overlays them on each image,
    and saves the figures to `output_dir`. Assumes JSON's file_name has no extension
    and appends `file_ext` when loading.

    Args:
        coco_json_path: Path to the COCO JSON file.
        image_dir: Directory where the images live.
        output_dir: Directory to save overlay figures (will be created if needed).
        image_key: JSON field in "images" giving the filename (without extension).
        file_ext: Extension to append when loading images (e.g. ".tif").
        seg_key: JSON field in each annotation with polygon coords.
        annotation_key: Top-level JSON key for annotations.
    """
    # Load JSON
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    # Index annotations by image_id
    anns_by_image = {}
    for ann in coco[annotation_key]:
        img_id = ann["image_id"]
        anns_by_image.setdefault(img_id, []).append(ann)

    os.makedirs(output_dir, exist_ok=True)

    # Iterate images
    for img in coco["images"]:
        img_id = img["id"]
        base_name = img[image_key]
        img_filename = base_name + file_ext
        img_path = os.path.join(image_dir, img_filename)

        # Load image
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: {img_path} not found, skipping.")
            continue

        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(pil_img)
        ax.axis("off")

        # Overlay polygons
        for ann in anns_by_image.get(img_id, []):
            for poly in ann[seg_key]:
                coords = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
                patch = MplPolygon(
                    coords,
                    closed=True,
                    edgecolor="yellow",
                    facecolor="none",
                    linewidth=1.5,
                    alpha=0.8
                )
                ax.add_patch(patch)

        # Save figure
        out_name = f"{base_name}_overlay.png"
        out_path = os.path.join(output_dir, out_name)
        fig.tight_layout(pad=0)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved overlay: {out_path}")

def visualize_json_annotations(json_path, image_folder, output_folder=None):
    """
    Visualizes annotations from a COCO JSON file for all images in the dataset.

    Args:
        json_path (str): Path to the JSON file.
        image_folder (str): Path to the folder containing the corresponding images.
        output_folder (str, optional): Folder to save the visualization. If None, the visualization is not saved.
    """
    # Load JSON data
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    # Create a mapping from image_id to image metadata
    image_id_to_info = {image["id"]: image for image in coco_data["images"]}

    # Assign random colors to each object ID
    object_colors = {}

    # Iterate over all images in the dataset
    for image_info in coco_data["images"]:
        image_filename = image_info["file_name"]
        image_filename = image_filename + ".jpg"
        image_id = image_info["id"]
        image_path = os.path.join(image_folder, image_filename)

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image {image_filename} not found in {image_folder}. Skipping...")
            continue

        # Create a blank overlay for visualization
        height, width = image_info["height"], image_info["width"]
        overlay = np.zeros((height, width, 3), dtype=np.uint8)

        # Get all annotations for this image
        image_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]

        # Draw each annotation
        for annotation in image_annotations:
            object_id = annotation["id"]
            segmentation = annotation["segmentation"]

            # Assign a random color if the object_id is new
            if object_id not in object_colors:
                object_colors[object_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            color = object_colors[object_id]

            # Draw each polygon in the segmentation
            for polygon in segmentation:
                points = np.array(polygon, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(overlay, [points], color)

        # Blend the overlay with the original image
        alpha = 0.5  # Transparency factor
        visualized_image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

        # Display the result
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(visualized_image, cv2.COLOR_BGR2RGB))
        plt.title(f"{image_filename}_Visualization of predictions")
        plt.axis("off")
        plt.show()

        # Save the visualization if an output folder is provided
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, f"{image_filename}_Visualization of predictions.png")
            cv2.imwrite(output_path, visualized_image)
            print(f"Visualization saved to {output_path}")
            # Save as SVG (vector) using Matplotlib
            output_path_svg = os.path.join(output_folder, f"{image_filename}_Visualization of predictions.svg")
            plt.figure(figsize=(visualized_image.shape[1]/100, visualized_image.shape[0]/100))
            plt.imshow(visualized_image)
            plt.axis('off')
            plt.savefig(output_path_svg, format='svg', bbox_inches='tight')
            plt.close()
            print(f"Visualization saved to {output_path_svg}")
            

    print("Visualization complete for all images.")



def visualize_json_annotations_svg(json_path, image_folder, output_folder=None,
                                   alpha=0.4, darken_factor=0.4):
    """
    Visualize COCO JSON annotations and save as SVG.
    - darken_factor: 0..1, multiplies original image to darken it
    """
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    object_colors = {}

    for image_info in coco_data["images"]:
        image_filename = image_info["file_name"] + ".jpg"
        image_id = image_info["id"]
        image_path = os.path.join(image_folder, image_filename)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Image {image_filename} not found. Skipping...")
            continue
        height, width = image.shape[:2]

        image_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]

        # Darken image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        darkened = (rgb_image * darken_factor).astype(np.uint8)

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.imshow(darkened)
        ax.axis('off')

        ax.set_position([0, 0, 1, 1])
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

        patches, colors = [], []

        for ann in image_annotations:
            object_id = ann["id"]
            if object_id not in object_colors:
                object_colors[object_id] = (random.random(), random.random(), random.random())
            color = object_colors[object_id]

            for polygon_coords in ann["segmentation"]:
                points = [(polygon_coords[i], polygon_coords[i + 1]) 
                          for i in range(0, len(polygon_coords), 2)]
                patches.append(MplPolygon(points, closed=True))
                colors.append(color)

        if patches:
            ax.add_collection(PatchCollection(
                patches, facecolor=colors, edgecolor='none',
                alpha=alpha, linewidth=2.5
            ))

        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            output_path_svg = os.path.join(output_folder, f"{image_filename}_masks_overlay.svg")
            plt.savefig(output_path_svg, format='svg', bbox_inches=None, pad_inches=0)
            plt.close(fig)
            print(f"Saved SVG with masks overlay to {output_path_svg}")

            
def visualize_json_annotations_bitmap(json_path, image_folder, output_folder=None, alpha=0.5):
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    object_colors = {}

    for image_info in coco_data["images"]:
        image_filename = image_info["file_name"] + ".jpg"
        image_id = image_info["id"]
        image_path = os.path.join(image_folder, image_filename)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Image {image_filename} not found. Skipping...")
            continue
        height, width = image.shape[:2]

        # blank mask (same size as image)
        mask_overlay = np.zeros_like(image, dtype=np.uint8)

        image_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]

        for ann in image_annotations:
            object_id = ann["id"]
            if object_id not in object_colors:
                object_colors[object_id] = tuple([int(255*random.random()) for _ in range(3)])
            color = object_colors[object_id]

            for polygon_coords in ann["segmentation"]:
                # Convert to Nx2 int32 array
                pts = np.array(polygon_coords, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask_overlay, [pts], color)

        # blend original image with mask overlay
        blended = cv2.addWeighted(image, 1 - alpha, mask_overlay, alpha, 0)

        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            output_path_png = os.path.join(output_folder, f"{image_filename}_masks_overlay.png")
            cv2.imwrite(output_path_png, blended)
            print(f"Saved rasterized bitmap with masks overlay to {output_path_png}")

        # Optional: also show
        plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()


def visualize_json_masks_only_svg(json_path, output_folder=None, alpha=0.5, bg_color='black'):
    import os
    import json
    import random
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    with open(json_path, "r") as f:
        coco_data = json.load(f)

    object_colors = {}

    for image_info in coco_data["images"]:
        image_filename = image_info["file_name"]+ ".jpg"
        image_id = image_info["id"]
        width, height = image_info["width"], image_info["height"]

        image_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.axis('off')
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        # fill whole figure, no padding
        ax.set_position([0, 0, 1, 1])
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

        patches, colors = [], []

        for ann in image_annotations:
            object_id = ann["id"]
            if object_id not in object_colors:
                object_colors[object_id] = (
                    random.random() * 0.7 + 0.3,
                    random.random() * 0.7 + 0.3,
                    random.random() * 0.7 + 0.3
                )
            color = object_colors[object_id]

            for polygon_coords in ann["segmentation"]:
                points = [(polygon_coords[i], polygon_coords[i+1]) for i in range(0, len(polygon_coords), 2)]
                patches.append(MplPolygon(points, closed=True))
                colors.append(color)

        if patches:
            ax.add_collection(PatchCollection(patches, facecolor=colors, edgecolor='none',
                                              alpha=alpha)) #, linewidth=0.1
            ax.autoscale_view()

        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            output_path_svg = os.path.join(output_folder, f"{image_filename}_masks_only.svg")
            plt.savefig(output_path_svg, format='svg', facecolor=fig.get_facecolor(),
                        bbox_inches=None, pad_inches=0)
            plt.close(fig)
            print(f"Saved SVG with masks only to {output_path_svg}")