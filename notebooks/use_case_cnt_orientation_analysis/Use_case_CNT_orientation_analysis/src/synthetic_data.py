import numpy as np
import pandas as pd
import cv2
import json

def line_to_polygon(x1, y1, x2, y2, thickness):
    """
    Convert a line segment to a polygon (rectangle) for COCO segmentation.
    """
    # Compute direction vector
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    if length == 0:
        return [x1, y1, x1, y1, x1, y1, x1, y1]  # Degenerate case
    
    # Unit perpendicular vector
    ux = -dy / length
    uy = dx / length

    # Half thickness
    ht = thickness / 2

    # Four corners
    p1 = [x1 + ux*ht, y1 + uy*ht]
    p2 = [x1 - ux*ht, y1 - uy*ht]
    p3 = [x2 - ux*ht, y2 - uy*ht]
    p4 = [x2 + ux*ht, y2 + uy*ht]

    # Flatten to list for COCO
    polygon = [coord for point in [p1, p2, p3, p4] for coord in point]
    return polygon

def generate_segments_coco(
    img_size=256,
    num_segments=150,
    main_angle_deg=45,
    angle_std_deg=5,
    length_range=(10, 80),
    thickness_range=(1, 4),
    base_image=None,
    category_id=1,
    annotation_start_id=1,
    image_id=1
):
    """
    Generate synthetic line segments as polygons for COCO format.
    Returns image and list of COCO annotations.
    """
    if base_image is None:
        image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    else:
        image = base_image.copy()

    main_angle = np.deg2rad(main_angle_deg)
    angle_std = np.deg2rad(angle_std_deg)

    annotations = []
    ann_id = annotation_start_id

    for _ in range(num_segments):
        x1, y1 = np.random.randint(0, img_size, 2)
        angle = np.random.normal(main_angle, angle_std)
        length = np.random.randint(*length_range)
        thickness = np.random.randint(*thickness_range)

        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        x2 = np.clip(x2, 0, img_size - 1)
        y2 = np.clip(y2, 0, img_size - 1)

        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), thickness)

        # COCO polygon
        poly = line_to_polygon(x1, y1, x2, y2, thickness)
        x_coords = poly[0::2]
        y_coords = poly[1::2]
        x_min, y_min = min(x_coords), min(y_coords)
        width, height = max(x_coords) - x_min, max(y_coords) - y_min
        area = width * height

        annotation = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": [poly],
            "bbox": [x_min, y_min, width, height],
            "area": area,
            "iscrowd": 0
        }
        annotations.append(annotation)
        ann_id += 1

    return image, annotations



import numpy as np
import cv2

def generate_segments_image(
    img_size=256,
    num_segments=150,
    main_angle_deg=45,
    angle_std_deg=5,
    length_range=(10, 80),
    thickness_range=(1, 4),
    color=(255, 255, 255),
    background_color=(0, 0, 0),
    base_image=None
):
    """
    Generate a synthetic image containing random line segments with a preferred orientation.
    Can optionally draw on an existing image to combine multiple orientations.

    Parameters:
        img_size (int): Size of the square image (pixels).
        num_segments (int): Number of line segments.
        main_angle_deg (float): Main orientation in degrees.
        angle_std_deg (float): Angular deviation (standard deviation in degrees).
        length_range (tuple): (min_length, max_length) of segments.
        thickness_range (tuple): (min_thickness, max_thickness) of lines.
        color (tuple): BGR color of the line segments.
        background_color (tuple): BGR background color.
        base_image (np.ndarray or None): Optional image to draw on (must be same size).

    Returns:
        np.ndarray: Generated image (uint8).
    """
    # Create a blank image or use the provided base image
    if base_image is None:
        image = np.full((img_size, img_size, 3), background_color, dtype=np.uint8)
    else:
        image = base_image.copy()

    # Convert angles to radians
    main_angle = np.deg2rad(main_angle_deg)
    angle_std = np.deg2rad(angle_std_deg)

    for _ in range(num_segments):
        x1, y1 = np.random.randint(0, img_size, 2)
        angle = np.random.normal(main_angle, angle_std)
        length = np.random.randint(*length_range)
        thickness = np.random.randint(*thickness_range)

        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        x2 = np.clip(x2, 0, img_size - 1)
        y2 = np.clip(y2, 0, img_size - 1)

        cv2.line(image, (x1, y1), (x2, y2), color, thickness)

    return image

