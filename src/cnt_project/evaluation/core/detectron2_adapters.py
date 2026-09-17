import numpy as np
import cv2


# utility functions to convert predictions and gt from detectron2 to multiclass numpy arrays
def preprocess_predictions(predictions):
    """
    Convert dictionary-based predictions (from Detectron2) to multiclass NumPy arrays.
    """
    processed_predictions = []
    for pred in predictions:
        if isinstance(pred, dict) and "instances" in pred:
            # Detectron2 format: Convert to multiclass image
            masks = pred["instances"].pred_masks.cpu().numpy()  # Shape: (N, H, W)
            height, width = masks.shape[1], masks.shape[2]
            multiclass_image = np.zeros((height, width), dtype=np.uint8)
            for i, mask in enumerate(masks):
                multiclass_image[mask] = i + 1  # Assign unique ID (starting from 1)
            processed_predictions.append(multiclass_image)
        else:
            # Assume it's already a multiclass NumPy array
            processed_predictions.append(pred)
    return processed_predictions

def preprocess_ground_truth(ground_truth, image_height, image_width):
    """
    Convert ground truth annotations (from Detectron2 dataset format) to multiclass NumPy arrays.
    """
    processed_ground_truth = []
    for gt in ground_truth:
        if isinstance(gt, dict) and "annotations" in gt:
            # Detectron2 format: Convert to multiclass image
            gt_annotations = gt["annotations"]
            gt_multiclass_image = np.zeros((image_height, image_width), dtype=np.uint8)
            for i, ann in enumerate(gt_annotations):
                polygons = ann["segmentation"]  # List of polygons
                mask = np.zeros((image_height, image_width), dtype=np.uint8)
                for polygon in polygons:
                    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 2))
                    cv2.fillPoly(mask, [polygon], color=1)
                gt_multiclass_image[mask == 1] = i + 1  # Assign unique ID (starting from 1)
            processed_ground_truth.append(gt_multiclass_image)
        else:
            # Assume it's already a multiclass NumPy array
            processed_ground_truth.append(gt)
    return processed_ground_truth

