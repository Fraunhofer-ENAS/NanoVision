from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import pycocotools.mask as mask
from skimage.draw import polygon as sk_polygon


def pack_array(array: Sequence[float]) -> np.ndarray:
    return np.array([[int(array[index]), int(array[index + 1])] for index in range(0, len(array), 2)])


def gt_polygons_to_mask(segmentation, height, width):
    mask_out = np.zeros((height, width), dtype=np.uint8)

    if not segmentation:
        return mask_out

    polygons = [pack_array(poly) for poly in segmentation]
    cv2.fillPoly(mask_out, polygons, 1)
    return mask_out

def rle_segmentation_to_mask(segmentation: dict, height: int, width: int) -> np.ndarray:
    rle = dict(segmentation)

    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("utf-8")

    decoded = mask.decode(rle)

    if len(decoded.shape) == 3:
        decoded = np.sum(decoded, axis=2) > 0

    return decoded.astype(np.uint8)

def extract_predictions_from_json(score_thres, coco_gt, coco_dt):
    det_scores_per_image = {}
    det_masks_per_image = []
    det_boxes_per_image = []

    img_id_to_idx = {img_id: idx for idx, img_id in enumerate(coco_gt.getImgIds())}
    predictions_by_image = {}

    for det in coco_dt["annotations"]:
        score = det.get("score", 1.0)
        if score < score_thres:
            continue
        img_id = det["image_id"]
        predictions_by_image.setdefault(img_id, []).append(det)

    for img_id in coco_gt.getImgIds():
        img_info = coco_gt.imgs[img_id]
        height, width = img_info["height"], img_info["width"]
        dets = predictions_by_image.get(img_id, [])

        scores = []
        masks_list = []
        boxes = []
        for det in dets:
            scores.append(det.get("score", 1.0))
            boxes.append(det["bbox"])

            seg = det["segmentation"]

            if isinstance(seg, dict):
                decoded = rle_segmentation_to_mask(seg, height, width)
            else:
                rle = mask.frPyObjects(seg, height, width)
                decoded = mask.decode(rle)
                if len(decoded.shape) == 3:
                    decoded = np.sum(decoded, axis=2) > 0
                decoded = decoded.astype(np.uint8)

            masks_list.append(decoded.astype(np.uint8))

        det_scores_per_image[img_id_to_idx[img_id]] = scores
        det_masks_per_image.append(masks_list)
        det_boxes_per_image.append(boxes)

    return det_scores_per_image, det_masks_per_image, det_boxes_per_image


def coco_polygons_json_to_instance_masks(
    coco_json_path: str,
    *,
    image_stems: Sequence[str],
    image_shapes: Sequence[Tuple[int, int] | Tuple[int, int, int]],
    segmentation_index: int = 0,
    dtype: np.dtype = np.uint16,
) -> List[np.ndarray]:
    with open(coco_json_path, "r", encoding="utf-8") as handle:
        coco_data = json.load(handle)

    stem_to_id = {}
    for image in coco_data.get("images", []):
        stem = Path(image["file_name"]).stem
        stem_to_id[stem] = image["id"]

    id_to_annotations = {}
    for annotation in coco_data.get("annotations", []):
        image_id = annotation["image_id"]
        id_to_annotations.setdefault(image_id, []).append(annotation)

    masks: List[np.ndarray] = []
    for stem, shape in zip(image_stems, image_shapes):
        height, width = int(shape[0]), int(shape[1])
        mask_out = np.zeros((height, width), dtype=dtype)

        image_id = stem_to_id.get(stem)
        if image_id is None:
            masks.append(mask_out)
            continue

        annotations = id_to_annotations.get(image_id, [])
        inst_id = 1
        for annotation in annotations:
            seg = annotation.get("segmentation")
            if not seg:
                continue

            seg_flat = seg
            if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], (list, tuple)):
                if segmentation_index >= len(seg):
                    continue
                seg_flat = seg[segmentation_index]

            if not isinstance(seg_flat, (list, tuple)) or len(seg_flat) < 6:
                continue

            poly = np.asarray(seg_flat, dtype=float).reshape(-1, 2)
            rr, cc = sk_polygon(poly[:, 1], poly[:, 0], (height, width))
            mask_out[rr, cc] = inst_id
            inst_id += 1

        masks.append(mask_out)

    return masks