from cnt_project.evaluation.core.kaggle_dsb_metrics import (
    dsb_precision,
    dsb_precision_from_length_eval_dsb,
)
from cnt_project.evaluation.core.cldice import (
    cldice_object_f1,
)
import json
from typing import Any
import numpy as np
from pycocotools.coco import COCO
import scipy.optimize as opt

from cnt_project.evaluation.core.metrics_binary_masks import mask_iou, dice_coefficient
from cnt_project.coco.masks import extract_predictions_from_json, gt_polygons_to_mask

def evaluate_dsb_map(
    gt_json_path: str,
    pred_json_path: str,
    *,
    score_thresh: float = 0.0,
    iou_threshs: list[float] | None = None,
    cldice_threshs: list[float] | None = None,
    precision_impl: str = "dsb_precision",
) -> dict[str, Any]:
    """
    Core DSB evaluation on a full dataset.

    This matches the *behavior* of the legacy inline DSB evaluation:
      - decode predicted masks per image using extract_predictions_from_json
      - build GT masks using gt_polygons_to_mask
      - compute IoU matrix and run DSB precision over IoU thresholds
      - also compute mean Dice over Hungarian assignment

    Returns:
      Returns:
        {
            "DSB_mAP": float,
            "DSB_APs": list[float],
            "object_F1s": list[float],
            "IoU_thresholds": list[float],
            "mean_dice": float,
            "cldice_object_F1s": list[float],
            "cldice_thresholds": list[float],
        }

    precision_impl:
      - "dsb_precision" (default): returns TP/FP/FN diagnostics too (we only use mean_ap + precisions)
      - "dsb_precision_from_length_eval_dsb": alternative implementation
    """
    if iou_threshs is None:
        iou_threshs = np.round(np.arange(0.0, 1.0, 0.05), 2).tolist()

    if cldice_threshs is None:
        cldice_threshs = []

    cldice_threshs = [ float(threshold) for threshold in cldice_threshs ]

    for threshold in cldice_threshs:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "clDice thresholds must be between 0 and 1. "
                f"Got: {threshold}"
            )

    coco_gt = COCO(gt_json_path)
    img_ids = sorted(coco_gt.getImgIds())

    # predictions json might be "dict" (already loaded) in some runners; here we keep file-based API
    with open(pred_json_path, "r", encoding="utf-8") as f:
        pred_dataset_dict = json.load(f)

    # decode predictions per GT image order
    _scores, det_masks_per_image, _boxes = extract_predictions_from_json(
        score_thresh, coco_gt, pred_dataset_dict
    )

    dataset_dsb_map_sum = 0.0
    dataset_dsb_ap_sums = np.zeros(len(iou_threshs), dtype=float)
    dataset_object_f1_sums = np.zeros(len(iou_threshs), dtype=float)
    dataset_cldice_object_f1_sums = np.zeros( len(cldice_threshs), dtype=float, )

    dice_scores_all: list[float] = []

    n_eval = 0

    for img_idx, img_id in enumerate(img_ids):
        img_info = coco_gt.imgs[img_id]
        H, W = img_info["height"], img_info["width"]

        anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))

        pred_masks = det_masks_per_image[img_idx] if img_idx < len(det_masks_per_image) else []
        if pred_masks is None:
            pred_masks = []

        gt_masks = [gt_polygons_to_mask(ann["segmentation"], H, W) for ann in anns]

        # Optional clDice-based object F1.
        # For every requested clDice threshold:
        # 1. calculate pairwise instance clDice;
        # 2. perform threshold-aware Hungarian matching;
        # 3. calculate object F1 for this image;
        # 4. add it to the dataset accumulator.
        for threshold_index, cldice_threshold in enumerate(
            cldice_threshs
        ):
            cldice_result = cldice_object_f1(
                ground_truth_masks=gt_masks,
                prediction_masks=pred_masks,
                threshold=cldice_threshold,
            )

            dataset_cldice_object_f1_sums[
                threshold_index
            ] += cldice_result.object_f1

        # IoU matrix
        iou_matrix_size = max(len(gt_masks), len(pred_masks), 1)
        iou_matrix = np.zeros((iou_matrix_size, iou_matrix_size), dtype=float)

        for gi in range(len(gt_masks)):
            for pj in range(len(pred_masks)):
                iou_matrix[gi, pj] = mask_iou(pred_masks[pj], gt_masks[gi])

        # Precision over thresholds
        if precision_impl == "dsb_precision_from_length_eval_dsb":
            mean_ap, precisions = dsb_precision_from_length_eval_dsb(
                iou_matrix, len(gt_masks), len(pred_masks), iou_threshs
            )
            mean_ap = float(mean_ap)
            precisions_arr = np.asarray(precisions, dtype=float)
            object_f1_arr = np.full( len(iou_threshs), np.nan, dtype=float, )
        else:
            res = dsb_precision( iou_matrix, len(gt_masks), len(pred_masks), iou_threshs, )

            mean_ap = float(res.mean_ap)

            precisions_arr = np.asarray( res.precisions, dtype=float, )

            tp_arr = np.asarray( res.tp, dtype=float, )

            fp_arr = np.asarray( res.fp, dtype=float, )

            fn_arr = np.asarray( res.fn, dtype=float, )

            object_precision_arr = np.divide( tp_arr, tp_arr + fp_arr, out=np.zeros_like( tp_arr, dtype=float, ), where=(tp_arr + fp_arr) > 0, )

            object_recall_arr = np.divide( tp_arr, tp_arr + fn_arr, out=np.zeros_like( tp_arr, dtype=float, ), where=(tp_arr + fn_arr) > 0, )

            object_f1_arr = np.divide( 2.0 * object_precision_arr * object_recall_arr, object_precision_arr + object_recall_arr, out=np.zeros_like( object_precision_arr, dtype=float, ), where=( object_precision_arr + object_recall_arr ) > 0, )

        dataset_dsb_map_sum += mean_ap
        dataset_dsb_ap_sums += precisions_arr

        if not np.isnan( object_f1_arr ).all():
            dataset_object_f1_sums += object_f1_arr

        n_eval += 1

        # Dice on assignment (diagnostic)
        gt_assignment, pred_assignment = opt.linear_sum_assignment(1 - iou_matrix)
        for gt_ass, pred_ass in zip(gt_assignment, pred_assignment):
            if gt_ass < len(gt_masks) and pred_ass < len(pred_masks):
                dice_scores_all.append(dice_coefficient(pred_masks[pred_ass], gt_masks[gt_ass]))

    denom = max(n_eval, 1)
    dsb_map_avg = dataset_dsb_map_sum / denom
    dsb_ap_avg = (dataset_dsb_ap_sums / denom).tolist()
    dice_avg = float(np.mean(dice_scores_all)) if dice_scores_all else 0.0
    object_f1_avg = ( dataset_object_f1_sums / denom ).tolist()
    cldice_object_f1_avg = ( dataset_cldice_object_f1_sums / denom ).tolist()

    return {
        "DSB_mAP": float(dsb_map_avg),
        "DSB_APs": [ float(value) for value in dsb_ap_avg ],
        "object_F1s": [ float(value) for value in object_f1_avg ],
        "IoU_thresholds": [ float(value) for value in iou_threshs ],
        "mean_dice": float(dice_avg),
        "cldice_object_F1s": [ float(value) for value in cldice_object_f1_avg ],
        "cldice_thresholds": [ float(value) for value in cldice_threshs ],
    }

