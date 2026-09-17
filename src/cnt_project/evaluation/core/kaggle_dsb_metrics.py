from __future__ import annotations

import numpy as np
import scipy.optimize as opt
from dataclasses import dataclass


"""This is a full dataset-level evaluation pipeline for DSB-style instance segmentation metrics
How it evaluates
dsb_precision

This:

takes an IoU matrix between GT objects and predicted objects
uses Hungarian assignment (linear_sum_assignment)
at each IoU threshold, counts TP/FP/FN
computes DSB precision:
tp / (tp + fp + fn)
averages across thresholds
GT starts as COCO polygons
predictions start as COCO detections
both are converted into masks
evaluation is then done as instance-mask IoU matching
"""

@dataclass
class DSBPrecisionResult:
    mean_ap: float
    precisions: np.ndarray
    tp: list
    fp: list
    fn: list


def dsb_precision(result_matrix, gt_len, pred_len, thresholds):
    precisions = np.zeros(len(thresholds))
    tp_list, fp_list, fn_list = [], [], []

    for thresh_id, thresh in enumerate(thresholds):
        gt_assignment, pred_assignment = opt.linear_sum_assignment(1 - result_matrix)
        tp_count, fp_count, fn_count = 0, 0, 0

        for ass in range(len(gt_assignment)):
            fp_fn_found = False
            gt_ass = gt_assignment[ass]
            pred_ass = pred_assignment[ass]

            if gt_ass >= gt_len:
                fp_count += 1
                fp_fn_found = True
            if pred_ass >= pred_len:
                fn_count += 1
                fp_fn_found = True

            if not fp_fn_found:
                res_iou = result_matrix[gt_ass, pred_ass]
                if res_iou > thresh:
                    tp_count += 1
                else:
                    fp_count += 1
                    fn_count += 1

        tp_list.append(tp_count)
        fp_list.append(fp_count)
        fn_list.append(fn_count)
        precisions[thresh_id] = tp_count / (tp_count + fp_count + fn_count)

    mean_ap = precisions.mean()
    return DSBPrecisionResult(mean_ap, precisions, tp_list, fp_list, fn_list)


def dsb_precision_from_length_eval_dsb(result_matrix, gt_len, pred_len, thresholds):
    # https://www.kaggle.com/competitions/data-science-bowl-2018/overview/evaluation
    # StarDist and WormSwin

    precisions = np.zeros(len(thresholds))

    for thresh_id, thresh in enumerate(thresholds):
        gt_assignment, pred_assignment = opt.linear_sum_assignment(1 - result_matrix)

        tp_count, fp_count, fn_count = 0, 0, 0

        for ass in range(len(gt_assignment)):
            fp_fn_found = False

            gt_ass = gt_assignment[ass]
            pred_ass = pred_assignment[ass]

            if gt_ass >= gt_len:
                fp_count += 1
                fp_fn_found = True
            if pred_ass >= pred_len:
                fn_count += 1
                fp_fn_found = True

            if not fp_fn_found:
                res_iou = result_matrix[gt_ass, pred_ass]
                if res_iou > thresh:
                    tp_count += 1
                else:
                    fp_count += 1
                    fn_count += 1

        precisions[thresh_id] = tp_count / (tp_count + fp_count + fn_count)
    
    return ((1 / len(thresholds)) * precisions.sum(), precisions)

