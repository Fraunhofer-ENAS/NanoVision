import numpy as np
"""This is a small low-level mask-metric utility module. It works directly on binary masks 
and computes only pairwise overlap metrics.
Object level evaluation metrics"""

def mask_iou(pred_mask, gt_mask):
    intersection = np.sum((pred_mask + gt_mask) > 1)
    union = np.sum((pred_mask + gt_mask) > 0)
    return intersection / float(union) if union > 0 else 0

def dice_coefficient(pred_mask, gt_mask):
    intersection = np.sum((pred_mask == 1) & (gt_mask == 1))
    total = np.sum(pred_mask == 1) + np.sum(gt_mask == 1)
    return 2 * intersection / total if total > 0 else 1.0

def dice_coefficient_from_length_eval_dsb(pred_mask, gt_mask):
    intersection = np.sum((pred_mask + gt_mask) > 1)
    pred_area = np.sum(pred_mask)
    gt_area = np.sum(gt_mask)
    return 2 * intersection / (pred_area + gt_area) if (pred_area + gt_area) > 0 else 0
