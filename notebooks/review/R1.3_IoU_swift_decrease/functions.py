"""
functions.py

Consolidated module for the NanoVision R2#2 rebuttal (overlap_score vs. rotation
mismatch) and the related DSB-Hungarian matching / orientation-mismatch
diagnostics. Combines three previously separate pieces so they can be
imported as one unit from the analysis notebook.

Sections:
  1. overlap_score vs. rotation-angle analysis (empirical real-mask curve + rod-model
     bound), including run_rotation_sweep and plot_results.
  2. DSB-style precision/AP via OPTIMAL (Hungarian) one-to-one matching,
     with interchangeable IoU / clDice scoring.
  3. Real GT<->prediction orientation-mismatch measurement, using the same
     Hungarian matcher as (2), plus a debug plot of matched mask pairs.

Import from the notebook as:
    import functions as fn
    fn.debug_flag = False   # module-attribute assignment -- see note below
    df = fn.run_rotation_sweep(gt_path, max_instances=1000)

Note on debug_flag: several functions below check the module-level
`debug_flag` to optionally render intermediate debug figures. Because
these functions look up `debug_flag` in THIS module's global namespace
(not the caller's), toggling it from the notebook requires
`fn.debug_flag = False` (a module-attribute assignment) -- a bare
`debug_flag = False` typed in the notebook after `from functions import *`
creates a separate notebook-local variable and will NOT reach the
functions defined here.
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

import cldice 
import importlib


importlib.reload(cldice)
from cldice import clDice

# Project-specific import (kept as in the original notebook cell).
# Adjust the relative path if functions.py is not colocated with the
# original notebook, since sys.path.append is relative to the current
# working directory at import time, not to this file's location.
sys.path.append("..\\..\\..\\src")
from cnt_project.features.core.geodesic_length import polygon_mask_geodesic_length

# Module-level debug toggle, checked inside the functions below. See the
# module docstring note above for how to set this from a notebook.
debug_flag = False
MICRONS_PER_PIXEL = 5.0 / 256.0

# ===========================================================================
# Section 1: overlap_score vs. rotation-angle analysis (Reviewer 2, comment 2)
# ===========================================================================
#
# Replaces the schematic ("5 deg -> overlap_score=0.05") with:
#   (A) an empirical curve: real GT instance masks rotated by a swept angle
#       about their own centroid, overlap_score(original, rotated) measured directly.
#   (B) an idealized rod-model curve, computed per-instance from each
#       instance's own (length, width) -- not one fixed (L, w) like the
#       cartoon -- aggregated the same way, shown as a bound.
#   (C, optional) real GT<->prediction matched pairs: observed orientation
#       mismatch vs observed IoU, scattered on top of (A)/(B) as validation
#       (see Section 3, match_gt_pred_observed_points).

# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

def ann_to_mask(coco: COCO, ann: dict) -> np.ndarray:
    """Full-image binary mask (H, W) uint8 for one COCO annotation."""
    img_info = coco.imgs[ann["image_id"]]
    h, w = img_info["height"], img_info["width"]
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(seg["counts"], list):
        rle = mask_utils.frPyObjects(seg, h, w)
    else:
        rle = seg
    return mask_utils.decode(rle).astype(np.uint8)


def crop_to_bbox(mask: np.ndarray, margin: int = 5):
    """Crop a full-image mask to its bounding box + margin. Returns crop and offset."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, None
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, mask.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, mask.shape[1])
    return mask[y0:y1, x0:x1], (y0, x0)


def estimate_mean_width(mask):
    mask = mask.astype(bool)
    skeleton = skeletonize(mask)
    distance = distance_transform_edt(mask)
    widths = 2 * distance[skeleton]
    return widths.mean(), widths


def supersampled_rotate(mask: np.ndarray, theta_deg: float, upsample: int = 8) -> np.ndarray:
    """
    Rotate a small (cropped) binary mask about its own centroid with
    sub-pixel accuracy: upsample -> rotate (linear) -> threshold ->
    downsample (area) -> threshold. Needed because these masks are only
    a few pixels wide, so naive nearest-neighbour rotation on the raw
    raster is dominated by aliasing.
    """
    h, w = mask.shape
    big = cv2.resize(mask.astype(np.float32), (w * upsample, h * upsample),
                      interpolation=cv2.INTER_NEAREST)
    M = cv2.moments(mask.astype(np.uint8))
    if M["m00"] == 0:
        return np.zeros_like(mask)
    cx, cy = (M["m10"] / M["m00"]) * upsample, (M["m01"] / M["m00"]) * upsample
    rot_mat = cv2.getRotationMatrix2D((cx, cy), theta_deg, 1.0)
    rotated_big = cv2.warpAffine(big, rot_mat, (w * upsample, h * upsample),
                                  flags=cv2.INTER_LINEAR, borderValue=0)
    rotated_big_bin = (rotated_big > 0.5).astype(np.float32)
    rotated = cv2.resize(rotated_big_bin, (w, h), interpolation=cv2.INTER_AREA)

    if debug_flag:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        ax[0].imshow(mask, cmap="gray")
        ax[0].set_title("Original mask")
        ax[0].axis("off")

        ax[1].imshow(rotated, cmap="gray")
        ax[1].set_title(f"Rotated: {theta_deg}\u00b0")
        ax[1].axis("off")

        plt.tight_layout()
        plt.show()

    return (rotated > 0.5).astype(np.uint8)


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return inter / union if union > 0 else 0.0


def orientation_deg(mask: np.ndarray) -> float:
    """Principal-axis angle (deg, mod 180) via PCA on foreground pixel coords."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 2:
        return np.nan
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    pts -= pts.mean(axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_vec = eigvecs[:, np.argmax(eigvals)]
    ang = np.degrees(np.arctan2(main_vec[1], main_vec[0])) % 180
    return ang


def angle_diff_mod180(a: float, b: float) -> float:
    d = abs(a - b) % 180
    return min(d, 180 - d)


# ---------------------------------------------------------------------------
# Rod model (idealized bound), computed per-instance (L, w), not one fixed value
# ---------------------------------------------------------------------------

def rod_model(length: float, width: float, theta_deg: float, scoring_method: str = "iou", upsample: int = 4) -> float:
    """overlap_score of a straight L x w rectangle vs. itself rotated by theta about its centre,
    computed by direct high-resolution rasterization (numerically 'carrying the
    integral through' rather than assuming a closed-form)."""
    pad = 1.6  # canvas margin so the rotated rod never clips
    canvas = int(np.ceil(length * pad))
    hi = canvas * upsample
    base = np.zeros((hi, hi), dtype=np.uint8)
    cx = cy = hi // 2
    half_l, half_w = (length / 2) * upsample, (width / 2) * upsample
    cv2.rectangle(
        base,
        (int(cx - half_l), int(cy - half_w)),
        (int(cx + half_l), int(cy + half_w)),
        1, thickness=-1,
    )
    rot_mat = cv2.getRotationMatrix2D((cx, cy), theta_deg, 1.0)
    rotated = cv2.warpAffine(base.astype(np.float32), rot_mat, (hi, hi),
                              flags=cv2.INTER_LINEAR, borderValue=0)
    rotated = (rotated > 0.5).astype(np.uint8)

    if scoring_method == "iou":
        score = iou(base, rotated)
    elif scoring_method == "clDice":
        score = clDice(base, rotated)
    else:
        raise ValueError(f"Unknown scoring method: {scoring_method}")

    if debug_flag:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        ax[0].imshow(base, cmap="gray")
        ax[0].set_title(f"Rod\nL={length:.1f}, W={width:.1f}")
        ax[0].axis("off")

        ax[1].imshow(rotated, cmap="gray")
        ax[1].set_title(f"Rotated {theta_deg}\u00b0")
        ax[1].axis("off")

        ax[2].imshow(base, cmap="gray", alpha=0.5)
        ax[2].imshow(rotated, cmap="Reds", alpha=0.5)
        ax[2].set_title(f"Overlay\n{scoring_method} = {score:.3f}")
        ax[2].axis("off")

        plt.tight_layout()
        plt.show()
    return score


# ---------------------------------------------------------------------------
# Main sweep: Experiment A + B (real masks + rod-model bound)
# ---------------------------------------------------------------------------

def run_rotation_sweep(gt_json_path: str, thetas=np.arange(0, 21, 1), scoring_method: str = "iou", max_instances=None, debug_flag=False):
    coco = COCO(gt_json_path)
    ann_ids = coco.getAnnIds()
    if max_instances is not None:
        ann_ids = ann_ids[:max_instances]

    rows = []
    for ann_id in ann_ids:
        ann = coco.anns[ann_id]
        full_mask = ann_to_mask(coco, ann)
        cropped, _ = crop_to_bbox(full_mask, margin=5)
        if cropped is None or cropped.sum() < 5:
            continue

        
        length, length_um = polygon_mask_geodesic_length(cropped, microns_per_pixel=MICRONS_PER_PIXEL)
        mean_width, widths = estimate_mean_width(cropped)
        if debug_flag:
            print(f"Instance {ann_id}: length={length:.2f} (lw_um={length_um:.2f}), width={mean_width:.2f} (width_um={mean_width * MICRONS_PER_PIXEL:.2f})")
            if debug_flag:

                length_px, length_um, path, debug = polygon_mask_geodesic_length(
                    full_mask,
                    microns_per_pixel=1.0,
                    return_path=True,
                    return_debug=True
                )

                skeleton = skeletonize(cropped)
                distance = distance_transform_edt(cropped)

                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(full_mask, cmap="gray")
                ax[0].plot(path[:, 1], path[:, 0], "r-", linewidth=2)
                ax[0].scatter(
                    [path[0, 1], path[-1, 1]],
                    [path[0, 0], path[-1, 0]],
                    c="blue"
                )
                ax[1].imshow(cropped, cmap="gray")
                ax[1].scatter(
                    np.where(skeleton)[1],
                    np.where(skeleton)[0],
                    c=2 * distance[skeleton],
                    s=5,
                    cmap="viridis"
                )
                fig.colorbar(ax[1].collections[0], ax=ax[1], label="Local width (px)")

                ax[0].axis("equal")
                ax[0].set_title(f"Length = {length_px:.1f} px ({length_um:.2f} \u00b5m)")
                ax[1].axis("equal")
                ax[1].set_title(f"Width = {width:.1f} px ({width * MICRONS_PER_PIXEL:.2f} \u00b5m)")
                plt.show()

        if length is None or mean_width is None:
            continue
        width = mean_width

        for theta in thetas:
            # empirical: rotate the real (possibly curved/irregular) mask both directions
            overlap_scores_signed = []
            for sign in (1, -1) if theta != 0 else (1,):

                rotated = supersampled_rotate(cropped, sign * theta)

                if scoring_method == "iou":
                    overlap_scores_signed.append(iou(cropped, rotated))
                elif scoring_method == "clDice":
                    overlap_scores_signed.append(clDice(cropped, rotated))
                else:
                    raise ValueError(f"Unsupported scoring method: {scoring_method}")
                
                empirical_overlap_score = float(np.mean(overlap_scores_signed))

                # idealized bound: straight rod with this instance's own (L, w)
                bound_overlap_score = rod_model(length, width, theta, scoring_method=scoring_method)
                rows.append({
                    "ann_id": ann_id, "theta": theta,
                    "length": length, "width": width, "aspect_ratio": length / width,
                    "empirical_overlap_score": empirical_overlap_score, "rod_model": bound_overlap_score,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot -- ADAPTED per request: now actually saves to out_path (previously
# only plt.show() ran; fig.savefig was present but commented out, so the
# two distinct filenames passed by the notebook's plot_results calls were
# never actually written to disk).
# ---------------------------------------------------------------------------

def plot_results(df: pd.DataFrame, observed_df: "pd.DataFrame | None", scoring_method: str, out_path: str, figsize=(7, 5)):
    agg = df.groupby("theta").agg(
        emp_mean=("empirical_overlap_score", "mean"),
        emp_p10=("empirical_overlap_score", lambda x: np.percentile(x, 10)),
        emp_p90=("empirical_overlap_score", lambda x: np.percentile(x, 90)),
        rod_mean=("rod_model", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=figsize)

    # a handful of individual real-instance curves, lightly, for texture
    # for ann_id, g in df.groupby("ann_id"):
    #     ax.plot(g["theta"], g["empirical_overlap_score"], color="steelblue", alpha=0.15, linewidth=1)

        
    ax.fill_between(agg["theta"], agg["emp_p10"], agg["emp_p90"],
                     color="steelblue", alpha=0.25, label="Real GT masks (10-90th pct)")
    if observed_df is not None and len(observed_df):
        norm = plt.Normalize(
            observed_df["observed_length_diff"].min(),
            observed_df["observed_length_diff"].max()
        )
        cmap = plt.cm.Blues

        ax.scatter(
            observed_df["observed_theta_diff"],
            observed_df["observed_overlap_score"],
            s=14,
            color=cmap(norm(observed_df["observed_length_diff"])),
            alpha=0.7,
            label="Observed GT<->prediction matches"
        )

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(r"$\Delta length$ (um)")

    ax.plot(agg["theta"], agg["emp_mean"], color="steelblue", linewidth=2.5, alpha=1.0,
             label="Real GT masks (mean)")
    ax.plot(agg["theta"], agg["rod_mean"], color="firebrick", linewidth=2,
             linestyle="--", label="Idealized rod model (per-instance L, w)")



    ax.set_xlabel(r"$\Delta\theta$ (um)")
    ax.set_ylabel("Overlap score")
    ax.set_ylim(0, 1.02)
    ax.set_title(r"{scoring_method} vs. \theta mismatch, actual dataset aspect-ratio distribution")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")
    plt.show()


# ===========================================================================
# Section 2: DSB-style precision/AP via OPTIMAL (Hungarian) matching
# ===========================================================================
#
# Two interchangeable scoring functions are provided for the match matrix:
#   - mask_iou     (existing metric)
#   - mask_clDice  (Shit et al., CVPR 2021) -- per Reviewer 2's request
# Both feed into the same generic dsb_precision_from_scores, so switching
# metrics is a one-line change at the call site.
#
# Why Hungarian instead of greedy: a greedy "assign each GT to its best-IoU
# pred" approach is order-dependent and can leave a pred unmatched to a GT
# it overlaps more with, if that pred was already grabbed by another GT
# with a slightly higher IoU. The Hungarian algorithm finds the assignment
# that maximizes total score across the whole image in one shot, which is
# the correct notion of "best" matching and removes that order-dependence
# -- relevant here since merged fragments can create ambiguous overlaps.

def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return inter / union if union > 0 else 0.0


def _skeleton(mask: np.ndarray) -> np.ndarray:
    return skeletonize(mask.astype(bool))





def build_iou_matrix(gt_masks: list, pred_masks: list) -> np.ndarray:
    n_gt, n_pred = len(gt_masks), len(pred_masks)
    matrix = np.zeros((n_gt, n_pred), dtype=float)
    for gi in range(n_gt):
        for pj in range(n_pred):
            matrix[gi, pj] = mask_iou(gt_masks[gi], pred_masks[pj])
    return matrix


def build_clDice_matrix(gt_masks: list, pred_masks: list) -> np.ndarray:
    n_gt, n_pred = len(gt_masks), len(pred_masks)
    matrix = np.zeros((n_gt, n_pred), dtype=float)
    for gi in range(n_gt):
        for pj in range(n_pred):
            matrix[gi, pj] = clDice(pred_masks[pj], gt_masks[gi])
    return matrix


def hungarian_match(score_matrix: np.ndarray):
    """
    Optimal one-to-one assignment maximizing total score.
    score_matrix: (n_gt, n_pred), any non-negative similarity (IoU, clDice, ...).
    Returns: list of (gt_idx, pred_idx, score) for the matched pairs.
             Unmatched GT/pred (when n_gt != n_pred) are simply absent
             from the returned list.
    """
    n_gt, n_pred = score_matrix.shape
    if n_gt == 0 or n_pred == 0:
        return []
    cost = -score_matrix  # linear_sum_assignment minimizes cost
    row_idx, col_idx = linear_sum_assignment(cost)
    return [(int(r), int(c), float(score_matrix[r, c])) for r, c in zip(row_idx, col_idx)]


def dsb_precision_from_scores(score_matrix: np.ndarray, n_gt: int, n_pred: int,
                               thresholds: list) -> dict:
    """
    DSB-style AP sweep, using ONE global optimal matching (Hungarian),
    then applying each threshold to that fixed matching.

    Note: matching is computed once (order/threshold independent);
    only the TP/FP/FN counts change per threshold, by keeping only
    matched pairs whose score clears that threshold.
    """
    matches = hungarian_match(score_matrix)  # [(gt_idx, pred_idx, score), ...]
    #match_scores = np.array([m[2] for m in matches]) if matches else np.array([])

    return {
        "thresholds": list(thresholds),
        "matches": matches,
    }

def dsb_matching(gt_masks, pred_masks, overlap_threshs, scoring_method):
    if scoring_method=='iou':
        score_matrix = build_iou_matrix(gt_masks, pred_masks)
        return dsb_precision_from_scores(score_matrix, len(gt_masks), len(pred_masks), overlap_threshs)
    elif scoring_method=='clDice':
        score_matrix = build_clDice_matrix(gt_masks, pred_masks)
        return dsb_precision_from_scores(score_matrix, len(gt_masks), len(pred_masks), overlap_threshs)
    else:
        raise ValueError(f"Unknown scoring method: {scoring_method}")   


# ===========================================================================
# Section 3: Real GT<->prediction orientation-mismatch measurement
# ===========================================================================
#
# Uses the SAME Hungarian matcher (dsb_matching_iou_hungarian, Section 2)
# as the official metric pipeline -- so this diagnostic can't silently
# disagree with how matches are defined elsewhere in the paper.
#
# Produces:
#   - a DataFrame of (image_id, observed_theta, observed_overlap_score) for every
#     matched pair that clears thresh_gate
#   - an optional debug figure showing the actual matched GT/pred mask
#     pairs (e.g. the worst orientation mismatches), so the numbers can be
#     visually sanity-checked against real masks rather than trusted blind.

def _decode_pred_segmentation(seg, h, w):
    if isinstance(seg, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
    elif isinstance(seg["counts"], list):
        rle = mask_utils.frPyObjects(seg, h, w)
    else:
        rle = seg
    return mask_utils.decode(rle).astype(np.uint8)


def get_image_id_for_filename(coco: COCO, filename: str) -> int:
    """Return the COCO image_id corresponding to a file_name."""
    for image_id, image_info in coco.imgs.items():
        if image_info.get("file_name") == filename:
            return image_id
    raise ValueError(f"Filename not found in COCO file: {filename}")


def get_annotations_for_filename(coco: COCO, filename: str) -> list[dict]:
    """Return all annotations belonging to filename."""
    image_id = get_image_id_for_filename(coco, filename)
    ann_ids = coco.getAnnIds(imgIds=image_id)
    return coco.loadAnns(ann_ids)


def match_gt_pred_for_filename(
    gt_coco: COCO,
    pred_coco: COCO,
    filename: str,
    scoring_method: str = "iou",
    thresh_gate: float = 0.01,
    debug_plot: bool = True,
):
    """
    Debug the Hungarian GT<->prediction matching for ONE selected filename.

    GT and prediction image_ids are looked up independently from file_name,
    so the two COCO files do not need to share image_id values.

    Returns:
        result: DSB-style matching result containing the IoU matrix and matches.
    """
    gt_image_id = get_image_id_for_filename(gt_coco, filename)
    pred_image_id = get_image_id_for_filename(pred_coco, filename)

    gt_anns = get_annotations_for_filename(gt_coco, filename)
    pred_anns = get_annotations_for_filename(pred_coco, filename)

    gt_masks = [ann_to_mask(gt_coco, ann) for ann in gt_anns]
    pred_masks = [ann_to_mask(pred_coco, ann) for ann in pred_anns]

    print(f"Filename: {filename}")
    print(f"GT image_id:   {gt_image_id}")
    print(f"Pred image_id: {pred_image_id}")
    print(f"GT instances:   {len(gt_masks)}")
    print(f"Pred instances: {len(pred_masks)}")

    if not gt_masks or not pred_masks:
        print("No GT or prediction masks found.")
        return None

    if scoring_method == "iou":
        score_matrix = build_iou_matrix(gt_masks, pred_masks)
    elif scoring_method == "clDice":
        score_matrix = build_clDice_matrix(gt_masks, pred_masks)
    else:
        raise ValueError(f"Unsupported scoring method: {scoring_method}")

    matches = hungarian_match(score_matrix)

    print(f"\n{scoring_method.upper()} matrix:")
    print(np.round(score_matrix, 3))

    print("\nHungarian matches:")
    for gt_idx, pred_idx, score in matches:
        print(f"GT {gt_idx} -> Pred {pred_idx}: {scoring_method} = {score:.3f}")

    if debug_plot:
        _plot_single_image_matches(
            gt_masks,
            pred_masks,
            matches,
            filename,
            scoring_method
        )

    return {
        "filename": filename,
        "gt_image_id": gt_image_id,
        "pred_image_id": pred_image_id,
        "gt_anns": gt_anns,
        "pred_anns": pred_anns,
        "gt_masks": gt_masks,
        "pred_masks": pred_masks,
        "overlap_score_matrix": score_matrix,
        "matches": matches,
    }


def match_gt_pred_observed_points(
    gt_json_path: str,
    pred_json_path: str,
    thresh_gate: float = 0.01,
    scoring_method: str = "iou",
    debug_plot_path: str = None,
    debug_n_examples: int = 8,
    debug_sort_by: str = "observed_theta_diff",
):
    """
    Match GT to predictions by FILENAME, not by shared image_id.

    Both JSON files are loaded as pycocotools COCO objects. For each filename
    present in GT, the corresponding image_id is independently found in GT
    and prediction COCO files before retrieving annotations.

    This is important because independently generated COCO files may use
    different image_id values for the same filename.
    """
    gt_coco = COCO(gt_json_path)
    pred_coco = COCO(pred_json_path)

    points = []
    debug_crops = []

    # Match images by filename.
    gt_files = {
        image_info["file_name"]: image_id
        for image_id, image_info in gt_coco.imgs.items()
    }
    pred_files = {
        image_info["file_name"]: image_id
        for image_id, image_info in pred_coco.imgs.items()
    }

    common_filenames = sorted(set(gt_files) & set(pred_files))
    #common_filenames = common_filenames[0:2]
    for filename in tqdm(common_filenames, desc="Processing"):
        gt_image_id = gt_files[filename]
        pred_image_id = pred_files[filename]

        gt_ann_ids = gt_coco.getAnnIds(imgIds=gt_image_id)
        pred_ann_ids = pred_coco.getAnnIds(imgIds=pred_image_id)

        gt_masks = [ann_to_mask(gt_coco, gt_coco.anns[aid]) for aid in gt_ann_ids]
        pred_masks = [ann_to_mask(pred_coco, pred_coco.anns[aid]) for aid in pred_ann_ids]

        if not gt_masks or not pred_masks:
            continue

        result = dsb_matching(gt_masks, pred_masks, overlap_threshs=[0.5], scoring_method=scoring_method)
        #result['matches'] = result['matches'][0:8]
        for gt_idx, pred_idx, score in tqdm(result["matches"], desc="Matching"):
            if score < thresh_gate:
                continue

            gt_mask = gt_masks[gt_idx]
            pred_mask = pred_masks[pred_idx]

            gt_ori = orientation_deg(gt_mask)
            pred_ori = orientation_deg(pred_mask)
            cropped_gt, _ = crop_to_bbox(gt_mask, margin=5)
            cropped_pred, _ = crop_to_bbox(pred_mask, margin=5)
            _, gt_length_um = polygon_mask_geodesic_length(cropped_gt, microns_per_pixel=MICRONS_PER_PIXEL)
            _, pred_length_um = polygon_mask_geodesic_length(cropped_pred, microns_per_pixel=MICRONS_PER_PIXEL)

            if np.isnan(gt_ori) or np.isnan(pred_ori):
                continue

            observed_theta_diff = angle_diff_mod180(gt_ori, pred_ori)
            observed_length_diff = abs(gt_length_um - pred_length_um)

            points.append({
                "filename": filename,
                "gt_image_id": gt_image_id,
                "pred_image_id": pred_image_id,
                "gt_ann_id": gt_ann_ids[gt_idx],
                "pred_ann_id": pred_ann_ids[pred_idx],
                "gt_ori": gt_ori,
                "pred_ori": pred_ori,
                "gt_length": gt_length_um,
                "pred_length": pred_length_um,
                "observed_theta_diff": observed_theta_diff,
                "observed_length_diff": observed_length_diff,
                "observed_overlap_score": score,
            })

            if debug_plot_path is not None:
                if score < 0.5:
                    if observed_theta_diff <20:
                        if observed_length_diff < 0.5:
                        
                            debug_crops.append({
                                "filename": filename,
                                "gt_image_id": gt_image_id,
                                "pred_image_id": pred_image_id,
                                "gt_ann_id": gt_ann_ids[gt_idx],
                                "pred_ann_id": pred_ann_ids[pred_idx],
                                "gt_mask": gt_mask,
                                "pred_mask": pred_mask,
                                "gt_ori": gt_ori,
                                "pred_ori": pred_ori,
                                "gt_length": gt_length_um,
                                "pred_length": pred_length_um,
                                "observed_theta_diff": observed_theta_diff,
                                "observed_length_diff": observed_length_diff,
                                "observed_overlap_score": score,
                            })

    df = pd.DataFrame(points)

    if debug_plot_path is not None and debug_crops:
        _plot_matched_examples(
            debug_crops,
            debug_plot_path,
            debug_n_examples,
            debug_sort_by,
            scoring_method,
        )

    return df


def _plot_single_image_matches(gt_masks, pred_masks, matches, filename, scoring_method):
    """Visualize GT/PRED indices and every Hungarian matched pair for one image."""

    # First: all instances with their indices.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(np.max(np.stack(gt_masks), axis=0), cmap="gray")
    for i, mask in enumerate(gt_masks):
        ys, xs = np.where(mask)
        if len(xs):
            axes[0].text(xs.mean(), ys.mean(), str(i), color="red", fontsize=14)
    axes[0].set_title("GT instances")
    axes[0].axis("off")

    axes[1].imshow(np.max(np.stack(pred_masks), axis=0), cmap="gray")
    for i, mask in enumerate(pred_masks):
        ys, xs = np.where(mask)
        if len(xs):
            axes[1].text(xs.mean(), ys.mean(), str(i), color="red", fontsize=14)
    axes[1].set_title("Prediction instances")
    axes[1].axis("off")

    fig.suptitle(filename)
    plt.tight_layout()
    plt.show()

    # Then: one visualization per Hungarian pair.
    for gt_idx, pred_idx, score in matches:
        fig, ax = plt.subplots(figsize=(6, 6))

        gt_mask = gt_masks[gt_idx]
        pred_mask = pred_masks[pred_idx]

        rgb = np.ones((*gt_mask.shape, 3))
        gt_only = np.logical_and(gt_mask, ~pred_mask)
        pred_only = np.logical_and(pred_mask, ~gt_mask)
        both = np.logical_and(gt_mask, pred_mask)

        rgb[gt_only] = [0.25, 0.45, 0.85]
        rgb[pred_only] = [0.90, 0.45, 0.15]
        rgb[both] = [0.55, 0.25, 0.65]

        union = (gt_mask | pred_mask).astype(np.uint8)
        cropped, (y0, x0) = crop_to_bbox(union, margin=10)

        ax.imshow(
            rgb[y0:y0 + cropped.shape[0], x0:x0 + cropped.shape[1]],
            origin="upper",
        )

        ax.set_title(
            f"GT {gt_idx} -> Pred {pred_idx}\n"
            f"{scoring_method} = {score:.3f}"
        )
        ax.axis("off")
        plt.tight_layout()
        plt.show()


def _orientation_line(mask, ori_deg, length_frac=0.4):
    """Endpoints of a short line through the mask centroid at angle ori_deg,
    for overlaying the measured orientation on the debug plot."""
    ys, xs = np.where(mask > 0)
    cx, cy = xs.mean(), ys.mean()
    diag = np.hypot(mask.shape[0], mask.shape[1])
    half_len = diag * length_frac / 2
    rad = np.radians(ori_deg)
    dx, dy = np.cos(rad) * half_len, np.sin(rad) * half_len
    return (cx - dx, cx + dx), (cy - dy, cy + dy)


def _plot_matched_examples(debug_crops, out_path, n_examples, sort_by, scoring_method, figsize=(6.0, 4.0)):
    ascending = sort_by == "observed_overlap_score"  # worst overlap_score = smallest first; worst theta = largest first
    debug_crops = sorted(debug_crops, key=lambda d: d[sort_by], reverse=not ascending)
    if len(debug_crops)<n_examples:
        selected = debug_crops[:max(len(debug_crops), n_examples)]
    else:
        selected = debug_crops[:n_examples]

    n_cols = min(4, len(selected))
    n_rows = int(np.ceil(len(selected) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows), squeeze=False)

    for idx, d in enumerate(selected):
        ax = axes[idx // n_cols][idx % n_cols]

        gt_mask, pred_mask = d["gt_mask"], d["pred_mask"]
        union_mask = np.logical_or(gt_mask, pred_mask).astype(np.uint8)
        cropped, (y0, x0) = crop_to_bbox(union_mask, margin=10)
        y1, x1 = y0 + cropped.shape[0], x0 + cropped.shape[1]
        gt_crop = gt_mask[y0:y1, x0:x1]
        pred_crop = pred_mask[y0:y1, x0:x1]

        # RGB composite: GT-only in blue, pred-only in orange/red, overlap in purple
        h, w = gt_crop.shape
        rgb = np.ones((h, w, 3))
        gt_only = np.logical_and(gt_crop, np.logical_not(pred_crop))
        pred_only = np.logical_and(pred_crop, np.logical_not(gt_crop))
        both = np.logical_and(gt_crop, pred_crop)
        rgb[gt_only] = [0.25, 0.45, 0.85]    # blue = GT only
        rgb[pred_only] = [0.90, 0.45, 0.15]  # orange = pred only
        rgb[both] = [0.55, 0.25, 0.65]       # purple = overlap

        ax.imshow(rgb, origin="upper")

        # overlay measured orientation lines, in local crop coordinates
        for mask_full, ori, color in [(gt_mask, d["gt_ori"], "navy"),
                                       (pred_mask, d["pred_ori"], "darkred")]:
            (x_a, x_b), (y_a, y_b) = _orientation_line(mask_full[y0:y1, x0:x1], ori)
            ax.plot([x_a, x_b], [y_a, y_b], color=color, linewidth=1.5, linestyle=":")

        ax.set_title(
            f"fn: {d['filename'][:-4]} ann {d['gt_ann_id']}\n"
            f"{scoring_method}={d['observed_overlap_score']:.2f}  |  \u0394\u03b8={d['observed_theta_diff']:.1f}\u00b0"  +  f"  |  \u0394L={d['observed_length_diff']:.2f}",
            fontsize=10,
        )
        ax.set_xticks([]); ax.set_yticks([])

    # blank out any unused subplot slots
    for idx in range(len(selected), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    handles = [
        plt.Line2D([0], [0], color=[0.25, 0.45, 0.85], lw=6, label="GT only"),
        plt.Line2D([0], [0], color=[0.90, 0.45, 0.15], lw=6, label="Pred only"),
        plt.Line2D([0], [0], color=[0.55, 0.25, 0.65], lw=6, label="Overlap"),
        plt.Line2D([0], [0], color="navy", lw=1.5, linestyle=":", label="GT orientation"),
        plt.Line2D([0], [0], color="darkred", lw=1.5, linestyle=":", label="Pred orientation"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Matched GT/pred pairs, sorted by {sort_by}", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved debug plot to {out_path}")
