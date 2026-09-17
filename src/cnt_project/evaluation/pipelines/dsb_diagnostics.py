from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize as opt
from pycocotools.coco import COCO

from cnt_project.coco.masks import extract_predictions_from_json, gt_polygons_to_mask
from cnt_project.evaluation.core.kaggle_dsb_metrics import dsb_precision
from cnt_project.evaluation.core.metrics_binary_masks import dice_coefficient, mask_iou
from cnt_project.io.paths import OutputPaths, ProjectPaths
from cnt_project.evaluation.utils import safe_progress_iter

def run_dsb_full_dataset_diagnostics(
    *,
    run_name: str,
    pred_run: str,
    pred_filename: str = "predicted_annotations_poly.json",
    gt_json_path: str | Path | None = None,
    eval_tag: str = "diagnostics",
    global_outputs_root: str | Path | None = None,
) -> None:
    # ------------------------------------------------------------------
    # Resolve output structure
    # ------------------------------------------------------------------

    if global_outputs_root is not None:
        output_paths = OutputPaths.from_root(
            global_outputs_root
        )
        project_paths = None
    else:
        project_paths = ProjectPaths.from_here(__file__)
        output_paths = project_paths.output_paths

    output_paths.ensure()


    # ------------------------------------------------------------------
    # Resolve evaluation output directory
    # ------------------------------------------------------------------

    out_dir = output_paths.eval_subdir( run_name, "dsb", eval_tag, )


    # ------------------------------------------------------------------
    # Resolve prediction JSON
    # ------------------------------------------------------------------

    pred_json_path = output_paths.predicted_poly_json( pred_run, filename=pred_filename, )


    # ------------------------------------------------------------------
    # Resolve ground-truth JSON
    # ------------------------------------------------------------------

    if gt_json_path is None:
        if project_paths is None:
            raise ValueError(
                "--gt-json-path is required when using a custom "
                "global outputs root."
            )

        gt_json_path = project_paths.test_coco_json

    else:
        gt_json_path = Path( gt_json_path ).resolve()

    if not pred_json_path.exists():
        raise FileNotFoundError(
            "Prediction JSON does not exist: "
            f"{pred_json_path}"
        )

    if not Path(gt_json_path).exists():
        raise FileNotFoundError(
            "Ground-truth COCO JSON does not exist: "
            f"{gt_json_path}"
        )
    print(f"Prediction JSON: {pred_json_path}")
    print(f"Ground-truth JSON: {gt_json_path}")
    print(f"Evaluation outputs: {out_dir}")

    score_thresh = 0.0
    # iou_threshs = np.round(np.arange(0.0, 0.95 + 0.05, 0.05), 2).tolist()
    iou_threshs = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

    coco_gt = COCO(gt_json_path)
    img_ids = coco_gt.getImgIds()

    with open(pred_json_path, "r", encoding="utf-8") as f:
        coco_dt = json.load(f)

    _, det_masks_per_image, _ = extract_predictions_from_json(score_thresh, coco_gt, coco_dt)

    dataset_dsb_map_sum = 0.0
    dataset_dsb_ap_sums = np.zeros(len(iou_threshs), dtype=float)
    detailed_rows = []
    n_eval = 0

    progress_iter = safe_progress_iter(len(det_masks_per_image))
    for img_count, bar in enumerate(progress_iter):
        img_id = img_ids[img_count]
        img_info = coco_gt.imgs[img_id]
        img_name = os.path.splitext(img_info["file_name"])[0]
        H, W = img_info["height"], img_info["width"]

        anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        pred_masks = det_masks_per_image[img_count] or []

        gt_masks = [gt_polygons_to_mask(ann["segmentation"], H, W) for ann in anns]

        size = max(len(anns), len(pred_masks), 1)
        iou_matrix = np.zeros((size, size), dtype=float)
        for gi in range(len(anns)):
            for pj in range(len(pred_masks)):
                iou_matrix[gi, pj] = mask_iou(pred_masks[pj], gt_masks[gi])

        result = dsb_precision(iou_matrix, len(anns), len(pred_masks), iou_threshs)

        gt_assignment, pred_assignment = opt.linear_sum_assignment(1 - iou_matrix)
        dice_scores = []
        for gt_idx, pred_idx in zip(gt_assignment, pred_assignment):
            if gt_idx < len(anns) and pred_idx < len(pred_masks):
                dice_scores.append(dice_coefficient(pred_masks[pred_idx], gt_masks[gt_idx]))
        avg_dice = float(np.mean(dice_scores)) if dice_scores else 0.0

        for t, dsb_precision_value, tp, fp, fn in zip(
            iou_threshs,
            result.precisions,
            result.tp,
            result.fp,
            result.fn,
        ):
            classical_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = (
                2 * classical_precision * recall / (classical_precision + recall)
                if (classical_precision + recall) > 0
                else 0.0
            )
            fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0

            detailed_rows.append(
                {
                    "image_id": img_name,
                    "IoU_thresh": t,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "num_gt_objects": int(len(gt_masks)),
                    "num_pred_objects": int(len(pred_masks)),
                    "avg_dice": avg_dice,
                    "dsb_precision": float(dsb_precision_value),
                    "precision": float(classical_precision),
                    "recall": float(recall),
                    "f1_score": float(f1_score),
                    "fdr": float(fdr),
                }
            )

        dataset_dsb_map_sum += float(result.mean_ap)
        dataset_dsb_ap_sums += result.precisions
        n_eval += 1

        if bar is not None:
            bar()

    metadata = {
        "run_name": run_name,
        "pred_run": pred_run,
        "pred_json_path": str(pred_json_path),
        "pred_filename": pred_filename,
        "gt_json_path": str(gt_json_path),
        "iou_thresholds": iou_threshs,
    }

    (Path(out_dir) / "dsb_diagnostics_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    summary_df = pd.DataFrame({"IoU_thresh": iou_threshs,
                               "average_ap": [round(float(ap / max(n_eval, 1)), 4) for ap in dataset_dsb_ap_sums]})
    summary_df.to_csv(os.path.join(out_dir, "dsb_ap_summary.csv"), index=False)

    detailed_df = pd.DataFrame(detailed_rows)
    detailed_df.to_csv(os.path.join(out_dir, "dsb_per_image_threshold_metrics.csv"), index=False)

    per_image_average_df = (
        detailed_df
        .groupby("image_id", as_index=False)
        .agg(
            num_gt_objects=("num_gt_objects", "first"),
            num_pred_objects=("num_pred_objects", "first"),
            mean_dice=("avg_dice", "mean"),
            mAP_dsb=("dsb_precision", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1_score", "mean"),
            mean_fdr=("fdr", "mean"),
        )
    )

    per_image_average_df.to_csv(
        os.path.join(out_dir, "dsb_per_image_average_metrics.csv"),
        index=False,
    )
