from __future__ import annotations

import itertools
import json
import os
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize as opt
from pycocotools.coco import COCO

from cnt_project.coco.filters import subselect_coco_by_filenames
from cnt_project.coco.masks import extract_predictions_from_json, gt_polygons_to_mask
from cnt_project.evaluation.core.kaggle_dsb_metrics import dsb_precision
from cnt_project.evaluation.core.metrics_binary_masks import dice_coefficient, mask_iou
from cnt_project.io.paths import OutputPaths, ProjectPaths
from cnt_project.evaluation.inputs.evaluation_grouping import normalize_noise_selection, load_noise_classification_csv
from cnt_project.evaluation.utils import safe_progress_iter

def run_dsb_by_noise_class(
    selected_noise: str | None = None,
    *,
    run_name: str = "evaluation_runner_test",
    pred_run: str = "evaluation_runner_test",
    pred_filename: str = "predicted_annotations_poly.json",
    gt_json_path: str | None = None,
    eval_tag: str = "noise_eval",
    noise_csv_path: str | None = None,
    noise_label_column: str = "noise_class",
    global_outputs_root: str | Path | None = None,
) -> None:
    # Resolve output structure
    if global_outputs_root is not None:
        output_paths = OutputPaths.from_root(
            global_outputs_root
        )
        project_paths = None
    else:
        project_paths = ProjectPaths.from_here(__file__)
        output_paths = project_paths.output_paths

    output_paths.ensure()

    # Resolve evaluation output directory
    out_dir = output_paths.eval_subdir(
        run_name,
        "dsb",
        eval_tag,
    )

    # Resolve prediction JSON
    pred_json_path = output_paths.predicted_poly_json(
        pred_run,
        filename=pred_filename,
    )

    # Resolve ground-truth JSON
    if gt_json_path is None:
        if project_paths is None:
            raise ValueError(
                "--gt-json-path is required when using a custom "
                "global outputs root."
            )

        gt_json_path = project_paths.test_coco_json
    else:
        gt_json_path = Path(gt_json_path).resolve()

    # Resolve noise-classification CSV
    if noise_csv_path is None:
        if project_paths is None:
            raise ValueError(
                "--noise-csv-path is required when using a custom "
                "global outputs root."
            )

        noise_csv_path = project_paths.metadata_file(
            "test",
            "noise_evaluation_results_with_labels.csv",
        )
    else:
        noise_csv_path = Path(noise_csv_path).resolve()

    if not pred_json_path.exists():
        raise FileNotFoundError(
            f"Prediction JSON does not exist: {pred_json_path}"
        )

    if not Path(gt_json_path).exists():
        raise FileNotFoundError(
            f"Ground-truth COCO JSON does not exist: {gt_json_path}"
        )

    if not Path(noise_csv_path).exists():
        raise FileNotFoundError(
            f"Noise classification CSV does not exist: {noise_csv_path}"
        )

    selected_noise = normalize_noise_selection(selected_noise)

    score_thresh = 0.0
    iou_threshs = np.round(np.arange(0.0, 0.95 + 0.05, 0.05), 2).tolist()

    metadata = {
        "mode": "noise",
        "run_name": run_name,
        "pred_run": pred_run,
        "pred_json_path": str(pred_json_path),
        "pred_filename": pred_filename,
        "gt_json_path": str(gt_json_path),
        "noise_csv_path": str(noise_csv_path),
        "noise_label_column": noise_label_column,
        "selected_noise": selected_noise,
        "iou_thresholds": iou_threshs,
    }

    (Path(out_dir) / "dsb_noise_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    coco_gt_full = COCO(gt_json_path)
    coco_dt_full = COCO(pred_json_path)

    noise_groups = load_noise_classification_csv(
        noise_csv_path,
        label_column=noise_label_column,
    )

    if selected_noise == "ALL":
        groups_to_run = sorted(noise_groups.keys())
    else:
        groups_to_run = [selected_noise]

    missing = [g for g in groups_to_run if g not in noise_groups]
    if missing:
        raise ValueError(
            f"Noise group(s) {missing} not found in {noise_csv_path} column '{noise_label_column}'. "
            f"Found: {sorted(noise_groups.keys())}"
        )

    for noise_key in groups_to_run:
        target_filenames = noise_groups[noise_key]

        filtered_gt_dataset = subselect_coco_by_filenames(coco_gt_full, target_filenames)
        filtered_gt_coco = COCO()
        filtered_gt_coco.dataset = filtered_gt_dataset
        filtered_gt_coco.createIndex()

        filtered_dt_dataset = subselect_coco_by_filenames(coco_dt_full, target_filenames)
        filtered_dt_coco = COCO()
        filtered_dt_coco.dataset = filtered_dt_dataset
        filtered_dt_coco.createIndex()

        img_ids = sorted(filtered_gt_coco.getImgIds())

        _det_scores_per_image, det_masks_per_image, _det_boxes_per_image = extract_predictions_from_json(
            score_thresh, filtered_gt_coco, filtered_dt_coco.dataset
        )

        dataset_dsb_map_sum = 0.0
        dataset_dsb_ap_sums = np.zeros(len(iou_threshs), dtype=float)
        dataset_object_precision_sums = np.zeros(len(iou_threshs), dtype=float)
        dataset_object_f1_sums = np.zeros(len(iou_threshs), dtype=float)
        dataset_object_fdr_sums = np.zeros(len(iou_threshs), dtype=float)
        per_image_results: list[dict[str, Any]] = []

        progress_iter = safe_progress_iter(len(det_masks_per_image))
        for img_count, bar in enumerate(progress_iter):
            img_id = img_ids[img_count]
            img_info = filtered_gt_coco.imgs[img_id]
            H, W = img_info["height"], img_info["width"]

            ann_ids = filtered_gt_coco.getAnnIds(imgIds=img_id)
            anns = filtered_gt_coco.loadAnns(ann_ids)

            pred_masks = det_masks_per_image[img_count]

            if pred_masks is None or len(pred_masks) == 0:
                if bar is not None:
                    bar()
                continue

            gt_masks = [gt_polygons_to_mask(ann["segmentation"], H, W) for ann in anns]

            iou_matrix_size = max(len(gt_masks), len(pred_masks))
            iou_matrix = np.zeros((iou_matrix_size, iou_matrix_size), dtype=float)
            for gi, pj in itertools.product(range(len(gt_masks)), range(len(pred_masks))):
                iou_matrix[gi, pj] = mask_iou(pred_masks[pj], gt_masks[gi])

            res = dsb_precision(iou_matrix, len(gt_masks), len(pred_masks), iou_threshs)
            dsb_map = float(res.mean_ap)
            dsb_precisions = np.asarray(res.precisions, dtype=float)

            tp_arr = np.asarray(res.tp, dtype=float)
            fp_arr = np.asarray(res.fp, dtype=float)
            fn_arr = np.asarray(res.fn, dtype=float)

            object_precision_arr = np.divide(
                tp_arr,
                tp_arr + fp_arr,
                out=np.zeros_like(tp_arr, dtype=float),
                where=(tp_arr + fp_arr) > 0,
            )
            object_recall_arr = np.divide(
                tp_arr,
                tp_arr + fn_arr,
                out=np.zeros_like(tp_arr, dtype=float),
                where=(tp_arr + fn_arr) > 0,
            )
            object_f1_arr = np.divide(
                2.0 * object_precision_arr * object_recall_arr,
                object_precision_arr + object_recall_arr,
                out=np.zeros_like(object_precision_arr, dtype=float),
                where=(object_precision_arr + object_recall_arr) > 0,
            )
            object_fdr_arr = np.divide(
                fp_arr,
                tp_arr + fp_arr,
                out=np.zeros_like(fp_arr, dtype=float),
                where=(tp_arr + fp_arr) > 0,
            )

            dataset_dsb_ap_sums += dsb_precisions
            dataset_object_precision_sums += object_precision_arr
            dataset_object_f1_sums += object_f1_arr
            dataset_object_fdr_sums += object_fdr_arr

            dice_scores: list[float] = []
            gt_assignment, pred_assignment = opt.linear_sum_assignment(1 - iou_matrix)
            for gt_ass, pred_ass in zip(gt_assignment, pred_assignment):
                if gt_ass < len(gt_masks) and pred_ass < len(pred_masks):
                    dice_scores.append(dice_coefficient(pred_masks[pred_ass], gt_masks[gt_ass]))
            mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0

            per_image_results.append(
                {
                    "image_id": int(img_id),
                    "filename": img_info.get("file_name", ""),
                    "num_gt_objects": int(len(gt_masks)),
                    "mAP": dsb_map,
                    "mean_dice": round(mean_dice, 4),
                    "object_precision": round(float(np.mean(object_precision_arr)), 4),
                    "object_f1_score": round(float(np.mean(object_f1_arr)), 4),
                    "object_fdr": round(float(np.mean(object_fdr_arr)), 4),
                    "APs": {f"IoU@{iou_threshs[i]:.2f}": round(float(p), 4) for i, p in enumerate(dsb_precisions)},
                    **{
                        f"dsb_precision_IoU@{iou_threshs[i]:.2f}": round(float(v), 4)
                        for i, v in enumerate(dsb_precisions)
                    },
                    **{
                        f"object_precision_IoU@{iou_threshs[i]:.2f}": round(float(v), 4)
                        for i, v in enumerate(object_precision_arr)
                    },
                    **{
                        f"object_f1_score_IoU@{iou_threshs[i]:.2f}": round(float(v), 4)
                        for i, v in enumerate(object_f1_arr)
                    },
                    **{
                        f"object_fdr_IoU@{iou_threshs[i]:.2f}": round(float(v), 4)
                        for i, v in enumerate(object_fdr_arr)
                    },
                }
            )

            dataset_dsb_map_sum += dsb_map

            if bar is not None:
                bar()

        denom = max(1, len(det_masks_per_image))

        df_summary = pd.DataFrame(
            {
                "IoU_thresh": [round(float(t), 2) for t in iou_threshs],
                "average_ap": [round(float(s / denom), 2) for s in dataset_dsb_ap_sums],
                "average_object_precision": [round(float(s / denom), 4) for s in dataset_object_precision_sums],
                "average_object_f1_score": [round(float(s / denom), 4) for s in dataset_object_f1_sums],
                "average_object_fdr": [round(float(s / denom), 4) for s in dataset_object_fdr_sums],
            }
        )
        df_summary.to_csv(os.path.join(out_dir, f"ap_results_poly_{noise_key}.csv"), index=False)

        df_per_image = pd.DataFrame(per_image_results)
        df_per_image.to_csv(os.path.join(out_dir, f"ap_dice_results_poly_{noise_key}.csv"), index=False)
