from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import scipy.optimize as opt
from pycocotools.coco import COCO

from cnt_project.coco.filters import subselect_coco_by_filenames
from cnt_project.coco.masks import extract_predictions_from_json, gt_polygons_to_mask
from cnt_project.evaluation.core.kaggle_dsb_metrics import dsb_precision_from_length_eval_dsb
from cnt_project.evaluation.core.metrics_binary_masks import dice_coefficient, mask_iou
from cnt_project.io.paths import OutputPaths, ProjectPaths
from cnt_project.evaluation.utils import safe_progress_iter
from cnt_project.evaluation.inputs.evaluation_grouping import (
    load_length_groups,
)

def evaluate_dsb_subset_length(
    coco_gt: COCO,
    pred_dataset_dict: dict,
    iou_threshs: list[float],
    score_thresh: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs DSB evaluation on the (already subsetted) GT COCO and prediction dataset dict.

    Returns:
      - summary_df: IoU_thresh, average_ap
      - per_image_df: per-image mAP + mean dice + num_gt_objects
    """
    img_ids = sorted(coco_gt.getImgIds())

    # Predictions decoded per GT image order
    _, det_masks_per_image, _ = extract_predictions_from_json(score_thresh, coco_gt, pred_dataset_dict)

    dataset_dsb_map_sum = 0.0
    dataset_dsb_ap_sums = np.zeros(len(iou_threshs), dtype=float)
    n_eval = 0

    per_image_rows = []

    progress_iter = safe_progress_iter(len(img_ids))
    for img_idx, bar in enumerate(progress_iter):
        img_id = img_ids[img_idx]
        img_info = coco_gt.imgs[img_id]
        H, W = img_info["height"], img_info["width"]

        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)

        pred_masks = det_masks_per_image[img_idx] if img_idx < len(det_masks_per_image) else []
        if pred_masks is None:
            pred_masks = []

        gt_masks = []
        for ann in anns:
            gt_masks.append(gt_polygons_to_mask(ann["segmentation"], H, W))

        # IoU matrix
        iou_matrix_size = max(len(gt_masks), len(pred_masks), 1)
        iou_matrix = np.zeros((iou_matrix_size, iou_matrix_size), dtype=float)
        for gi in range(len(gt_masks)):
            for pj in range(len(pred_masks)):
                iou_matrix[gi, pj] = mask_iou(pred_masks[pj], gt_masks[gi])

        # DSB precision over thresholds (length-class impl)
        dsb_map, dsb_precisions = dsb_precision_from_length_eval_dsb(
            iou_matrix, len(gt_masks), len(pred_masks), iou_threshs
        )

        dataset_dsb_ap_sums += np.asarray(dsb_precisions, dtype=float)
        dataset_dsb_map_sum += float(dsb_map)
        n_eval += 1

        # Dice on assignment (diagnostic)
        dice_scores = []
        gt_assignment, pred_assignment = opt.linear_sum_assignment(1 - iou_matrix)
        for gt_ass, pred_ass in zip(gt_assignment, pred_assignment):
            if gt_ass < len(gt_masks) and pred_ass < len(pred_masks):
                dice_scores.append(dice_coefficient(pred_masks[pred_ass], gt_masks[gt_ass]))
        mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0

        per_image_rows.append(
            {
                "image_id": int(img_id),
                "filename": img_info.get("file_name", ""),
                "num_gt_objects": int(len(gt_masks)),
                "mAP": float(dsb_map),
                "mean_dice": round(mean_dice, 4),
            }
        )

        if bar is not None:
            bar()

    denom = max(n_eval, 1)
    summary_df = pd.DataFrame(
        {
            "IoU_thresh": [round(float(t), 2) for t in iou_threshs],
            "average_ap": [round(float(v / denom), 4) for v in dataset_dsb_ap_sums],
        }
    )
    per_image_df = pd.DataFrame(per_image_rows)

    return summary_df, per_image_df

def run_dsb_by_length_class(
    *,
    run_name: str = "evaluation_runner_test",
    pred_run: str = "evaluation_runner_test",
    pred_filename: str = "predicted_annotations_poly.json",
    gt_json_path: str | Path | None = None,
    eval_tag: str = "length_eval",
    dataset_root: str | Path,
    split_manifest_path: str | Path,
    subset: str = "test",
    length_source: str = "canonical_metadata",
    length_measurement: str = "geodesic_px",
    length_percentile: float = 98.5,
    legacy_length_csv: str | Path | None = None,
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
    out_root = output_paths.eval_subdir(
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


    # Validate inputs
    if not pred_json_path.exists():
        raise FileNotFoundError(
            f"Prediction JSON does not exist: {pred_json_path}"
        )

    if not Path(gt_json_path).exists():
        raise FileNotFoundError(
            f"Ground-truth COCO JSON does not exist: {gt_json_path}"
        )

    dataset_root = Path(dataset_root).resolve()
    split_manifest_path = Path(split_manifest_path).resolve()

    object_lengths_csv = ( dataset_root / "metadata" / "object_lengths.csv" )

    image_lengths_csv = ( dataset_root / "metadata" / "image_lengths.csv" )

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {dataset_root}"
        )

    if not split_manifest_path.exists():
        raise FileNotFoundError(
            f"Split manifest does not exist: {split_manifest_path}"
        )

    if not object_lengths_csv.exists():
        raise FileNotFoundError(
            f"Object length metadata does not exist: {object_lengths_csv}"
        )

    if not image_lengths_csv.exists():
        raise FileNotFoundError(
            f"Image length metadata does not exist: {image_lengths_csv}"
        )

    if (
        length_source == "legacy_metadata"
        and legacy_length_csv is None
    ):
        raise ValueError(
            "--legacy-length-csv is required when "
            "--length-source legacy_metadata."
        )


    score_thresh = 0.0
    iou_threshs = np.round(
        np.arange(0.0, 0.95 + 0.05, 0.05),
        2,
    ).tolist()

    categories_to_files, length_threshold = load_length_groups(
        length_source=length_source,
        object_lengths_csv=object_lengths_csv,
        image_lengths_csv=image_lengths_csv,
        split_manifest_path=split_manifest_path,
        subset=subset,
        measurement=length_measurement,
        percentile=length_percentile,
        legacy_length_csv=legacy_length_csv,
    )

    coco_gt_full = COCO(gt_json_path)
    coco_dt = COCO(pred_json_path)

    for category, filenames in categories_to_files.items():
        gt_subset_dataset = subselect_coco_by_filenames(coco_gt_full, filenames)
        coco_gt_subset = COCO()
        coco_gt_subset.dataset = gt_subset_dataset
        coco_gt_subset.createIndex()

        dt_subset_dataset = subselect_coco_by_filenames(coco_dt, filenames)
        pred_subset_dataset_dict = dt_subset_dataset

        summary_df, per_image_df = evaluate_dsb_subset_length(
            coco_gt=coco_gt_subset,
            pred_dataset_dict=pred_subset_dataset_dict,
            iou_threshs=iou_threshs,
            score_thresh=score_thresh,
        )

        cat_out = out_root / category
        cat_out.mkdir(parents=True, exist_ok=True)

        summary_df.to_csv(cat_out / "dsb_ap_summary.csv", index=False)
        per_image_df.to_csv(cat_out / "dsb_per_image_metrics.csv", index=False)
