from __future__ import annotations

from typing import Sequence

import numpy as np
from pycocotools.coco import COCO

from cnt_project.evaluation.core.coco_eval_custom import (
    CustomCOCOeval,
)


def evaluate_coco_annotations(input_json_file, output_json_file, thresholds=None):
    """
    Evaluate COCO format annotations and extract TP, FP, FN for each image and IoU threshold.
    """
    coco_gt = COCO(input_json_file)
    coco_dt = COCO(output_json_file)

    # https://cocodataset.org/#detection-eval
    coco_eval = CustomCOCOeval(coco_gt, coco_dt, "segm")
    coco_eval.params.maxDets = [30, 80, 100]
    coco_eval.params.areaRng = [
        [0, 10000],
        [0, 30],
        [31, 100],
        [100, 10000],
    ]
    if thresholds is not None:
        coco_eval.params.iouThrs = thresholds

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def make_iou_thresholds(
    thresholds: Sequence[float] | None,
    *,
    start: float = 0.50,
    stop: float = 0.95,
    step: float = 0.05,
) -> np.ndarray | None:
    """
    If thresholds is provided, use it.
    Otherwise default to COCO-style 0.50:0.95 step 0.05.
    """
    if thresholds is None:
        return np.round(np.arange(start, stop + 1e-9, step), 2)
    return np.asarray(list(thresholds), dtype=float)
