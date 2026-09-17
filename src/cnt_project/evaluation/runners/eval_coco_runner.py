from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from pycocotools.coco import COCO

from cnt_project.evaluation.core.coco_eval_custom import CustomCOCOeval
from cnt_project.io.paths import ProjectPaths
from cnt_project.evaluation.pipelines.coco_evaluation import (
    evaluate_coco_annotations,
    make_iou_thresholds,
)

"""eval_coco_runner.py
This is the runner/orchestration layer for COCO evaluation.

How it evaluates

The flow is:

load GT COCO JSON
load prediction COCO JSON
instantiate CustomCOCOeval(..., "segm")
optionally set IoU thresholds
run:
evaluate()
accumulate()
summarize()
save a summary JSON/CSV

This is clearly the COCO-style evaluation family.
"""


def run_coco_eval(
    *,
    run_name: str = "evaluation_runner_test",
    pred_run: str = "evaluation_runner_test",
    pred_filename: str = "predicted_annotations_poly.json",
    thresholds: Sequence[float] | None = None,
    save_outputs: bool = True,
) -> CustomCOCOeval:
    """
    Runner-style entry point:
    - Reads GT from dataset test split
    - Reads preds from global_outputs/runs/<pred_run>/inference/<pred_filename>
    - Writes results to global_outputs/runs/<run_name>/eval/coco/coco_eval/
    """
    P = ProjectPaths.from_here(__file__)
    P.ensure_outputs()

    gt_json_path = str(P.coco_json("test"))
    pred_json_path = str(P.predicted_poly_json(pred_run, filename=pred_filename))

    out_dir = P.eval_subdir(run_name, "coco", "coco_eval")
    iou_thrs = make_iou_thresholds(thresholds)

    coco_eval = evaluate_coco_annotations(
        input_json_file=gt_json_path,
        output_json_file=pred_json_path,
        thresholds=iou_thrs,
    )

    if save_outputs:
        payload = {
            "gt_json": gt_json_path,
            "pred_json": pred_json_path,
            "params": {
                "iouThrs": [float(x) for x in getattr(coco_eval.params, "iouThrs", [])],
                "maxDets": list(getattr(coco_eval.params, "maxDets", [])),
                "areaRng": list(getattr(coco_eval.params, "areaRng", [])),
            },
            "stats": [float(x) for x in getattr(coco_eval, "stats", [])]
            if hasattr(coco_eval, "stats")
            else None,
        }

        (Path(out_dir) / "coco_eval_summary.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        if payload["stats"] is not None:
            csv_lines = ["index,value"]
            csv_lines += [f"{i},{v}" for i, v in enumerate(payload["stats"])]
            (Path(out_dir) / "coco_eval_stats.csv").write_text(
                "\n".join(csv_lines) + "\n",
                encoding="utf-8",
            )

        print(f"\nSaved COCOeval outputs to: {out_dir}")

    return coco_eval


def main() -> None:
    run_coco_eval(
        run_name="detectron2_trial_1_LB",
        pred_run="detectron2_trial_1_LB",
        pred_filename="predicted_annotations_poly.json",
        thresholds=None,
        save_outputs=True,
    )


if __name__ == "__main__":
    main()