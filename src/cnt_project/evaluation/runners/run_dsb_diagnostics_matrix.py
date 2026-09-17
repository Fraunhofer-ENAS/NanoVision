from __future__ import annotations

from cnt_project.evaluation.pipelines.dsb_diagnostics import (
    run_dsb_full_dataset_diagnostics,
)
from cnt_project.io.paths import ProjectPaths


def main() -> None:
    P = ProjectPaths.from_here(__file__)

    pred_run = "nanovision_20260610"
    eval_run = f"{pred_run}"

    pred_files = {
        "mask": "predicted_annotations_mask.json",
        "poly": "predicted_annotations_poly.json",
        "rle": "predicted_annotations_rle.json",
    }

    gt_files = {
        "old": P.test_coco_root / "annotations_old.json",
        "chain_simple": P.test_coco_root / "annotations_chain_approx_simple.json",
        "chain_none": P.test_coco_root / "annotations_chain_approx_none.json",
        # "lb": P.test_coco_root / "annotations_LB.json",
    }

    for pred_name, pred_filename in pred_files.items():
        for gt_name, gt_path in gt_files.items():
            eval_tag = f"Diagnostic_eval__gt-{gt_name}__pred-{pred_name}"

            print("=" * 80)
            print("pred_run:", pred_run)
            print("pred_filename:", pred_filename)
            print("gt:", gt_path)
            print("eval_tag:", eval_tag)

            run_dsb_full_dataset_diagnostics(
                run_name=eval_run,
                pred_run=pred_run,
                pred_filename=pred_filename,
                gt_json_path=str(gt_path),
                eval_tag=eval_tag,
            )


if __name__ == "__main__":
    main()