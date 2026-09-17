from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import json
from pycocotools.coco import COCO

from cnt_project.coco.filters import (
    subselect_coco_by_filenames,
)
from cnt_project.evaluation.pipelines.dsb_evaluation import (
    evaluate_dsb_map,
)
from cnt_project.evaluation.pipelines.coco_evaluation import (
    evaluate_coco_annotations,
)
from cnt_project.model_development.inference.export_predictions import (
    export_predictions_to_coco_json,
)
from cnt_project.model_development.inference.predictor import (
    predict_dataset,
)



@dataclass(frozen=True)
class ThresholdEvalResult:
    """
    Result for one probability-threshold candidate.

    A candidate can either be fully evaluated or skipped by an explicitly
    configured resource guard. Skipped candidates remain part of the sweep
    results for reproducibility.
    """

    threshold: float

    status: str
    skip_reason: str | None
    prediction_count: int

    dsb_map: float | None
    mean_dice: float | None

    iou_thresholds: tuple[float, ...]
    dsb_aps: tuple[float, ...]
    object_f1s: tuple[float, ...]
    cldice_thresholds: tuple[float, ...]
    cldice_object_f1s: tuple[float, ...]

    coco_stats: tuple[float, ...]

    prediction_json_path: Path


@dataclass(frozen=True)
class ThresholdOptimizationResult:
    """
    Results of a probability-threshold sweep.

    ``best_threshold`` is the candidate with the highest selected objective
    value. All candidate evaluations are retained for inspection.
    """

    best_threshold: float
    best_metric: float
    objective: str
    evaluations: tuple[ThresholdEvalResult, ...]
    results_csv_path: Path

    n_requested: int
    n_evaluated: int
    n_skipped: int


def _metric_for_objective(
    *,
    objective: str,
    dsb_map: float,
    mean_dice: float,
    iou_thresholds: tuple[float, ...],
    dsb_aps: tuple[float, ...],
    object_f1s: tuple[float, ...],
    cldice_thresholds: tuple[float, ...],
    cldice_object_f1s: tuple[float, ...],
    coco_stats: tuple[float, ...],
) -> float:
    """
    Resolve the scalar metric used to rank probability thresholds.
    """
    objective_norm = (
        str(objective)
        .strip()
        .lower()
    )

    if objective_norm == "dsb_map":
        return float(dsb_map)

    if objective_norm == "mean_dice":
        return float(mean_dice)
    
    if objective_norm == "coco_ap":
        if len(coco_stats) <= 0:
            raise ValueError(
                "COCO stats do not contain index 0 required for coco_ap."
            )
        return float(coco_stats[0])

    if objective_norm == "coco_ap50":
        if len(coco_stats) <= 1:
            raise ValueError(
                "COCO stats do not contain index 1 required for coco_ap50."
            )
        return float(coco_stats[1])

    if objective_norm == "coco_ap75":
        if len(coco_stats) <= 2:
            raise ValueError(
                "COCO stats do not contain index 2 required for coco_ap75."
            )
        return float(coco_stats[2])

    if objective_norm.startswith("cldice_object_f1@"):
        threshold_text = objective_norm.split(
            "@",
            maxsplit=1,
        )[1]

        requested_cldice = float(threshold_text)

        matches = [
            index
            for index, value in enumerate(cldice_thresholds)
            if abs(value - requested_cldice) < 1e-9
        ]

        if not matches:
            raise ValueError(
                f"Objective {objective!r} requests "
                f"clDice={requested_cldice}, but available "
                "clDice thresholds are "
                f"{list(cldice_thresholds)}."
            )

        return float(
            cldice_object_f1s[matches[0]]
        )


    if objective_norm.startswith(
        "dsb_ap@"
    ):
        metric_values = dsb_aps
        threshold_text = objective_norm.split(
            "@",
            maxsplit=1,
        )[1]

    elif objective_norm.startswith(
        "object_f1@"
    ):
        metric_values = object_f1s
        threshold_text = objective_norm.split(
            "@",
            maxsplit=1,
        )[1]

    else:
        raise ValueError(
            "Unsupported threshold objective: "
            f"{objective!r}. Use 'dsb_map', 'mean_dice', "
            "'dsb_ap@<iou>', 'object_f1@<iou>', "
            "'cldice_object_f1@<cldice>', "
            "'coco_ap', 'coco_ap50', or 'coco_ap75'."
        )

    requested_iou = float(
        threshold_text
    )

    matches = [
        index
        for index, value
        in enumerate(iou_thresholds)
        if abs(value - requested_iou)
        < 1e-9
    ]

    if not matches:
        raise ValueError(
            f"Objective {objective!r} requests "
            f"IoU={requested_iou}, but available "
            "IoU thresholds are "
            f"{list(iou_thresholds)}."
        )

    return float(
        metric_values[matches[0]]
    )

def metric_for_evaluation(
    result: ThresholdEvalResult,
    *,
    objective: str,
) -> float:
    """
    Return the scalar objective value for an evaluated threshold.
    """
    if result.status != "evaluated":
        raise ValueError(
            "Cannot resolve an objective metric for a threshold that was "
            f"not evaluated. threshold={result.threshold}, "
            f"status={result.status!r}, "
            f"reason={result.skip_reason!r}"
        )

    if result.dsb_map is None or result.mean_dice is None:
        raise ValueError(
            "Evaluated threshold is missing required metric values: "
            f"threshold={result.threshold}."
        )

    return _metric_for_objective(
        objective=objective,
        dsb_map=result.dsb_map,
        mean_dice=result.mean_dice,
        iou_thresholds=result.iou_thresholds,
        dsb_aps=result.dsb_aps,
        object_f1s=result.object_f1s,
        cldice_thresholds=result.cldice_thresholds,
        cldice_object_f1s=result.cldice_object_f1s,
        coco_stats=result.coco_stats,
    )

def evaluate_probability_threshold(
    *,
    threshold: float,
    model: Any,
    images: list,
    filenames: list[str],
    gt_json_path: str | Path,
    output_dir: str | Path,
    apply_smoothing: bool | None = None,
    dsb_score_thresh: float = 0.0,
    dsb_iou_threshs: list[float] | None = None,
    cldice_threshs: list[float] | None = None,
    max_prediction_annotations: int | None = None,
) -> ThresholdEvalResult:
    """
    Evaluate one probability-threshold candidate using the canonical CNT
    inference and postprocessing pipeline.

    The model's stored ``thresholds.json`` is not modified.

    Parameters
    ----------
    threshold:
        Probability threshold to test.

    model:
        Loaded patched StarDist model.

    images:
        Input images used for threshold calibration.

    filenames:
        Filenames corresponding to ``images``.

    gt_json_path:
        COCO ground-truth annotations for the same image subset.

    output_dir:
        Directory in which the candidate prediction JSON is written.

    objective:
        Metric used to score this candidate:
        ``"dsb_map"`` or ``"mean_dice"``.

    apply_smoothing:
        Optional override for CNT polygon smoothing.

    dsb_score_thresh:
        Prediction-score filtering threshold used during DSB evaluation.

    dsb_iou_threshs:
        Optional DSB IoU thresholds.

    Returns
    -------
    ThresholdEvalResult
        Threshold, objective value, and generated prediction JSON path.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "threshold must be strictly between 0 and 1. "
            f"Got: {threshold}"
        )

    if ( max_prediction_annotations is not None and max_prediction_annotations <= 0 ):
        raise ValueError(
            "max_prediction_annotations must be greater than 0 or None. "
            f"Got: {max_prediction_annotations}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Run the actual CNT inference/postprocessing path with a temporary
    # probability-threshold override.
    batch = predict_dataset(
        model=model,
        images=images,
        filenames=filenames,
        prob_thresh=threshold,
        apply_smoothing=apply_smoothing,
    )

    threshold_tag = f"{threshold:.6f}".rstrip("0").rstrip(".")

    polygon_json_path = (
        output_dir
        / f"predicted_annotations_poly_prob_{threshold_tag}.json"
    )

    rle_json_path = (
        output_dir
        / f"predicted_annotations_rle_prob_{threshold_tag}.json"
    )

    export_predictions_to_coco_json(
        images=batch.images,
        filenames=batch.filenames,
        polygons=batch.polygons,
        scores=batch.scores,
        out_json_path=polygon_json_path,
        out_rle_json_path=rle_json_path,
        category_id=1,
    )
    prediction_payload = json.loads( polygon_json_path.read_text( encoding="utf-8", ) )

    prediction_count = len( prediction_payload.get( "annotations", [], ) )

    print( "  Exported prediction annotations: " f"{prediction_count}" )

    if ( max_prediction_annotations is not None and prediction_count > max_prediction_annotations ):
        skip_reason = (
            "Prediction annotation count exceeds configured resource limit: "
            f"count={prediction_count}, "
            f"limit={max_prediction_annotations}."
        )
        print( "  Skipping metric evaluation: " f"{skip_reason}" )

        return ThresholdEvalResult(
            threshold=float(threshold),
            status="skipped_resource_limit",
            skip_reason=skip_reason,
            prediction_count=prediction_count,
            dsb_map=None,
            mean_dice=None,
            iou_thresholds=(),
            dsb_aps=(),
            object_f1s=(),
            cldice_thresholds=(),
            cldice_object_f1s=(),
            coco_stats=(),
            prediction_json_path=polygon_json_path,
        )

    metrics = evaluate_dsb_map(
        str(gt_json_path),
        str(polygon_json_path),
        score_thresh=dsb_score_thresh,
        iou_threshs=dsb_iou_threshs,
        cldice_threshs=cldice_threshs,
    )
    coco_eval = evaluate_coco_annotations( input_json_file=str(gt_json_path), output_json_file=str(polygon_json_path), )

    coco_stats = tuple( float(value) for value in getattr(coco_eval, "stats", []) )

    dsb_map = float( metrics["DSB_mAP"] )

    mean_dice = float( metrics["mean_dice"] )

    iou_thresholds = tuple( float(value) for value in metrics["IoU_thresholds"] )

    dsb_aps = tuple( float(value) for value in metrics["DSB_APs"] )

    object_f1s = tuple( float(value) for value in metrics["object_F1s"] )

    cldice_thresholds = tuple( float(value) for value in metrics["cldice_thresholds"] )

    cldice_object_f1s = tuple( float(value) for value in metrics["cldice_object_F1s"] )


    return ThresholdEvalResult(
        threshold=float(threshold),
        status="evaluated",
        skip_reason=None,
        prediction_count=prediction_count,
        dsb_map=dsb_map,
        mean_dice=mean_dice,
        iou_thresholds=iou_thresholds,
        dsb_aps=dsb_aps,
        object_f1s=object_f1s,
        cldice_thresholds=cldice_thresholds,
        cldice_object_f1s=cldice_object_f1s,
        coco_stats=coco_stats,
        prediction_json_path=polygon_json_path,
    )


def prepare_ground_truth_subset(
    *,
    gt_json_path: str | Path,
    filenames: list[str],
    output_path: str | Path,
) -> Path:
    """
    Create a COCO ground-truth JSON containing exactly the images being
    evaluated.

    This makes threshold evaluation safe when the dataset is limited with
    max_samples / max_val_samples.

    If all images from the source GT are being evaluated, the function still
    writes a run-local snapshot for reproducibility.
    """
    gt_json_path = Path(gt_json_path)
    output_path = Path(output_path)

    if not gt_json_path.exists():
        raise FileNotFoundError(
            f"Ground-truth COCO JSON not found: {gt_json_path}"
        )

    if not filenames:
        raise ValueError(
            "At least one filename is required to prepare the GT subset."
        )

    coco_gt = COCO(str(gt_json_path))

    subset_dataset = subselect_coco_by_filenames(
        coco_gt,
        filenames,
    )

    selected_images = subset_dataset.get(
        "images",
        [],
    )

    requested_stems = {
        Path(str(filename)).stem.lower()
        for filename in filenames
    }

    selected_stems = {
        Path(str(image["file_name"])).stem.lower()
        for image in selected_images
    }

    missing = sorted(
        requested_stems - selected_stems
    )

    unexpected = sorted(
        selected_stems - requested_stems
    )

    if missing or unexpected:
        raise ValueError(
            "Ground-truth subset does not match the requested evaluation "
            "filenames. "
            f"Missing={missing}, unexpected={unexpected}"
        )

    if len(selected_images) != len(requested_stems):
        raise ValueError(
            "Ground-truth subset image count does not match the requested "
            "evaluation image count: "
            f"requested={len(requested_stems)}, "
            f"selected={len(selected_images)}."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            subset_dataset,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return output_path

def optimize_probability_threshold(
    *,
    thresholds: Sequence[float],
    model: Any,
    images: list,
    filenames: list[str],
    gt_json_path: str | Path,
    predictions_dir: str | Path,
    results_dir: str | Path,
    objective: str = "dsb_map",
    apply_smoothing: bool | None = None,
    dsb_score_thresh: float = 0.0,
    dsb_iou_threshs: list[float] | None = None,
    max_prediction_annotations: int | None = None,
) -> ThresholdOptimizationResult:
    """
    Evaluate a collection of probability thresholds and recommend the
    candidate with the highest selected objective value.

    This function does not modify the model's ``thresholds.json``.

    All candidate metrics are retained and written to CSV so that the
    automatically selected threshold can also be inspected manually.
    """
    threshold_values = tuple(
        float(value)
        for value in thresholds
    )

    if not threshold_values:
        raise ValueError(
            "At least one probability threshold must be supplied."
        )

    if ( max_prediction_annotations is not None and max_prediction_annotations <= 0 ):
        raise ValueError(
            "max_prediction_annotations must be greater than 0 or None. "
            f"Got: {max_prediction_annotations}"
        )

    for threshold in threshold_values:
        if not 0.0 < threshold < 1.0:
            raise ValueError(
                "All probability thresholds must be strictly between "
                f"0 and 1. Got: {threshold}"
            )


    predictions_dir = Path(predictions_dir)
    results_dir = Path(results_dir)

    predictions_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    evaluation_gt_json_path = prepare_ground_truth_subset(
        gt_json_path=gt_json_path,
        filenames=filenames,
        output_path=( results_dir / "ground_truth_subset.json" ),
    )

    print( "Evaluation GT subset: " f"{evaluation_gt_json_path}" )

    # Match the existing evaluate_dsb_map() default when the caller
    # does not explicitly specify IoU thresholds.
    resolved_iou_threshs = (
        list(dsb_iou_threshs)
        if dsb_iou_threshs is not None
        else [
            round(index * 0.05, 2)
            for index in range(20)
        ]
    )

    objective_norm = str(objective).strip().lower()

    if objective_norm.startswith("cldice_object_f1@"):
        threshold_text = objective_norm.split(
            "@",
            maxsplit=1,
        )[1]

        try:
            requested_cldice_threshold = float(
                threshold_text
            )
        except ValueError as error:
            raise ValueError(
                "The clDice objective must use the form "
                "'cldice_object_f1@<threshold>'. "
                f"Received: {objective!r}"
            ) from error

        if not 0.0 <= requested_cldice_threshold <= 1.0:
            raise ValueError(
                "The clDice threshold must be between 0 and 1. "
                f"Received: {requested_cldice_threshold}"
            )

        resolved_cldice_threshs = [
            requested_cldice_threshold
        ]
    else:
        resolved_cldice_threshs = []

    evaluations: list[ThresholdEvalResult] = []

    for index, threshold in enumerate(
        threshold_values,
        start=1,
    ):
        print("")
        print(
            f"Threshold {index}/{len(threshold_values)}: "
            f"{threshold:.6f}"
        )

        result = evaluate_probability_threshold(
            threshold=threshold,
            model=model,
            images=images,
            filenames=filenames,
            gt_json_path=evaluation_gt_json_path,
            output_dir=predictions_dir,
            apply_smoothing=apply_smoothing,
            dsb_score_thresh=dsb_score_thresh,
            dsb_iou_threshs=resolved_iou_threshs,
            cldice_threshs=resolved_cldice_threshs,
            max_prediction_annotations=max_prediction_annotations,
        )

        evaluations.append(result)

        if result.status == "evaluated":
            objective_metric = metric_for_evaluation( result, objective=objective, )

            print( f"  DSB mAP:   {result.dsb_map:.6f}" )
            print( f"  Mean Dice: {result.mean_dice:.6f}" )
            print( f"  Objective ({objective}): " f"{objective_metric:.6f}" )
        else:
            print( f"  Status: {result.status}" )
            print( f"  Reason: {result.skip_reason}" )


    evaluated_results = [ result for result in evaluations if result.status == "evaluated" ]

    if not evaluated_results:
        raise RuntimeError(
            "No probability-threshold candidate completed metric evaluation. "
            "All candidates were skipped by the configured resource guard."
        )

    best_result = max( evaluated_results, key=lambda result: metric_for_evaluation( result, objective=objective, ), )

    best_metric = metric_for_evaluation( best_result, objective=objective, )

    rows: list[dict[str, Any]] = []

    for result in evaluations:
        if result.status == "evaluated":
            objective_metric = metric_for_evaluation( result, objective=objective, )
        else:
            objective_metric = None

        row: dict[str, Any] = {
            "prob_threshold": result.threshold,
            "status": result.status,
            "skip_reason": result.skip_reason,
            "prediction_count": result.prediction_count,
            "objective": objective,
            "objective_metric": objective_metric,
            "DSB_mAP": result.dsb_map,
            "mean_dice": result.mean_dice,
            "prediction_json_path": str(
                result.prediction_json_path
            ),
            "is_best": result is best_result,
        }
        if len(result.coco_stats) > 0:
            row["COCO_AP"] = result.coco_stats[0]

        if len(result.coco_stats) > 1:
            row["COCO_AP50"] = result.coco_stats[1]

        if len(result.coco_stats) > 2:
            row["COCO_AP75"] = result.coco_stats[2]

        for iou_threshold, ap in zip(
            result.iou_thresholds,
            result.dsb_aps,
        ):
            row[ f"DSB_AP@{iou_threshold:.2f}" ] = ap

        for iou_threshold, f1_value in zip(
            result.iou_thresholds,
            result.object_f1s,
        ):
            row[ f"Object_F1@{iou_threshold:.2f}" ] = f1_value

        for cldice_threshold, f1_value in zip(
            result.cldice_thresholds,
            result.cldice_object_f1s,
        ):
            row[ f"clDice_Object_F1@{cldice_threshold:.2f}" ] = f1_value

        rows.append(row)

    results_csv_path = (
        results_dir
        / "threshold_sweep_results.csv"
    )

    pd.DataFrame(rows).to_csv(
        results_csv_path,
        index=False,
    )

    print("")
    print("Threshold optimization completed")
    print("--------------------------------")
    print(
        f"Objective:      {objective}"
    )
    print(
        f"Best threshold: {best_result.threshold:.6f}"
    )
    print(
        f"Best metric:    {best_metric:.6f}"
    )
    print(
        f"Results CSV:    {results_csv_path}"
    )

    return ThresholdOptimizationResult(
        best_threshold=best_result.threshold,
        best_metric=best_metric,
        objective=objective,
        evaluations=tuple(evaluations),
        results_csv_path=results_csv_path,
        n_requested=len(threshold_values),
        n_evaluated=len(evaluated_results),
        n_skipped=len(evaluations) - len(evaluated_results),
    )


def save_probability_threshold(
    *,
    model: Any,
    model_dir: str | Path,
    probability_threshold: float,
) -> Path:
    """
    Persist the optimized CNT probability threshold to thresholds.json.

    Only the probability threshold is optimized by the CNT workflow.
    The existing/default StarDist NMS threshold is preserved for model
    compatibility.
    """
    if not 0.0 < probability_threshold < 1.0:
        raise ValueError(
            "probability_threshold must be strictly between 0 and 1. "
            f"Got: {probability_threshold}"
        )

    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}"
        )

    thresholds_path = model_dir / "thresholds.json"

    # Preserve an existing NMS value when available.
    if thresholds_path.exists():
        payload = json.loads(
            thresholds_path.read_text(
                encoding="utf-8",
            )
        )

        nms_threshold = float(
            payload.get(
                "nms",
                model.thresholds.nms,
            )
        )
    else:
        nms_threshold = float(
            model.thresholds.nms
        )

    payload = {
        "prob": float(probability_threshold),
        "nms": nms_threshold,
    }

    thresholds_path.write_text(
        json.dumps(
            payload,
            indent=4,
        )
        + "\n",
        encoding="utf-8",
    )

    return thresholds_path