from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from cellpose import io, models


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )

from cnt_project.evaluation.core.cldice import (  # noqa: E402
    cldice_object_f1,
)


VALIDATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "cellpose_dataset"
    / "validation_legacy_seed_42"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "cellpose_dataset"
    / "training_runs"
    / "cellpose_cyto_original_legacy_seed42_150epochs_patience40"
    / "models"
    / (
        "cellpose_cyto_original_legacy_seed42_"
        "150epochs_patience40_best_validation"
    )
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "cellpose_dataset"
    / "threshold_optimization"
    / (
        "cellpose_cyto_original_legacy_seed42_"
        "best_validation_refined"
    )
)

CELLDICE_THRESHOLD = 0.50

CELLPROB_THRESHOLDS = [
    -1.50,
    -1.25,
    -1.00,
    -0.75,
    -0.50,
]

FLOW_THRESHOLDS = [
    0.6,
    0.8,
    1.0,
]

MIN_SIZE = 15


def instance_masks(
    label_mask: np.ndarray,
) -> list[np.ndarray]:
    object_ids = np.unique(
        label_mask
    )

    object_ids = object_ids[
        object_ids != 0
    ]

    return [
        label_mask == object_id
        for object_id in object_ids
    ]


def load_validation_samples() -> list[dict]:
    image_paths = sorted(
        path
        for path in VALIDATION_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in {".tif", ".tiff"}
            and not path.name.lower().endswith(
                "_flows.tif"
            )
        )
    )

    samples = []

    for image_path in image_paths:
        seg_path = (
            VALIDATION_DIR
            / f"{image_path.stem}_seg.npy"
        )

        if not seg_path.is_file():
            raise FileNotFoundError(
                seg_path
            )

        payload = np.load(
            seg_path,
            allow_pickle=True,
        ).item()

        gt_mask = np.asarray(
            payload["masks"]
        )

        samples.append(
            {
                "image_name": image_path.name,
                "image": io.imread(
                    str(image_path)
                ),
                "gt_mask": gt_mask,
                "gt_objects": instance_masks(
                    gt_mask
                ),
            }
        )

    if not samples:
        raise RuntimeError(
            "No validation samples found."
        )

    return samples


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            MODEL_PATH
        )

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            "Output directory already exists: "
            f"{OUTPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True
    )

    samples = load_validation_samples()

    print(f"Model: {MODEL_PATH}")
    print(
        f"Validation images: {len(samples)}"
    )
    print(
        "Configurations: "
        f"{len(CELLPROB_THRESHOLDS) * len(FLOW_THRESHOLDS)}"
    )
    print(
        f"clDice matching threshold: "
        f"{CELLDICE_THRESHOLD}"
    )
    print(f"Output: {OUTPUT_DIR}")

    model = models.CellposeModel(
        gpu=True,
        pretrained_model=str(
            MODEL_PATH
        ),
    )

    configuration_rows = []
    per_image_rows = []

    configuration_number = 0
    total_configurations = (
        len(CELLPROB_THRESHOLDS)
        * len(FLOW_THRESHOLDS)
    )

    for cellprob_threshold in CELLPROB_THRESHOLDS:
        for flow_threshold in FLOW_THRESHOLDS:
            configuration_number += 1
            start_time = time.time()

            print()
            print(
                f"Configuration "
                f"{configuration_number}/"
                f"{total_configurations}: "
                f"cellprob={cellprob_threshold}, "
                f"flow={flow_threshold}"
            )

            image_f1_scores = []
            total_tp = 0
            total_fp = 0
            total_fn = 0
            total_gt = 0
            total_predictions = 0

            for image_index, sample in enumerate(
                samples,
                start=1,
            ):
                predicted_mask, flows, styles = (
                    model.eval(
                        sample["image"],
                        batch_size=1,
                        channels=[0, 0],
                        channel_axis=-1,
                        normalize=True,
                        diameter=None,
                        flow_threshold=(
                            flow_threshold
                        ),
                        cellprob_threshold=(
                            cellprob_threshold
                        ),
                        min_size=MIN_SIZE,
                        bsize=256,
                    )
                )

                predicted_objects = instance_masks(
                    np.asarray(
                        predicted_mask
                    )
                )

                result = cldice_object_f1(
                    ground_truth_masks=(
                        sample["gt_objects"]
                    ),
                    prediction_masks=(
                        predicted_objects
                    ),
                    threshold=(
                        CELLDICE_THRESHOLD
                    ),
                )

                gt_count = len(
                    sample["gt_objects"]
                )

                prediction_count = len(
                    predicted_objects
                )

                total_tp += result.true_positives
                total_fp += result.false_positives
                total_fn += result.false_negatives
                total_gt += gt_count
                total_predictions += prediction_count

                image_f1_scores.append(
                    result.object_f1
                )

                per_image_rows.append(
                    {
                        "cellprob_threshold": (
                            cellprob_threshold
                        ),
                        "flow_threshold": (
                            flow_threshold
                        ),
                        "min_size": MIN_SIZE,
                        "image": sample[
                            "image_name"
                        ],
                        "gt_objects": gt_count,
                        "predicted_objects": (
                            prediction_count
                        ),
                        "true_positives": (
                            result.true_positives
                        ),
                        "false_positives": (
                            result.false_positives
                        ),
                        "false_negatives": (
                            result.false_negatives
                        ),
                        "object_f1": (
                            result.object_f1
                        ),
                    }
                )

                print(
                    f"  [{image_index:02d}/"
                    f"{len(samples):02d}] "
                    f"GT={gt_count:3d}, "
                    f"pred={prediction_count:3d}, "
                    f"F1={result.object_f1:.4f}"
                )

            macro_f1 = float(
                np.mean(
                    image_f1_scores
                )
            )

            micro_denominator = (
                2 * total_tp
                + total_fp
                + total_fn
            )

            micro_f1 = (
                2.0 * total_tp
                / micro_denominator
                if micro_denominator > 0
                else 0.0
            )

            elapsed_seconds = (
                time.time()
                - start_time
            )

            configuration_rows.append(
                {
                    "cellprob_threshold": (
                        cellprob_threshold
                    ),
                    "flow_threshold": (
                        flow_threshold
                    ),
                    "min_size": MIN_SIZE,
                    "cldice_threshold": (
                        CELLDICE_THRESHOLD
                    ),
                    "macro_object_f1": (
                        macro_f1
                    ),
                    "micro_object_f1": (
                        micro_f1
                    ),
                    "true_positives": total_tp,
                    "false_positives": total_fp,
                    "false_negatives": total_fn,
                    "total_gt_objects": total_gt,
                    "total_predicted_objects": (
                        total_predictions
                    ),
                    "count_difference": (
                        total_predictions
                        - total_gt
                    ),
                    "elapsed_seconds": (
                        elapsed_seconds
                    ),
                }
            )

            print(
                f"  Macro F1: {macro_f1:.6f}"
            )
            print(
                f"  Micro F1: {micro_f1:.6f}"
            )
            print(
                f"  Counts: GT={total_gt}, "
                f"predicted={total_predictions}"
            )

    results = pd.DataFrame(
        configuration_rows
    )

    best_index = (
        results.sort_values(
            [
                "macro_object_f1",
                "micro_object_f1",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .index[0]
    )

    results["is_best"] = False
    results.loc[
        best_index,
        "is_best",
    ] = True

    best = results.loc[
        best_index
    ]

    results_path = (
        OUTPUT_DIR
        / "threshold_sweep_results.csv"
    )

    per_image_path = (
        OUTPUT_DIR
        / "per_image_results.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    pd.DataFrame(
        per_image_rows
    ).to_csv(
        per_image_path,
        index=False,
    )

    best_configuration = {
        "model": str(MODEL_PATH),
        "selection_subset": "validation",
        "objective": (
            "macro_object_f1"
            f"@cldice_{CELLDICE_THRESHOLD:.2f}"
        ),
        "cellprob_threshold": float(
            best["cellprob_threshold"]
        ),
        "flow_threshold": float(
            best["flow_threshold"]
        ),
        "min_size": int(
            best["min_size"]
        ),
        "macro_object_f1": float(
            best["macro_object_f1"]
        ),
        "micro_object_f1": float(
            best["micro_object_f1"]
        ),
        "total_gt_objects": int(
            best["total_gt_objects"]
        ),
        "total_predicted_objects": int(
            best["total_predicted_objects"]
        ),
    }

    best_path = (
        OUTPUT_DIR
        / "best_configuration.json"
    )

    best_path.write_text(
        json.dumps(
            best_configuration,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Optimization completed."
    )
    print(
        "Best cellprob threshold:",
        best_configuration[
            "cellprob_threshold"
        ],
    )
    print(
        "Best flow threshold:",
        best_configuration[
            "flow_threshold"
        ],
    )
    print(
        "Best macro object F1:",
        best_configuration[
            "macro_object_f1"
        ],
    )
    print(
        "Results:",
        results_path,
    )
    print(
        "Per-image results:",
        per_image_path,
    )
    print(
        "Best configuration:",
        best_path,
    )


if __name__ == "__main__":
    main()