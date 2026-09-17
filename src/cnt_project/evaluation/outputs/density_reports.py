from __future__ import annotations

from pathlib import Path

import pandas as pd
from cnt_project.evaluation.utils import display_density_label




def append_summary_and_pixel_rows_from_density_dir(
    *,
    out_dir: Path,
    run_name: str,
    density: str,
    algorithms: tuple[str, ...],
    pixel_metrics_algorithm: str,
    need_summary: bool,
    need_pixel_rows: bool,
    summary_rows: list[dict],
    pixel_metric_rows: list[dict],
) -> None:
    if need_summary:
        for algo in algorithms:
            image_metrics_csv = out_dir / f"image_metrics_{algo}.csv"
            if not image_metrics_csv.exists():
                continue

            df = pd.read_csv(image_metrics_csv)
            if df.empty:
                continue

            summary_rows.append(
                {
                    "run_name": run_name,
                    "density": density,
                    "matching_algorithm": algo,
                    "n_images": int(df.shape[0]),
                    "object_mAP_mean": float(df["Object mAP"].mean()),
                    "pixel_precision_mean": float(df["Pixel Precision"].mean()),
                    "pixel_recall_mean": float(df["Pixel Recall"].mean()),
                    "pixel_f1_mean": float(df["Pixel F1"].mean()),
                    "pixel_iou_mean": float(df["Pixel IoU"].mean()),
                    "pixel_dice_mean": float(df["Pixel Dice"].mean()),
                }
            )

    if need_pixel_rows:
        pixel_metrics_csv = out_dir / f"image_metrics_{pixel_metrics_algorithm}.csv"
        if pixel_metrics_csv.exists():
            pixel_df = pd.read_csv(pixel_metrics_csv)
            if not pixel_df.empty:
                for _, row in pixel_df.iterrows():
                    pixel_metric_rows.append(
                        {
                            "run_name": run_name,
                            "density": display_density_label(density),
                            "filename": row.get("filename"),
                            "pixel_precision": float(row["Pixel Precision"]),
                            "pixel_recall": float(row["Pixel Recall"]),
                            "pixel_f1": float(row["Pixel F1"]),
                            "pixel_dice": float(row["Pixel Dice"]),
                        }
                    )

