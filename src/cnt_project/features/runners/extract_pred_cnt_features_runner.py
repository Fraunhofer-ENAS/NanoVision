from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cnt_project.features.plotting.prediction_feature_debug import (
    build_prediction_feature_debug_visualizations,
)
from cnt_project.features.pipelines.prediction_feature_extraction import (
    IMAGE_FEATURE_KEYS,
    OBJECT_FEATURE_KEYS,
    extract_features_from_prediction_json,
)
from cnt_project.features.plotting.histograms import (
    plot_feature_table_histograms,
)
from cnt_project.io.paths import ProjectPaths


def run_feature_extraction_for_run(
    *,
    pred_run: str,
    output_run: str | None = None,
    pred_filename: str = "predicted_annotations_poly.json",
    output_subdir: str = "pred_cnt_features",
    default_image_height: int = 256,
    default_image_width: int = 256,
    make_debug_visualizations: bool = False,
    debug_max_images: int = 6,
    debug_max_objects_per_image: int = 8,
    debug_row_index: int | None = None,
) -> dict[str, Any]:
    P = ProjectPaths.from_here(__file__)
    P.ensure_outputs()

    out_run = output_run or pred_run

    pred_json_path = P.predicted_poly_json(pred_run, filename=pred_filename)

    if not pred_json_path.exists():
        raise FileNotFoundError(f"Prediction JSON not found: {pred_json_path}")

    out_dir = P.eval_subdir(out_run, "features", output_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    object_df, image_df = extract_features_from_prediction_json(
        pred_json_path,
        default_image_height=default_image_height,
        default_image_width=default_image_width,
    )

    object_csv = out_dir / "object_level_features.csv"
    image_csv = out_dir / "image_level_features.csv"

    object_df.to_csv(object_csv, index=False)
    image_df.to_csv(image_csv, index=False)

    object_hist_dir = out_dir / "histograms" / "object_level"
    image_hist_dir = out_dir / "histograms" / "image_level"

    object_hist_paths = plot_feature_table_histograms(
        object_df,
        feature_columns=OBJECT_FEATURE_KEYS,
        out_dir=object_hist_dir,
        title_prefix="Object-level",
    )

    image_feature_cols = IMAGE_FEATURE_KEYS + [
        f"mean_{k}"
        for k in OBJECT_FEATURE_KEYS
    ]

    image_hist_paths = plot_feature_table_histograms(
        image_df,
        feature_columns=image_feature_cols,
        out_dir=image_hist_dir,
        title_prefix="Image-level",
    )

    debug_viz_paths: list[str] = []

    if make_debug_visualizations:
        debug_viz_paths = build_prediction_feature_debug_visualizations(
            pred_json_path=pred_json_path,
            output_dir=out_dir,
            image_roots=[P.test_images_root, P.train_images_root],
            default_image_height=default_image_height,
            default_image_width=default_image_width,
            max_images=debug_max_images,
            max_objects_per_image=debug_max_objects_per_image,
            row_index=debug_row_index,
        )

    summary = {
        "pred_run": pred_run,
        "output_run": out_run,
        "prediction_json": str(pred_json_path),
        "output_dir": str(out_dir),
        "object_csv": str(object_csv),
        "image_csv": str(image_csv),
        "n_images": int(image_df["image_id"].nunique()) if not image_df.empty else 0,
        "n_objects": int(len(object_df)),
        "n_object_hist_files": len(object_hist_paths),
        "n_image_hist_files": len(image_hist_paths),
        "n_debug_visualizations": len(debug_viz_paths),
        "debug_visualizations": debug_viz_paths,
        "object_feature_keys": OBJECT_FEATURE_KEYS,
        "image_feature_keys": IMAGE_FEATURE_KEYS,
    }

    summary_path = out_dir / "feature_extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("CNT feature extraction finished.")
    print(f"Prediction JSON: {pred_json_path}")
    print(f"Output directory: {out_dir}")
    print(f"Object-level CSV: {object_csv}")
    print(f"Image-level CSV: {image_csv}")
    if make_debug_visualizations:
        print(f"Debug visualizations generated: {len(debug_viz_paths)} files")
        print(f"Debug directory: {out_dir / 'debug_visualizations'}")
    print(f"Summary JSON: {summary_path}")

    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract canonical CNT features from a run prediction JSON "
            "and save object/image CSVs plus histograms."
        ),
    )

    parser.add_argument(
        "--pred-run",
        required=True,
        help="Run name that contains inference prediction JSON.",
    )
    parser.add_argument(
        "--output-run",
        default=None,
        help="Run name to write outputs into. Defaults to --pred-run.",
    )
    parser.add_argument(
        "--pred-filename",
        default="predicted_annotations_poly.json",
        help="Prediction filename under inference folder.",
    )
    parser.add_argument(
        "--output-subdir",
        default="pred_cnt_features",
        help="Subdirectory name under eval/features/.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=256,
        help="Fallback image height if missing in JSON image record.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=256,
        help="Fallback image width if missing in JSON image record.",
    )
    parser.add_argument(
        "--make-debug-viz",
        action="store_true",
        help=(
            "Generate debug overlays and line-density row ownership plots "
            "for selected test images."
        ),
    )
    parser.add_argument(
        "--debug-max-images",
        type=int,
        default=6,
        help="Maximum number of images to visualize for debug outputs.",
    )
    parser.add_argument(
        "--debug-max-objects",
        type=int,
        default=8,
        help="Maximum number of objects annotated per debug image overlay.",
    )
    parser.add_argument(
        "--debug-row-index",
        type=int,
        default=None,
        help=(
            "Optional fixed row index for line-density diagnostics. "
            "If omitted, uses the max line-density row per image."
        ),
    )

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    run_feature_extraction_for_run(
        pred_run=args.pred_run,
        output_run=args.output_run,
        pred_filename=args.pred_filename,
        output_subdir=args.output_subdir,
        default_image_height=args.image_height,
        default_image_width=args.image_width,
        make_debug_visualizations=args.make_debug_viz,
        debug_max_images=args.debug_max_images,
        debug_max_objects_per_image=args.debug_max_objects,
        debug_row_index=args.debug_row_index,
    )


if __name__ == "__main__":
    main()