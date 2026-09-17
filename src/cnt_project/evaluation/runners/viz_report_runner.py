from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2  

from cnt_project.io.paths import ProjectPaths

from cnt_project.visualizations.evaluation.data_analysis.metrics_vs_object_count import plot_metrics_vs_object_count
from cnt_project.visualizations.evaluation.data_analysis.model_comparisons import (
    ModelDisplaySpec,
    apply_science_style,
    plot_ap_vs_iou,
    plot_avg_dice_across_models,
    plot_ap_vs_noise,
)
from cnt_project.coco import (
    index_coco_by_image,
    load_coco,
    save_image_with_annotations,
)

# -----------------------------
#  binned metrics vs object count
# -----------------------------
from cnt_project.visualizations.evaluation.data_analysis.metrics_vs_object_count_binned import (
    apply_science_style as apply_science_style_binned,
    load_and_merge_csvs,
    plot_metrics_vs_gt_objects_binned,
)


def _resolve_noise_csv(P: ProjectPaths, noise_csv: str | None) -> str | None:
    """
    Resolve noise CSV path using ProjectPaths conventions.

    Priority:
      1) explicit noise_csv argument (absolute or project-root relative)
      2) canonical test metadata file: noise_evaluation_results.csv
      3) first metadata file matching noise*.csv
    """
    if noise_csv:
        p = Path(noise_csv)
        if not p.is_absolute():
            p = P.project_root / p
        return str(p.resolve())

    canonical = P.metadata_file("test", "noise_evaluation_results.csv")
    if canonical.exists():
        return str(canonical)

    candidates = sorted(P.test_metadata_root.glob("noise*.csv"))
    if candidates:
        return str(candidates[0])

    return None


# -----------------------------
# Task 1: density trends
# -----------------------------
def task_density_trends(
    *,
    P: ProjectPaths,
    run: str,
    selected_density: str,
    eval_tag: str,
) -> None:
    # new agreed layout:
    # input:  runs/<run>/eval/dsb/density_eval/ap_dice_results_poly_<DENSITY>.csv
    # output: runs/<run>/viz/density_trends/
    out_dir = P.viz_dir(run, "density_trends")
    density_eval_dir = P.eval_subdir(run, "dsb", eval_tag)
    csv_path = density_eval_dir / f"ap_dice_results_poly_{selected_density}.csv"

    if not csv_path.exists():
        available = sorted(p.name for p in density_eval_dir.glob("ap_dice_results_poly_*.csv"))
        raise FileNotFoundError(
            f"Missing CSV for density trends: {csv_path}\n"
            f"Available in {density_eval_dir}: {available if available else 'none'}"
        )

    out_map, out_dice = plot_metrics_vs_object_count(str(csv_path), str(out_dir), tag=selected_density)

    print(f"Saved: {out_map}")
    print(f"Saved: {out_dice}")

# -----------------------------
# Task 1b: metrics vs GT objects (binned)
# -----------------------------
def task_metrics_vs_object_count_binned(
    *,
    P: ProjectPaths,
    run: str,
    densities: list[str],
    bin_size: int,
    eval_tag: str,
) -> None:
    """
    Consumes per-image DSB density CSVs and produces binned mean/std curves.

    Option 1 inputs:
      runs/<run>/eval/dsb/density_eval/ap_dice_results_poly_<DENSITY>.csv

    Output:
      runs/<run>/viz/metrics_vs_object_count_binned/
        - mean_dice_binned.pdf + .pgf
        - mAP_binned.pdf + .pgf
    """
    out_dir = P.viz_dir(run, "metrics_vs_object_count_binned")
    density_eval_dir = P.eval_subdir(run, "dsb", eval_tag)

    csv_paths: list[Path] = []
    missing: list[str] = []
    for d in densities:
        p = density_eval_dir / f"ap_dice_results_poly_{d}.csv"
        if not p.exists():
            missing.append(d)
            continue
        csv_paths.append(p)

    if missing:
        print(f"Warning: missing density CSVs for {missing}.")

    if not csv_paths:
        available = sorted(p.name for p in density_eval_dir.glob("ap_dice_results_poly_*.csv"))
        raise FileNotFoundError(
            f"No density CSVs found for run '{run}'.\n"
            f"Expected under: {density_eval_dir}\n"
            f"Requested densities: {densities}\n"
            f"Available files: {available if available else 'none'}\n"
            "Run DSB density evaluation first (run_dsb_by_density with selected_density='ALL') for this run."
        )

    # keep plotting logic unchanged: call the module helpers as-is
    apply_science_style_binned()
    df = load_and_merge_csvs([str(p) for p in csv_paths])

    plot_metrics_vs_gt_objects_binned(
        df,
        metrics=("mean_dice", "mAP"),
        bin_size=bin_size,
        output_prefixes=("mean_dice_binned", "mAP_binned"),
        out_dir=out_dir,
    )

    print(f"Saved binned plots to: {out_dir}")

# -----------------------------
# Task 2: model comparison
# -----------------------------
def task_model_comparison(
    *,
    P: ProjectPaths,
    report: str,
    runs: list[str],
    length_classes: list[str],
    noise_csv: str | None,
) -> None:
    # output location (report):
    summary_output_base = P.report_dir(report, "figures")

    spec = ModelDisplaySpec(
        model_name_map={
            "fluo_centerness": "SYM",
            "fluo_edt_standard": "EDT",
            "fluo_new_edt": "EDT-SYM",
            "fluo_edt_LB": "EDT-FULL",
        },
        color_map={
            "SYM": "#a6cee3",
            "EDT": "#1f78b4",
            "EDT-SYM": "#6baed6",
            "EDT-FULL": "#08306b",
        },
    )

    apply_science_style()

    # "path" points to runs/<run>/eval/dsb (where the CSVs live).
    models = {run_name: str(P.eval_dir(run_name, "dsb")) for run_name in runs}

    # ---- 1) AP vs IoU (dataset) ----
    for lc in length_classes:
        # we now write:
        # runs/<run>/eval/dsb/length_eval/<lc>/dsb_ap_summary.csv
        csv_paths = {name: os.path.join(path, "length_eval", lc, "dsb_ap_summary.csv") for name, path in models.items()}
        out = summary_output_base / f"comparison_Average_AP_{lc}.png"
        plot_ap_vs_iou(csv_paths, spec=spec, output_path=str(out))

    # ---- 2) Avg Dice across models ----
    for lc in length_classes:
        plot_avg_dice_across_models(models, length_class=lc, spec=spec, output_dir=str(summary_output_base))

    # ---- 3) Noise vs AP ----
    resolved_noise_csv = _resolve_noise_csv(P, noise_csv)
    if resolved_noise_csv is not None and Path(resolved_noise_csv).exists():
        plot_ap_vs_noise(
            models,
            noise_csv=resolved_noise_csv,
            length_classes=length_classes,
            spec=spec,
            output_dir=str(summary_output_base),
        )
    else:
        if noise_csv is not None:
            print(f"noise_csv not found at: {Path(noise_csv)} -> skipping noise-vs-AP plot.")
        else:
            print(
                "noise_csv not provided and no canonical noise CSV found under "
                f"{P.test_metadata_root} -> skipping noise-vs-AP plot."
            )


# -----------------------------
# Task 3: COCO overlays
# -----------------------------
def task_coco_overlays(
    *,
    images_dir: Path,
    json_path: Path,
    output_dir: Path,
    image_ext: str,
    alpha: float,
    dpi: int,
) -> None:
    print(f"CWD: {os.getcwd()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    coco = load_coco(json_path)
    images_by_id, anns_by_image_id = index_coco_by_image(coco)

    for image_id, img_info in images_by_id.items():
        filename = img_info.get("file_name", "")
        if not filename.endswith(image_ext):
            continue

        image_path = images_dir / filename
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            continue

        bgr = cv2.imread(str(image_path))
        if bgr is None:
            print(f"Failed to read: {image_path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        anns = anns_by_image_id.get(image_id, [])
        out_path = output_dir / f"annotated_{Path(filename).name}"

        save_image_with_annotations(
            image_rgb=rgb,
            anns=anns,
            out_path=out_path,
            alpha=alpha,
            dpi=dpi,
        )
        print(f"Saved: {out_path}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main(
    task: str = "density_trends",
    *,
    # outputs root override (defaults to ./global_outputs)
    outputs_root: str | None = None,
    # density_trends params
    run: str = "evaluation_runner_test",
    density: str = "LOW_DENSITY",
    eval_tag: str = "density_eval",
    # binned params
    densities: list[str] | None = None,
    bin_size: int = 10,
    # model_comparison params
    runs: list[str] | None = None,
    report: str = "model_comparison",
    length_classes: list[str] | None = None,
    noise_csv: str | None = None,
    # coco_overlays params
    images_dir: str | None = None,
    json_path: str | None = None,
    out: str | None = None,
    image_ext: str = ".jpg",
    alpha: float = 0.5,
    dpi: int = 150,
) -> None:
    P = ProjectPaths.from_here(__file__, outputs_root=outputs_root or "global_outputs")
    P.ensure_outputs()

    if task == "density_trends":
        task_density_trends(
            P=P,
            run=run,
            selected_density=density,
            eval_tag=eval_tag,
        )
        return
    
    if task == "metrics_vs_object_count_binned":
        if densities is None:
            densities = ["LOW_DENSITY", "MID_DENSITY", "HIGH_DENSITY"]
        task_metrics_vs_object_count_binned(
            P=P,
            run=run,
            densities=densities,
            bin_size=bin_size,
            eval_tag=eval_tag,
        )
        return
    
    if task == "model_comparison":
        if runs is None:
            raise ValueError("task='model_comparison' requires runs=[...]")
        if length_classes is None:
            length_classes = ["long_objects", "short_objects"]

        task_model_comparison(
            P=P,
            report=report,
            runs=runs,
            length_classes=length_classes,
            noise_csv=noise_csv,
        )
        return

    if task == "coco_overlays":
        resolved_images_dir = Path(images_dir) if images_dir is not None else P.images_root("test")
        resolved_json_path = Path(json_path) if json_path is not None else P.coco_json("test")
        resolved_out = Path(out) if out is not None else P.misc_dir("coco_overlays")

        task_coco_overlays(
            images_dir=resolved_images_dir,
            json_path=resolved_json_path,
            output_dir=resolved_out,
            image_ext=image_ext,
            alpha=alpha,
            dpi=dpi,
        )
        return

    raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualization/report runner. Task must be explicit to avoid unintended output creation.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["density_trends", "metrics_vs_object_count_binned", "model_comparison", "coco_overlays"],
    )

    parser.add_argument("--outputs-root", default=None)

    parser.add_argument("--run", default="evaluation_runner_test")
    parser.add_argument("--density", default="LOW_DENSITY")
    parser.add_argument("--eval-tag", default="density_eval")
    parser.add_argument("--densities", nargs="*", default=None)
    parser.add_argument("--bin-size", type=int, default=10)

    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--report", default="model_comparison")
    parser.add_argument("--length-classes", nargs="*", default=None)
    parser.add_argument("--noise-csv", default=None)

    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--json-path", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--image-ext", default=".jpg")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--dpi", type=int, default=150)

    if len(sys.argv) == 1:
        parser.print_help()
        raise SystemExit(0)

    args = parser.parse_args()

    main(
        task=args.task,
        outputs_root=args.outputs_root,
        run=args.run,
        density=args.density,
        eval_tag=args.eval_tag,
        densities=args.densities,
        bin_size=args.bin_size,
        runs=args.runs,
        report=args.report,
        length_classes=args.length_classes,
        noise_csv=args.noise_csv,
        images_dir=args.images_dir,
        json_path=args.json_path,
        out=args.out,
        image_ext=args.image_ext,
        alpha=args.alpha,
        dpi=args.dpi,
    )