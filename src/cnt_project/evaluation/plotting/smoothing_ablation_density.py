from __future__ import annotations

from pathlib import Path

import pandas as pd

from cnt_project.visualizations.experiments.paper_figures.density_dice_plots import (
    plot_density_metric,
)
from cnt_project.evaluation.utils import (
    display_density_label,
)


SCORING_COLOR_MAP = {
    "EDT with smoothing": "#1f78b4",
    "EDT without smoothing": "#a6cee3",
    "EDT-SYM with smoothing": "#6baed6",
    "EDT-SYM without smoothing": "#c6dbef",
    "EDT-FULL with smoothing": "#08306b",
    "EDT-FULL without smoothing": "#4292c6",
    "SYM with smoothing": "#2171b5",
    "SYM without smoothing": "#9ecae1",
}



def save_smoothing_ablation_object_metric_figures(
    rows: list[dict],
    *,
    run_names: list[str],
    run_label_map: dict[str, str] | None,
    output_dir: Path,
    figure_formats: tuple[str, ...],
    figure_dpi: int,
) -> None:
    if not rows:
        print("No rows available for plotting.")
        return

    df = pd.DataFrame(rows)

    legend_name_map = {
        run_name: (run_label_map or {}).get(run_name, display_density_label(run_name))
        for run_name in run_names
    }

    plot_df = df.copy()
    plot_df["run_label"] = plot_df["run_name"].map(legend_name_map)

    labels = list(dict.fromkeys(legend_name_map.values()))
    color_map = {
        label: SCORING_COLOR_MAP.get(label, None)
        for label in labels
    }
    color_map = {k: v for k, v in color_map.items() if v is not None}

    metric_specs = [
        {
            "metric_column": "object_mAP",
            "ylabel": "Object mAP",
            "filename": "object_map_by_density",
        },
        {
            "metric_column": "object_mean_dice",
            "ylabel": "Object Mean Dice",
            "filename": "object_mean_dice_by_density",
        },
    ]

    density_order = ["Low", "Mid", "High"]

    for spec in metric_specs:
        metric_col = spec["metric_column"]
        metric_df = plot_df.dropna(subset=[metric_col]).copy()
        if metric_df.empty:
            continue

        for fmt in figure_formats:
            out_path = output_dir / f"{spec['filename']}.{fmt}"
            plot_density_metric(
                metric_df,
                color_map=color_map,
                legend_name_map={label: label for label in labels},
                metric_col=metric_col,
                ylabel=spec["ylabel"],
                experiment_col="run_label",
                order=density_order,
                figsize=(4.8, 2.9),
                ylim=(0, 1),
                legend_outside=True,
                save_path=out_path,
                save_dpi=figure_dpi,
                show=False,
            )
     # Per-model figures: compare with vs without smoothing for each model separately.
    for model_base in sorted(plot_df["model_base"].dropna().unique()):
        model_df = plot_df[plot_df["model_base"] == model_base].copy()
        if model_df.empty:
            continue

        model_dir = output_dir / "per_model" / str(model_base)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_run_names = list(dict.fromkeys(model_df["run_name"].tolist()))

        model_legend_name_map = {
            run_name: legend_name_map.get(run_name, display_density_label(run_name))
            for run_name in model_run_names
        }

        model_plot_df = model_df.copy()
        model_plot_df["run_label"] = model_plot_df["run_name"].map(model_legend_name_map)

        model_labels = list(dict.fromkeys(model_plot_df["run_label"].tolist()))

        model_color_map = {
            label: SCORING_COLOR_MAP.get(label, None)
            for label in model_labels
        }
        model_color_map = {k: v for k, v in model_color_map.items() if v is not None}

        for spec in metric_specs:
            metric_col = spec["metric_column"]
            metric_df = model_plot_df.dropna(subset=[metric_col]).copy()
            if metric_df.empty:
                continue

            for fmt in figure_formats:
                out_path = model_dir / f"{spec['filename']}.{fmt}"

                plot_density_metric(
                    metric_df,
                    color_map=model_color_map,
                    legend_name_map={label: label for label in model_labels},
                    metric_col=metric_col,
                    ylabel=spec["ylabel"],
                    experiment_col="run_label",
                    order=density_order,
                    figsize=(4.8, 2.9),
                    ylim=(0, 1),
                    legend_outside=False,
                    save_path=out_path,
                    save_dpi=figure_dpi,
                    show=False,
                )

