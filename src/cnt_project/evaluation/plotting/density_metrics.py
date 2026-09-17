from __future__ import annotations

from pathlib import Path

import pandas as pd

from cnt_project.visualizations.experiments.paper_figures.density_dice_plots import (
    plot_density_metric,
)
from cnt_project.evaluation.utils import display_density_label

SCORING_COLOR_MAP = {
    "SYM": "#a6cee3",
    "EDT": "#1f78b4",
    "EDT-SYM": "#6baed6",
    "EDT-FULL": "#08306b",
    "NanoVision": "#4169e1",
    "MaskRCNN": "#e34234",
    "WormSwin": "#9966cc",
}


EXTRA_BLUE_SHADES = [
    "#9ecae1",
    "#4292c6",
    "#2171b5",
    "#084594",
    "#6baed6",
    "#c6dbef",
]

VALID_FIGURE_FORMATS = {"png", "svg"}


def _metric_plot_specs() -> list[dict[str, str]]:
    return [
        {
            "metric_column": "pixel_dice",
            "ylabel": "Pixel DICE Score",
            "filename": "pixel_dice_by_density",
        },
        {
            "metric_column": "pixel_precision",
            "ylabel": "Pixel Precision",
            "filename": "pixel_precision_by_density",
        },
        {
            "metric_column": "pixel_recall",
            "ylabel": "Pixel Recall",
            "filename": "pixel_recall_by_density",
        },
        {
            "metric_column": "pixel_f1",
            "ylabel": "Pixel F1 Score",
            "filename": "pixel_f1_by_density",
        },
    ]


def normalize_figure_formats(formats: list[str] | None) -> tuple[str, ...]:
    if not formats:
        return ("png",)
    normalized = tuple(dict.fromkeys(str(fmt).strip().lower() for fmt in formats if str(fmt).strip()))
    unknown = set(normalized) - VALID_FIGURE_FORMATS
    if unknown:
        raise ValueError(f"Unsupported figure formats: {sorted(unknown)}")
    return normalized


def _resolve_run_display_specs(
    run_names: list[str],
    run_label_map: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    run_label_map = run_label_map or {}

    run_to_label = {
        run_name: run_label_map.get(run_name, run_name)
        for run_name in run_names
    }

    labels = list(run_to_label.values())
    if len(labels) != len(set(labels)):
        raise ValueError(
            "Each run must map to a unique display label for plotting. "
            f"Received labels: {labels}"
        )

    label_to_color: dict[str, str] = {}
    extra_idx = 0
    for label in run_to_label.values():
        if label in label_to_color:
            continue
        if label in SCORING_COLOR_MAP:
            label_to_color[label] = SCORING_COLOR_MAP[label]
        else:
            label_to_color[label] = EXTRA_BLUE_SHADES[extra_idx % len(EXTRA_BLUE_SHADES)]
            extra_idx += 1

    run_to_color = {
        run_name: label_to_color[label]
        for run_name, label in run_to_label.items()
    }
    return run_to_label, run_to_color


def save_density_metric_figures(
    pixel_metric_rows: list[dict],
    *,
    selected_densities: list[str],
    run_names: list[str],
    run_label_map: dict[str, str] | None,
    output_dir: Path,
    figure_formats: tuple[str, ...],
    figure_dpi: int,
    write_long_csv: bool,
) -> None:
    if not pixel_metric_rows:
        return

    plot_df = pd.DataFrame(pixel_metric_rows)
    if write_long_csv:
        plot_df.to_csv(output_dir / "pixel_metrics_by_density_long.csv", index=False)

    run_to_label, color_map = _resolve_run_display_specs(run_names, run_label_map)
    legend_name_map = {run_name: run_to_label[run_name] for run_name in run_names}
    density_order = [display_density_label(density) for density in selected_densities]

    for spec in _metric_plot_specs():
        metric_col = spec["metric_column"]
        if metric_col not in plot_df.columns:
            continue

        metric_df = plot_df.dropna(subset=[metric_col]).copy()
        if metric_df.empty:
            continue

        for figure_format in figure_formats:
            out_path = output_dir / f"{spec['filename']}.{figure_format}"
            plot_density_metric(
                metric_df,
                color_map,
                legend_name_map,
                metric_col=metric_col,
                ylabel=spec["ylabel"],
                experiment_col="run_name",
                order=density_order,
                figsize=(4.5, 2.8),
                ylim=(0, 1),
                save_path=out_path,
                save_dpi=figure_dpi,
                show=False,
            )

