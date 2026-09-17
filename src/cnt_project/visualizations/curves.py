from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd


def plot_series_vs_x(
    *,
    x_values: Sequence[float],
    series: dict[str, Sequence[float]],
    xlabel: str,
    ylabel: str,
    colors: dict[str, str] | None = None,
    dashes: dict[str, tuple] | None = None,
    legend_map: dict[str, str] | None = None,
    legend_title: str = "Model",
    legend_loc: str | None = None,
    linewidth: float | None = None,
    figsize: tuple[float, float] = (3.3, 2.5),
    ylim: tuple[float, float] | None = None,
    xticks: Sequence[float] | None = None,
    grid: bool = False,
    save_path: str | Path | None = None,
    save_format: str | None = None,
    dpi: int | None = None,
    show: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    for series_name, y_values in series.items():
        plot_kwargs = {
            "label": legend_map.get(series_name, series_name) if legend_map else series_name,
        }
        if colors is not None:
            plot_kwargs["color"] = colors.get(series_name, "#000000")
        if dashes is not None and series_name in dashes:
            plot_kwargs["dashes"] = dashes[series_name]
        if linewidth is not None:
            plot_kwargs["linewidth"] = linewidth

        ax.plot(x_values, y_values, **plot_kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(grid)

    if ylim is not None:
        ax.set_ylim(*ylim)

    if xticks is not None:
        ax.set_xticks(xticks)

    if legend_loc is not None:
        ax.legend(title=legend_title, loc=legend_loc)
    else:
        ax.legend(title=legend_title)

    fig.tight_layout()

    if save_path is not None:
        save_kwargs = {"bbox_inches": "tight"}
        if save_format is not None:
            save_kwargs["format"] = save_format
        if dpi is not None:
            save_kwargs["dpi"] = dpi
        fig.savefig(Path(save_path), **save_kwargs)

    if show:
        plt.show()


def plot_metric_vs_iou(
    *,
    mean_metrics_df: pd.DataFrame,
    color_map: dict[str, str],
    legend_map: dict[str, str],
    metric_col: str,
    experiment_col: str,
    iou_col: str = "IoU_thresh",
    xlabel: str = "IoU Threshold",
    ylabel: str = "Metric",
    figsize: tuple[float, float] = (3.3, 2.5),
    yticks: Sequence[float] | None = None,
    legend_title: str = "Models",
    legend_loc: str | None = None,
    linewidth: float | None = None,
    grid: bool = False,
    save_path: str | Path | None = None,
    dpi: int | None = None,
    show: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    experiment_values = mean_metrics_df[experiment_col].dropna().unique().tolist()

    for experiment in experiment_values:
        data = mean_metrics_df[mean_metrics_df[experiment_col] == experiment]
        plot_kwargs = {
            "label": legend_map.get(experiment, experiment),
            "color": color_map.get(experiment, "#000000"),
        }
        if linewidth is not None:
            plot_kwargs["linewidth"] = linewidth

        ax.plot(data[iou_col], data[metric_col], **plot_kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if yticks is not None:
        ax.set_yticks(yticks)

    ax.grid(grid)

    if legend_loc is not None:
        ax.legend(title=legend_title, loc=legend_loc)
    else:
        ax.legend(title=legend_title)

    fig.tight_layout()

    if save_path is not None:
        save_kwargs = {"bbox_inches": "tight"}
        if dpi is not None:
            save_kwargs["dpi"] = dpi
        fig.savefig(Path(save_path), **save_kwargs)

    if show:
        plt.show()


def prepare_metric_plot_dataframe(
    mean_metrics_df: pd.DataFrame,
    *,
    experiment_order: list[str] | None = None,
    experiment_col: str,
    iou_col: str = "IoU_thresh",
) -> pd.DataFrame:
    out = mean_metrics_df.copy()

    if experiment_order is not None:
        out[experiment_col] = pd.Categorical(
            out[experiment_col],
            categories=experiment_order,
            ordered=True,
        )
        out = out.sort_values([experiment_col, iou_col])

    return out
