from __future__ import annotations

from pathlib import Path
import string
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator

from .styles import apply_style


def _hide_first_tick(value: float, tick_number: int) -> str:
    if tick_number == 0:
        return ""
    if float(value).is_integer():
        return str(int(value))
    formatted = f"{value:.2f}".rstrip("0").rstrip(".")
    return formatted


def _restrict_orientation_to_90(angles: np.ndarray) -> np.ndarray:
    angles = np.where(angles < -90, angles + 180, angles)
    angles = np.where(angles > 90, angles - 180, angles)
    return angles


def _clean_feature_values(feature: str, values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if feature == "orientation_angle":
        arr = _restrict_orientation_to_90(arr)
    return arr


def _extract_feature_data(result: dict[str, Any], feature: str) -> np.ndarray:
    if feature == "line_density":
        return _clean_feature_values(feature, result["line_density"])
    return _clean_feature_values(feature, result["metrics"][feature])


def _prepare_ranges_and_bins(
    all_results: dict[str, Any],
    model_names: list[str],
    features: list[str],
    bins: int,
) -> tuple[dict[str, tuple[float, float]], dict[str, np.ndarray]]:
    feature_ranges: dict[str, tuple[float, float]] = {}
    feature_bin_edges: dict[str, np.ndarray] = {}

    for feature in features:
        all_feature_values = [_extract_feature_data(all_results["ground_truth"], feature)]
        for model_name in model_names:
            all_feature_values.append(_extract_feature_data(all_results[model_name], feature))

        non_empty = [v for v in all_feature_values if len(v) > 0]
        if not non_empty:
            fmin, fmax = 0.0, 1.0
        else:
            merged = np.concatenate(non_empty)
            fmin, fmax = float(np.nanmin(merged)), float(np.nanmax(merged))
            if abs(fmax - fmin) < 1e-12:
                fmin -= 0.5
                fmax += 0.5

        feature_ranges[feature] = (fmin, fmax)
        feature_bin_edges[feature] = np.linspace(fmin, fmax, bins + 1)

    return feature_ranges, feature_bin_edges


def _compute_shared_ymax(
    all_results: dict[str, Any],
    model_names: list[str],
    features: list[str],
    feature_bin_edges: dict[str, np.ndarray],
    skip_y_sharing_for: set[str] | None = None,
) -> dict[str, float]:
    skip_y_sharing_for = skip_y_sharing_for or set()
    feature_ymax: dict[str, float] = {}

    for feature in features:
        max_vals = []

        gt_data = _extract_feature_data(all_results["ground_truth"], feature)
        if len(gt_data) > 0:
            hist_counts, _ = np.histogram(gt_data, bins=feature_bin_edges[feature], density=True)
            if len(hist_counts) > 0:
                max_vals.append(float(np.max(hist_counts)))

        for model_name in model_names:
            if model_name in skip_y_sharing_for:
                continue
            data = _extract_feature_data(all_results[model_name], feature)
            if len(data) == 0:
                continue
            hist_counts, _ = np.histogram(data, bins=feature_bin_edges[feature], density=True)
            if len(hist_counts) > 0:
                max_vals.append(float(np.max(hist_counts)))

        feature_ymax[feature] = max(max_vals) * 1.05 if max_vals else 1.0

    return feature_ymax


def plot_feature_grid(
    all_results: dict[str, Any],
    *,
    features: list[str],
    feature_labels: list[str] | None = None,
    save_path: str | Path | None = None,
    color_map: dict[str, str] | None = None,
    model_name_map: dict[str, str] | None = None,
    gt_color: str = "#f0e68c",
    bins: int = 60,
    hist_stat: str = "density",
    orientation: str = "models_as_rows",
    include_gt_overlay: bool = True,
    include_gt_separate_axis: bool = False,
    skip_shared_y_for_models: set[str] | None = None,
    font_sizes: dict[str, int] | None = None,
    use_science_style: bool = True,
    use_latex: bool = False,
    style_overrides: dict[str, Any] | None = None,
    model_order: list[str] | None = None,
) -> None:
    import seaborn as sns

    if use_science_style:
        apply_style(style="science", no_latex=not use_latex)
    else:
        plt.rcdefaults()
    if style_overrides:
        plt.rcParams.update(style_overrides)

    if font_sizes is None:
        font_sizes = {
            "title": 16,
            "labels": 14,
            "ticks": 12,
            "legend": 13,
            "subplot_label": 12,
        }

    if feature_labels is None:
        feature_labels = features

    default_model_order = [
        "modelA",
        "modelB",
        "modelC",
        "modelD",
        "nanovision",
        "detectron2_trial_1",
        "wormswin",
        "nano1D",
    ]
    effective_model_order = model_order or default_model_order
    model_names = [m for m in effective_model_order if m in all_results]

    if not model_names:
        raise ValueError("No model results found in all_results.")

    feature_ranges, feature_bin_edges = _prepare_ranges_and_bins(
        all_results=all_results,
        model_names=model_names,
        features=features,
        bins=bins,
    )

    feature_ymax = _compute_shared_ymax(
        all_results=all_results,
        model_names=model_names,
        features=features,
        feature_bin_edges=feature_bin_edges,
        skip_y_sharing_for=skip_shared_y_for_models,
    )

    if orientation not in {"models_as_rows", "features_as_rows"}:
        raise ValueError("orientation must be 'models_as_rows' or 'features_as_rows'.")

    if orientation == "models_as_rows":
        n_rows = len(model_names) + (1 if include_gt_separate_axis else 0)
        n_cols = len(features)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 3.2 * n_rows))
    else:
        n_rows = len(features)
        n_cols = len(model_names) + (1 if include_gt_separate_axis else 0)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.2 * n_rows))

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    added_models: set[str] = set()

    def _format_axis(ax, feature: str, xmin: float, xmax: float) -> None:
        ax.set_xlim(xmin, xmax)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_formatter(FuncFormatter(_hide_first_tick))
        ax.tick_params(axis="both", labelsize=font_sizes["ticks"])

        if feature == "line_density":
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        elif feature == "orientation_angle":
            ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
        else:
            ax.set_xticks(np.linspace(xmin, xmax, 6))
            if feature in {"length_um", "width_um"}:
                ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    if orientation == "models_as_rows":
        for row_idx, model_name in enumerate(model_names):
            display_name = model_name_map.get(model_name, model_name) if model_name_map else model_name
            pred_color = color_map.get(display_name, "#4169e1") if color_map else "#4169e1"

            for col_idx, feature in enumerate(features):
                ax = axes[row_idx, col_idx]
                data_pred = _extract_feature_data(all_results[model_name], feature)
                data_true = _extract_feature_data(all_results["ground_truth"], feature)

                bin_edges = feature_bin_edges[feature]
                xmin, xmax = feature_ranges[feature]

                if include_gt_overlay and len(data_true) > 0:
                    sns.histplot(data_true, bins=bin_edges, color=gt_color, stat=hist_stat, alpha=0.7, ax=ax, edgecolor="black", linewidth=0.5)
                    if len(data_true) > 1:
                        sns.kdeplot(data_true, color=gt_color, ax=ax, linewidth=1.0, bw_adjust=0.8)
                    ax.axvline(float(np.mean(data_true)), color=gt_color, linestyle="--", linewidth=1.2)

                if len(data_pred) > 0:
                    sns.histplot(data_pred, bins=bin_edges, color=pred_color, stat=hist_stat, alpha=0.7, ax=ax, edgecolor="black", linewidth=0.5)
                    if len(data_pred) > 1:
                        sns.kdeplot(data_pred, color=pred_color, ax=ax, linewidth=1.0, bw_adjust=0.8)
                    ax.axvline(float(np.mean(data_pred)), color=pred_color, linestyle="--", linewidth=1.2)

                ax.set_xlabel("")
                ax.set_ylabel("")

                if skip_shared_y_for_models and model_name in skip_shared_y_for_models and len(data_pred) > 0:
                    hist_counts, _ = np.histogram(data_pred, bins=bin_edges, density=True)
                    ax.set_ylim(0, max(1e-9, float(np.max(hist_counts)) * 1.05))
                else:
                    ax.set_ylim(0, feature_ymax[feature])

                _format_axis(ax, feature, xmin, xmax)

                if row_idx == 0:
                    ax.set_title(feature_labels[col_idx], fontsize=font_sizes["title"], weight="bold", pad=15)

                subplot_label = f"{string.ascii_uppercase[row_idx]}{col_idx + 1}"
                ax.text(0.88, 0.95, subplot_label, transform=ax.transAxes, ha="left", va="top", fontsize=font_sizes["subplot_label"], weight="bold")

                if display_name not in added_models:
                    legend_handles.append(Patch(facecolor=pred_color, edgecolor="black", label=display_name))
                    legend_labels.append(display_name)
                    added_models.add(display_name)

        if include_gt_separate_axis:
            gt_row = len(model_names)
            for col_idx, feature in enumerate(features):
                ax = axes[gt_row, col_idx]
                data_true = _extract_feature_data(all_results["ground_truth"], feature)
                bin_edges = feature_bin_edges[feature]
                xmin, xmax = feature_ranges[feature]

                if len(data_true) > 0:
                    sns.histplot(data_true, bins=bin_edges, color=gt_color, stat=hist_stat, alpha=0.7, ax=ax, edgecolor="black", linewidth=0.5)
                    if len(data_true) > 1:
                        sns.kdeplot(data_true, color=gt_color, ax=ax, linewidth=1.0, bw_adjust=0.8)
                    ax.axvline(float(np.mean(data_true)), color=gt_color, linestyle="--", linewidth=1.2)

                ax.set_ylim(0, feature_ymax[feature])
                _format_axis(ax, feature, xmin, xmax)

                subplot_label = f"{string.ascii_uppercase[gt_row]}{col_idx + 1}"
                ax.text(0.92, 0.95, subplot_label, transform=ax.transAxes, ha="left", va="top", fontsize=font_sizes["subplot_label"], weight="bold")

    else:
        total_cols = len(model_names) + (1 if include_gt_separate_axis else 0)

        for row_idx, feature in enumerate(features):
            data_true = _extract_feature_data(all_results["ground_truth"], feature)
            bin_edges = feature_bin_edges[feature]
            xmin, xmax = feature_ranges[feature]

            for col_idx, model_name in enumerate(model_names):
                ax = axes[row_idx, col_idx]
                display_name = model_name_map.get(model_name, model_name) if model_name_map else model_name
                pred_color = color_map.get(display_name, "#4169e1") if color_map else "#4169e1"
                data_pred = _extract_feature_data(all_results[model_name], feature)

                if include_gt_overlay and len(data_true) > 0:
                    sns.histplot(data_true, bins=bin_edges, color=gt_color, stat=hist_stat, alpha=0.7, ax=ax, edgecolor="black", linewidth=0.5)
                    if len(data_true) > 1:
                        sns.kdeplot(data_true, color=gt_color, ax=ax, linewidth=1.0, bw_adjust=0.8)
                    ax.axvline(float(np.mean(data_true)), color=gt_color, linestyle="--", linewidth=1.2)

                if len(data_pred) > 0:
                    sns.histplot(data_pred, bins=bin_edges, color=pred_color, stat=hist_stat, alpha=0.7, ax=ax, edgecolor="black", linewidth=0.5)
                    if len(data_pred) > 1:
                        sns.kdeplot(data_pred, color=pred_color, ax=ax, linewidth=1.0, bw_adjust=0.8)
                    ax.axvline(float(np.mean(data_pred)), color=pred_color, linestyle="--", linewidth=1.2)

                ax.set_xlabel("")
                ax.set_ylabel("")

                if skip_shared_y_for_models and model_name in skip_shared_y_for_models and len(data_pred) > 0:
                    hist_counts, _ = np.histogram(data_pred, bins=bin_edges, density=True)
                    ax.set_ylim(0, max(1e-9, float(np.max(hist_counts)) * 1.05))
                else:
                    ax.set_ylim(0, feature_ymax[feature])

                _format_axis(ax, feature, xmin, xmax)

                subplot_label = f"{string.ascii_uppercase[col_idx]}{row_idx + 1}"
                ax.text(0.88, 0.95, subplot_label, transform=ax.transAxes, ha="left", va="top", fontsize=font_sizes["subplot_label"], weight="bold")

                if display_name not in added_models:
                    legend_handles.append(Patch(facecolor=pred_color, edgecolor="black", label=display_name))
                    legend_labels.append(display_name)
                    added_models.add(display_name)

            if include_gt_separate_axis:
                ax = axes[row_idx, -1]
                if len(data_true) > 0:
                    sns.histplot(data_true, bins=bin_edges, color=gt_color, stat=hist_stat, alpha=0.7, ax=ax, edgecolor="black", linewidth=0.5)
                    if len(data_true) > 1:
                        sns.kdeplot(data_true, color=gt_color, ax=ax, linewidth=1.0, bw_adjust=0.8)
                    ax.axvline(float(np.mean(data_true)), color=gt_color, linestyle="--", linewidth=1.2)

                ax.set_ylim(0, feature_ymax[feature])
                _format_axis(ax, feature, xmin, xmax)

                subplot_label = f"{string.ascii_uppercase[total_cols - 1]}{row_idx + 1}"
                ax.text(0.88, 0.95, subplot_label, transform=ax.transAxes, ha="left", va="top", fontsize=font_sizes["subplot_label"], weight="bold")

            mid_col = min(total_cols // 2, axes.shape[1] - 1)
            panel_letter = f"({string.ascii_lowercase[row_idx]})"
            axes[row_idx, mid_col].set_xlabel(f"{panel_letter} {feature_labels[row_idx]}", fontsize=font_sizes["title"], weight="bold", labelpad=10)

    gt_patch = Patch(facecolor=gt_color, edgecolor="black", label="GT")
    legend_handles.insert(0, gt_patch)
    legend_labels.insert(0, "GT")

    if orientation == "models_as_rows":
        fig.legend(legend_handles, legend_labels, loc="lower center", ncol=len(legend_labels), frameon=False, bbox_to_anchor=(0.5, -0.08), fontsize=font_sizes["legend"])
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.14)
    else:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=len(legend_labels), frameon=False, bbox_to_anchor=(0.5, 1.03), fontsize=font_sizes["legend"])
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format=save_path.suffix.lstrip(".") or None, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
