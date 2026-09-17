from __future__ import annotations

import os
from pathlib import Path
import string
from typing import Any, Dict, List, Sequence

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, FuncFormatter, FormatStrFormatter

# -------------------------------------------------------------
# Safe import for scienceplots style
# -------------------------------------------------------------
try:
    import scienceplots
    plt.style.use(["science", "no-latex"])
except Exception:
    pass


# -------------------------------------------------------------
# Tick formatting
# -------------------------------------------------------------
def format_tick_value(value, _tick_number):
    """Format tick labels by trimming trailing zeros while preserving integers."""
    if float(value).is_integer():
        return str(int(value))
    formatted = f"{value:.2f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def set_histogram_yticks(ax, feature, model_display=None):
    """Set feature-specific y-ticks for histogram readability."""
    ymin, ymax = ax.get_ylim()
    if ymax <= ymin:
        return

    # Special-case ticks for Nano1D histograms.
    if model_display == "Nano1D":
        if feature in ("orientation_angle", "orientation_angles"):
            ax.set_yticks(np.arange(0.02, 0.10 + 1e-9, 0.02))
            ax.yaxis.set_major_formatter(FuncFormatter(format_tick_value))
            return

        if feature == "line_density":
            ax.set_yticks(np.arange(1.5, 7.5 + 1e-9, 1.5))
            ax.yaxis.set_major_formatter(FuncFormatter(format_tick_value))
            return

        if feature == "length_um":
            ticks = np.arange(1.5, ymax + 1e-9, 1.5)
            if ticks.size == 0:
                ticks = np.array([ymax])
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(FuncFormatter(format_tick_value))
            return

    if feature == "line_density":
        ticks = np.arange(1.0, ymax + 1e-9, 1.0)
        if ticks.size == 0:
            ticks = np.array([ymax])
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(format_tick_value))
        return

    if feature == "length_um":
        ticks = np.arange(0.5, ymax + 1e-9, 0.5)
        if ticks.size == 0:
            ticks = np.array([ymax])
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(format_tick_value))
        return

    if ymin <= 0 <= ymax:
        start = ymax / 5.0
    else:
        start = ymin + (ymax - ymin) / 5.0

    ax.set_yticks(np.linspace(start, ymax, 5))
    ax.yaxis.set_major_formatter(FuncFormatter(format_tick_value))


# -------------------------------------------------------------
# Compute global ranges + bins across GT + all models
# -------------------------------------------------------------
def _compute_global_bins_and_ranges(
    all_results: Dict[str, Dict],
    features: Sequence[str],
    bins: int,
):
    feature_ranges = {}
    feature_bins = {}

    gt_metrics = all_results["ground_truth"]["metrics"]
    gt_line_density = np.array(all_results["ground_truth"]["line_density"]) / 5.0

    model_keys = [k for k in all_results.keys() if k != "ground_truth"]

    for feature in features:
        all_data = []

        # Add GT
        if feature == "line_density":
            all_data.append(gt_line_density)
        else:
            all_data.append(np.asarray(gt_metrics[feature]))

        # Add models
        for mk in model_keys:
            model_metrics = all_results[mk]["metrics"]
            model_ld = np.array(all_results[mk]["line_density"]) / 5.0

            data = model_ld if feature == "line_density" else np.asarray(model_metrics[feature])
            all_data.append(data)

        all_concat = np.concatenate([arr for arr in all_data if arr.size > 0])
        fmin, fmax = np.nanmin(all_concat), np.nanmax(all_concat)

        if not np.isfinite(fmin) or not np.isfinite(fmax):
            fmin, fmax = 0.0, 1.0

        if abs(fmax - fmin) < 1e-10:
            fmin -= 0.5
            fmax += 0.5

        feature_ranges[feature] = (float(fmin), float(fmax))
        feature_bins[feature] = np.linspace(fmin, fmax, bins + 1)

    return feature_ranges, feature_bins


# -------------------------------------------------------------
# Main plotting function
# -------------------------------------------------------------
def plot_features_grid_from_dict(all_results,
                                 features=["orientation_angles", "line_density", "length_um"],
                                 feature_labels=None,
                                 filename_base="features_grid_colored",
                                 save_folder=".",
                                 color_map=None,
                                 model_name_map=None,
                                 gt_color="#f0e68c",
                                 bins=60,
                                 hist_stat="density",
                                 gt_in_separate_row=False,
                                 font_sizes=None):
    """
    Plot histograms in a grid: rows = models, columns = features, with optional GT overlay
    or separate GT row at the bottom. Shared legend at the bottom.
    """
    os.makedirs(save_folder, exist_ok=True)
    plt.style.use(['science'])

    # Default font sizes if not provided
    if font_sizes is None:
        font_sizes = {
            "title": 30,
            "labels": 28,
            "ticks": 28,
            "legend": 30,
            "subplot_label": 28
        }

    # Order models (only those present)
    model_order = ["modelA", "modelB", "modelC", "modelD"]
    model_names = [m for m in model_order if m in all_results]
    n_models = len(model_names)
    n_features = len(features)
    
    if feature_labels is None:
        feature_labels = features

    # Extract GT metrics
    gt_metrics = all_results["ground_truth"]["metrics"]
    gt_line_density = (np.array(all_results["ground_truth"]["line_density"]) / 5).tolist()


    # Compute global ranges for each feature
    feature_data_ranges = {}
    feature_bin_edges = {}
    for feature in features:
        data_all = []
        if feature == "line_density":
            data_all.append(gt_line_density)
        else:
            data_all.append(gt_metrics[feature])
        for model_name in model_names:
            model_metrics = all_results[model_name]["metrics"]
            model_line_density = (np.array(all_results[model_name]["line_density"]) / 5).tolist()
            if feature == "line_density":
                data_all.append(model_line_density)
            else:
                data_all.append(model_metrics[feature])
        all_data = np.concatenate(data_all)
        fmin, fmax = np.nanmin(all_data), np.nanmax(all_data)
        if fmax - fmin < 1e-10:
            fmin -= 0.5
            fmax += 0.5
        feature_data_ranges[feature] = (fmin, fmax)
        feature_bin_edges[feature] = np.linspace(fmin, fmax, bins + 1)

    # Shared y-limits
    feature_ymax = {}
    for feature in features:
        max_vals = []
        if feature == "line_density":
            data_true = gt_line_density
        else:
            data_true = gt_metrics[feature]
        hist_counts, _ = np.histogram(data_true, bins=feature_bin_edges[feature], density=True)
        max_vals.append(np.max(hist_counts))
        for model_name in model_names:
            if model_name == "modelD":
                continue
            model_metrics = all_results[model_name]["metrics"]
            model_line_density = (np.array(all_results[model_name]["line_density"]) / 5).tolist()
            data_pred = model_line_density if feature == "line_density" else model_metrics[feature]
            hist_counts, _ = np.histogram(data_pred, bins=feature_bin_edges[feature], density=True)
            max_vals.append(np.max(hist_counts))  # This line was missing!
        feature_ymax[feature] = np.max(max_vals) * 1.05

    # Total rows
    total_rows = n_models + 1 if gt_in_separate_row else n_models
    
    # Optimize figure dimensions for better horizontal space utilization
    fig_width = 6 * n_features  # Increased width to match legend span
    fig_height = 3.2 * total_rows
    
    # Create figure with adjusted dimensions
    fig, axes = plt.subplots(total_rows, n_features, figsize=(fig_width, fig_height))
    
    if total_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_features == 1:
        axes = np.expand_dims(axes, axis=1)

    legend_handles = []
    legend_labels = []
    added_models = set()

    # Plot models
    for row_idx, model_name in enumerate(model_names):
        model_metrics = all_results[model_name]["metrics"]
        model_line_density = (np.array(all_results[model_name]["line_density"]) / 5).tolist()
        display_name = model_name_map.get(model_name, model_name) if model_name_map else model_name
        pred_color = color_map.get(display_name, "#4169e1") if color_map else "#4169e1"

        for col_idx, feature in enumerate(features):
            ax = axes[row_idx, col_idx]
            if feature == "line_density":
                data_pred = model_line_density
                data_true = gt_line_density if not gt_in_separate_row else None
            else:
                data_pred = model_metrics[feature]
                data_true = gt_metrics[feature] if not gt_in_separate_row else None

            bin_edges = feature_bin_edges[feature]
            xmin, xmax = feature_data_ranges[feature]

            if data_true is not None:
                sns.histplot(data_true, bins=bin_edges, color=gt_color, stat=hist_stat, alpha=0.7, ax=ax,
                             edgecolor='black', linewidth=0.5)
                sns.kdeplot(data_true, color=gt_color, ax=ax, linewidth=1.0, alpha=1.0, bw_adjust=0.8)
                ax.axvline(np.mean(data_true), color=gt_color, linestyle='--', linewidth=1.2)

            sns.histplot(data_pred, bins=bin_edges, color=pred_color, stat=hist_stat, alpha=0.7, ax=ax,
                         edgecolor='black', linewidth=0.5)
            sns.kdeplot(data_pred, color=pred_color, ax=ax, linewidth=1.0, alpha=1.0, bw_adjust=0.8)
            ax.axvline(np.mean(data_pred), color=pred_color, linestyle='--', linewidth=1.2)

            ax.set_xlabel("", fontsize=font_sizes["labels"])
            ax.set_ylabel("", fontsize=font_sizes["labels"])
            ax.tick_params(axis="both", labelsize=font_sizes["ticks"])
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(0, feature_ymax[feature])
            
            # Add the same x-tick formatting as in the first function
            if feature == "line_density":
                # ax.xaxis.set_major_locator(MaxNLocator(integer=False))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            elif feature == "orientation_angles" or feature == "orientation_angle":
                ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
            else:
                ax.set_xticks(np.linspace(xmin, xmax, 6))
                if feature == "length_um":
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            
            if row_idx == 0:
                ax.set_title(feature_labels[col_idx] if feature_labels else feature,
                             fontsize=font_sizes["title"], weight="bold", pad=15)

            if display_name not in added_models:
                pred_patch = Patch(facecolor=pred_color, edgecolor="black", label=display_name)
                legend_handles.append(pred_patch)
                legend_labels.append(display_name)
                added_models.add(display_name)

            if model_name == "modelD":
                hist_counts, _ = np.histogram(data_pred, bins=bin_edges, density=True)
                ylim_max = np.max(hist_counts) * 1.05
                ax.set_ylim(0, ylim_max)
            else:
                ax.set_ylim(0, feature_ymax[feature])
            set_histogram_yticks(ax, feature, display_name)
            subplot_label = f"{string.ascii_uppercase[row_idx]}{col_idx+1}"
            ax.text(0.88, 0.95, subplot_label, transform=ax.transAxes,
                    ha="left", va="top", fontsize=font_sizes["subplot_label"], weight="bold")

    # GT separate row
    if gt_in_separate_row:
        for col_idx, feature in enumerate(features):
            ax = axes[-1, col_idx]
            data_true = gt_line_density if feature == "line_density" else gt_metrics[feature]
            bin_edges = feature_bin_edges[feature]
            xmin, xmax = feature_data_ranges[feature]

            sns.histplot(data_true, bins=bin_edges, color=gt_color, stat=hist_stat, alpha=0.7, ax=ax,
                         edgecolor='black', linewidth=0.5)
            sns.kdeplot(data_true, color=gt_color, ax=ax, linewidth=1.0, alpha=1.0, bw_adjust=0.8)
            ax.axvline(np.mean(data_true), color=gt_color, linestyle='--', linewidth=1.2)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(0, feature_ymax[feature])
            ax.tick_params(axis="both", labelsize=font_sizes["ticks"])
            
            # Add the same x-tick formatting as in the first function
            if feature == "line_density":
                ax.xaxis.set_major_locator(MaxNLocator(integer=False))
            elif feature == "orientation_angles" or feature == "orientation_angle":
                ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
            else:
                ax.set_xticks(np.linspace(xmin, xmax, 6))
                if feature == "length_um":
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            
            set_histogram_yticks(ax, feature)

            subplot_label = f"{string.ascii_uppercase[total_rows-1]}{col_idx+1}"
            ax.text(0.92, 0.95, subplot_label, transform=ax.transAxes,
                    ha="left", va="top", fontsize=font_sizes["subplot_label"], weight="bold")

    # Shared legend - arrange in two rows to match histogram width
    gt_patch = Patch(facecolor=gt_color, edgecolor="black", label="GT")
    legend_handles.insert(0, gt_patch)
    legend_labels.insert(0, "GT")
    
    # Calculate optimal number of columns for legend to match histogram width
    # For 5 items (GT + 4 models), we'll use 3 columns in first row and 2 in second
    ncol_legend = 5  # First row will have 3 items, second row will have 2
    
    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=ncol_legend,
               frameon=False, bbox_to_anchor=(0.5, -0.08), fontsize=font_sizes["legend"])
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.02)  # Increase bottom margin to accommodate legend
    
    save_path = os.path.join(save_folder, f"{filename_base}_grid_colored_poly_new_line_density.svg")
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.show()



# plot single histogram
def save_individual_histograms_per_model_feature(
    all_results: Dict[str, Dict[str, Any]],
    features: Sequence[str],
    feature_labels: Sequence[str],
    save_folder: Path,
    color_map: Dict[str, str],
    model_name_map: Dict[str, str],
    bins: int = 60,
    hist_stat: str = "density",
    figsize: tuple = (2.0, 1.8),
    gt_color: str = "#f0e68c",
    filename_prefix: str = "hist"
) -> Dict[str, Path]:

    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import MaxNLocator, FuncFormatter, FormatStrFormatter

    # Style: scienceplots + controlled font sizes (frozen per-figure)
    try:
        import scienceplots  # noqa: F401
        base_styles = ["science", "no-latex"]
        plt.style.use(base_styles)
    except Exception:
        base_styles = []

    save_folder.mkdir(parents=True, exist_ok=True)
    out_paths = {}

    # Fixed, per your spec
    YLIMS = {
        "orientation_angle": {
            "Nanovision": (0, 0.065),
            "MaskRCNN":   (0, 0.065),
            "WormSwin":   (0, 0.065),
            "Nano1D":     (0, 0.3),
        },
        "line_density": {
            "Nanovision": (0, 5.5),
            "MaskRCNN":   (0, 5.5),
            "WormSwin":   (0, 5.5),
            "Nano1D":     (0, 8.8),
        },
        "length_um": {
            "Nanovision": (0, 3.8),
            "MaskRCNN":   (0, 3.8),
            "WormSwin":   (0, 3.8),
            "Nano1D":     (0, 17),
        }
    }

    def _vals(entry, feat):
        if feat == "line_density":
            return np.asarray(entry["line_density"], float) / 5.0
        return np.asarray(entry["metrics"][feat], float)

    # GT (shared)
    gt_vals = {feat: _vals(all_results["ground_truth"], feat) for feat in features}

    # Common bin edges/xrange from GT + all models
    feature_bin_edges, feature_xrange = {}, {}
    model_keys = [k for k in all_results if k != "ground_truth"]

    for feat in features:
        all_data = [gt_vals[feat]] + [_vals(all_results[k], feat) for k in model_keys]
        concat = np.concatenate(all_data)
        fmin, fmax = float(np.min(concat)), float(np.max(concat))
        if abs(fmax - fmin) < 1e-12:
            fmin -= 0.5; fmax += 0.5
        feature_xrange[feat] = (fmin, fmax)
        feature_bin_edges[feat] = np.linspace(fmin, fmax, bins + 1)

    # Frozen font sizes (per your small panel look)
    rc_lock = {
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.constrained_layout.use": False,  # avoid auto changes
    }

    # MAIN LOOP
    for internal_key in model_keys:
        model_display = model_name_map[internal_key]
        model_color = color_map[model_display]

        for feat, feat_label in zip(features, feature_labels):
            model_vals = _vals(all_results[internal_key], feat)
            gt_vals_f = gt_vals[feat]
            xmin, xmax = feature_xrange[feat]
            bin_edges = feature_bin_edges[feat]

            # Per-figure rc_context to prevent drift AND avoid ‘tight’ changing viewBox
            with plt.rc_context(rc_lock):
                fig, ax = plt.subplots()
                fig.set_size_inches(*figsize)         # hard-assert size
                fig.set_constrained_layout(False)

                # GT
                sns.histplot(
                    gt_vals_f, bins=bin_edges, color=gt_color,
                    stat=hist_stat, alpha=0.7, edgecolor="black",
                    linewidth=0.5, ax=ax
                )
                sns.kdeplot(gt_vals_f, color=gt_color, linewidth=1.0, alpha=1.0, bw_adjust=0.8, ax=ax)
                ax.axvline(np.mean(gt_vals_f), color=gt_color, linestyle="--", linewidth=1.2)

                # Model
                sns.histplot(
                    model_vals, bins=bin_edges, color=model_color,
                    stat=hist_stat, alpha=0.7, edgecolor="black",
                    linewidth=0.5, ax=ax
                )
                sns.kdeplot(model_vals, color=model_color, linewidth=1.0, alpha=1.0, bw_adjust=0.8, ax=ax)
                ax.axvline(np.mean(model_vals), color=model_color, linestyle="--", linewidth=1.2)

                # Axes
                ax.set_xlim(xmin, xmax)
                if feat not in YLIMS or model_display not in YLIMS[feat]:
                    raise ValueError(f"Missing ylim mapping for {feat} / {model_display}")
                ax.set_ylim(*YLIMS[feat][model_display])

                # X ticks
                if feat in ("orientation_angle", "orientation_angles"):
                    ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
                elif feat == "line_density":
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
                else:
                    ax.set_xticks(np.linspace(xmin, xmax, 6))
                    if feat == "length_um":
                        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

                # Y ticks
                set_histogram_yticks(ax, feat, model_display)

                # Labels
                ax.set_xlabel(feat_label)
                ax.set_ylabel("Density" if hist_stat == "density" else "Count")
                ax.tick_params(direction="in")
                ax.xaxis.set_ticks_position("both")
                ax.yaxis.set_ticks_position("both")

                # Save with fixed viewBox (no tight)
                out_path = save_folder / f"{filename_prefix}_{model_display}_{feat}.svg"
                fig.savefig(out_path, format="svg", bbox_inches=None, pad_inches=0)
                plt.close(fig)

            out_paths[f"{model_display}|{feat}"] = out_path

    return out_paths
