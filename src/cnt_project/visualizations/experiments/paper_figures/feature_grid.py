
# src/plotting/feature_grid.py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter, FormatStrFormatter
from matplotlib.patches import Patch
import string

from .constants import COLOR_MAP

def _hide_first_tick(value, tick_number):
    if tick_number == 0:
        return ""
    if float(value).is_integer():
        return str(int(value))
    s = f"{value:.2f}"
    return s.rstrip('0').rstrip('.') if '.' in s else s

def _bin_edges(data, bins):
    mn, mx = float(np.nanmin(data)), float(np.nanmax(data))
    if mx - mn < 1e-10:
        mn -= 0.5; mx += 0.5
    return np.linspace(mn, mx, bins + 1), (mn, mx)

def plot_features_grid_from_dict(
    all_results: dict,
    features=('orientation_angle', 'line_density', 'length_um'),
    feature_labels=(r"Orientation ($^\circ$)", r"Line Density (CNTs/$\mu$m)", r"Length ($\mu$m)"),
    filename_base="features_grid",
    save_folder=".",
    color_map=None,
    model_name_map=None,
    gt_color="#f0e68c",
    bins=60,
    hist_stat="density",
    gt_in_separate_row=False,
    font_sizes=None
):
    if color_map is None:
        color_map = COLOR_MAP
    if font_sizes is None:
        font_sizes = {"title": 18, "labels": 14, "ticks": 12, "legend": 14, "subplot_label": 14}

    model_order = ['modelA', 'modelB', 'modelC', 'modelD']
    model_names = [m for m in model_order if m in all_results]

    gt_metrics = all_results['ground_truth']['metrics']
    gt_line_density = (np.array(all_results['ground_truth']['line_density']) / 5.0).tolist()

    feature_bins = {}
    feature_ranges = {}
    for f in features:
        pieces = [gt_line_density if f == 'line_density' else gt_metrics[f]]
        for m in model_names:
            mm = all_results[m]['metrics']
            ld = (np.array(all_results[m]['line_density']) / 5.0).tolist()
            pieces.append(ld if f == 'line_density' else mm[f])
        all_data = np.concatenate(pieces)
        be, rng = _bin_edges(all_data, int(bins))
        feature_bins[f] = be; feature_ranges[f] = rng

    feature_ymax = {}
    for f in features:
        def hist_max(x):
            h, _ = np.histogram(x, bins=feature_bins[f], density=True)
            return float(np.max(h)) if h.size else 0.0
        maxima = [hist_max(gt_line_density if f == 'line_density' else gt_metrics[f])]
        for m in model_names:
            mm = all_results[m]['metrics']
            ld = (np.array(all_results[m]['line_density']) / 5.0).tolist()
            maxima.append(hist_max(ld if f == 'line_density' else mm[f]))
        feature_ymax[f] = 1.05 * max(maxima) if maxima else 1.0

    total_rows = len(model_names) + (1 if gt_in_separate_row else 0)
    fig, axes = plt.subplots(total_rows if total_rows else 1, len(features), figsize=(6*len(features), 3.2*max(total_rows,1)))
    if total_rows == 1:
        axes = np.expand_dims(axes, 0)
    if len(features) == 1:
        axes = np.expand_dims(axes, 1)

    legend_handles, legend_labels, added = [], [], set()

    for r, m in enumerate(model_names):
        mm = all_results[m]['metrics']
        ld = (np.array(all_results[m]['line_density']) / 5.0).tolist()
        disp = model_name_map.get(m, m) if model_name_map else m
        col = color_map.get(disp, '#4169e1')

        for c, f in enumerate(features):
            ax = axes[r, c]
            pred = ld if f == 'line_density' else mm[f]
            true = None if gt_in_separate_row else (gt_line_density if f == 'line_density' else gt_metrics[f])
            be = feature_bins[f]; (xmin, xmax) = feature_ranges[f]

            if true is not None:
                sns.histplot(true, bins=be, color=gt_color, stat=hist_stat, alpha=0.7, ax=ax, edgecolor='black', linewidth=0.5)
                sns.kdeplot(true, color=gt_color, ax=ax, linewidth=1.0, alpha=1.0, bw_adjust=0.8)
                ax.axvline(np.mean(true), color=gt_color, linestyle='--', linewidth=1.2)

            sns.histplot(pred, bins=be, color=col, stat=hist_stat, alpha=0.7, ax=ax, edgecolor='black', linewidth=0.5)
            sns.kdeplot(pred, color=col, ax=ax, linewidth=1.0, alpha=1.0, bw_adjust=0.8)
            ax.axvline(np.mean(pred), color=col, linestyle='--', linewidth=1.2)

            ax.set_xlim(xmin, xmax); ax.set_ylim(0, feature_ymax[f])
            ax.set_xlabel("", fontsize=font_sizes['labels']); ax.set_ylabel("", fontsize=font_sizes['labels'])
            ax.tick_params(axis='both', labelsize=font_sizes['ticks'])

            if f == 'line_density':
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            elif f in ('orientation_angles', 'orientation_angle'):
                ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
            else:
                ax.set_xticks(np.linspace(xmin, xmax, 6))
                if f == 'length_um':
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            if r == 0:
                ax.set_title(feature_labels[c] if feature_labels else f, fontsize=font_sizes['title'], weight='bold', pad=15)

            if disp not in added:
                legend_handles.append(Patch(facecolor=col, edgecolor='black', label=disp))
                legend_labels.append(disp); added.add(disp)

            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_major_formatter(FuncFormatter(_hide_first_tick))
            subplot_label = f"{string.ascii_uppercase[r]}{c+1}"
            ax.text(0.88, 0.95, subplot_label, transform=ax.transAxes, ha='left', va='top', fontsize=font_sizes['subplot_label'], weight='bold')

    if gt_in_separate_row:
        r = total_rows - 1
        for c, f in enumerate(features):
            ax = axes[r, c]
            true = gt_line_density if f == 'line_density' else gt_metrics[f]
            be = feature_bins[f]; (xmin, xmax) = feature_ranges[f]
            sns.histplot(true, bins=be, color=gt_color, stat=hist_stat, alpha=0.7, ax=ax, edgecolor='black', linewidth=0.5)
            sns.kdeplot(true, color=gt_color, ax=ax, linewidth=1.0, alpha=1.0, bw_adjust=0.8)
            ax.axvline(np.mean(true), color=gt_color, linestyle='--', linewidth=1.2)
            ax.set_xlim(xmin, xmax); ax.set_ylim(0, feature_ymax[f])
            ax.tick_params(axis='both', labelsize=font_sizes['ticks'])
            if f == 'line_density':
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            elif f in ('orientation_angles', 'orientation_angle'):
                ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
            else:
                ax.set_xticks(np.linspace(xmin, xmax, 6))
                if f == 'length_um':
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_major_formatter(FuncFormatter(_hide_first_tick))
            subplot_label = f"{string.ascii_uppercase[r]}{c+1}"
            ax.text(0.92, 0.95, subplot_label, transform=ax.transAxes, ha='left', va='top', fontsize=font_sizes['subplot_label'], weight='bold')

    gt_patch = Patch(facecolor=gt_color, edgecolor='black', label='GT')
    legend_handles.insert(0, gt_patch); legend_labels.insert(0, 'GT')
    ncol = max(3, len(legend_labels))
    fig = plt.gcf()
    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=ncol, frameon=False, bbox_to_anchor=(0.5, -0.08), fontsize=font_sizes['legend'])
    plt.tight_layout(); plt.subplots_adjust(bottom=0.06)

    import os
    os.makedirs(save_folder, exist_ok=True)
    path = os.path.join(save_folder, f"{filename_base}_grid.svg")
    plt.savefig(path, format='svg', bbox_inches='tight')
    return path
