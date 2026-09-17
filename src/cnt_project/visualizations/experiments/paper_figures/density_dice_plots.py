import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch


def plot_density_metric(
    df,
    color_map,
    legend_name_map,
    *,
    metric_col,
    ylabel,
    experiment_col="experiment",
    density_col="density",
    order=("Low", "Mid", "High"),
    figsize=(4.5, 2.8),
    ylim=(0, 1),
    box_width=0.8,
    legend_loc="best",
    legend_outside: bool = False,
    save_path=None,
    save_dpi=None,
    show=True,
):
    """
    Box + strip plot of an arbitrary metric grouped by density.

    The figure styling matches the existing density DICE paper figure.
    """

    plt.figure(figsize=figsize)
    plt.style.use(["science", "no-latex"])

    hue_levels = list(df[experiment_col].dropna().unique())
    fallback_colors = iter(sns.color_palette(n_colors=len(hue_levels)).as_hex())
    plot_color_map = {
        experiment: color_map.get(experiment) or next(fallback_colors)
        for experiment in hue_levels
    }

    sns.boxplot(
        x=density_col,
        y=metric_col,
        hue=experiment_col,
        data=df,
        palette=plot_color_map,
        dodge=True,
        width=box_width,
        fliersize=0,
        order=order,
        legend=False,
    )

    sns.stripplot(
        x=density_col,
        y=metric_col,
        hue=experiment_col,
        data=df,
        dodge=True,
        jitter=False,
        palette=plot_color_map,
        linewidth=0.5,
        edgecolor="gray",
        alpha=0.7,
        order=order,
        legend=False,
    )

    grouped = df.groupby([density_col, experiment_col])[metric_col].median().reset_index()
    n_hues = len(hue_levels)

    for i, dens in enumerate(order):
        for j, exp in enumerate(hue_levels):
            row = grouped.query(f"{density_col} == @dens and {experiment_col} == @exp")
            if row.empty:
                continue

            base_color = plot_color_map[exp]
            rgb = to_rgb(base_color)
            darker = tuple(max(0, c * 0.7) for c in rgb)

            # Place marker at the exact center of the corresponding dodged box.
            dodge_amount = box_width / n_hues if n_hues > 0 else box_width
            x_pos = i - box_width / 2 + (j + 0.5) * dodge_amount
            plt.scatter(
                x_pos,
                row[metric_col].values[0],
                color=darker,
                marker="x",
                s=70,
                zorder=10,
            )

    handles = [
        Patch(facecolor=plot_color_map[k], label=legend_name_map.get(k, k))
        for k in hue_levels
    ]
    if legend_outside:
        plt.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            borderaxespad=0.0,
        )
    else:
        plt.legend(handles=handles, loc=legend_loc, frameon=True)

    plt.xlabel("")
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.2)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.tight_layout()
    if save_path:
        save_kwargs = {"bbox_inches": "tight"}
        if save_dpi is not None:
            save_kwargs["dpi"] = int(save_dpi)
        plt.savefig(save_path, **save_kwargs)

    if show:
        plt.show()
    else:
        plt.close()


def plot_density_dice(
    df,
    color_map,
    legend_name_map,
    order=("Low", "Mid", "High"),
    figsize=(4.5, 2.8),
    save_path=None,
    marker_stat="median",
    box_width=0.8,
    marker_visual_nudge=None,
):
    """
    Box + strip plot of DICE grouped by density
    df needs columns: density, avg_dice, experiment
    """

    plt.figure(figsize=figsize)
    plt.style.use(["science", "no-latex"])

    hue_order = [k for k in color_map.keys() if k in set(df["experiment"].dropna())]

    # --- BOX ---
    sns.boxplot(
        x="density",
        y="avg_dice",
        hue="experiment",
        data=df,
        palette=color_map,
        hue_order=hue_order,
        dodge=True,
        width=box_width,
        fliersize=0,
        order=order,
        legend=False
    )


    # --- STRIP ---
    sns.stripplot(
        x="density",
        y="avg_dice",
        hue="experiment",
        data=df,
        dodge=True,
        jitter=False,
        palette=color_map,
        hue_order=hue_order,
        linewidth=0.5,
        edgecolor="gray",
        alpha=0.7,
        order=order,
        legend=False
    )

    # --- STATISTIC MARKERS ---
    if marker_stat not in {"median", "mean"}:
        raise ValueError("marker_stat must be either 'median' or 'mean'")
    if marker_visual_nudge is None:
        marker_visual_nudge = {}

    exps = hue_order
    n_hues = len(exps)
    marker_palette = {
        exp: tuple(max(0, c * 0.7) for c in to_rgb(color_map[exp]))
        for exp in exps
    }
    estimator = np.nanmedian if marker_stat == "median" else np.nanmean

    # Match point markers to box centers from a dodged boxplot.
    # For n hues, box centers are spaced by box_width / n.
    # pointplot's dodge spans the full left-right marker spread.
    point_dodge = 0.0 if n_hues <= 1 else box_width * (n_hues - 1) / n_hues

    ax = plt.gca()
    line_count_before = len(ax.lines)
    collection_count_before = len(ax.collections)
    sns.pointplot(
        x="density",
        y="avg_dice",
        hue="experiment",
        data=df,
        order=order,
        hue_order=hue_order,
        dodge=point_dodge,
        markers="o",
        linestyles="none",
        estimator=estimator,
        errorbar=None,
        palette=marker_palette,
        legend=False,
        zorder=10,
    )

    # Use pointplot only for exact coordinates, then draw explicit X markers.
    marker_lines = ax.lines[line_count_before:]
    marker_collections = ax.collections[collection_count_before:]
    marker_rgb = {exp: np.array(marker_palette[exp]) for exp in exps}

    for line in marker_lines:
        line_color = np.array(to_rgb(line.get_color()))
        exp = min(exps, key=lambda e: np.linalg.norm(line_color - marker_rgb[e]))

        dx = float(marker_visual_nudge.get(exp, 0.0))
        xdata = np.asarray(line.get_xdata(), dtype=float) + dx
        ydata = np.asarray(line.get_ydata(), dtype=float)

        ax.scatter(
            xdata,
            ydata,
            color=marker_palette[exp],
            marker="x",
            s=70,
            linewidths=1.2,
            zorder=11,
        )

    for line in marker_lines:
        line.remove()
    for coll in marker_collections:
        coll.remove()

    # --- LEGEND ---
    handles = [
        Patch(facecolor=color_map[k], label=legend_name_map.get(k, k))
        for k in exps
    ]
    plt.legend(handles=handles, loc="upper right", frameon=True)

    plt.xlabel("")
    plt.ylabel("Average DICE Score")
    plt.grid(axis="y", linestyle="--", alpha=0.2)
    plt.ylim(0, 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="svg", bbox_inches="tight")

    plt.show()
