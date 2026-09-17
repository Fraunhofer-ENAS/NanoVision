from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Plotting style
# ---------------------------------------------------------

try:
    import scienceplots
except ImportError:
    scienceplots = None


FIGURE_SIZE = (3.3, 2.5)

PRIMARY_COLOR = "#4169E1"

POINT_SIZE = 42

VALIDATED_ALPHA = 0.75
HIGH_DENSITY_ALPHA = 0.75

APPLICATION_LINE_COLOR = "0.65"
BOUNDARY_LINE_COLOR = "0.25"
REFERENCE_LINE_COLOR = "0.25"

EXTRAPOLATION_SHADE = "0.95"


def _save_svg(
    fig,
    save_path,
):
    save_svg = str(save_path)

    if not save_svg.lower().endswith(".svg"):
        save_svg += ".svg"

    fig.savefig(
        save_svg,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    save_png = (
        save_svg[:-4]
        + ".png"
    )

    fig.savefig(
        save_png,
        format="png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)

    return save_svg


def _apply_axis_style(
    ax,
):
    ax.tick_params(
        direction="in",
    )

    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")


def _shade_extrapolation_region(
    ax,
    validated_density_max,
):
    """
    Shade the region outside the validated test-density domain.
    """

    x_max = ax.get_xlim()[1]

    if x_max > validated_density_max:
        ax.axvspan(
            validated_density_max,
            x_max,
            facecolor=EXTRAPOLATION_SHADE,
            edgecolor="none",
            zorder=0,
        )


def _plot_application_density_markers(
    ax,
    application_densities,
):
    for density in application_densities:
        ax.axvline(
            density,
            linestyle=":",
            linewidth=0.7,
            color=APPLICATION_LINE_COLOR,
            alpha=0.65,
            zorder=1,
        )


# =========================================================
# Absolute density error
# =========================================================

def plot_operational_limit(
    df,
    save_path,
    validated_density_max=6.0,
    application_densities=(
        0.38,
        0.75,
        1.02,
        2.0,
    ),
):
    """
    Plot absolute density error against reference density.

    Fully annotated test data define the validated operating domain.
    High-density measurements are shown as an extrapolation experiment.
    """

    df = df.copy()

    reference = (
        df["manual_density_um"]
        .astype(float)
    )

    prediction = (
        df["predicted_density_um"]
        .astype(float)
    )

    valid = (
        np.isfinite(reference)
        & np.isfinite(prediction)
    )

    plot_df = (
        df.loc[valid]
        .copy()
    )

    plot_df["absolute_error_um"] = (
        plot_df["predicted_density_um"]
        .astype(float)
        .sub(
            plot_df["manual_density_um"]
            .astype(float)
        )
        .abs()
    )

    with plt.style.context(
        [
            "science",
            "no-latex",
        ]
    ):
        fig, ax = plt.subplots(
            figsize=FIGURE_SIZE,
        )

        if "dataset_type" in plot_df.columns:
            test_mask = (
                plot_df["dataset_type"]
                .eq("test")
            )
        else:
            test_mask = np.ones(
                len(plot_df),
                dtype=bool,
            )

        # -------------------------------------------------
        # Validated test set
        # -------------------------------------------------

        ax.scatter(
            plot_df.loc[
                test_mask,
                "manual_density_um",
            ],
            plot_df.loc[
                test_mask,
                "absolute_error_um",
            ],
            s=POINT_SIZE,
            marker="o",
            facecolor=PRIMARY_COLOR,
            edgecolor="black",
            linewidth=0.45,
            alpha=VALIDATED_ALPHA,
            label="Test set",
            zorder=3,
        )

        # -------------------------------------------------
        # High-density extrapolation
        # -------------------------------------------------

        if "dataset_type" in plot_df.columns:
            high_mask = (
                plot_df["dataset_type"]
                .eq("high_density")
            )

            if high_mask.any():
                ax.scatter(
                    plot_df.loc[
                        high_mask,
                        "manual_density_um",
                    ],
                    plot_df.loc[
                        high_mask,
                        "absolute_error_um",
                    ],
                    s=POINT_SIZE,
                    marker="s",
                    facecolor="none",
                    edgecolor=PRIMARY_COLOR,
                    linewidth=1.0,
                    alpha=HIGH_DENSITY_ALPHA,
                    label="High-density extrapolation",
                    zorder=3,
                )

        # Establish x-limits before shading.
        # ax.relim()
        # ax.autoscale_view()
        x_max = float(
            plot_df["manual_density_um"].max()
        )

        x_upper = x_max * 1.04

        ax.set_xlim(
            0.0,
            x_upper,
        )

    


        # -------------------------------------------------
        # Application densities
        # -------------------------------------------------

        

        ax.set_xlabel(
            "Mean Line Density "
            "(CNT/$\\mu$m)"
        )

        ax.set_ylabel(
            "Abs Density Error "
            "CNT/$\\mu$m"
        )

        _apply_axis_style(
            ax,
        )

        ax.legend(
            frameon=False,
            fontsize=6.5,
            loc="upper left",
        )

        fig.tight_layout()

        return _save_svg(
            fig,
            save_path,
        )


# =========================================================
# Relative density error
# =========================================================

def plot_relative_density_error(
    df,
    save_path,
    validated_density_max=6.0,
    primary_threshold_pct=15.0,
    application_densities=(
        0.38,
        0.75,
        1.02,
        2.0,
    ),
):
    """
    Plot absolute relative density error against reference density.

    Relative error:
        |prediction - reference| / reference * 100

    The 15% line is the application-motivated accuracy criterion.
    """

    df = df.copy()

    reference = (
        df["manual_density_um"]
        .astype(float)
    )

    prediction = (
        df["predicted_density_um"]
        .astype(float)
    )

    valid = (
        np.isfinite(reference)
        & np.isfinite(prediction)
        & (reference > 0)
    )

    plot_df = (
        df.loc[valid]
        .copy()
    )

    plot_df["relative_error_pct"] = (
        (
            plot_df["predicted_density_um"]
            .astype(float)
            - plot_df["manual_density_um"]
            .astype(float)
        )
        .abs()
        / plot_df["manual_density_um"]
        .astype(float)
        * 100.0
    )

    with plt.style.context(
        [
            "science",
            "no-latex",
        ]
    ):
        fig, ax = plt.subplots(
            figsize=FIGURE_SIZE,
        )

        if "dataset_type" in plot_df.columns:
            test_mask = (
                plot_df["dataset_type"]
                .eq("test")
            )
        else:
            test_mask = np.ones(
                len(plot_df),
                dtype=bool,
            )

        # -------------------------------------------------
        # Validated test data
        # -------------------------------------------------

        ax.scatter(
            plot_df.loc[
                test_mask,
                "manual_density_um",
            ],
            plot_df.loc[
                test_mask,
                "relative_error_pct",
            ],
            s=POINT_SIZE,
            marker="o",
            facecolor=PRIMARY_COLOR,
            edgecolor="black",
            linewidth=0.45,
            alpha=VALIDATED_ALPHA,
            label="Test set",
            zorder=3,
        )

        # -------------------------------------------------
        # High-density extrapolation
        # -------------------------------------------------

        if "dataset_type" in plot_df.columns:
            high_mask = (
                plot_df["dataset_type"]
                .eq("high_density")
            )

            if high_mask.any():
                ax.scatter(
                    plot_df.loc[
                        high_mask,
                        "manual_density_um",
                    ],
                    plot_df.loc[
                        high_mask,
                        "relative_error_pct",
                    ],
                    s=POINT_SIZE,
                    marker="s",
                    facecolor="none",
                    edgecolor=PRIMARY_COLOR,
                    linewidth=1.0,
                    alpha=HIGH_DENSITY_ALPHA,
                    label="High-density extrapolation",
                    zorder=3,
                )

        # ax.relim()
        # ax.autoscale_view()
        x_max = float(
            plot_df["manual_density_um"].max()
        )

        x_upper = x_max * 1.04

        ax.set_xlim(
            0.0,
            x_upper,
        )


        # -------------------------------------------------
        # 15% application-motivated criterion
        # -------------------------------------------------

        ax.axhline(
            primary_threshold_pct,
            linestyle="--",
            linewidth=1.0,
            color=REFERENCE_LINE_COLOR,
            label=(
                f"{primary_threshold_pct:g}% "
                "relative-error criterion"
            ),
            zorder=2,
        )


        # -------------------------------------------------
        # Application densities
        # -------------------------------------------------

        ax.set_xlabel(
            "Mean Line Density "
            "(CNT/$\\mu$m)"
        )

        ax.set_ylabel(
            "Abs Relative Error (%)"
        )

        _apply_axis_style(
            ax,
        )

        ax.legend(
            frameon=False,
            fontsize=6.3,
            loc="upper left",
        )

        fig.tight_layout()

        return _save_svg(
            fig,
            save_path,
        )


# =========================================================
# Relative-error compliance
# =========================================================

def plot_relative_error_compliance(
    df,
    save_path,
    bin_edges=(
        0.0,
        0.5,
        1.0,
        2.0,
        6.0,
        10.0,
        15.0,
        22.0,
    ),
    threshold=15.0,
    validated_density_max=6.0,
):
    """
    Plot relative-error threshold compliance as a function
    of reference-density interval.

    Test-set bins within <= validated_density_max are treated
    as validated-domain measurements.

    High-density measurements above validated_density_max are
    displayed separately as extrapolation bins.
    """

    df = df.copy()

    reference = (
        df["manual_density_um"]
        .astype(float)
    )

    prediction = (
        df["predicted_density_um"]
        .astype(float)
    )

    valid = (
        np.isfinite(reference)
        & np.isfinite(prediction)
        & (reference > 0)
    )

    work = (
        df.loc[valid]
        .copy()
    )

    work["relative_error_pct"] = (
        (
            work["predicted_density_um"]
            .astype(float)
            - work["manual_density_um"]
            .astype(float)
        )
        .abs()
        / work["manual_density_um"]
        .astype(float)
        * 100.0
    )

    work["density_bin"] = pd.cut(
        work["manual_density_um"],
        bins=bin_edges,
        include_lowest=True,
        right=True,
    )

    rows = []

    for density_bin, group in work.groupby(
        "density_bin",
        observed=False,
    ):
        if len(group) == 0:
            compliance = np.nan
            median_error = np.nan
            p90_error = np.nan
            n = 0

        else:
            compliance = (
                (
                    group["relative_error_pct"]
                    <= threshold
                )
                .mean()
                * 100.0
            )

            median_error = (
                group["relative_error_pct"]
                .median()
            )

            p90_error = (
                group["relative_error_pct"]
                .quantile(0.90)
            )

            n = len(group)

        rows.append(
            {
                "density_bin":
                    density_bin,
                "n":
                    n,
                "median_relative_error_pct":
                    median_error,
                "p90_relative_error_pct":
                    p90_error,
                f"compliance_le_{threshold:g}_pct":
                    compliance,
            }
        )

    summary = pd.DataFrame(
        rows
    )

    labels = [
        f"{left:g}-{right:g}"
        for left, right in zip(
            bin_edges[:-1],
            bin_edges[1:],
        )
    ]

    summary["bin_label"] = labels

    # A bin is considered validated if its upper edge
    # does not exceed validated_density_max.
    upper_edges = np.asarray(
        bin_edges[1:],
        dtype=float,
    )

    summary["validated_domain"] = (
        upper_edges
        <= validated_density_max
    )

    compliance_column = (
        f"compliance_le_{threshold:g}_pct"
    )

    x = np.arange(
        len(summary)
    )

    with plt.style.context(
        [
            "science",
            "no-latex",
        ]
    ):
        fig, ax = plt.subplots(
            figsize=FIGURE_SIZE,
        )

        for i, row in summary.iterrows():

            value = row[
                compliance_column
            ]

            if not np.isfinite(value):
                continue

            is_validated = bool(
                row[
                    "validated_domain"
                ]
            )

            if is_validated:
                facecolor = PRIMARY_COLOR
                edgecolor = "black"
                hatch = None
                alpha = 0.80

            else:
                facecolor = "none"
                edgecolor = PRIMARY_COLOR
                hatch = "///"
                alpha = 1.0

            bar = ax.bar(
                i,
                value,
                width=0.62,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=0.8,
                hatch=hatch,
                alpha=alpha,
                zorder=3,
            )[0]

            bar_center = (
                bar.get_x()
                + bar.get_width() / 2
            )

            # -------------------------------------------------
            # Bar annotations
            # -------------------------------------------------

            if value > 0:

                # Compliance percentage above the bar.
                ax.text(
                    bar_center,
                    value + 1.5,
                    f"{value:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

                # Sample count inside the upper part of the bar.
                ax.text(
                    bar_center,
                    max(
                        value - 4.0,
                        2.0,
                    ),
                    f"n={int(row['n'])}",
                    ha="center",
                    va="top",
                    fontsize=6.5,
                    color=(
                        "white"
                        if is_validated
                        and value > 20
                        else PRIMARY_COLOR
                    ),
                )

            else:

                # Zero-compliance bins have no visible bar
                # height, so keep both annotations above the
                # x-axis instead of allowing them to fall below it.
                ax.text(
                    bar_center,
                    4.0,
                    "0%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

                ax.text(
                    bar_center,
                    1.0,
                    f"n={int(row['n'])}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color=PRIMARY_COLOR,
                )

                # Small horizontal marker to make the zero-height
                # extrapolation bar visually identifiable.
                ax.hlines(
                    y=0.0,
                    xmin=i - 0.31,
                    xmax=i + 0.31,
                    color=PRIMARY_COLOR,
                    linewidth=1.3,
                    zorder=4,
                )

        # -------------------------------------------------
        # Reference line
        # -------------------------------------------------

        ax.axhline(
            100.0,
            linestyle=":",
            linewidth=0.7,
            color="0.70",
        )

        # -------------------------------------------------
        # Separate validated and extrapolation bins
        # -------------------------------------------------

        validated_indices = np.where(
            upper_edges
            <= validated_density_max
        )[0]

        if len(validated_indices) > 0:
            validation_boundary_index = (
                validated_indices[-1]
                + 0.5
            )

            ax.axvline(
                validation_boundary_index,
                linestyle="--",
                linewidth=0.9,
                color="0.35",
            )

            ax.text(
                validation_boundary_index - 0.1,
                8,
                "",
                rotation=90,
                ha="right",
                va="bottom",
                fontsize=6.5,
                color="0.35",
            )

            ax.text(
                validation_boundary_index + 0.1,
                8,
                "",
                rotation=90,
                ha="left",
                va="bottom",
                fontsize=6.5,
                color="0.35",
            )

        # -------------------------------------------------
        # Axes
        # -------------------------------------------------

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            summary[
                "bin_label"
            ],
            rotation=25,
            ha="right",
        )

        ax.set_ylim(
            0,
            108,
        )

        ax.set_xlabel(
            "Line Density Interval "
            "(CNT/$\\mu$m)"
        )

        ax.set_ylabel(
            f"Images Within {threshold:g}% Error"
        )

        ax.tick_params(
            direction="in",
        )

        ax.xaxis.set_ticks_position(
            "both"
        )

        ax.yaxis.set_ticks_position(
            "both"
        )

        fig.tight_layout()

        # -------------------------------------------------
        # Save
        # -------------------------------------------------

        save_svg = str(
            save_path
        )

        if not save_svg.lower().endswith(
            ".svg"
        ):
            save_svg += ".svg"

        fig.savefig(
            save_svg,
            format="svg",
            bbox_inches="tight",
            pad_inches=0.02,
        )

        save_png = (
            save_svg[:-4]
            + ".png"
        )

        fig.savefig(
            save_png,
            format="png",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.02,
        )

        plt.close(fig)

    return (
        save_svg,
        summary,
    )
# =========================================================
# Bland-Altman
# =========================================================

def plot_bland_altman_density(
    df,
    save_path,
    validated_density_max=6.0,
    show_high_density=False,
    restrict_to_validated_domain=True,
    return_stats=False,
):
    """
    Bland-Altman agreement analysis.

    Statistics are calculated from fully annotated test images
    within the validated density range.

    For the final manuscript figure, high-density extrapolation
    data are hidden by default because they are excluded from the
    agreement statistics.
    """

    df = df.copy()

    reference = (
        df["manual_density_um"]
        .astype(float)
    )

    prediction = (
        df["predicted_density_um"]
        .astype(float)
    )

    valid_pair = (
        np.isfinite(reference)
        & np.isfinite(prediction)
    )

    stats_mask = (
        valid_pair
        & (reference <= validated_density_max)
    )

    if "dataset_type" in df.columns:
        stats_mask &= (
            df["dataset_type"]
            .eq("test")
        )

    stats_df = (
        df.loc[stats_mask]
        .copy()
    )

    if len(stats_df) < 2:
        raise ValueError(
            "At least two valid paired measurements are "
            "required for Bland-Altman analysis."
        )

    ref_stats = (
        stats_df["manual_density_um"]
        .astype(float)
    )

    pred_stats = (
        stats_df["predicted_density_um"]
        .astype(float)
    )

    mean_density = (
        ref_stats
        + pred_stats
    ) / 2.0

    difference = (
        pred_stats
        - ref_stats
    )

    bias = float(
        difference.mean()
    )

    sd_difference = float(
        difference.std(
            ddof=1
        )
    )

    loa_lower = (
        bias
        - 1.96
        * sd_difference
    )

    loa_upper = (
        bias
        + 1.96
        * sd_difference
    )

    with plt.style.context(
        [
            "science",
            "no-latex",
        ]
    ):
        fig, ax = plt.subplots(
            figsize=FIGURE_SIZE,
        )

        ax.scatter(
            mean_density,
            difference,
            s=POINT_SIZE,
            marker="o",
            facecolor=PRIMARY_COLOR,
            edgecolor="black",
            linewidth=0.45,
            alpha=VALIDATED_ALPHA,
            zorder=3,
        )

        # -------------------------------------------------
        # Optional high-density extrapolation
        # -------------------------------------------------

        if (
            show_high_density
            and "dataset_type" in df.columns
        ):
            high_mask = (
                valid_pair
                & df["dataset_type"]
                .eq("high_density")
            )

            if high_mask.any():
                ref_high = (
                    df.loc[
                        high_mask,
                        "manual_density_um",
                    ]
                    .astype(float)
                )

                pred_high = (
                    df.loc[
                        high_mask,
                        "predicted_density_um",
                    ]
                    .astype(float)
                )

                mean_high = (
                    ref_high
                    + pred_high
                ) / 2.0

                diff_high = (
                    pred_high
                    - ref_high
                )

                ax.scatter(
                    mean_high,
                    diff_high,
                    s=POINT_SIZE,
                    marker="s",
                    facecolor="none",
                    edgecolor=PRIMARY_COLOR,
                    linewidth=1.0,
                    alpha=HIGH_DENSITY_ALPHA,
                    zorder=3,
                )

        # Zero-difference reference
        ax.axhline(
            0.0,
            color="0.70",
            linestyle=":",
            linewidth=0.8,
            zorder=1,
        )

        # Bias
        ax.axhline(
            bias,
            color=REFERENCE_LINE_COLOR,
            linestyle="-",
            linewidth=1.0,
            label=(
                f"Bias = {bias:.3f}"
            ),
        )

        # Limits of agreement
        ax.axhline(
            loa_upper,
            color=REFERENCE_LINE_COLOR,
            linestyle="--",
            linewidth=0.9,
            label=(
                "95\% LoA "
                f"[{loa_lower:.3f}, "
                f"{loa_upper:.3f}]"
            ),
        )

        ax.axhline(
            loa_lower,
            color=REFERENCE_LINE_COLOR,
            linestyle="--",
            linewidth=0.9,
        )

        ax.set_xlabel(
            "Mean Density "
            "(CNT/$\\mu$m)"
        )

        ax.set_ylabel(
            "Prediction $-$ Reference "
            "(CNT/$\\mu$m)"
        )

        _apply_axis_style(
            ax,
        )

        ax.legend(
            frameon=False,
            fontsize=6.5,
            loc="best",
        )

        fig.tight_layout()

        save_svg = _save_svg(
            fig,
            save_path,
        )

    stats = {
        "n":
            len(stats_df),
        "bias":
            bias,
        "sd_difference":
            sd_difference,
        "loa_lower":
            loa_lower,
        "loa_upper":
            loa_upper,
    }

    if return_stats:
        return (
            save_svg,
            stats,
        )

    return save_svg