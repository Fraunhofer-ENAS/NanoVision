from __future__ import annotations

import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401
except Exception:
    scienceplots = None  # type: ignore


def use_science_style() -> None:
    plt.style.use(["science"])


def use_science_no_latex_style() -> None:
    plt.style.use(["science", "no-latex"])


def use_science_ieee_style() -> None:
    plt.style.use(["science", "ieee", "no-latex"])


def use_science_notebook_style() -> None:
    plt.style.use(["science", "notebook", "grid", "no-latex"])


def set_large_text_rcparams(
    *,
    axes_labelsize: int = 24,
    axes_titlesize: int = 24,
    xtick_labelsize: int = 22,
    ytick_labelsize: int = 22,
    lines_linewidth: float = 1.5,
    legend_fontsize: int | None = None,
) -> None:
    updates: dict[str, float | int] = {
        "axes.labelsize": axes_labelsize,
        "axes.titlesize": axes_titlesize,
        "xtick.labelsize": xtick_labelsize,
        "ytick.labelsize": ytick_labelsize,
        "lines.linewidth": lines_linewidth,
    }

    if legend_fontsize is not None:
        updates["legend.fontsize"] = legend_fontsize

    plt.rcParams.update(updates)


def set_extra_large_text_rcparams() -> None:
    plt.rcParams.update(
        {
            "axes.labelsize": 46,
            "axes.labelweight": "bold",
            "axes.titlesize": 46,
            "axes.titleweight": "bold",
            "xtick.labelsize": 46,
            "ytick.labelsize": 46,
            "legend.fontsize": 14,
        }
    )


def use_large_science_style() -> None:
    use_science_style()
    set_large_text_rcparams()


def use_medium_science_style() -> None:
    use_science_style()
    plt.rcParams.update(
        {
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )


def use_single_column_style() -> None:
    use_science_ieee_style()
    plt.rcParams.update(
        {
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "lines.linewidth": 1.5,
        }
    )


def reset_plot_rcparams() -> None:
    plt.rcdefaults()


def apply_style(
    *,
    style: str = "science",
    no_latex: bool = False,
    ieee: bool = False,
    notebook: bool = False,
    large_text: bool = False,
    extra_large_text: bool = False,
) -> None:
    styles = [style]

    if ieee:
        styles.append("ieee")
    if notebook:
        styles.extend(["notebook", "grid"])
    if no_latex:
        styles.append("no-latex")

    plt.style.use(styles)

    if extra_large_text:
        set_extra_large_text_rcparams()
    elif large_text:
        set_large_text_rcparams()
