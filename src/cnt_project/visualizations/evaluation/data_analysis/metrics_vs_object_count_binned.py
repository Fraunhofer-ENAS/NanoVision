from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Optional dependency used in your legacy script.
# Keep import here so the style call works when available.
import scienceplots  # noqa: F401


def apply_science_style() -> None:
    """
    Applies the same matplotlib/scienceplots styling as the legacy script.

    Note:
      - You set text.usetex=True. This requires a working LaTeX installation.
      - If you run into LaTeX issues, switch text.usetex to False in the *runner*.
    """
    plt.style.use(["science", "no-latex"])
    matplotlib.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
            # legacy config kept:
            "text.usetex": True,
            "pgf.texsystem": "pdflatex",
            "pgf.preamble": r"\usepackage{amsmath}\usepackage{siunitx}",
        }
    )


def load_and_merge_csvs(csv_paths: Sequence[str | Path]) -> pd.DataFrame:
    """
    Load and merge CSVs without duplicate filenames.
    Legacy behavior: drop duplicates by 'filename' column.
    """
    dfs = [pd.read_csv(str(path)) for path in csv_paths]
    combined = pd.concat(dfs, ignore_index=True)
    return combined.drop_duplicates(subset="filename").reset_index(drop=True)


def plot_metrics_vs_gt_objects_binned(
    merged_df: pd.DataFrame,
    *,
    metrics: Sequence[str] = ("mean_dice", "mAP"),
    bin_size: int = 10,
    output_prefixes: Sequence[str] = ("mean_dice_binned", "mAP_binned"),
    out_dir: str | Path = ".",
) -> None:
    """
    Plot selected metrics vs number of GT objects, binned into ranges.
    Saves: <out_dir>/<output_prefix>.pdf and <out_dir>/<output_prefix>.pgf

    Required columns:
      - num_gt_objects
      - metric columns (e.g., mean_dice, mAP)
    """
    if "num_gt_objects" not in merged_df.columns:
        raise ValueError("DataFrame missing required column: 'num_gt_objects'")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_obj = int(merged_df["num_gt_objects"].max())
    bins = range(0, max_obj + bin_size, bin_size)
    labels = [f"{b}-{b + bin_size - 1}" for b in list(bins)[:-1]]

    merged_df = merged_df.copy()
    merged_df["binned_objects"] = pd.cut(
        merged_df["num_gt_objects"],
        bins=bins,
        labels=labels,
        right=False,
    )

    for metric, out_name in zip(metrics, output_prefixes):
        if metric not in merged_df.columns:
            print(f"Warning: '{metric}' not found in DataFrame, skipping.")
            continue

        grouped = (
            merged_df.groupby("binned_objects", observed=False)[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        # Legacy behavior: interpolate missing means to avoid gaps
        grouped["mean"] = grouped["mean"].interpolate(method="linear", limit_direction="both")

        fig, ax = plt.subplots(figsize=(10, 6))

        x = range(len(labels))
        ax.plot(x, grouped["mean"], marker="o", label="Mean")
        ax.fill_between(
            x,
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.2,
            label="±1 std. dev.",
        )

        ax.set_xlabel("Number of Ground Truth Objects (binned)")
        ax.set_ylabel(metric)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45)
        ax.legend()
        ax.grid(False)

        plt.tight_layout()
        pdf_path = out_dir / f"{out_name}.pdf"
        pgf_path = out_dir / f"{out_name}.pgf"
        plt.savefig(str(pdf_path), dpi=300)
        plt.savefig(str(pgf_path))
        plt.close(fig)
