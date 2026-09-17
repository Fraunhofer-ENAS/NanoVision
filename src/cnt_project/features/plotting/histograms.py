from __future__ import annotations

import os
from typing import Any
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cnt_project.features.plotting.utils import slugify_plot_name



def plot_feature_table_histograms(
    df: pd.DataFrame,
    feature_columns: list[str],
    out_dir: Path,
    *,
    title_prefix: str,
) -> list[str]:
    """
    Plot one histogram per selected feature column in a feature table.

    Both PNG and SVG versions are saved for each valid feature.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    for col in feature_columns:
        if col not in df.columns:
            continue

        values = (
            pd.to_numeric(df[col], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )

        if values.size == 0:
            continue

        bins = min(
            40,
            max(
                10,
                int(np.sqrt(values.size) * 2),
            ),
        )

        fig, ax = plt.subplots(
            figsize=(6.0, 4.0)
        )

        ax.hist(
            values,
            bins=bins,
            color="#1f77b4",
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
        )

        ax.set_title(
            f"{title_prefix}: {col}"
        )
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.25,
        )

        fig.tight_layout()

        stem = slugify_plot_name(col)

        png_path = out_dir / f"{stem}.png"
        svg_path = out_dir / f"{stem}.svg"

        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        fig.savefig(
            svg_path,
            bbox_inches="tight",
            pad_inches=0.02,
        )

        plt.close(fig)

        saved.extend(
            [
                str(png_path),
                str(svg_path),
            ]
        )

    return saved