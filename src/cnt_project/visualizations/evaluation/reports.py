from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_density_vs_angle(
    density_vs_angle: dict[float, float],
    *,
    dst_dir: str | Path,
    file_stem: str,
) -> Path:
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_path = dst_dir / f"{file_stem}_MeanLineDensity_vs_Angle.png"

    plt.plot(list(density_vs_angle.keys()), list(density_vs_angle.values()), marker="o", linestyle="-")
    plt.xlabel("Orientation Angle (degrees)")
    plt.ylabel("Mean Line Density")
    plt.title("Mean Line Density vs Orientation Angle")
    plt.savefig(out_path)
    plt.close()

    return out_path


def plot_performance_by_density(
    *,
    num_labels: Iterable[float],
    mean_errors: Iterable[float],
    rel_mean_errors: Iterable[float],
    std_errors: Iterable[float],
    rel_std_errors: Iterable[float],
) -> None:
    df_plot = pd.DataFrame(
        {
            "Number of Labels (Objects)": list(num_labels),
            "MAE Mean Line Density": list(mean_errors),
            "MRE Mean Line Density": list(rel_mean_errors),
            "MAE Std Line Density": list(std_errors),
            "MRE Std Line Density": list(rel_std_errors),
        }
    )
    sns.lineplot(data=df_plot, x="Number of Labels (Objects)", y="MAE Mean Line Density", label="MAE Mean Line Density")
    sns.lineplot(data=df_plot, x="Number of Labels (Objects)", y="MRE Mean Line Density", label="MRE Mean Line Density")
    sns.lineplot(data=df_plot, x="Number of Labels (Objects)", y="MAE Std Line Density", label="MAE Std Line Density")
    sns.lineplot(data=df_plot, x="Number of Labels (Objects)", y="MRE Std Line Density", label="MRE Std Line Density")
    plt.title("Performance by Density")
    plt.xlabel("Number of Labels (Objects)")
    plt.ylabel("Error Metrics")
    plt.legend()
    plt.show()
    plt.close()
