
# src/plotting/ap_curves.py
import numpy as np
from typing import Dict, Sequence
from cnt_project.visualizations.curves import plot_series_vs_x

def plot_ap_curves(ious: Sequence[float], series: Dict[str, Sequence[float]],
                   colors: Dict[str, str] | None = None,
                   dashes: Dict[str, tuple] | None = None,
                   ylim: tuple[float, float] = (0.0, 0.7),
                   figsize: tuple = (3.3, 2.5),  
                   save_path: str | None = None):
    """Plot AP (or precision) vs IoU threshold for multiple models."""
    plot_series_vs_x(
        x_values=ious,
        series=series,
        xlabel="IoU Threshold",
        ylabel="AP",
        colors=colors,
        dashes=dashes,
        legend_title="Model",
        legend_loc="upper right",
        figsize=figsize,
        ylim=ylim,
        xticks=np.arange(0, 0.85, 0.2),
        grid=False,
        save_path=save_path,
        save_format="svg" if save_path else None,
        show=True,
    )
