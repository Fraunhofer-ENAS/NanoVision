from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_density_classification_csv(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def summarize_cnt_count_by_density(
    df: pd.DataFrame,
    *,
    density_col: str = "Density_Tertile",
    count_col: str = "GT Number of Cnts",
) -> pd.DataFrame:
    density_order = {
        "LOW_DENSITY": 0,
        "MID_DENSITY": 1,
        "HIGH_DENSITY": 2,
        "LOW": 0,
        "MID": 1,
        "HIGH": 2,
    }

    display_map = {
        "LOW_DENSITY": "Low",
        "MID_DENSITY": "Mid",
        "HIGH_DENSITY": "High",
        "LOW": "Low",
        "MID": "Mid",
        "HIGH": "High",
    }

    out = df.copy()
    out[density_col] = out[density_col].astype(str).str.strip().str.upper()
    out["density_display"] = out[density_col].map(display_map).fillna(out[density_col])

    summary = (
        out.groupby("density_display", as_index=False)
        .agg(
            mean_cnt_count=(count_col, "mean"),
            std_cnt_count=(count_col, "std"),
            n_images=(count_col, "count"),
        )
    )

    summary["density_order"] = summary["density_display"].map(
        {"Low": 0, "Mid": 1, "High": 2}
    )
    return summary.sort_values("density_order").reset_index(drop=True)