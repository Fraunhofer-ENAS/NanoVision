import pandas as pd
from pathlib import Path


def load_dice_metrics(model_names, outputs_root, metric_file="per_image_threshold_metrics.csv"):
    """
    Loads per-image DICE metrics for each model.
    Returns a DataFrame:
        image_id, avg_dice, experiment
    """
    dfs = []

    for name in model_names:
        p = Path(outputs_root) / name / metric_file
        if not p.exists():
            print(f"[warn] missing metrics for: {name}")
            continue

        df = pd.read_csv(p)
        if "image_id" not in df.columns or "avg_dice" not in df.columns:
            print(f"[warn] missing required columns in {name}")
            continue

        out = df.drop_duplicates(subset=["image_id"])[["image_id", "avg_dice"]].copy()
        out["experiment"] = name
        dfs.append(out)

    if not dfs:
        raise ValueError("No valid DICE files found")

    return pd.concat(dfs, ignore_index=True)



def load_density_tertiles(csv_path):
    """
    Loads the Low/Mid/High density classification file:
        CSV column: Image Index, Density_Tertile

    Returns dict:
        filename → "Low"/"Mid"/"High"
    """
    df = pd.read_csv(csv_path)

    mapping = dict(zip(df["Image Index"], df["Density_Tertile"]))
    return mapping
