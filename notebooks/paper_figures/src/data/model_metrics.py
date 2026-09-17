
import os
from pathlib import Path
import pandas as pd

def load_meso_results(outputs_root: str | Path, exp_name: str) -> pd.DataFrame:
    """Load threshold-wise AP results for an experiment."""
    outputs_root = Path(outputs_root)
    csv_file = outputs_root / exp_name / "ap_results_poly_ALL.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Missing: {csv_file}")
    df = pd.read_csv(csv_file)
    pref = exp_name
    return df.rename(columns={c: f"{pref}_{c}" for c in df.columns if c != 'IoU_thresh'})

def normalize_filename(filename: str) -> str:
    import os
    name = os.path.basename(filename)
    return os.path.splitext(name)[0].lower()

def read_filegroups(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if 'image_id' in df.columns:
        df['image_id'] = df['image_id'].apply(normalize_filename)
    return df

def merge_model_metrics(model_names, outputs_root, metrics_file, description_df, description_column: str):
    """Merge per-image metrics across models with a description column (e.g., noise_std)."""
    all_list = []
    for name in model_names:
        p = Path(outputs_root) / name / metrics_file
        if not p.exists():
            print(f"[warn] metrics not found for {name}: {p}")
            continue
        df = pd.read_csv(p)
        if 'image_id' not in df.columns:
            print(f"[warn] missing image_id in {name}")
            continue
        df['experiments'] = name
        df['image_id'] = df['image_id'].apply(normalize_filename)

        desc = description_df[['image_id', description_column]].copy()
        merged = df.merge(desc, on='image_id', how='left')
        all_list.append(merged)
    if not all_list:
        raise ValueError("No valid metrics files found.")
    return pd.concat(all_list, ignore_index=True)
