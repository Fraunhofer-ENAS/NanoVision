import os
import pandas as pd

def list_images(folder, extensions=['.png', '.jpg', '.jpeg']):
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in extensions]

def save_features_csv(df, out_path):
    df.to_csv(out_path, index=False)

def load_features_csv(path):
    return pd.read_csv(path)

def ensure_dir_exists(path):
    os.makedirs(path, exist_ok=True)
