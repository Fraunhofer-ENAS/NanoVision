import os
import cv2
import numpy as np
import pandas as pd
import pywt
from tqdm import tqdm
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans

def estimate_noise_wavelet(image: np.ndarray, wavelet: str = 'db1') -> float:
    """
    Estimate the noise standard deviation (sigma) of an image using wavelet-based
    high-frequency detail coefficients.

    Steps:
    1. Convert RGB/BGR images to grayscale (if needed).
    2. Perform a single-level 2D wavelet decomposition.
    3. Extract the diagonal detail coefficients `cD`.
    4. Compute sigma = median(|cD|) / 0.6745  (MAD normalization factor).

    Returns
    -------
    float
        Estimated noise standard deviation (sigma).
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coeffs = pywt.wavedec2(image.astype(np.float32), wavelet=wavelet, level=1)
    _, (_, _, cD) = coeffs
    sigma = float(np.median(np.abs(cD)) / 0.6745)
    return sigma

def evaluate_noise_in_folder(folder_path: str, out_csv: str | None = None) -> pd.DataFrame:
    """Compute noise std for all 256x256 .tif images in a folder."""
    rows = []
    for filename in tqdm(os.listdir(folder_path)):
        if not filename.lower().endswith('.tif'):
            continue
        path = os.path.join(folder_path, filename)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[:2] != (256, 256):
            continue
        rows.append({'filename': filename, 'noise_std': estimate_noise_wavelet(img)})
    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df

def add_noise_classes(df: pd.DataFrame, method: str = 'otsu', random_state: int = 42) -> pd.DataFrame:
    """
    Classify images into noise categories based on their measured noise standard deviation (`noise_std`).

    This function adds two categorical columns to the input DataFrame:
    
    - **`Noise_Otsu`**: assigns `"Clean"` or `"Noisy"` based on an Otsu threshold computed over `noise_std`.
    - **`Noise_KMeans`**: assigns `"Clean"` or `"Noisy"` using a 2-cluster K-Means model trained on `noise_std`.
      The cluster with the **higher mean noise** is labeled `"Noisy"`.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with two additional columns:
        `'Noise_Otsu'` and `'Noise_KMeans'`, containing `"Clean"` or `"Noisy"` labels.

    Raises
    ------
    ValueError
        If the input DataFrame does not contain the column `'noise_std'`.
    """
    if 'noise_std' not in df.columns:
        raise ValueError("DataFrame must contain 'noise_std'")
    out = df.copy()
    otsu = threshold_otsu(out['noise_std'].values)
    out['Noise_Otsu'] = out['noise_std'].apply(lambda x: 'Clean' if x <= otsu else 'Noisy')

    km = KMeans(n_clusters=2, n_init=20, max_iter=300, random_state=random_state)
    labels = km.fit_predict(out['noise_std'].values.reshape(-1, 1))
    means = out.groupby(labels)['noise_std'].mean()
    noisy_lab = means.idxmax()
    out['Noise_KMeans'] = ['Noisy' if lab == noisy_lab else 'Clean' for lab in labels]
    return out
