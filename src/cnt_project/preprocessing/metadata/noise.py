from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pywt
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans

from cnt_project.preprocessing.metadata.schemas import (
    FILENAME_COL,
    NOISE_CLASS_COL,
    NOISE_CLASS_KMEANS_COL,
    NOISE_CLASS_OTSU_COL,
    NOISE_SIGMA_COL,
)
from cnt_project.preprocessing.metadata.validation import (
    DatasetValidationError,
    ensure_metadata_coverage,
    validate_image_mask_pairs,
)


def estimate_noise_wavelet(image: np.ndarray, wavelet: str = "db1") -> float:
    """Estimate image noise sigma from wavelet diagonal detail coefficients."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    coeffs = pywt.wavedec2(image.astype(np.float32), wavelet=wavelet, level=1)
    _cA, (_cH, _cV, cD) = coeffs
    return float(np.median(np.abs(cD)) / 0.6745)


def classify_noise_otsu(sigmas: pd.Series) -> pd.Series:
    thr = float(threshold_otsu(sigmas.to_numpy(dtype=float)))
    return sigmas.apply(lambda x: "Clean" if float(x) <= thr else "Noisy")


def classify_noise_kmeans(sigmas: pd.Series, random_seed: int = 42) -> pd.Series:
    X = sigmas.to_numpy(dtype=float).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=random_seed, n_init=20, max_iter=300)
    labels = kmeans.fit_predict(X)

    tmp = pd.DataFrame({"label": labels, "sigma": sigmas.to_numpy(dtype=float)})
    means = tmp.groupby("label")["sigma"].mean()
    noisy_cluster = int(means.idxmax())

    mapped = ["Noisy" if int(lbl) == noisy_cluster else "Clean" for lbl in labels]
    return pd.Series(mapped, index=sigmas.index)


def generate_noise_metadata(
    dataset_root: Path,
    output_csv: Path,
    *,
    method: str = "otsu",
    random_seed: int = 42,
    wavelet: str = "db1",
    expected_shape: tuple[int, int] | None = None,
    include_comparison_columns: bool = True,
) -> pd.DataFrame:
    """
    Generate canonical noise metadata from the unsplit dataset.

    Output always includes:
      - filename
      - noise_sigma
      - noise_class
    """
    if method not in {"otsu", "kmeans"}:
        raise ValueError("method must be one of: 'otsu', 'kmeans'")

    images_root = dataset_root / "images"
    masks_root = dataset_root / "masks"

    sample_filenames = validate_image_mask_pairs(images_root, masks_root)

    df = evaluate_noise_in_folder(
        image_dir=images_root,
        wavelet=wavelet,
        expected_shape=expected_shape,
        file_extensions=(".tif",),
    )

    otsu = classify_noise_otsu(df[NOISE_SIGMA_COL])
    kmeans = classify_noise_kmeans(df[NOISE_SIGMA_COL], random_seed=random_seed)

    df[NOISE_CLASS_OTSU_COL] = otsu
    df[NOISE_CLASS_KMEANS_COL] = kmeans
    df[NOISE_CLASS_COL] = df[NOISE_CLASS_OTSU_COL] if method == "otsu" else df[NOISE_CLASS_KMEANS_COL]

    ensure_metadata_coverage(
        sample_filenames=sample_filenames,
        metadata_filenames=df[FILENAME_COL].astype(str).tolist(),
        metadata_name="noise metadata",
    )

    if not include_comparison_columns:
        df = df[[FILENAME_COL, NOISE_SIGMA_COL, NOISE_CLASS_COL]]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def evaluate_noise_in_folder(
    image_dir: str | Path,
    output_csv: str | Path | None = None,
    *,
    wavelet: str = "db1",
    expected_shape: tuple[int, int] | None = None,
    file_extensions: tuple[str, ...] = (".tif", ".tiff"),
) -> pd.DataFrame:
    """
    Estimate wavelet noise sigma for every supported image in a folder.

    Unlike ``generate_noise_metadata``, this function does not require masks
    or a canonical dataset structure. It is intended for arbitrary image
    folders, exploratory analysis, and notebook workflows.

    Parameters
    ----------
    image_dir
        Folder containing the images to evaluate.

    output_csv
        Optional CSV output path. When omitted, no file is written.

    wavelet
        Wavelet used for the level-one decomposition.

    expected_shape
        Optional required spatial image shape as ``(height, width)``.

    file_extensions
        Supported filename extensions, including the leading dot.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing ``filename`` and ``noise_sigma``.
    """
    image_dir = Path(image_dir)

    if not image_dir.exists() or not image_dir.is_dir():
        raise DatasetValidationError(
            f"Image directory does not exist: {image_dir}"
        )

    normalized_extensions = {
        extension.lower()
        if extension.startswith(".")
        else f".{extension.lower()}"
        for extension in file_extensions
    }

    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in normalized_extensions
    )

    if not image_paths:
        raise DatasetValidationError(
            f"No supported images found in {image_dir}. "
            f"Supported extensions: {sorted(normalized_extensions)}"
        )

    rows: list[dict[str, float | str]] = []

    for image_path in image_paths:
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:
            raise DatasetValidationError(
                f"Failed to read image: {image_path}"
            )

        spatial_shape = (
            int(image.shape[0]),
            int(image.shape[1]),
        )

        if (
            expected_shape is not None
            and spatial_shape != expected_shape
        ):
            raise DatasetValidationError(
                f"Unexpected image shape for {image_path.name}: "
                f"{spatial_shape} != {expected_shape}"
            )

        rows.append(
            {
                FILENAME_COL: image_path.name,
                NOISE_SIGMA_COL: estimate_noise_wavelet(
                    image,
                    wavelet=wavelet,
                ),
            }
        )

    result = pd.DataFrame(rows)

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        result.to_csv(
            output_csv,
            index=False,
        )

    return result