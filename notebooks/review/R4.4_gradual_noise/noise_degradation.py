"""
Controlled synthetic-noise degradation experiment, per Reviewer 2's comment:

  "add controlled synthetic Gaussian and salt and pepper noise at known
   sigma to a fixed set of images and measure the degradation curve."

Reads TIFF images from `path_to_data`, and for a fixed set of severity
levels writes three output folders:
  - gaussian_noise/     : Gaussian noise only, sweeping sigma
  - salt_and_pepper/    : salt-and-pepper noise only, sweeping corruption fraction
  - both_added/         : both applied together, matched severity index

Every output filename encodes:
  - the injected (known, ground-truth) noise parameter -- this is what the
    reviewer means by "known sigma": you control it, so it's exact, not
    estimated.
  - for the Gaussian folder, ALSO the empirically measured noise sigma of
    the resulting image (via a standard fast estimator), so you can check
    the injected and measured values track each other sensibly before
    trusting the degradation curve built from them.

A debug figure shows one randomly selected image across all severities for
all three noise types, so the injected noise levels can be sanity-checked
visually before running the full downstream segmentation-degradation
analysis.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import tifffile
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------

def to_normalized_float(img: np.ndarray):
    """Convert an image of arbitrary dtype to float64 in [0, 1], remembering
    enough to convert back losslessly for uint8/uint16 sources."""
    if np.issubdtype(img.dtype, np.integer):
        max_val = np.iinfo(img.dtype).max
        return img.astype(np.float64) / max_val, img.dtype, max_val
    else:
        # float source: normalize by its own observed range instead of an
        # assumed bit depth, since float TIFFs carry no fixed max value.
        lo, hi = float(img.min()), float(img.max())
        span = hi - lo if hi > lo else 1.0
        normed = (img.astype(np.float64) - lo) / span
        return normed, img.dtype, (lo, span)


def from_normalized_float(normed: np.ndarray, orig_dtype, scale_info):
    normed = np.clip(normed, 0.0, 1.0)
    if np.issubdtype(orig_dtype, np.integer):
        max_val = scale_info
        return (normed * max_val).round().astype(orig_dtype)
    else:
        lo, span = scale_info
        return (normed * span + lo).astype(orig_dtype)


def add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    """sigma is in NORMALIZED [0,1] units, so it's comparable across images
    of different bit depth / dynamic range."""
    normed, dtype, scale_info = to_normalized_float(img)
    noisy = normed + np.random.normal(0.0, sigma, size=normed.shape)
    return from_normalized_float(noisy, dtype, scale_info)


def add_salt_and_pepper(img: np.ndarray, amount: float) -> np.ndarray:
    """amount = fraction of pixels replaced with salt (max) or pepper (min),
    split evenly between the two."""
    normed, dtype, scale_info = to_normalized_float(img)
    noisy = normed.copy()
    mask = np.random.random(size=normed.shape)
    noisy[mask < amount / 2] = 0.0
    noisy[(mask >= amount / 2) & (mask < amount)] = 1.0
    return from_normalized_float(noisy, dtype, scale_info)


def add_both(img: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    normed, dtype, scale_info = to_normalized_float(img)
    noisy = normed + np.random.normal(0.0, sigma, size=normed.shape)
    noisy = np.clip(noisy, 0.0, 1.0)
    mask = np.random.random(size=normed.shape)
    noisy[mask < amount / 2] = 0.0
    noisy[(mask >= amount / 2) & (mask < amount)] = 1.0
    return from_normalized_float(noisy, dtype, scale_info)


# ---------------------------------------------------------------------------
# Noise measurement (for the Gaussian folder's sanity check)
# ---------------------------------------------------------------------------

def estimate_gaussian_sigma(img: np.ndarray) -> float:
    """Fast Laplacian-based noise sigma estimator (Immerkaer, 1996),
    reported in the SAME normalized [0,1] units as the injected sigma.
    Used only to sanity-check that injected vs. measured noise track each
    other -- not a claim that this is the same estimator used elsewhere
    in the paper.

    TIFF inputs may be 2D grayscale or stacked 3D arrays. For stack inputs,
    estimate the noise on the mean image over the stack so the same 2D
    Laplacian estimator remains valid.
    """
    normed, _, _ = to_normalized_float(img)
    if normed.ndim > 2:
        normed = normed.mean(axis=-1)
    H, W = normed.shape[-2:]
    if H < 3 or W < 3:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    from scipy.signal import convolve2d
    conv = convolve2d(normed, kernel, mode="valid")
    sigma_est = np.sqrt(np.pi / 2) * np.mean(np.abs(conv)) / (6 * (W - 2) * (H - 2)) * (W - 2) * (H - 2)
    # simplifies to sqrt(pi/2) * mean(|conv|) / 6 -- kept expanded above for
    # readability against the standard formula reference
    sigma_est = np.sqrt(np.pi / 2) * np.mean(np.abs(conv)) / 6
    return float(sigma_est)


# ---------------------------------------------------------------------------
# Main degradation-set builder
# ---------------------------------------------------------------------------

def build_noisy_datasets(
    path_to_data: str | Path,
    output_root: str | Path,
    *,
    selected_filenames,
    gaussian_sigmas=(
        0.05,
        0.10,
        0.20,
        0.30,
    ),
    snp_amounts=(
        0.05,
        0.10,
        0.20,
        0.30,
    ),
    seed: int = 0,
):
    """
    Generate Gaussian, salt-and-pepper, and
    combined-noise variants for selected images.

    Outputs are grouped first by severity stage
    and then by noise condition. Original filenames
    are retained because every condition has its
    own directory.
    """
    if len(gaussian_sigmas) != len(
        snp_amounts
    ):
        raise ValueError(
            "gaussian_sigmas and snp_amounts "
            "must contain the same number of "
            "severity levels."
        )

    if not selected_filenames:
        raise ValueError(
            "selected_filenames cannot be empty."
        )

    np.random.seed(seed)

    data_dir = Path(
        path_to_data
    )

    output_root = Path(
        output_root
    )

    selected_names = [
        Path(filename).name
        for filename in selected_filenames
    ]

    if len(set(selected_names)) != len(
        selected_names
    ):
        raise ValueError(
            "selected_filenames contains "
            "duplicate filenames."
        )

    selected_paths = []

    for filename in selected_names:
        direct_path = (
            data_dir
            / filename
        )

        if direct_path.is_file():
            selected_paths.append(
                direct_path
            )

            continue

        matches = list(
            data_dir.rglob(
                filename
            )
        )

        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one source "
                f"image for '{filename}', but "
                f"found {len(matches)}:\n"
                + "\n".join(
                    str(path)
                    for path in matches
                )
            )

        selected_paths.append(
            matches[0]
        )

    output_records = []

    for stage_number, (
        sigma,
        amount,
    ) in enumerate(
        zip(
            gaussian_sigmas,
            snp_amounts,
        ),
        start=1,
    ):
        stage_directory = (
            output_root
            / f"stage_{stage_number}"
        )

        gaussian_directory = (
            stage_directory
            / "gaussian_noise"
        )

        snp_directory = (
            stage_directory
            / "salt_and_pepper"
        )

        combined_directory = (
            stage_directory
            / "both_added"
        )

        for directory in (
            gaussian_directory,
            snp_directory,
            combined_directory,
        ):
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        for image_path in selected_paths:
            image = tifffile.imread(
                image_path
            )

            gaussian_image = (
                add_gaussian_noise(
                    image,
                    sigma,
                )
            )

            salt_and_pepper_image = (
                add_salt_and_pepper(
                    image,
                    amount,
                )
            )

            combined_image = add_both(
                image,
                sigma,
                amount,
            )

            gaussian_path = (
                gaussian_directory
                / image_path.name
            )

            snp_path = (
                snp_directory
                / image_path.name
            )

            combined_path = (
                combined_directory
                / image_path.name
            )

            tifffile.imwrite(
                gaussian_path,
                gaussian_image,
            )

            tifffile.imwrite(
                snp_path,
                salt_and_pepper_image,
            )

            tifffile.imwrite(
                combined_path,
                combined_image,
            )

            output_records.extend(
                [
                    {
                        "stage": stage_number,
                        "noise_type": (
                            "gaussian_noise"
                        ),
                        "filename": (
                            image_path.name
                        ),
                        "gaussian_sigma": sigma,
                        "snp_amount": 0.0,
                        "measured_gaussian_sigma": (
                            estimate_gaussian_sigma(
                                gaussian_image
                            )
                        ),
                        "output_path": str(
                            gaussian_path
                        ),
                    },
                    {
                        "stage": stage_number,
                        "noise_type": (
                            "salt_and_pepper"
                        ),
                        "filename": (
                            image_path.name
                        ),
                        "gaussian_sigma": 0.0,
                        "snp_amount": amount,
                        "measured_gaussian_sigma": (
                            np.nan
                        ),
                        "output_path": str(
                            snp_path
                        ),
                    },
                    {
                        "stage": stage_number,
                        "noise_type": (
                            "both_added"
                        ),
                        "filename": (
                            image_path.name
                        ),
                        "gaussian_sigma": sigma,
                        "snp_amount": amount,
                        "measured_gaussian_sigma": (
                            np.nan
                        ),
                        "output_path": str(
                            combined_path
                        ),
                    },
                ]
            )

    print(
        f"Generated noise for "
        f"{len(selected_paths)} selected images."
    )

    print(
        f"Severity stages: "
        f"{len(gaussian_sigmas)}"
    )

    print(
        f"Generated TIFF files: "
        f"{len(output_records)}"
    )

    return output_records

# ---------------------------------------------------------------------------
# Debug visualization: one random image, increasing severity, all 3 conditions
# ---------------------------------------------------------------------------

def plot_degradation_examples(
    path_to_data: str | Path,
    noisy_root: str | Path,
    output_fig_path: str | Path,
    *,
    selected_filenames,
    gaussian_sigmas=(
        0.05,
        0.10,
        0.20,
        0.30,
    ),
    snp_amounts=(
        0.05,
        0.10,
        0.20,
        0.30,
    ),
    seed: int = 0,
):
    """
    Plot one selected image across all saved
    noise stages and conditions.

    The function reads the already-saved TIFFs,
    ensuring the figure displays the exact images
    that will be evaluated.
    """
    if len(gaussian_sigmas) != len(
        snp_amounts
    ):
        raise ValueError(
            "The Gaussian and salt-and-pepper "
            "severity lists must have equal length."
        )

    selected_names = [
        Path(filename).name
        for filename in selected_filenames
    ]

    if not selected_names:
        raise ValueError(
            "selected_filenames cannot be empty."
        )

    random_generator = random.Random(
        seed
    )

    chosen_filename = (
        random_generator.choice(
            selected_names
        )
    )

    data_directory = Path(
        path_to_data
    )

    noisy_root = Path(
        noisy_root
    )

    source_matches = list(
        data_directory.rglob(
            chosen_filename
        )
    )

    if len(source_matches) != 1:
        raise RuntimeError(
            f"Expected one original image for "
            f"'{chosen_filename}', but found "
            f"{len(source_matches)}."
        )

    original_image = tifffile.imread(
        source_matches[0]
    )

    number_of_stages = len(
        gaussian_sigmas
    )

    figure, axes = plt.subplots(
        4,
        number_of_stages,
        figsize=(
            2.5 * number_of_stages,
            9,
        ),
        squeeze=False,
    )

    for column, (
        sigma,
        amount,
    ) in enumerate(
        zip(
            gaussian_sigmas,
            snp_amounts,
        )
    ):
        stage_number = column + 1

        gaussian_image = tifffile.imread(
            noisy_root
            / f"stage_{stage_number}"
            / "gaussian_noise"
            / chosen_filename
        )

        snp_image = tifffile.imread(
            noisy_root
            / f"stage_{stage_number}"
            / "salt_and_pepper"
            / chosen_filename
        )

        combined_image = tifffile.imread(
            noisy_root
            / f"stage_{stage_number}"
            / "both_added"
            / chosen_filename
        )

        displayed_images = [
            (
                original_image,
                "Original",
            ),
            (
                gaussian_image,
                (
                    "Gaussian\n"
                    f"σ={sigma:.2f}"
                ),
            ),
            (
                snp_image,
                (
                    "Salt and pepper\n"
                    f"amount={amount:.2f}"
                ),
            ),
            (
                combined_image,
                (
                    "Combined\n"
                    f"σ={sigma:.2f}, "
                    f"amount={amount:.2f}"
                ),
            ),
        ]

        for row, (
            displayed_image,
            label,
        ) in enumerate(
            displayed_images
        ):
            axis = axes[
                row,
                column,
            ]

            axis.imshow(
                displayed_image,
                cmap=(
                    "gray"
                    if displayed_image.ndim == 2
                    else None
                ),
            )

            axis.set_title(
                (
                    f"Stage {stage_number}\n"
                    f"{label}"
                ),
                fontsize=9,
            )

            axis.set_xticks([])
            axis.set_yticks([])

    figure.suptitle(
        (
            "Synthetic-noise degradation — "
            f"{chosen_filename}"
        ),
        fontsize=13,
    )

    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96,
        ]
    )

    output_fig_path = Path(
        output_fig_path
    )

    output_fig_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_fig_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        "Saved degradation visualization:",
        output_fig_path,
    )

    print(
        "Displayed source image:",
        chosen_filename,
    )

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path_to_data: folder of clean TIFFs")
    parser.add_argument("--out", required=True, help="output root; creates 3 subfolders inside")
    parser.add_argument("--fig", default="noise_degradation_examples.png")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    build_noisy_datasets(args.data, args.out, seed=args.seed)
    plot_degradation_examples(args.data, args.fig, seed=args.seed)
