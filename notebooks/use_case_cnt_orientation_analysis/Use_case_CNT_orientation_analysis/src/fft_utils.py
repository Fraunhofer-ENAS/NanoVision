import numpy as np
import cv2
import os
import importlib
import sys, os

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.orientation_metrics import extract_orientations_from_profile

def generate_fft_meshgrid(shape):
    """Generate a centered FFT meshgrid for visualization or computation."""
    y, x = np.indices(shape)
    center = (shape[0] // 2, shape[1] // 2)
    return x - center[1], y - center[0]

def remove_low_frequencies(fshift, p=0.01, use_hanning=False):
    """
    Removes low frequencies from a 2D Fourier transform (acts as a high-pass filter).

    Parameters:
        fshift (ndarray): Shifted 2D Fourier transform.
        p (float): Fraction of the lowest frequencies to remove (e.g. 0.01 = 1%).
        use_hanning (bool): Apply a smooth Hanning window transition.

    Returns:
        ndarray: Filtered Fourier transform.
    """
    h, w = fshift.shape
    cx, cy = w // 2, h // 2

    y, x = np.ogrid[:h, :w]
    center_dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    radius = min(h, w) * p

    mask = np.ones_like(fshift, dtype=float)
    mask[center_dist < radius] = 0  # remove low frequencies

    if use_hanning:
        # smooth transition with a 2D Hanning window
        window_y = np.hanning(h)[:, np.newaxis]
        window_x = np.hanning(w)[np.newaxis, :]
        window_2d = np.sqrt(window_y * window_x)
        mask *= window_2d

    return fshift * mask


def remove_high_frequencies(fshift, p=0.01, use_hanning=False, steep=False):
    """
    Removes high frequencies from a 2D Fourier transform (acts as a low-pass filter).

    Parameters:
        fshift (ndarray): Shifted 2D Fourier transform.
        p (float): Fraction of the highest frequencies to remove (e.g. 0.01 = 1%).
        use_hanning (bool): Apply a smooth Hanning window.
        steep (bool): Use a steeper cutoff (less than 5% of high freq removed).

    Returns:
        ndarray: Filtered Fourier transform.
    """
    h, w = fshift.shape
    cx, cy = w // 2, h // 2

    y, x = np.ogrid[:h, :w]
    center_dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    radius = min(h, w) * (1 - p)

    mask = np.zeros_like(fshift, dtype=float)
    mask[center_dist < radius] = 1  # keep low frequencies

    if steep:
        # Sharper roll-off by compressing the cutoff region
        transition_width = int(min(h, w) * 0.02)
        transition_mask = (center_dist > radius - transition_width) & (center_dist < radius)
        mask[transition_mask] = np.linspace(1, 0, transition_mask.sum())

    if use_hanning:
        # Apply smooth Hanning weighting
        window_y = np.hanning(h)[:, np.newaxis]
        window_x = np.hanning(w)[np.newaxis, :]
        window_2d = np.sqrt(window_y * window_x)
        mask *= window_2d

    return fshift * mask

def image_scaling(image):
    """Scale image to [0, 1] range."""
    img_min = np.min(image)
    img_max = np.max(image)
    if img_max - img_min == 0:
        return np.zeros_like(image)
    return (image - img_min) / (img_max - img_min)


import numpy as np

def compute_fft(image, mode="log"):
    """
    Compute the frequency magnitude spectrum of an image using FFT.
    
    Parameters:
        image (np.ndarray): Input grayscale image.
        mode (str): How to scale the magnitude. Options:
            - "log": logarithmic magnitude (default)
            - "energy": squared magnitude
            - "normalized_spectrum": normalized magnitude [0,1]
    
    Returns:
        np.ndarray: Processed FFT magnitude spectrum.
    """
    # --- Normalize image to [0, 1] for numerical stability ---
    image = image_scaling(image)

    # --- Compute 2D FFT and shift zero-frequency component to center ---
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # --- Apply frequency filters (if defined) ---
    fshift = remove_low_frequencies(fshift, p=0.01, use_hanning=True)
    fshift = remove_high_frequencies(fshift, p=0.05, steep=True)

    # --- Compute magnitude spectrum ---
    magnitude = np.abs(fshift)

    # --- Scale based on selected mode ---
    if mode == "log":
        spectrum = 20 * np.log10(magnitude + 1e-8)
    elif mode == "energy":
        spectrum = magnitude ** 2
    elif mode == "normalized_spectrum":
        spectrum = image_scaling(magnitude)
    elif mode == "default":
        spectrum = magnitude
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return spectrum


def compute_angular_profile(log_magnitude, num_bins=36):
    # https://github.com/lastnpcalex/FFT
    # https://github.com/lastnpcalex/FFT/blob/main/fft_analysis.py

    # WARNING:
    # rotate FFT plot so angular profile is the orientation profile
    # The rotated fft plot is for convenience
    # to get the true FFT plot, do not rotate it 
    # the orientations angles become: rotated_bins = (angle_bins + 90) % 180

    log_magnitude = np.rot90(log_magnitude, k=1)
    rows, cols = log_magnitude.shape
    center = np.array([rows // 2, cols // 2])
    Y, X = np.indices(log_magnitude.shape)
    angles = np.arctan2(Y - center[0], X - center[1])
    angles_deg = np.degrees(angles) % 180
    angles_flat = angles_deg.ravel()
    energy_flat = log_magnitude.ravel()
    bins = np.linspace(0, 180, num_bins+1)
    bin_indices = np.digitize(angles_flat, bins) - 1
    angular_sum = np.zeros(num_bins)
    angular_count = np.zeros(num_bins)
    for i in range(num_bins):
        mask = (bin_indices == i)
        angular_sum[i] = energy_flat[mask].sum()
        angular_count[i] = np.sum(mask)
    angular_profile = angular_sum / (angular_count + 1e-8)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    return bin_centers, angular_profile



def extract_fft_features(image):
    """Load image, compute FFT orientation profile, and return top peaks + profile."""
    spectrum = compute_fft(image, mode="default")
    angle_bins, angular_profile = compute_angular_profile(spectrum, num_bins=36)

    smoothed_profile, peaks = extract_orientations_from_profile(
        angular_profile,
        angle_bins,
        n_peaks=5,
        sigma=1,
        min_distance_deg=20,
        wrap=True
    )
    

    return smoothed_profile, peaks, angle_bins




    
