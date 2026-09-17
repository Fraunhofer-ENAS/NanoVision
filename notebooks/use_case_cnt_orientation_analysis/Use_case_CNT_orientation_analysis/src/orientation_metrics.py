import numpy as np
import pandas as pd

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def circular_error(angle1, angle2):
    """Compute absolute circular error between two angles (in degrees)."""
    return np.abs(((angle1 - angle2 + 90) % 180) - 90)

def circular_mae(angles1, angles2):
    """Compute circular mean absolute error."""
    errors = [circular_error(a1, a2) for a1, a2 in zip(angles1, angles2)]
    return np.mean(errors)

def aggregate_metrics(df_fft, df_mask, df_gt):
    """Aggregate metrics comparing FFT and mask orientations to GT."""
    merged_fft = df_fft.merge(df_gt, on='filename', suffixes=('_fft', '_gt'))
    merged_mask = df_mask.merge(df_gt, on='filename', suffixes=('_mask', '_gt'))
    metrics = {
        'fft_circ_mae': circular_mae(merged_fft['fft_orientation'], merged_fft['gt_orientation']),
        'mask_circ_mae': circular_mae(merged_mask['mask_orientation'], merged_mask['gt_orientation'])
    }
    return metrics




import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

def extract_orientations_from_profile(
    profile, angle_bins, n_peaks=5, sigma=1, min_distance_deg=20, wrap=True
):
#     # Circular convolution mode with "wrap"
#     # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html
    smoothed_profile = gaussian_filter1d(
        profile.astype(float), sigma=sigma, mode='wrap'
    )

    bin_size_deg = 180 / len(angle_bins)
    distance_bins = int(np.ceil(min_distance_deg / bin_size_deg))

    if wrap:
        # Duplicate signal for circular continuity
        extended_profile = np.concatenate([smoothed_profile, smoothed_profile])
        peaks, _ = find_peaks(extended_profile, distance=distance_bins)
        
        # Map back into original 0-180 domain
        mapped_peaks = peaks % len(smoothed_profile)
        mapped_vals = extended_profile[peaks]
        
        # Merge duplicates by taking the maximum value for each angle bin
        unique_peaks, idx = np.unique(mapped_peaks, return_index=True)
        peak_vals = np.zeros_like(unique_peaks, dtype=float)
        for i, p in enumerate(unique_peaks):
            mask = mapped_peaks == p
            peak_vals[i] = mapped_vals[mask].max()
        peaks = unique_peaks

    else:
        peaks, _ = find_peaks(smoothed_profile, distance=distance_bins)
        peak_vals = smoothed_profile[peaks]

    return smoothed_profile, peaks






def find_main_orientation(peaks, angle_bins, weights, max_orientations=2, energy_threshold=0.5):
    """Find dominant image orientations explaining up to 50% of angular energy (max 2)."""
    import numpy as np
    
    #weights /= weights.sum()
    # Extract weights and corresponding angles at peak positions
    peak_weights = np.asarray(weights)[peaks]
    peak_angles = np.asarray(angle_bins)[peaks]

    # Sort peaks by descending intensity
    sorted_idx = np.argsort(peak_weights)[::-1]
    sorted_weights = peak_weights[sorted_idx]
    sorted_angles = peak_angles[sorted_idx]

    # Compute cumulative normalized energy
    cumulative_energy = np.cumsum(sorted_weights)
    idx_threshold = np.searchsorted(cumulative_energy, energy_threshold)
    n_select = min(idx_threshold + 1, max_orientations)

    # Select orientations contributing up to 90% of energy, limited to max 4
    dominant_image_orientations = sorted_angles[:n_select]

    return dominant_image_orientations
