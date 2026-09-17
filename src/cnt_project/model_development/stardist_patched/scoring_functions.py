from __future__ import print_function, unicode_literals, absolute_import, division

from csbdeep.utils.tf import keras_import, IS_TF_1, BACKEND as K
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from typing import Tuple
from datetime import datetime
import scienceplots
plt.style.use(["science", "no-latex"])
from matplotlib import cm
from scipy import ndimage
import warnings
import numpy as np
import warnings
import os, sys
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from scipy.ndimage import find_objects


try:
    from edt import edt
    _edt_available = True

    if hasattr(os, "sched_getaffinity"):
        try:
            _edt_parallel_max = len(os.sched_getaffinity(0))
        except Exception:
            _edt_parallel_max = os.cpu_count() or 1
    else:
        _edt_parallel_max = os.cpu_count() or 1

    _edt_parallel_default = 4
    _edt_parallel_env = os.environ.get("STARDIST_EDT_NUM_THREADS", str(_edt_parallel_default))

    try:
        _edt_parallel = min(_edt_parallel_max, int(_edt_parallel_env))
    except ValueError:
        warnings.warn(
            f"Invalid value ({_edt_parallel_env}) for STARDIST_EDT_NUM_THREADS. "
            f"Using default value ({_edt_parallel_default}) instead."
        )
        _edt_parallel = min(_edt_parallel_max, _edt_parallel_default)

    del _edt_parallel_default, _edt_parallel_max, _edt_parallel_env

except ImportError:
    _edt_available = False
    _edt_parallel = 1

def edt_prob(
    lbl_img,
    dist=None,
    anisotropy=None,
    mode='standard',
    output_prefix="edt",
    debug_dir=None,
):
    """
    Dispatch between fast EDT backend and scipy/custom scoring backend.

    Rules:
    - None mode can use the edt package when available
    - custom modes (standard ,centerness, length_biased, new_edt, edt_LB, ...) must use scipy path
    """
    if mode == "None" and _edt_available:
        return _edt_prob_edt(
            lbl_img,
            dist=dist,
            anisotropy=anisotropy,
            output_prefix=output_prefix,
            debug_dir=debug_dir,
        )

    return _edt_prob_scipy(
        lbl_img,
        dist,
        anisotropy=anisotropy,
        mode=mode,
        output_prefix=output_prefix,
        debug_dir=debug_dir,
    )

def _edt_prob_edt(lbl_img, dist=None, anisotropy=None):
    """Perform EDT on each labeled object and normalize.
    Internally uses https://github.com/seung-lab/euclidean-distance-transform-3d
    that can handle multiple labels at once
    """
    lbl_img = np.ascontiguousarray(lbl_img)
    constant_img = lbl_img.min() == lbl_img.max() and lbl_img.flat[0] > 0
    if constant_img:
        warnings.warn("EDT of constant label image is ill-defined. (Assuming background around it.)")
    # we just need to compute the edt once but then normalize it for each object
    prob = edt(lbl_img, anisotropy=anisotropy, black_border=constant_img, parallel=_edt_parallel)
    objects = find_objects(lbl_img)
    for i,sl in enumerate(objects,1):
        # i: object label id, sl: slices of object in lbl_img
        if sl is None: continue
        _mask = lbl_img[sl]==i
        # normalize it
        prob[sl][_mask] /= np.max(prob[sl][_mask]+1e-10)
    plt.imsave('old_edt_result.png', prob, cmap='viridis')  
    return prob


# ─── Helpers: slice growth/shrink ──────────────────────────────────────────────

def _grow_standard(sl, interior):
    """Grow slices for bounding box exactly as in old version."""
    return tuple(slice(s.start - int(w[0]), s.stop + int(w[1])) for s, w in zip(sl, interior))

def _shrink_standard(interior):
    """Shrink slices for bounding box exactly as in old version."""
    # Note: slice int(w[0]) to (-1 if w[1] else None) matches old behavior
    return tuple(slice(int(w[0]), (-1 if w[1] else None)) for w in interior)


# ─── Core EDT Variants ─────────────────────────────────────────────────────────

def edt_standard(lbl_img: np.ndarray, anisotropy=None) -> np.ndarray:
    """
    Compute the normalized Euclidean Distance Transform (EDT) per object.

    For each labeled object in the image:
    - Grows the bounding box to ensure stable boundary computation.
    - Applies EDT within the grown region.
    - Shrinks the result to avoid border artifacts.
    - Normalizes the distance values per object (maximum distance is 1).
    - Returns a full image where each object's interior is filled with its normalized EDT.

    This function avoids writing into background or overlapping object regions.
    """
    # Check for constant label image edge case (optional, copy from old)
    constant_img = lbl_img.min() == lbl_img.max() and lbl_img.flat[0] > 0
    if constant_img:
        lbl_img = np.pad(lbl_img, ((1, 1),) * lbl_img.ndim, mode='constant')
        warnings.warn("EDT of constant label image is ill-defined. (Assuming background around it.)")

    objects = find_objects(lbl_img)
    out = np.zeros_like(lbl_img, np.float32)

    for i, sl in enumerate(objects, 1):
        if sl is None:
            continue

        # Determine if slice is at image boundary edges
        interior = [(s.start > 0, s.stop < sz) for s, sz in zip(sl, lbl_img.shape)]

        # Grow and shrink slices exactly as in old function
        grow_slice = _grow_standard(sl, interior)
        shrink_slice = _shrink_standard(interior)

        # Extract grown mask and then shrunk mask (should match shape of lbl_img[sl])
        grown_mask = (lbl_img[grow_slice] == i)
        shrunk_mask = grown_mask[shrink_slice]

        # Compute distance transform EDT on grown mask, then crop to shrunk region
        edt_result = distance_transform_edt(grown_mask, sampling=anisotropy)[shrink_slice]

        # Normalize per object (only pixels within object mask)
        max_dist = np.max(edt_result[shrunk_mask])
        edt_norm = edt_result / (max_dist + 1e-10)

        # Assign normalized values back to output using original slice and mask
        out[sl][shrunk_mask] = edt_norm[shrunk_mask]

    if constant_img:
        # Remove padding if added earlier
        out = out[(slice(1, -1),) * lbl_img.ndim]

    return out

def edt_centerness_old(lbl_img: np.ndarray, dist: np.ndarray, anisotropy=None) -> np.ndarray:
    """
    Combined centerness: C(p) = (sum |d_i - d_{i+N/2}|)/max_asym
    """
    objects = find_objects(lbl_img)
    C = np.zeros_like(lbl_img, np.float32)

    def symmetry_score(dist_patch):
        """Calculate the sum of absolute differences between opposite angle pairs."""
        num_angles = dist_patch.shape[-1]
        assert num_angles % 2 == 0, "Number of angles must be even to pair opposites."

        dist_sum = np.zeros(dist_patch.shape[:-1], np.float32)
        half_angles = num_angles // 2

        for i in range(half_angles):
            dist_1 = dist_patch[..., i]
            dist_2 = dist_patch[..., i + half_angles]
            dist_sum += np.abs(dist_1 - dist_2)

        return dist_sum

    for i, sl in enumerate(objects, 1):
        if sl is None:
            continue

        interior = [(s.start > 0, s.stop < sz) for s, sz in zip(sl, lbl_img.shape)]
        grow_slice = _grow_standard(sl, interior)
        shrink_slice = _shrink_standard(interior)

        grown_mask = lbl_img[grow_slice] == i
        shrunk_mask = grown_mask[shrink_slice]

        # Extract and compute asymmetry
        dist_patch = dist[grow_slice]
        asym = symmetry_score(dist_patch)
        asym_crop = asym[shrink_slice]
        asym_norm = asym_crop / (np.max(asym_crop[shrunk_mask]) + 1e-10)

        # Invert asymmetry to get centerness
        sym_score = 1.0 - asym_norm

        # Safe write-back
        buf = np.zeros_like(sym_score, dtype=np.float32)
        buf[shrunk_mask] = sym_score[shrunk_mask]

        region = C[grow_slice][shrink_slice]
        region[buf > 0] = buf[buf > 0]
        C[grow_slice][shrink_slice] = region

    # Global normalization
    min_val, max_val = np.min(C), np.max(C)
    if max_val > min_val:
        C = (C - min_val) / (max_val - min_val + 1e-10)

    return C

def edt_centerness(lbl_img: np.ndarray, dist: np.ndarray, anisotropy=None) -> np.ndarray:
    """
    Compute an improved centerness score using a combination of EDT and radial symmetry.

    For each object:
    - Computes normalized Euclidean Distance Transform (EDT) on a grown region.
    - Calculates a radial symmetry score based on differences between opposite directions.
    - Inverts and normalizes the symmetry score to emphasize symmetry.
    - Multiplies the two scores to obtain a refined centerness map.
    - Writes back values only for the object's true pixels.
    """
    objects = find_objects(lbl_img)
    accum = np.zeros_like(lbl_img, np.float32)

    def symmetry_score(dist_patch: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
        """
        Compute symmetry score:
        - For each pixel, compare opposite angular directions.
        - Invert and normalize asymmetry.
        - Pixels in the background (object_mask == False) get score 0.
        - Symmetric pixels get values near 1.

        Parameters:
        - dist_patch: (H, W, N_angles)
        - object_mask: (H, W) boolean mask for valid pixels (e.g., shrunk_mask)

        Returns:
        - sym_score: (H, W) float32 values in [0, 1]
        """
        num_angles = dist_patch.shape[-1]
        assert num_angles % 2 == 0, "Number of angles must be even."

        half = num_angles // 2
        asym = np.zeros(dist_patch.shape[:-1], np.float32)

        for i in range(half):
            diff = np.abs(dist_patch[..., i] - dist_patch[..., i + half])
            asym += diff

        # Normalize asymmetry only within the object
        max_asym = np.max(asym[object_mask]) + 1e-10
        asym_norm = np.zeros_like(asym, dtype=np.float32)
        asym_norm[object_mask] = asym[object_mask] / max_asym

        # Invert to get symmetry score
        sym_score = np.zeros_like(asym_norm)
        sym_score[object_mask] = 1.0 - asym_norm[object_mask]

        return sym_score

    for i, sl in enumerate(objects, 1):
        if sl is None:
            continue

        interior = [(s.start > 0, s.stop < sz) for s, sz in zip(sl, lbl_img.shape)]
        gsl, ssl = _grow_standard(sl, interior), _shrink_standard(interior)

        grown_mask = lbl_img[gsl] == i
        shrunk_mask = grown_mask[ssl]

        # 1. EDT
        D = distance_transform_edt(grown_mask, sampling=anisotropy)
        D_crop = D[ssl]
        D_crop = D_crop / (np.max(D_crop[shrunk_mask]) + 1e-10)
        # D_norm = D / (D.max() + 1e-10)

        # 2. Symmetry
        dist_patch = dist[gsl]
        # S_patch = symmetry_score(dist_patch)
        # S_crop = S_patch[ssl]
        # S_crop = S_crop / (np.max(S_crop[shrunk_mask]) + 1e-10)
        # S_crop = np.where(S_crop != 0, 1 - S_crop, S_crop)
        S_crop = symmetry_score(dist_patch[ssl], shrunk_mask)




        # 3. Combine and store only valid object pixels
        # C_patch = D_norm * S_patch
        C_crop = S_crop
        # 4. Per-object normalization of C_crop
        valid_vals = C_crop[shrunk_mask]
        c_min, c_max = valid_vals.min(), valid_vals.max()
        if c_max > c_min:
            C_crop = (C_crop - c_min) / (c_max - c_min + 1e-10)

        patch_buf = np.zeros_like(C_crop, dtype=np.float32)
        patch_buf[shrunk_mask] = C_crop[shrunk_mask]

        accum_view = accum[gsl]
        patch_region = accum_view[ssl]

        # Only overwrite where patch_buf is non-zero
        patch_region[patch_buf > 0] = patch_buf[patch_buf > 0]

        # Write it back
        accum_view[ssl] = patch_region
        accum[gsl] = accum_view

    # 4. Global normalization
    mn, mx = accum.min(), accum.max()
    if mx > mn:
        accum = (accum - mn) / (mx - mn + 1e-10)

    return accum

def edt_length_biased(lbl_img: np.ndarray, dist: np.ndarray, anisotropy=None) -> np.ndarray:
    """
    Compute a length-biased centerness score combining EDT and symmetry.

    For each object:
    - Computes the normalized Euclidean Distance Transform (EDT).
    - Measures radial asymmetry by comparing opposite directions in `dist`.
    - Inverts and normalizes the symmetry score.
    - Combines the EDT and symmetry: `score = EDT * (1 - asymmetry)`.
    - Writes back values only inside the object region.

    This function emphasizes central, symmetric, and well-insulated regions of each object.
    """
    from scipy.ndimage import distance_transform_edt, find_objects

    def symmetry_score(dist_patch):
        """Sum of differences between opposite angle pairs."""
        num_angles = dist_patch.shape[-1]
        half = num_angles // 2
        asym = np.zeros(dist_patch.shape[:-1], np.float32)
        for i in range(half):
            asym += np.abs(dist_patch[..., i] - dist_patch[..., i + half])
        return asym

    accum = np.zeros_like(lbl_img, np.float32)
    objects = find_objects(lbl_img)

    for i, sl in enumerate(objects, 1):
        if sl is None:
            continue

        interior = [(s.start > 0, s.stop < sz) for s, sz in zip(sl, lbl_img.shape)]
        gsl = _grow_standard(sl, interior)
        ssl = _shrink_standard(interior)

        grown_mask = lbl_img[gsl] == i
        shrunk_mask = grown_mask[ssl]

        # 1. Distance transform
        D = distance_transform_edt(grown_mask, sampling=anisotropy)
        D_crop = D[ssl]
        D_norm = D_crop / (np.max(D_crop[shrunk_mask]) + 1e-10)

        # 2. Symmetry
        dist_patch = dist[gsl]
        asym = symmetry_score(dist_patch)
        S_crop = asym[ssl]
        S_crop = S_crop / (np.max(S_crop[shrunk_mask]) + 1e-10)
        Sym_crop = 1.0 - S_crop

        # 3. Combine
        C_crop = D_norm * Sym_crop

        # 4. Safe write-back
        patch_buf = np.zeros_like(C_crop)
        patch_buf[shrunk_mask] = C_crop[shrunk_mask]

        accum_view = accum[gsl]
        region = accum_view[ssl]
        region[patch_buf > 0] = patch_buf[patch_buf > 0]
        accum_view[ssl] = region
        accum[gsl] = accum_view

    # 5. Optional global normalization (comment out if you want raw per-object contrast)
    mn, mx = accum.min(), accum.max()
    if mx > mn:
        accum = (accum - mn) / (mx - mn + 1e-10)

    return accum

def new_edt(lbl_img: np.ndarray, dist: np.ndarray, anisotropy=None) -> np.ndarray:
    """
    Compute an improved centerness score using a combination of EDT and radial symmetry.

    For each object:
    - Computes normalized Euclidean Distance Transform (EDT) on a grown region.
    - Calculates a radial symmetry score based on differences between opposite directions.
    - Inverts and normalizes the symmetry score to emphasize symmetry.
    - Multiplies the two scores to obtain a refined centerness map.
    - Writes back values only for the object's true pixels.
    """
    objects = find_objects(lbl_img)
    accum = np.zeros_like(lbl_img, np.float32)

    def symmetry_score(dist_patch: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
        """
        Compute symmetry score:
        - For each pixel, compare opposite angular directions.
        - Invert and normalize asymmetry.
        - Pixels in the background (object_mask == False) get score 0.
        - Symmetric pixels get values near 1.

        Parameters:
        - dist_patch: (H, W, N_angles)
        - object_mask: (H, W) boolean mask for valid pixels (e.g., shrunk_mask)

        Returns:
        - sym_score: (H, W) float32 values in [0, 1]
        """
        num_angles = dist_patch.shape[-1]
        assert num_angles % 2 == 0, "Number of angles must be even."

        half = num_angles // 2
        asym = np.zeros(dist_patch.shape[:-1], np.float32)

        for i in range(half):
            diff = np.abs(dist_patch[..., i] - dist_patch[..., i + half])
            asym += diff

        # Normalize asymmetry only within the object
        max_asym = np.max(asym[object_mask]) + 1e-10
        asym_norm = np.zeros_like(asym, dtype=np.float32)
        asym_norm[object_mask] = asym[object_mask] / max_asym

        # Invert to get symmetry score
        sym_score = np.zeros_like(asym_norm)
        sym_score[object_mask] = 1.0 - asym_norm[object_mask]

        return sym_score

    for i, sl in enumerate(objects, 1):
        if sl is None:
            continue

        interior = [(s.start > 0, s.stop < sz) for s, sz in zip(sl, lbl_img.shape)]
        gsl, ssl = _grow_standard(sl, interior), _shrink_standard(interior)

        grown_mask = lbl_img[gsl] == i
        shrunk_mask = grown_mask[ssl]

        # 1. EDT
        D = distance_transform_edt(grown_mask, sampling=anisotropy)
        D_crop = D[ssl]
        D_crop = D_crop / (np.max(D_crop[shrunk_mask]) + 1e-10)
        # D_norm = D / (D.max() + 1e-10)

        # 2. Symmetry
        dist_patch = dist[gsl]
        # S_patch = symmetry_score(dist_patch)
        # S_crop = S_patch[ssl]
        # S_crop = S_crop / (np.max(S_crop[shrunk_mask]) + 1e-10)
        # S_crop = np.where(S_crop != 0, 1 - S_crop, S_crop)
        S_crop = symmetry_score(dist_patch[ssl], shrunk_mask)




        # 3. Combine and store only valid object pixels
        # C_patch = D_norm * S_patch
        C_crop = D_crop * S_crop
        # 4. Per-object normalization of C_crop
        valid_vals = C_crop[shrunk_mask]
        c_min, c_max = valid_vals.min(), valid_vals.max()
        if c_max > c_min:
            C_crop = (C_crop - c_min) / (c_max - c_min + 1e-10)

        patch_buf = np.zeros_like(C_crop, dtype=np.float32)
        patch_buf[shrunk_mask] = C_crop[shrunk_mask]

        accum_view = accum[gsl]
        patch_region = accum_view[ssl]

        # Only overwrite where patch_buf is non-zero
        patch_region[patch_buf > 0] = patch_buf[patch_buf > 0]

        # Write it back
        accum_view[ssl] = patch_region
        accum[gsl] = accum_view

    # 4. Global normalization
    mn, mx = accum.min(), accum.max()
    if mx > mn:
        accum = (accum - mn) / (mx - mn + 1e-10)

    return accum

def edt_LB(lbl_img: np.ndarray, dist: np.ndarray, anisotropy=None, mode="pixel") -> np.ndarray:
    """
    New EDT variant combining elongation, symmetry, and EDT scores:
    edt_LB = elongated_score * symm_score * edt_score

    Parameters:
    - lbl_img: label image with unique integer labels per object
    - dist: star-convex radial distances (H, W, N_angles)
    - anisotropy: optional sampling for EDT
    - mode: 'object' or 'pixel' to control how elongated_score is computed

    Returns:
    - Float image of same size as input, with per-pixel edt_LB score
    """
    from scipy.ndimage import distance_transform_edt, find_objects

    epsilon = 1e-10
    accum = np.zeros_like(lbl_img, np.float32)
    objects = find_objects(lbl_img)

    # Compute max over all dist values (only valid pixels)
    global_dist_max = np.max(dist[dist > 0]) + epsilon

    def symmetry_score(dist_patch):
        num_angles = dist_patch.shape[-1]
        assert num_angles % 2 == 0, "Number of angles must be even to pair opposites."
        half = num_angles // 2
        asym = np.zeros(dist_patch.shape[:-1], np.float32)
        for i in range(half):
            asym += np.abs(dist_patch[..., i] - dist_patch[..., i + half])
        return asym

    for i, sl in enumerate(objects, 1):
        if sl is None:
            continue

        interior = [(s.start > 0, s.stop < sz) for s, sz in zip(sl, lbl_img.shape)]
        gsl = _grow_standard(sl, interior)
        ssl = _shrink_standard(interior)

        grown_mask = lbl_img[gsl] == i
        shrunk_mask = grown_mask[ssl]

        # === 1. Elongated Score ===
        dist_patch = dist[gsl]
        dist_crop = dist_patch[ssl]

        if mode == "object":
            max_dist_obj = np.max(dist_crop[shrunk_mask]) + epsilon
            elongated_score = max_dist_obj / global_dist_max
            elongated_map = np.full_like(shrunk_mask, elongated_score, dtype=np.float32)
        elif mode == "pixel":
            pixel_max_dist = np.max(dist_crop, axis=-1)
            elongated_map = (pixel_max_dist / global_dist_max).astype(np.float32)
            elongated_map[~shrunk_mask] = 0  # suppress values outside the object

        else:
            raise ValueError("mode must be 'object' or 'pixel'")

        # === 2. Symmetry Score ===
        asym = symmetry_score(dist_patch)
        asym_crop = asym[ssl]
        asym_max = np.max(asym_crop[shrunk_mask]) + epsilon
        symm_map = 1.0 - (asym_crop / asym_max)
        symm_map[~shrunk_mask] = 0

        # === 3. EDT Score ===
        D = distance_transform_edt(grown_mask, sampling=anisotropy)
        D_crop = D[ssl]
        edt_max = np.max(D_crop[shrunk_mask]) + epsilon
        edt_norm = D_crop / edt_max
        edt_norm[~shrunk_mask] = 0

        # === 4. Combine scores ===
        score = elongated_map * symm_map * edt_norm

        # Optional per-object normalization (commented out)
        # score_vals = score[shrunk_mask]
        # s_min, s_max = score_vals.min(), score_vals.max()
        # if s_max > s_min:
        #     score = (score - s_min) / (s_max - s_min + epsilon)

        # === 5. Safe write-back ===
        patch_buf = np.zeros_like(score)
        patch_buf[shrunk_mask] = score[shrunk_mask]

        accum_view = accum[gsl]
        patch_region = accum_view[ssl]
        patch_region[patch_buf > 0] = patch_buf[patch_buf > 0]
        accum_view[ssl] = patch_region
        accum[gsl] = accum_view

    # Optional global normalization (commented out)
    a_min, a_max = accum.min(), accum.max()
    if a_max > a_min:
        accum = (accum - a_min) / (a_max - a_min + epsilon)

    return accum

# ─── Dispatcher ────────────────────────────────────────────────────────────────


def _edt_prob_scipy(
    lbl_img: np.ndarray,
    dist: np.ndarray = None,
    anisotropy=None,
    mode: str = 'standard',
    output_prefix: str = 'edt',
    debug_dir: str | None = None,
) -> Tuple[np.ndarray, None]:
    """
    Dispatch to one of EDT-based centerness calculators, then save a visualization.    
    """
    def _object_highscore_points(label_img: np.ndarray, score_map: np.ndarray):
        """Return per-object strong local peaks and the strongest candidate peak."""
        peak_filter_size = 3
        peak_rel_threshold = 0.9

        strongest_points = []
        strongest_values = []
        candidate_points = []

        for obj_id, sl in enumerate(find_objects(label_img), 1):
            if sl is None:
                continue

            obj_mask = (label_img[sl] == obj_id)
            if not np.any(obj_mask):
                continue

            patch_scores = np.array(score_map[sl], copy=True)
            patch_scores[~obj_mask] = 0
            local_scores = patch_scores[obj_mask]

            if local_scores.size == 0:
                continue

            obj_max = float(np.max(local_scores))
            if obj_max <= 0:
                continue

            local_max = ndimage.maximum_filter(patch_scores, size=peak_filter_size, mode='constant')
            peak_mask = (patch_scores == local_max) & obj_mask & (patch_scores >= peak_rel_threshold * obj_max)
            local_peak_coords = np.argwhere(peak_mask)

            if local_peak_coords.size == 0:
                local_peak_coords = np.argwhere(obj_mask)

            peak_scores = patch_scores[local_peak_coords[:, 0], local_peak_coords[:, 1]]
            best_idx = int(np.argmax(peak_scores))
            best_y, best_x = local_peak_coords[best_idx]

            strongest_points.append((sl[0].start + best_y, sl[1].start + best_x))
            strongest_values.append(float(patch_scores[best_y, best_x]))

            for py, px in local_peak_coords:
                candidate_points.append((sl[0].start + py, sl[1].start + px))

        return strongest_points, strongest_values, candidate_points

    constant_img = lbl_img.min() == lbl_img.max() and lbl_img.flat[0] > 0
    if constant_img:
        lbl_img = np.pad(lbl_img, ((1, 1),) * lbl_img.ndim, mode='constant')
        warnings.warn("EDT of constant label image is ill-defined. (Assuming background around it.)")

    # 1) Compute the map
    if mode == 'standard':
        edt_map = edt_standard(lbl_img, anisotropy)
    elif mode == 'centerness':
        if dist is None:
            raise ValueError("mode='centerness' requires a `dist` argument")
        edt_map = edt_centerness(lbl_img, dist, anisotropy)
    elif mode == 'length_biased':
        if dist is None:
            raise ValueError("mode='length_biased' requires a `dist` argument")
        edt_map = edt_length_biased(lbl_img, dist, anisotropy)
    elif mode == 'new_edt':
        if dist is None:
            raise ValueError("needs `dist` for new edt")
        edt_map = new_edt(lbl_img, dist, anisotropy)
    elif mode == 'edt_LB':
        if dist is None:
            raise ValueError("needs `dist` for new edt")
        edt_map = edt_LB(lbl_img, dist, anisotropy)
    else:
        raise ValueError(f"Unknown EDT mode '{mode}'")

    # 2) Save visualization (hardcoded folder + mode subfolder)
    if debug_dir is not None:
        mode_dir = os.path.join(debug_dir, str(mode))
        os.makedirs(mode_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = os.path.join(mode_dir, f"{output_prefix}_{timestamp}.png")
        strongest_points, strongest_values, candidate_points = _object_highscore_points(lbl_img, edt_map)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(edt_map, cmap='viridis')
        fig.colorbar(im, ax=ax, label='EDT Distance')

        if candidate_points:
            ys, xs = zip(*candidate_points)
            ax.scatter(
                xs,
                ys,
                s=26,
                c='yellow',
                edgecolors='black',
                linewidths=0.4,
                alpha=0.35,
                label='Local Peak Candidates',
            )

        if strongest_points:
            ys, xs = zip(*strongest_points)
            ax.scatter(
                xs,
                ys,
                s=42,
                c='red',
                marker='x',
                linewidths=1.3,
                alpha=0.45,
                label='Strongest Local Peak',
            )

        for (py, px), peak_val in zip(strongest_points, strongest_values):
            ax.text(
                px + 2,
                py - 2,
                f"{peak_val:.3f}",
                color='red',
                fontsize=7,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.15'),
            )

        if strongest_points or candidate_points:
            ax.legend(loc='upper right', framealpha=0.85)

        ax.set_title(f'EDT Map - {mode}')
        ax.axis('off')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    return edt_map
