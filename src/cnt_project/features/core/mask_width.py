"""Canonical CNT width measurement from a binary object mask."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label

from cnt_project.features.core.geodesic_length import (
    polygon_mask_geodesic_length,
)


def calculate_mask_width_px(
    mask: np.ndarray,
) -> float:
    """
    Calculate canonical CNT width in pixels.

    Width is twice the mean Euclidean distance-transform value
    sampled along the center-weighted geodesic path of every
    8-connected component.

    When no valid geodesic path is available, EDT values over
    the complete foreground mask are used as a fallback.
    """
    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError(
            "mask must be two-dimensional; "
            f"received shape {mask.shape}."
        )

    mask = mask.astype(
        bool,
        copy=False,
    )

    if not mask.any():
        raise ValueError(
            "mask must contain at least one foreground pixel."
        )

    labeled_mask = label(
        mask,
        connectivity=2,
    )

    number_of_components = int(
        labeled_mask.max()
    )

    component_paths = []

    for component_id in range(
        1,
        number_of_components + 1,
    ):
        component_mask = (
            labeled_mask == component_id
        )

        (
            _,
            _,
            component_path_rc,
            _,
        ) = polygon_mask_geodesic_length(
            component_mask,
            microns_per_pixel=1.0,
            alpha=2.0,
            eps=1e-3,
            return_path=True,
            return_debug=True,
        )

        if (
            component_path_rc is not None
            and len(component_path_rc) > 0
        ):
            component_paths.append(
                np.asarray(
                    component_path_rc,
                    dtype=int,
                )
            )

    distance_map = distance_transform_edt(
        mask
    )

    if component_paths:
        path_rc = np.concatenate(
            component_paths,
            axis=0,
        )

        rows = path_rc[:, 0]
        columns = path_rc[:, 1]

        sampled_distances = distance_map[
            rows,
            columns,
        ]
    else:
        sampled_distances = distance_map[
            mask
        ]

    if sampled_distances.size == 0:
        raise RuntimeError(
            "Width measurement produced no EDT samples."
        )

    width_px = float(
        2.0
        * np.mean(sampled_distances)
    )

    if (
        not np.isfinite(width_px)
        or width_px <= 0.0
    ):
        raise ValueError(
            "Calculated width must be finite and positive; "
            f"received {width_px}."
        )

    return width_px