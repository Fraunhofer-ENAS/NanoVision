from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from sklearn.decomposition import PCA


"""
Legacy Shapely-polygon PCA orientation extraction. mainly used in the merging conditions

Purpose
-------
This module preserves the CNT orientation calculation that operated
directly on Shapely ``Polygon`` and ``MultiPolygon`` objects.


Input representation
--------------------
This function accepts:

- ``shapely.geometry.Polygon``
- ``shapely.geometry.MultiPolygon``

It does not accept a COCO annotation dictionary and does not decode or
rasterize a binary mask.

Orientation calculation
-----------------------
For a ``Polygon``:

1. Read ``polygon.exterior.coords``.
2. Convert the exterior coordinates to a NumPy array.
3. Center the coordinates by subtracting their arithmetic mean.
4. Fit ``sklearn.decomposition.PCA`` with two components.
5. Use the first principal component as the dominant orientation vector.
6. Calculate the angle with ``atan2(y, x)``.
7. Convert the angle from radians to degrees.
8. Normalize the result to the range ``[-90, 90]``.

For a ``MultiPolygon``:

1. Calculate the orientation of every constituent ``Polygon`` independently.
2. Return the arithmetic mean of the resulting angles.

Important reproducibility details
---------------------------------
The following historical behaviors are intentionally retained:

- The repeated closing coordinate from ``polygon.exterior.coords`` is included.
- ``sklearn.decomposition.PCA`` is used rather than a direct covariance
  eigendecomposition.
- Each ``MultiPolygon`` component contributes equally, regardless of its area
  or number of vertices.
- ``MultiPolygon`` angles are combined using a normal arithmetic mean rather
  than an axial or circular mean.
- No polygon validation, simplification, resampling, area weighting, or
  coordinate-system conversion is performed.
- The angle follows the coordinate system of the supplied Shapely geometry.

These behaviors may not be ideal for a new canonical orientation metric, but
changing them would prevent exact reproduction of historical outputs.

Difference from annotation-based orientation functions
-------------------------------------------------------
This implementation is retained for historical/reproducibility purposes.
The canonical feature-extraction pipeline uses the orientation definition
implemented in cnt_feature_extraction.py.

This legacy module:
    Operates directly on Shapely geometry and performs no explicit image-to-
    Cartesian coordinate conversion. Its sign convention therefore depends on
    the coordinate system used to construct the Shapely geometry.


"""


def calculate_shapely_polygon_pca_orientation_legacy(
    polygon: Polygon | MultiPolygon,
) -> float:
    """
    Calculate the historical PCA orientation of a Shapely polygon geometry.

    Parameters
    ----------
    polygon
        A Shapely ``Polygon`` or ``MultiPolygon``.

    Returns
    -------
    float
        Orientation angle in degrees, normalized to ``[-90, 90]``.

        For a ``MultiPolygon``, the returned value is the arithmetic mean of
        the independently calculated component angles.

    Raises
    ------
    TypeError
        If ``polygon`` is neither a Shapely ``Polygon`` nor ``MultiPolygon``.

    Notes
    -----
    This function intentionally reproduces the historical implementation.
    Do not modify its calculation without explicitly accepting that previously
    generated orientation results may no longer be reproducible.
    """
    if isinstance(polygon, MultiPolygon):
        angles = []

        for poly in polygon.geoms:
            angles.append(
                calculate_shapely_polygon_pca_orientation_legacy(poly)
            )

        return np.mean(angles)

    if isinstance(polygon, Polygon):
        # Preserve historical behavior:
        # polygon.exterior.coords contains the repeated closing coordinate.
        vertices = np.array(polygon.exterior.coords)

        # Center the boundary vertices.
        mean_vertex = np.mean(vertices, axis=0)
        centered_vertices = vertices - mean_vertex

        # Preserve historical sklearn PCA implementation.
        pca = PCA(n_components=2)
        pca.fit(centered_vertices)

        # The first principal component represents the dominant orientation.
        principal_components = pca.components_
        orientation_vector = principal_components[0]

        orientation_angle_radians = np.arctan2(
            orientation_vector[1],
            orientation_vector[0],
        )
        orientation_angle_degrees = np.degrees(
            orientation_angle_radians
        )

        # Preserve historical normalization to [-90, 90].
        if orientation_angle_degrees > 90:
            orientation_angle_degrees -= 180
        elif orientation_angle_degrees < -90:
            orientation_angle_degrees += 180

        return orientation_angle_degrees

    raise TypeError("Input must be a Polygon or MultiPolygon")

def calculate_all_shapely_polygon_pca_orientations_legacy(
    polygons: list[Polygon | MultiPolygon],
) -> list[float]:
    """
    Calculate the historical PCA orientation for multiple Shapely polygon
    geometries.

    This is a convenience wrapper around
    ``calculate_shapely_polygon_pca_orientation_legacy``. The orientation of
    each input geometry is calculated independently, preserving the historical
    implementation used by the legacy CNT postprocessing pipeline.

    Parameters
    ----------
    polygons
        List of Shapely ``Polygon`` and/or ``MultiPolygon`` objects.

    Returns
    -------
    list[float]
        One orientation angle, in degrees, for each input geometry. The order
        of the returned angles matches the order of the input geometries.
    """
    return [
        calculate_shapely_polygon_pca_orientation_legacy(polygon)
        for polygon in polygons
    ]

def get_shapely_polygon_pca_orientations_legacy(
    polygons: list[Polygon | MultiPolygon],
) -> list[float]:
    """
    Calculate historical PCA orientations for multiple Shapely geometries.

    This is the relocated equivalent of the former
    ``get_orientation_angles_for_polygons`` function.

    Parameters
    ----------
    polygons
        Sequence of Shapely ``Polygon`` or ``MultiPolygon`` geometries.

    Returns
    -------
    list[float]
        One orientation angle for each input geometry, in the same order.
    """
    orientation_angles = []

    for polygon in polygons:
        angle = calculate_shapely_polygon_pca_orientation_legacy(polygon)
        orientation_angles.append(angle)

    return orientation_angles