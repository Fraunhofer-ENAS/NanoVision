"""Distance measures for binary CNT centerline instance masks."""

from __future__ import annotations

from typing import Literal

import torch

DistanceMeasure = Literal[
    "chamfer",
    "hausdorff",
]


def _centerline_points(centerline: torch.Tensor) -> torch.Tensor:
    if centerline.ndim != 3 or centerline.shape[0] != 1:
        raise ValueError(
            "A centerline must have shape (1, H, W); "
            f"received {tuple(centerline.shape)}."
        )

    return torch.nonzero(centerline[0].bool(), as_tuple=False).to(torch.float64)


def point_to_curve_distance(
    source_centerline: torch.Tensor,
    target_centerline: torch.Tensor,
    *,
     reduction: Literal["mean", "max"] = "mean",
) -> float:
    """
    Return the directed nearest-point centerline distance.

    ``mean`` gives the directed Chamfer distance.
    ``max`` gives the directed Hausdorff distance.
    """
    if source_centerline.shape != target_centerline.shape:
        raise ValueError(
            "Centerlines must have the same shape; received "
            f"{tuple(source_centerline.shape)} and {tuple(target_centerline.shape)}."
        )

    source_points = _centerline_points(source_centerline)
    target_points = _centerline_points(target_centerline)

    if source_points.shape[0] == 0 and target_points.shape[0] == 0:
        return 0.0
    if source_points.shape[0] == 0 or target_points.shape[0] == 0:
        return float("inf")

    target_points = target_points.to(source_points.device)
    nearest_distances = torch.cdist(source_points, target_points).min(dim=1).values
    if reduction == "mean":
        distance = nearest_distances.mean()
    elif reduction == "max":
        distance = nearest_distances.max()
    else:
        raise ValueError(
            "reduction must be 'mean' or 'max'; "
            f"received {reduction!r}."
        )

    return float(distance.item())



def chamfer_distance(
    first_centerline: torch.Tensor,
    second_centerline: torch.Tensor,
) -> float:
    """Return the symmetric mean Chamfer distance in pixels."""
    forward = point_to_curve_distance(first_centerline, second_centerline)
    backward = point_to_curve_distance(second_centerline, first_centerline)
    return (forward + backward) / 2.0


def hausdorff_distance(
    first_centerline: torch.Tensor,
    second_centerline: torch.Tensor,
) -> float:
    """Return the symmetric Hausdorff distance in pixels."""
    forward = point_to_curve_distance( first_centerline, second_centerline, reduction="max", )
    backward = point_to_curve_distance( second_centerline, first_centerline, reduction="max", )

    return max(
        forward,
        backward,
    )

def centerline_distance(
    first_centerline: torch.Tensor,
    second_centerline: torch.Tensor,
    *,
    measure: DistanceMeasure = "chamfer",
) -> float:
    """Return the selected symmetric centerline distance."""
    if measure == "chamfer":
        return chamfer_distance(
            first_centerline,
            second_centerline,
        )

    if measure == "hausdorff":
        return hausdorff_distance(
            first_centerline,
            second_centerline,
        )

    raise ValueError(
        "measure must be 'chamfer' or 'hausdorff'; "
        f"received {measure!r}."
    )