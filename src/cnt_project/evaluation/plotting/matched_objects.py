"""Visualize matched segmentation objects and their features."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries
from skimage.morphology import skeletonize

from cnt_project.evaluation.core.matched_object_features import (
    ObjectFeatures,
    axial_orientation_error_deg,
)
from cnt_project.model_development.postprocessing.filtering.shape_heuristics import (
    count_endpoints,
    skeletonize_mask_with_polygon_smoothing,
)
from cnt_project.evaluation.core.object_matching import (
    ObjectMatch,
)
from cnt_project.evaluation.core.cldice import (
    centerline_coverage,
)


def _combined_bounds(
    first_mask: np.ndarray,
    second_mask: np.ndarray,
    *,
    padding: int = 5,
) -> tuple[slice, slice]:
    """Return a padded crop containing both masks."""
    combined = (
        np.asarray(first_mask).astype(bool)
        | np.asarray(second_mask).astype(bool)
    )

    rows, columns = np.nonzero(combined)

    if rows.size == 0:
        return (
            slice(0, combined.shape[0]),
            slice(0, combined.shape[1]),
        )

    row_start = max(
        int(rows.min()) - padding,
        0,
    )
    row_stop = min(
        int(rows.max()) + padding + 1,
        combined.shape[0],
    )
    column_start = max(
        int(columns.min()) - padding,
        0,
    )
    column_stop = min(
        int(columns.max()) + padding + 1,
        combined.shape[1],
    )

    return (
        slice(row_start, row_stop),
        slice(column_start, column_stop),
    )

def _combined_bounds_with_aspect_ratio(
    first_mask: np.ndarray,
    second_mask: np.ndarray,
    *,
    padding: int = 8,
    target_aspect_ratio: float = 3.3 / 2.5,
) -> tuple[slice, slice]:
    """
    Return a crop containing both masks with a fixed aspect ratio.

    The crop is expanded, rather than padded visually, so that its
    width / height matches target_aspect_ratio.

    The crop is kept inside the original image dimensions.
    """
    combined = (
        np.asarray(first_mask).astype(bool)
        | np.asarray(second_mask).astype(bool)
    )

    image_height, image_width = combined.shape

    rows, columns = np.nonzero(
        combined
    )

    if rows.size == 0:
        return (
            slice(0, image_height),
            slice(0, image_width),
        )

    # ---------------------------------------------------------
    # Initial object bounds + requested padding
    # ---------------------------------------------------------

    row_start = max(
        int(rows.min()) - padding,
        0,
    )

    row_stop = min(
        int(rows.max()) + padding + 1,
        image_height,
    )

    column_start = max(
        int(columns.min()) - padding,
        0,
    )

    column_stop = min(
        int(columns.max()) + padding + 1,
        image_width,
    )

    crop_height = (
        row_stop - row_start
    )

    crop_width = (
        column_stop - column_start
    )

    current_aspect_ratio = (
        crop_width / crop_height
    )

    # ---------------------------------------------------------
    # Expand crop to requested aspect ratio
    # ---------------------------------------------------------

    if current_aspect_ratio < target_aspect_ratio:
        # Crop is too narrow.
        desired_width = int(
            np.ceil(
                crop_height
                * target_aspect_ratio
            )
        )

        desired_width = min(
            desired_width,
            image_width,
        )

        center_column = (
            column_start + column_stop
        ) / 2.0

        column_start = int(
            round(
                center_column
                - desired_width / 2
            )
        )

        column_stop = (
            column_start
            + desired_width
        )

        # Shift crop back into image if necessary.
        if column_start < 0:
            column_stop -= column_start
            column_start = 0

        if column_stop > image_width:
            shift = (
                column_stop
                - image_width
            )

            column_start -= shift
            column_stop = image_width

        column_start = max(
            column_start,
            0,
        )

    elif current_aspect_ratio > target_aspect_ratio:
        # Crop is too wide.
        desired_height = int(
            np.ceil(
                crop_width
                / target_aspect_ratio
            )
        )

        desired_height = min(
            desired_height,
            image_height,
        )

        center_row = (
            row_start + row_stop
        ) / 2.0

        row_start = int(
            round(
                center_row
                - desired_height / 2
            )
        )

        row_stop = (
            row_start
            + desired_height
        )

        # Shift crop back into image if necessary.
        if row_start < 0:
            row_stop -= row_start
            row_start = 0

        if row_stop > image_height:
            shift = (
                row_stop
                - image_height
            )

            row_start -= shift
            row_stop = image_height

        row_start = max(
            row_start,
            0,
        )

    return (
        slice(
            row_start,
            row_stop,
        ),
        slice(
            column_start,
            column_stop,
        ),
    )


def _matching_display(
    masks: Sequence[np.ndarray],
    *,
    object_to_color: dict[int, int],
    colors: np.ndarray,
    fill_alpha: float = 0.30,
) -> np.ndarray:
    """
    Display every object with a translucent fill and solid contour.

    Matched GT/prediction objects receive the same color.
    Unmatched objects receive their own colors.
    """
    if not masks:
        raise ValueError(
            "At least one mask is required to determine image shape."
        )

    height, width = np.asarray(
        masks[0]
    ).shape

    display = np.zeros(
        (height, width, 3),
        dtype=np.float64,
    )

    for object_index, mask in enumerate(masks):
        binary_mask = np.asarray(
            mask
        ).astype(bool)

        color = colors[
            object_to_color[object_index]
        ]

        # Translucent-looking fill over the black background.
        display[binary_mask] = (
            (1.0 - fill_alpha)
            * display[binary_mask]
            + fill_alpha
            * color
        )

        # Fully colored object contour.
        boundary = find_boundaries(
            binary_mask,
            mode="outer",
        )

        display[boundary] = color

    return display

def save_object_matching_visualization(
    *,
    matching_metric: str,
    filename: str,
    height: int,
    width: int,
    ground_truth_masks: Sequence[np.ndarray],
    prediction_masks: Sequence[np.ndarray],
    matches: Sequence[ObjectMatch],
    threshold: float,
    output_path: str | Path,
) -> None:
    """Visualize thresholded one-to-one object matches."""
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shape = (
        int(height),
        int(width),
    )

    number_of_matches = len(matches)

    matched_ground_truth_indices = {
        match.ground_truth_index
        for match in matches
    }

    matched_prediction_indices = {
        match.prediction_index
        for match in matches
    }

    unmatched_ground_truth_indices = [
        index
        for index in range(
            len(ground_truth_masks)
        )
        if index not in matched_ground_truth_indices
    ]

    unmatched_prediction_indices = [
        index
        for index in range(
            len(prediction_masks)
        )
        if index not in matched_prediction_indices
    ]

    number_of_colors = (
        number_of_matches
        + len(unmatched_ground_truth_indices)
        + len(unmatched_prediction_indices)
    )

    # Golden-ratio hue spacing prevents consecutive objects from
    # receiving almost identical colors.
    color_positions = (
        np.arange(
            max(number_of_colors, 1),
            dtype=float,
        )
        * 0.618033988749895
    ) % 1.0

    colors = plt.cm.hsv(
        color_positions
    )[:, :3]

    ground_truth_to_color = {}
    prediction_to_color = {}

    # A matched pair receives the same color in both panels.
    for match_index, match in enumerate(matches):
        ground_truth_to_color[
            match.ground_truth_index
        ] = match_index

        prediction_to_color[
            match.prediction_index
        ] = match_index

    next_color_index = number_of_matches

    # Every unmatched GT object receives its own color.
    for object_index in unmatched_ground_truth_indices:
        ground_truth_to_color[
            object_index
        ] = next_color_index

        next_color_index += 1

    # Every unmatched prediction receives its own color.
    for object_index in unmatched_prediction_indices:
        prediction_to_color[
            object_index
        ] = next_color_index

        next_color_index += 1


    if ground_truth_masks:
        ground_truth_display = _matching_display(
            ground_truth_masks,
            object_to_color=ground_truth_to_color,
            colors=colors,
        )
    else:
        ground_truth_display = np.zeros(
            (*shape, 3),
            dtype=float,
        )

    if prediction_masks:
        prediction_display = _matching_display(
            prediction_masks,
            object_to_color=prediction_to_color,
            colors=colors,
        )
    else:
        prediction_display = np.zeros(
            (*shape, 3),
            dtype=float,
        )

    boundary_display = np.zeros(
        (*shape, 3),
        dtype=float,
    )

    for match_index, match in enumerate(matches):
        color = colors[match_index]

        ground_truth_boundary = find_boundaries(
            np.asarray(
                ground_truth_masks[
                    match.ground_truth_index
                ]
            ).astype(bool),
            mode="outer",
        )

        prediction_boundary = find_boundaries(
            np.asarray(
                prediction_masks[
                    match.prediction_index
                ]
            ).astype(bool),
            mode="outer",
        )

        boundary_display[
            ground_truth_boundary
        ] = color

        # Prediction boundaries are made brighter.
        boundary_display[
            prediction_boundary
        ] = np.minimum(
            color + 0.30,
            1.0,
        )

    true_positives = number_of_matches
    false_positives = (
        len(prediction_masks)
        - true_positives
    )
    false_negatives = (
        len(ground_truth_masks)
        - true_positives
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    axes[0].imshow(
        ground_truth_display
    )
    axes[0].set_title(
        "Ground truth\n translucent fill, solid contour"
    )

    axes[1].imshow(
        prediction_display
    )
    axes[1].set_title(
        "Predictions\n matched pairs share the same color"
    )

    axes[2].imshow(
        boundary_display
    )
    axes[2].set_title(
        "Matched boundaries\n"
        "GT: original color, prediction: brighter"
    )

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        f"{filename}\n"
        f"{matching_metric}; "
        f"threshold={threshold:.2f}; "
        f"TP={true_positives}, "
        f"FP={false_positives}, "
        f"FN={false_negatives}"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)



def save_cldice_pair_visualizations(
    *,
    filename: str,
    ground_truth_masks: Sequence[np.ndarray],
    prediction_masks: Sequence[np.ndarray],
    matches: Sequence[ObjectMatch],
    threshold: float,
    output_directory: str | Path,
    pairs_per_page: int = 12,
) -> None:
    """
    Save every accepted clDice match as a separate cropped overlay.

    Visualization:
        original AFM image:
            background

        GT-only mask:
            transparent red

        prediction-only mask:
            transparent blue

        GT/prediction overlap:
            clearly visible purple

        GT contour:
            solid red

        prediction contour:
            solid blue

        GT skeleton:
            dark-red dashed line

        prediction skeleton:
            dark-blue dashed line
    """
    if not matches:
        return

    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Original AFM image directory
    # ---------------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parents[4]
    )

    image_directory = (
        project_root
        / "data"
        / "cnt_segmentation"
        / "images"
    )

    if not image_directory.is_dir():
        raise FileNotFoundError(
            "CNT image directory does not exist:\n"
            f"{image_directory}"
        )

    # ---------------------------------------------------------
    # Locate original AFM image
    # ---------------------------------------------------------

    image_path = (
        image_directory
        / filename
    )

    if not image_path.is_file():
        matching_images = list(
            image_directory.rglob(filename)
        )

        if len(matching_images) == 1:
            image_path = matching_images[0]

        elif len(matching_images) == 0:
            raise FileNotFoundError(
                "Could not find original AFM image:\n"
                f"{filename}\n"
                f"under:\n{image_directory}"
            )

        else:
            raise RuntimeError(
                "Multiple original AFM images found for:\n"
                f"{filename}\n\n"
                + "\n".join(
                    str(path)
                    for path in matching_images
                )
            )

    original_image = plt.imread(
        image_path
    )

    # ---------------------------------------------------------
    # Visualization settings
    # ---------------------------------------------------------

    # Mask colors.
    GT_FILL_COLOR = (
        0.95,
        0.05,
        0.05,
    )

    PREDICTION_FILL_COLOR = (
        0.00,
        0.35,
        1.00,
    )

    # Strong saturated purple specifically for the overlap.
    OVERLAP_FILL_COLOR = (
        0.65,
        0.00,
        0.85,
    )

    # Skeleton colors.
    GT_SKELETON_COLOR = (
        0.50,
        0.00,
        0.00,
    )

    PREDICTION_SKELETON_COLOR = (
        0.00,
        0.10,
        0.55,
    )

    # GT-only and prediction-only regions remain transparent
    # so the AFM structure is clearly visible.
    MASK_ALPHA = 0.30

    # The overlap is intentionally more opaque so that it is
    # immediately distinguishable as a separate region.
    OVERLAP_ALPHA = 0.65

    CONTOUR_LINEWIDTH = 1.2

    SKELETON_LINEWIDTH = 1.2

    SKELETON_SHIFT = 0.2

    # ---------------------------------------------------------
    # Convert skeleton pixels into centerline graph paths
    # ---------------------------------------------------------

    def _skeleton_paths(
        skeleton,
    ):
        skeleton = np.asarray(
            skeleton,
            dtype=bool,
        )

        coordinates = {
            tuple(coordinate)
            for coordinate in np.argwhere(
                skeleton
            )
        }

        if not coordinates:
            return []

        neighbor_offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        def neighbors(pixel):
            row, column = pixel

            return [
                (
                    row + row_offset,
                    column + column_offset,
                )
                for row_offset, column_offset
                in neighbor_offsets
                if (
                    row + row_offset,
                    column + column_offset,
                ) in coordinates
            ]

        adjacency = {
            pixel: neighbors(pixel)
            for pixel in coordinates
        }

        # Endpoints and junctions.
        nodes = {
            pixel
            for pixel, pixel_neighbors
            in adjacency.items()
            if len(pixel_neighbors) != 2
        }

        visited_edges = set()

        def edge_key(
            pixel_a,
            pixel_b,
        ):
            return frozenset(
                (
                    pixel_a,
                    pixel_b,
                )
            )

        paths = []

        # -----------------------------------------------------
        # Paths between endpoints / junctions
        # -----------------------------------------------------

        for start_pixel in nodes:
            for next_pixel in adjacency[
                start_pixel
            ]:
                first_edge = edge_key(
                    start_pixel,
                    next_pixel,
                )

                if first_edge in visited_edges:
                    continue

                path = [
                    start_pixel,
                    next_pixel,
                ]

                visited_edges.add(
                    first_edge
                )

                previous_pixel = start_pixel
                current_pixel = next_pixel

                while (
                    current_pixel
                    not in nodes
                ):
                    candidates = [
                        pixel
                        for pixel in adjacency[
                            current_pixel
                        ]
                        if pixel
                        != previous_pixel
                    ]

                    if not candidates:
                        break

                    following_pixel = (
                        candidates[0]
                    )

                    current_edge = edge_key(
                        current_pixel,
                        following_pixel,
                    )

                    if (
                        current_edge
                        in visited_edges
                    ):
                        break

                    visited_edges.add(
                        current_edge
                    )

                    path.append(
                        following_pixel
                    )

                    previous_pixel = (
                        current_pixel
                    )

                    current_pixel = (
                        following_pixel
                    )

                if len(path) >= 2:
                    paths.append(path)

        # -----------------------------------------------------
        # Remaining closed loops
        # -----------------------------------------------------

        for start_pixel in coordinates:
            for next_pixel in adjacency[
                start_pixel
            ]:
                first_edge = edge_key(
                    start_pixel,
                    next_pixel,
                )

                if first_edge in visited_edges:
                    continue

                path = [
                    start_pixel,
                    next_pixel,
                ]

                visited_edges.add(
                    first_edge
                )

                previous_pixel = start_pixel
                current_pixel = next_pixel

                while True:
                    candidates = [
                        pixel
                        for pixel in adjacency[
                            current_pixel
                        ]
                        if pixel
                        != previous_pixel
                    ]

                    if not candidates:
                        break

                    following_pixel = (
                        candidates[0]
                    )

                    current_edge = edge_key(
                        current_pixel,
                        following_pixel,
                    )

                    if (
                        current_edge
                        in visited_edges
                    ):
                        break

                    visited_edges.add(
                        current_edge
                    )

                    path.append(
                        following_pixel
                    )

                    previous_pixel = (
                        current_pixel
                    )

                    current_pixel = (
                        following_pixel
                    )

                if len(path) >= 2:
                    paths.append(path)

        return paths

    # ---------------------------------------------------------
    # Plot actual skeleton centerline
    # ---------------------------------------------------------

    def _plot_skeleton(
        axis,
        skeleton,
        *,
        color,
        shift_x=0.0,
        shift_y=0.0,
    ):
        paths = _skeleton_paths(
            skeleton
        )

        for path in paths:
            coordinates = np.asarray(
                path,
                dtype=float,
            )

            y_coordinates = (
                coordinates[:, 0]
                + shift_y
            )

            x_coordinates = (
                coordinates[:, 1]
                + shift_x
            )

            axis.plot(
                x_coordinates,
                y_coordinates,
                color=color,
                linewidth=SKELETON_LINEWIDTH,
                linestyle=(
                    0,
                    (
                        3,
                        2,
                    ),
                ),
                dash_capstyle="round",
                solid_capstyle="round",
                zorder=7,
            )

    # ---------------------------------------------------------
    # Generate one figure per accepted pair
    # ---------------------------------------------------------

    for pair_index, match in enumerate(
        matches,
        start=1,
    ):
        ground_truth_mask = np.asarray(
            ground_truth_masks[
                match.ground_truth_index
            ]
        ).astype(bool)

        prediction_mask = np.asarray(
            prediction_masks[
                match.prediction_index
            ]
        ).astype(bool)

        ground_truth_skeleton = skeletonize(
            ground_truth_mask
        )

        prediction_skeleton = skeletonize(
            prediction_mask
        )

        # -----------------------------------------------------
        # Common crop
        # -----------------------------------------------------

        crop_rows, crop_columns = (
            _combined_bounds_with_aspect_ratio(
                ground_truth_mask,
                prediction_mask,
                padding=8,
                target_aspect_ratio=3.3 / 2.5,
            )
        )

        image_crop = original_image[
            crop_rows,
            crop_columns,
        ]

        ground_truth_crop = ground_truth_mask[
            crop_rows,
            crop_columns,
        ]

        prediction_crop = prediction_mask[
            crop_rows,
            crop_columns,
        ]

        ground_truth_skeleton_crop = (
            ground_truth_skeleton[
                crop_rows,
                crop_columns,
            ]
        )

        prediction_skeleton_crop = (
            prediction_skeleton[
                crop_rows,
                crop_columns,
            ]
        )

        # -----------------------------------------------------
        # Explicit mutually exclusive mask regions
        # -----------------------------------------------------

        gt_only = (
            ground_truth_crop
            & ~prediction_crop
        )

        prediction_only = (
            prediction_crop
            & ~ground_truth_crop
        )

        mask_overlap = (
            ground_truth_crop
            & prediction_crop
        )

        # -----------------------------------------------------
        # Skeleton overlap
        # -----------------------------------------------------

        skeleton_overlap = (
            ground_truth_skeleton_crop
            & prediction_skeleton_crop
        )

        if np.any(skeleton_overlap):
            gt_shift_x = (
                -SKELETON_SHIFT
            )

            prediction_shift_x = (
                SKELETON_SHIFT
            )
        else:
            gt_shift_x = 0.0
            prediction_shift_x = 0.0

        # -----------------------------------------------------
        # Figure
        # -----------------------------------------------------

        figure = plt.figure(
            figsize=(
                3.3,
                2.5,
            ),
        )

        axis = figure.add_axes(
            [
                0.0,
                0.0,
                1.0,
                1.0,
            ]
        )

        # -----------------------------------------------------
        # Original AFM background
        # -----------------------------------------------------

        if image_crop.ndim == 2:
            axis.imshow(
                image_crop,
                cmap="gray",
                interpolation="nearest",
                zorder=0,
            )
        else:
            axis.imshow(
                image_crop,
                interpolation="nearest",
                zorder=0,
            )

        # -----------------------------------------------------
        # Explicit GT / prediction / overlap overlay
        # -----------------------------------------------------

        overlay = np.zeros(
            (
                ground_truth_crop.shape[0],
                ground_truth_crop.shape[1],
                4,
            ),
            dtype=float,
        )

        # -----------------------------------------------------
        # GT only = RED
        # -----------------------------------------------------

        overlay[
            gt_only,
            :3,
        ] = GT_FILL_COLOR

        overlay[
            gt_only,
            3,
        ] = MASK_ALPHA

        # -----------------------------------------------------
        # Prediction only = BLUE
        # -----------------------------------------------------

        overlay[
            prediction_only,
            :3,
        ] = PREDICTION_FILL_COLOR

        overlay[
            prediction_only,
            3,
        ] = MASK_ALPHA

        # -----------------------------------------------------
        # Exact overlap = PURPLE
        #
        # This is explicitly assigned purple.
        # It is NOT produced by red/blue alpha blending.
        # -----------------------------------------------------

        overlay[
            mask_overlap,
            :3,
        ] = OVERLAP_FILL_COLOR

        overlay[
            mask_overlap,
            3,
        ] = OVERLAP_ALPHA

        axis.imshow(
            overlay,
            interpolation="nearest",
            zorder=2,
        )

        # -----------------------------------------------------
        # Solid GT contour
        # -----------------------------------------------------

        axis.contour(
            ground_truth_crop.astype(float),
            levels=[
                0.5,
            ],
            colors=[
                GT_FILL_COLOR,
            ],
            linewidths=CONTOUR_LINEWIDTH,
            linestyles="solid",
            zorder=5,
        )

        # -----------------------------------------------------
        # Solid prediction contour
        # -----------------------------------------------------

        axis.contour(
            prediction_crop.astype(float),
            levels=[
                0.5,
            ],
            colors=[
                PREDICTION_FILL_COLOR,
            ],
            linewidths=CONTOUR_LINEWIDTH,
            linestyles="solid",
            zorder=6,
        )

        # -----------------------------------------------------
        # Dashed GT skeleton
        # -----------------------------------------------------

        _plot_skeleton(
            axis,
            ground_truth_skeleton_crop,
            color=GT_SKELETON_COLOR,
            shift_x=gt_shift_x,
        )

        # -----------------------------------------------------
        # Dashed prediction skeleton
        # -----------------------------------------------------

        _plot_skeleton(
            axis,
            prediction_skeleton_crop,
            color=PREDICTION_SKELETON_COLOR,
            shift_x=prediction_shift_x,
        )

        # -----------------------------------------------------
        # Exact crop limits
        # -----------------------------------------------------

        axis.set_xlim(
            -0.5,
            image_crop.shape[1] - 0.5,
        )

        axis.set_ylim(
            image_crop.shape[0] - 0.5,
            -0.5,
        )

        axis.axis(
            "off"
        )

        # -----------------------------------------------------
        # Output filenames
        # -----------------------------------------------------

        output_stem = (
            f"pair_{pair_index:04d}"
            f"_gt_{match.ground_truth_index:04d}"
            f"_pred_{match.prediction_index:04d}"
            f"_cldice_{match.value:.3f}"
        )

        png_path = (
            output_directory
            / f"{output_stem}.png"
        )

        svg_path = (
            output_directory
            / f"{output_stem}.svg"
        )

        # -----------------------------------------------------
        # Save
        # -----------------------------------------------------

        figure.savefig(
            png_path,
            format="png",
            dpi=600,
            pad_inches=0,
        )

        figure.savefig(
            svg_path,
            format="svg",
            pad_inches=0,
        )

        plt.close(
            figure
        )

def _draw_orientation_axis(
    axis,
    mask: np.ndarray,
    angle_deg: float,
) -> None:
    """Draw the extracted PCA orientation through mask centroid."""
    if not np.isfinite(angle_deg):
        return

    rows, columns = np.nonzero(mask)

    if rows.size == 0:
        return

    center_row = float(np.mean(rows))
    center_column = float(np.mean(columns))

    angle_rad = np.deg2rad(
        angle_deg
    )

    axis_length = 0.4 * max(
        mask.shape
    )

    dx = axis_length * np.cos(
        angle_rad
    )

    # Convert Cartesian orientation back to image coordinates.
    dy_image = -axis_length * np.sin(
        angle_rad
    )

    axis.plot(
        [
            center_column - dx,
            center_column + dx,
        ],
        [
            center_row - dy_image,
            center_row + dy_image,
        ],
        color="yellow",
        linewidth=2,
    )


def save_matched_feature_visualization(
    *,
    filename: str,
    ground_truth_masks: Sequence[np.ndarray],
    prediction_masks: Sequence[np.ndarray],
    ground_truth_features: Sequence[ObjectFeatures],
    prediction_features: Sequence[ObjectFeatures],
    matches: Sequence[ObjectMatch],
    output_path: str | Path,
    maximum_pairs: int = 3,
    matching_metric: str,
    threshold: float,
) -> None:
    """Visualize length and orientation for selected matched pairs."""
    selected_matches = tuple(
        matches[:maximum_pairs]
    )

    if not selected_matches:
        return

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        len(selected_matches),
        2,
        figsize=(
            10,
            4 * len(selected_matches),
        ),
        squeeze=False,
    )

    for row_index, match in enumerate(
        selected_matches
    ):
        ground_truth_mask = np.asarray(
            ground_truth_masks[
                match.ground_truth_index
            ]
        ).astype(bool)

        prediction_mask = np.asarray(
            prediction_masks[
                match.prediction_index
            ]
        ).astype(bool)

        ground_truth_feature = (
            ground_truth_features[
                match.ground_truth_index
            ]
        )

        prediction_feature = (
            prediction_features[
                match.prediction_index
            ]
        )

        crop_rows, crop_columns = (
            _combined_bounds(
                ground_truth_mask,
                prediction_mask,
            )
        )

        cropped_ground_truth = ground_truth_mask[
            crop_rows,
            crop_columns,
        ]

        cropped_prediction = prediction_mask[
            crop_rows,
            crop_columns,
        ]

        length_error_um = abs(
            prediction_feature.length_um
            - ground_truth_feature.length_um
        )

        orientation_error_deg = (
            axial_orientation_error_deg(
                ground_truth_feature.orientation_deg,
                prediction_feature.orientation_deg,
            )
        )

        ground_truth_axis = axes[
            row_index,
            0,
        ]

        prediction_axis = axes[
            row_index,
            1,
        ]

        ground_truth_axis.imshow(
            cropped_ground_truth,
            cmap="gray",
        )

        prediction_axis.imshow(
            cropped_prediction,
            cmap="gray",
        )

        _draw_orientation_axis(
            ground_truth_axis,
            cropped_ground_truth,
            ground_truth_feature.orientation_deg,
        )

        _draw_orientation_axis(
            prediction_axis,
            cropped_prediction,
            prediction_feature.orientation_deg,
        )

        ground_truth_axis.set_title(
            f"Pair {row_index + 1}: GT object "
            f"{match.ground_truth_index}\n"
            f"length={ground_truth_feature.length_um:.3f} µm; "
            f"orientation="
            f"{ground_truth_feature.orientation_deg:.2f}°"
        )

        prediction_axis.set_title(
            f"Prediction object {match.prediction_index}; "
            f"matching value={match.value:.3f}\n"
            f"length={prediction_feature.length_um:.3f} µm; "
            f"orientation="
            f"{prediction_feature.orientation_deg:.2f}°\n"
            f"|Δlength|={length_error_um:.3f} µm; "
            f"axial |Δorientation|="
            f"{orientation_error_deg:.2f}°"
        )

        ground_truth_axis.axis("off")
        prediction_axis.axis("off")

    figure.suptitle(
        f"{filename}\n"
        f"{matching_metric}; threshold={threshold:.2f}\n"
        "Matched-object length and polygon-PCA orientation"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_rejected_prediction_visualization(
    *,
    filename: str,
    image: np.ndarray,
    rejected_masks: Sequence[np.ndarray],
    output_path: str | Path,
) -> None:
    """
    Overlay endpoint-filtered predictions on the original image.

    Red:
        rejected mask boundary

    Yellow:
        rejected mask skeleton
    """
    if not rejected_masks:
        return

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = np.asarray(image)

    if image.ndim == 3 and image.shape[0] in (
        1,
        3,
        4,
    ):
        image = np.moveaxis(
            image,
            0,
            -1,
        )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6),
    )

    axes[0].imshow(
        image,
        cmap="gray" if image.ndim == 2 else None,
    )
    axes[0].set_title("Original image")

    axes[1].imshow(
        image,
        cmap="gray" if image.ndim == 2 else None,
    )

    for rejected_mask in rejected_masks:
        rejected_mask = np.asarray(
            rejected_mask
        ).astype(bool)

        boundary = find_boundaries(
            rejected_mask,
            mode="outer",
        )

        # This is the same smoothed skeleton used by the
        # endpoint filter in the evaluation runner.
        skeleton = np.asarray(
            skeletonize_mask_with_polygon_smoothing(
                rejected_mask,
                smoothing_iterations=2,
            )
        ).astype(bool)

        endpoint_count = count_endpoints(
            skeleton
        )

        boundary_overlay = np.zeros(
            (*rejected_mask.shape, 4),
            dtype=float,
        )

        boundary_overlay[
            boundary
        ] = [
            1.0,
            0.0,
            0.0,
            1.0,
        ]

        skeleton_overlay = np.zeros(
            (*rejected_mask.shape, 4),
            dtype=float,
        )

        skeleton_overlay[
            skeleton
        ] = [
            1.0,
            1.0,
            0.0,
            1.0,
        ]

        axes[1].imshow(
            boundary_overlay
        )

        axes[1].imshow(
            skeleton_overlay
        )

        # Write the endpoint count at the object's center.
        mask_rows, mask_columns = np.nonzero(
            rejected_mask
        )

        if mask_rows.size > 0:
            center_row = float(
                np.mean(mask_rows)
            )

            center_column = float(
                np.mean(mask_columns)
            )

            axes[1].text(
                center_column,
                center_row,
                str(endpoint_count),
                color="cyan",
                fontsize=8,
                fontweight="bold",
                horizontalalignment="center",
                verticalalignment="center",
                bbox={
                    "facecolor": "black",
                    "alpha": 0.6,
                    "edgecolor": "none",
                    "pad": 1,
                },
            )
    axes[1].set_title(
        "Rejected predictions\n"
        "boundary: red, skeleton: yellow, "
        "cyan number: endpoint count"
    )

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        f"{filename}\n"
        f"Predictions rejected by endpoint filter: "
        f"{len(rejected_masks)}"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)