from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon


def visualize_shapely_polygons_single_image(
    polygons: Iterable,
    width: int,
    height: int,
    output_path: str | Path | None = None,
    alpha: float = 0.5,
    bg_color: str = "black",
) -> None:
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.axis("off")
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    patches = []
    colors = []
    object_colors = {}

    for idx, poly in enumerate(polygons):
        if poly.is_empty:
            continue

        if idx not in object_colors:
            object_colors[idx] = (
                random.random() * 0.7 + 0.3,
                random.random() * 0.7 + 0.3,
                random.random() * 0.7 + 0.3,
            )
        color = object_colors[idx]

        exterior_coords = list(poly.exterior.coords)
        patches.append(MplPolygon(exterior_coords, closed=True))
        colors.append(color)

        for interior in poly.interiors:
            interior_coords = list(interior.coords)
            patches.append(MplPolygon(interior_coords, closed=True))
            colors.append(bg_color if isinstance(bg_color, tuple) else (0, 0, 0))

    if patches:
        ax.add_collection(
            PatchCollection(
                patches,
                facecolor=colors,
                edgecolor="none",
                alpha=alpha,
            )
        )
        ax.autoscale_view()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_path,
            format="svg",
            facecolor=fig.get_facecolor(),
            bbox_inches=None,
            pad_inches=0,
        )
        plt.close(fig)
    else:
        plt.show()

def visualize_json_annotations_as_polygons_svg(
    json_path: str | Path,
    image_folder: str | Path,
    output_folder: str | Path | None = None,
    darken_factor: float = 0.4,
    polygon_alpha: float = 0.5,
    contour_width: float = 1.0,
) -> None:
    """
    Visualize COCO polygon annotations on top of the original image.

    The original image is darkened to make the annotations easier
    to distinguish.

    Each annotation/object receives one random color. If an object
    contains multiple segmentation polygons, all polygons belonging
    to that object are drawn using the same color.

    The actual segmentation polygons are shown rather than replacing
    them with straight PCA-derived lines.
    """

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as f:
        coco_data = json.load(f)

    object_colors = {}

    for image_info in coco_data["images"]:
        # -----------------------------------------------------
        # Locate image
        # -----------------------------------------------------

        image_filename = (
            image_info["file_name"]
            
        )

        image_id = image_info["id"]

        image_path = os.path.join(
            str(image_folder),
            image_filename,
        )

        image = cv2.imread(
            image_path
        )

        if image is None:
            print(
                f"Could not load image: "
                f"{image_path}"
            )
            continue

        height, width = image.shape[:2]

        # -----------------------------------------------------
        # Find annotations belonging to this image
        # -----------------------------------------------------

        image_annotations = [
            ann
            for ann in coco_data["annotations"]
            if ann["image_id"] == image_id
        ]

        # -----------------------------------------------------
        # Prepare darkened original image
        # -----------------------------------------------------

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        darkened = (
            rgb_image.astype(np.float32)
            * darken_factor
        )

        darkened = np.clip(
            darkened,
            0,
            255,
        ).astype(np.uint8)

        # -----------------------------------------------------
        # Figure
        # -----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(
                width / 100,
                height / 100,
            ),
            dpi=100,
        )

        ax.imshow(
            darkened
        )

        ax.axis(
            "off"
        )

        ax.set_position(
            [
                0,
                0,
                1,
                1,
            ]
        )

        ax.set_xlim(
            0,
            width,
        )

        ax.set_ylim(
            height,
            0,
        )

        # -----------------------------------------------------
        # Draw actual COCO polygons
        # -----------------------------------------------------

        for ann in image_annotations:
            object_id = ann["id"]

            # One color per CNT object.
            if object_id not in object_colors:
                base_color = np.array(
                [
                    random.random(),
                    random.random(),
                    random.random(),
                ]
            )

            white = np.ones(3)

            color = (
                0.45 * base_color
                + 0.55 * white
            )

            object_colors[object_id] = tuple(
                color
            )

            segmentation = ann.get(
                "segmentation",
                [],
            )

            # One annotation can contain multiple polygons.
            for seg in segmentation:
                if len(seg) < 6:
                    # Need at least 3 x/y points.
                    continue

                points = np.asarray(
                    seg,
                    dtype=float,
                ).reshape(
                    -1,
                    2,
                )

                polygon_patch = MplPolygon(
                    points,
                    closed=True,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=contour_width,
                    alpha=polygon_alpha,
                )

                ax.add_patch(
                    polygon_patch
                )

        # -----------------------------------------------------
        # Save / display
        # -----------------------------------------------------

        if output_folder:
            output_folder = Path(
                output_folder
            )

            output_folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path_svg = (
                output_folder
                / f"{image_filename}_polygons.svg"
            )

            plt.savefig(
                output_path_svg,
                format="svg",
                bbox_inches=None,
                pad_inches=0,
            )

            plt.close(
                fig
            )

        else:
            plt.show()

def find_line_endpoints(points):
    pts = np.array(points)
    if len(pts) < 2:
        return tuple(pts[0]), tuple(pts[0])

    pts_mean = pts.mean(axis=0)
    pts_centered = pts - pts_mean
    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    main_dir = eigvecs[:, np.argmax(eigvals)]

    proj = pts_centered @ main_dir
    idx_min, idx_max = np.argmin(proj), np.argmax(proj)
    return tuple(pts[idx_min]), tuple(pts[idx_max])


def visualize_json_annotations_as_straight_lines_svg(
    json_path: str | Path,
    image_folder: str | Path,
    output_folder: str | Path | None = None,
    darken_factor: float = 0.4,
    line_width: float = 2.0,
) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    object_colors = {}

    for image_info in coco_data["images"]:
        image_filename = image_info["file_name"] + ".jpg"
        image_id = image_info["id"]
        image_path = os.path.join(str(image_folder), image_filename)

        image = cv2.imread(image_path)
        if image is None:
            continue

        height, width = image.shape[:2]
        image_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        darkened = (rgb_image * darken_factor).astype(np.uint8)

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.imshow(darkened)
        ax.axis("off")
        ax.set_position([0, 0, 1, 1])
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

        for ann in image_annotations:
            object_id = ann["id"]
            if object_id not in object_colors:
                object_colors[object_id] = (random.random(), random.random(), random.random())
            color = object_colors[object_id]

            for seg in ann["segmentation"]:
                points = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                if len(points) >= 2:
                    pt1, pt2 = find_line_endpoints(points)
                    x = [pt1[0], pt2[0]]
                    y = [pt1[1], pt2[1]]
                    ax.plot(x, y, color=color, linewidth=line_width, solid_capstyle="round")

        if output_folder:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
            output_path_svg = output_folder / f"{image_filename}_straight_lines.svg"
            plt.savefig(output_path_svg, format="svg", bbox_inches=None, pad_inches=0)
            plt.close(fig)
        else:
            plt.show()
