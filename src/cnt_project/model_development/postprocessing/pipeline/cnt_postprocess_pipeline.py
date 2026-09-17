import numpy as np
from pathlib import Path

from cnt_project.model_development.postprocessing.filtering.nms import nms_polygons
from cnt_project.model_development.postprocessing.geometry.polygon_from_rays import construct_polygon_opp
from cnt_project.model_development.postprocessing.filtering.shape_heuristics import is_elongated_polygon
from cnt_project.model_development.postprocessing.geometry.polygon_merging_algorithm import merge_polygons_iterative_with_overlap_regions, merge_polygons_iterative_no_overlap_regions
from cnt_project.features.core.shapely_polygon_pca_orientation import (
    calculate_all_shapely_polygon_pca_orientations_legacy
)
from cnt_project.model_development.postprocessing.visualization.polygon_plot_util import plot_polygons
from cnt_project.model_development.postprocessing.visualization.postprocessing_debug_plots import (
    resolve_postprocessing_debug_plot_dir,
    _sanitize_tag,
    save_before_nms_score_peak_plot,
    save_postprocessing_stage_histograms,
)



def _sort_polygons_by_scores(polygons, scores, all_distances, all_points):
    # Combine all the lists using zip
    combined = list(zip(polygons, scores, all_distances, all_points))

    # Sort the combined list by scores in descending order (highest score first)
    combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)

    # Unzip the sorted list back into separate lists
    polygons_sorted, scores_sorted, all_distances_sorted, all_points_sorted = zip(*combined_sorted)

    # Convert the unzipped tuples back to lists
    return list(polygons_sorted), list(scores_sorted), list(all_distances_sorted), list(all_points_sorted)


# 2 main functions one for Approach A and the other for Approach B simply rename the function you want to use to "optimized_cell_segmentation"
# main function for Approach B (with overlap regions)
def merge_instances_with_overlap_regions(
    scores,
    dist,
    points_arr,
    thresh,
    use_kdtree=False,
    verbose=False,
    merge_overlap_threshold: float = 4.0,
    merge_angle_threshold: float = 50,
    merge_centroid_threshold_factor: float = 0.5,
    debug_plots: bool = False,
    debug_plot_dir: str | Path | None = None,
    debug_run_name: str | None = None,
    img_shape: tuple[int, int] | None = None,
):
    """
    Build polygon candidates from StarDist rays, suppress duplicates, and merge
    elongated instances using an overlap-aware merging strategy.

    Pipeline:
      1) construct polygons from (dist, points_arr) and sort by score
      2) run polygon NMS
      3) simplify polygons to reduce spiky boundaries
      4) split into elongated vs. non-elongated shapes
      5) merge only elongated polygons with overlap-region logic
      6) compute orientation angles for final polygons

    Parameters
    ----------
    scores : (N,) array-like
        Confidence score per candidate.
    dist : (N, R) array-like
        Ray distances per candidate (R rays).
    points_arr : (N, 2) array-like
        Candidate center points (y, x).
    thresh : float
        Threshold forwarded to downstream logic (currently not used directly here).
    use_kdtree : bool
        Whether to use KD-tree acceleration in downstream merging.
    verbose : bool
        Print progress messages.

    Returns
    -------
    merged_polygons : list[shapely.geometry.Polygon]
        Final merged polygons (elongated merged + non-elongated kept).
    orientation_angles : array-like
        Orientation angle per merged polygon.
    """
    # Check if scores, dist, or points_arr are empty
    if scores is None or dist is None or points_arr is None or len(scores) == 0 or len(dist) == 0 or len(points_arr) == 0:
        if verbose:
            print("Input arrays are empty, returning None or empty lists.")
        return [], []

    # Check if dist and points_arr have compatible dimensions
    if dist.shape[0] != len(points_arr):
        raise ValueError("dist and points_arr dimensions do not match.")
    n_rays = dist.shape[-1]
    plot_dir = None
    image_tag = f"image_n{len(points_arr)}_sig{int(np.sum(points_arr)) % 100000}"
    if debug_plots:
        plot_dir = resolve_postprocessing_debug_plot_dir(
            run_name=debug_run_name,
            explicit_dir=debug_plot_dir,
            approach_tag="with_overlap",
        )

    results = [construct_polygon_opp(i, dist, points_arr, scores) for i in range(len(points_arr))]
    polygons, all_distances, all_points, all_scores = zip(*results)

    # Convert tuples to lists if needed
    polygons = list(polygons)
    all_distances = list(all_distances)
    all_points = list(all_points)
    all_scores = list(all_scores)
    polygons, scores, all_distances, all_points = _sort_polygons_by_scores(polygons, scores,all_distances,all_points)

    if debug_plots:
        save_before_nms_score_peak_plot(
            debug_root=plot_dir,
            image_tag=image_tag,
            points=np.asarray(all_points, dtype=np.float32),
            scores=np.asarray(all_scores, dtype=np.float32),
            polygons=polygons,
            img_shape=img_shape,
        )
        plot_polygons(
            polygons,
            "Original Polygons",
            n_rays,
            dist=all_distances,
            points_arr=all_points,
            save_path=plot_dir,
        )
        save_postprocessing_stage_histograms(
            debug_root=plot_dir,
            stage_tag="before_nms",
            image_tag=image_tag,
            polygons=polygons,
            raw_distances=all_distances,
        )

    # Perform NMS (returns non-suppressed polygons, their scores, distances, and points)
    polygons, all_scores, all_distances, all_points = nms_polygons(polygons, all_scores, all_distances, all_points, 0.5)

    if debug_plots:
        plot_polygons(
            polygons,
            "Polygons After NMS",
            n_rays,
            dist=all_distances,
            points_arr=all_points,
            scores=all_scores,
            save_path=plot_dir,
        )
        save_postprocessing_stage_histograms(
            debug_root=plot_dir,
            stage_tag="after_nms",
            image_tag=image_tag,
            polygons=polygons,
            raw_distances=all_distances,
        )

    # Apply smoothing to each polygon because at the moment polygons have spikes
    polygons = [poly.simplify(1.2, preserve_topology=True) for poly in polygons]
    # Filter polygons based on shape properties
    elongated_polygons = []
    elongated_distances = []
    elongated_points = []
    elongated_scores = []

    star_like_polygons = []
    star_like_distances = []
    star_like_points = []
    star_like_scores = []

    for polygon, distances, points, score in zip(polygons, all_distances, all_points, all_scores):
        if is_elongated_polygon(polygon):
            elongated_polygons.append(polygon)
            elongated_distances.append(distances)
            elongated_points.append(points)
            elongated_scores.append(score)
        else:
            star_like_polygons.append(polygon)
            star_like_distances.append(distances)
            star_like_points.append(points)
            star_like_scores.append(score)

    if debug_plots:
        if elongated_polygons:
            plot_polygons(
                elongated_polygons,
                "Elongated Polygons",
                n_rays,
                save_path=plot_dir,
            )
        if star_like_polygons:
            plot_polygons(
                star_like_polygons,
                "Star-Like Polygons",
                n_rays,
                save_path=plot_dir,
            )

    # Perform polygon merging
    merged_polygons = merge_polygons_iterative_with_overlap_regions(
        elongated_polygons,
        elongated_points,
        elongated_distances,
        elongated_scores,
        n_rays,
        overlap_threshold=merge_overlap_threshold,
        angle_threshold=merge_angle_threshold,
        centroid_threshold_factor=merge_centroid_threshold_factor,
        debug_plots=debug_plots,
        debug_plot_dir=plot_dir,
    )
    # Combine merged elongated polygons with star-like polygons
    merged_polygons = merged_polygons + star_like_polygons

    # Calculate orientation angles
    orientation_angles = calculate_all_shapely_polygon_pca_orientations_legacy(merged_polygons)

    if debug_plots and merged_polygons:
        plot_polygons(
            merged_polygons,
            "Merged Polygons",
            n_rays,
            orientation_angles=orientation_angles,
            save_path=plot_dir,
        )
        save_postprocessing_stage_histograms(
            debug_root=plot_dir,
            stage_tag="after_merging",
            image_tag=image_tag,
            polygons=merged_polygons,
            raw_distances=None,
        )


    return merged_polygons, orientation_angles

# main function for Approach A (no overlap) labeling approach
def merge_instances_no_overlap_regions(
    scores,
    dist,
    points_arr,
    thresh,
    use_kdtree=False,
    verbose=False,
    fname=None,
    merge_overlap_threshold: float = 0.0,
    merge_angle_threshold: float = 50,
    merge_alignment_tolerance: float = 1.3,
    merge_aspect_ratio_threshold: float = 5,
    polygon_nms_overlap_threshold: float = 0.5,
    normalized_centroid_threshold: float = 0.5,
    debug_plots: bool = False,
    debug_plot_dir: str | Path | None = None,
    debug_run_name: str | None = None,
    debug_plot_level: str = "basic",
    img_shape: tuple[int, int] | None = None,
    return_stage_outputs: bool = False,
    apply_smoothing: bool = True,
):
    """
    Build polygon candidates from StarDist rays, suppress duplicates, smooth boundaries,
    and merge polygons using a no-overlap merging strategy.

    If return_stage_outputs=True, also returns:
        stage_outputs["after_nms"]
        stage_outputs["after_smoothing"]

    These are captured from the actual pipeline objects, not recomputed.
    """
    fname_prefix = ""
    if fname is not None:
        fname_prefix = f"{fname}__"

    stage_outputs = {}

    plot_dir = None
    if debug_plots:
        plot_dir = resolve_postprocessing_debug_plot_dir(
            run_name=debug_run_name,
            explicit_dir=debug_plot_dir,
            approach_tag="no_overlap",
        )

    if scores is None or dist is None or points_arr is None or len(scores) == 0 or len(dist) == 0 or len(points_arr) == 0:
        if verbose:
            print("Input arrays are empty, returning None or empty lists.")
        if return_stage_outputs:
            return [], [], [], [], stage_outputs
        return [], [], [], []

    if dist.shape[0] != len(points_arr):
        raise ValueError("dist and points_arr dimensions do not match.")

    n_rays = dist.shape[-1]

    results = [
        construct_polygon_opp(i, dist, points_arr, scores)
        for i in range(len(points_arr))
    ]
    polygons, all_distances, all_points, all_scores = zip(*results)

    polygons = list(polygons)
    all_distances = list(all_distances)
    all_points = list(all_points)
    all_scores = list(all_scores)

    polygons, scores, all_distances, all_points = _sort_polygons_by_scores(
        polygons,
        scores,
        all_distances,
        all_points,
    )

    if len(polygons) > 0:
        largest_poly = max(
            (p for p in polygons if p is not None and not p.is_empty),
            key=lambda p: p.area,
            default=None,
        )

        if largest_poly is not None:
            print(
                f"[{fname_prefix.rstrip('__')}] "
                f"Largest polygon area: {largest_poly.area:.2f} px^2"
            )

    def max_distance(dist_list):
        if dist_list is None or len(dist_list) == 0:
            return 0.0
        return float(np.max(dist_list))

    combined_largest = list(zip(polygons, all_scores, all_points, all_distances))
    combined_largest.sort(key=lambda t: t[0].area, reverse=True)
    combined_top_longest = combined_largest[:3]

    combined_longest = list(zip(polygons, all_scores, all_distances, all_points))
    combined_longest.sort(
        key=lambda t: max_distance(t[2]),
        reverse=True,
    )
    combined_longest_top = combined_longest[:3]

    if combined_longest_top:
        polygons_longest_top, all_scores_longest_top, all_distances_longest_top, all_points_longest_top = map(
            list, zip(*combined_longest_top)
        )
    else:
        polygons_longest_top, all_scores_longest_top, all_distances_longest_top, all_points_longest_top = [], [], [], []

    if combined_top_longest:
        polygons_largest_top, all_scores_largest_top, all_points_largest_top, all_dist_largest_top = map(
            list, zip(*combined_top_longest)
        )
    else:
        polygons_largest_top, all_scores_largest_top, all_points_largest_top, all_dist_largest_top = [], [], [], []

    if debug_plots:
        save_before_nms_score_peak_plot(
            debug_root=plot_dir,
            image_tag=_sanitize_tag(fname, default="image"),
            points=np.asarray(all_points, dtype=np.float32),
            scores=np.asarray(all_scores, dtype=np.float32),
            polygons=polygons,
            img_shape=img_shape,
        )
        plot_polygons(
            polygons,
            f"{fname_prefix}Original Polygons",
            n_rays,
            dist=None,
            points_arr=all_points,
            flip_vertical=True,
            save_path=plot_dir,
        )
        save_postprocessing_stage_histograms(
            debug_root=plot_dir,
            stage_tag="before_nms",
            image_tag=_sanitize_tag(fname, default="image"),
            polygons=polygons,
            raw_distances=all_distances,
        )
        plot_polygons(
            polygons,
            f"{fname_prefix}Original Polygons with Scores",
            n_rays,
            dist=None,
            points_arr=all_points,
            scores=all_scores,
            flip_vertical=True,
            save_path=plot_dir,
        )

        if debug_plot_level == "full":
            if polygons_largest_top:
                plot_polygons(
                    polygons_largest_top,
                    f"{fname_prefix}3 largest polygon from Original Polygons",
                    n_rays,
                    dist=None,
                    points_arr=None,
                    flip_vertical=True,
                    save_path=plot_dir,
                )
                plot_polygons(
                    polygons_largest_top,
                    f"{fname_prefix}3 largest polygon from Original Polygons with Scores",
                    n_rays,
                    dist=None,
                    points_arr=all_points_largest_top,
                    scores=all_scores_largest_top,
                    flip_vertical=True,
                    save_path=plot_dir,
                )

            if polygons_longest_top:
                plot_polygons(
                    polygons_longest_top,
                    f"{fname_prefix}3 longest polygon from Original Polygons",
                    n_rays,
                    dist=None,
                    points_arr=None,
                    flip_vertical=True,
                    save_path=plot_dir,
                )
                plot_polygons(
                    polygons_longest_top,
                    f"{fname_prefix}3 longest polygon from Original Polygons with Scores",
                    n_rays,
                    dist=None,
                    points_arr=all_points_longest_top,
                    scores=all_scores_longest_top,
                    flip_vertical=True,
                    save_path=plot_dir,
                )

    # Existing pipeline NMS.
    polygons, all_scores, all_distances, all_points = nms_polygons(
        polygons,
        all_scores,
        all_distances,
        all_points,
        overlap_threshold=(
            polygon_nms_overlap_threshold
        ),
    )

    if return_stage_outputs:
        stage_outputs["after_nms"] = {
            "polygons": list(polygons),
            "scores": list(all_scores),
            "distances": list(all_distances),
            "points": list(all_points),
        }

    if debug_plots:
        plot_polygons(
            polygons,
            f"{fname_prefix}After NMS Polygons",
            n_rays,
            dist=None,
            points_arr=all_points,
            scores=None,
            flip_vertical=True,
            save_path=plot_dir,
        )
        plot_polygons(
            polygons,
            f"{fname_prefix}After NMS Polygons with Scores",
            n_rays,
            dist=None,
            points_arr=all_points,
            scores=all_scores,
            flip_vertical=True,
            save_path=plot_dir,
        )
        save_postprocessing_stage_histograms(
            debug_root=plot_dir,
            stage_tag="after_nms",
            image_tag=_sanitize_tag(fname, default="image"),
            polygons=polygons,
            raw_distances=all_distances,
        )

    # Existing pipeline smoothing.
    if apply_smoothing:
        polygons = [
            poly.simplify(1.2, preserve_topology=True)
            for poly in polygons
        ]

        if return_stage_outputs:
            stage_outputs["after_smoothing"] = {
                "polygons": list(polygons),
                "scores": list(all_scores),
                "distances": list(all_distances),
                "points": list(all_points),
            }

    # Existing pipeline merging.
    merged_polygons, merged_scores, merged_coords = merge_polygons_iterative_no_overlap_regions(
        polygons,
        all_points,
        all_distances,
        all_scores,
        n_rays,
        overlap_threshold=merge_overlap_threshold,
        angle_threshold=merge_angle_threshold,
        normalized_centroid_threshold=( normalized_centroid_threshold ),
        alignment_tolerance=merge_alignment_tolerance,
        aspect_ratio_threshold=merge_aspect_ratio_threshold,
        fname=fname,
        debug_plots=debug_plots,
        debug_plot_dir=plot_dir,
    )

    orientation_angles = calculate_all_shapely_polygon_pca_orientations_legacy(merged_polygons)

    if debug_plots and merged_polygons:
        plot_polygons(
            merged_polygons,
            f"{fname_prefix}masked CNTs",
            n_rays,
            orientation_angles=orientation_angles,
            flip_vertical=True,
            save_path=plot_dir,
        )
        save_postprocessing_stage_histograms(
            debug_root=plot_dir,
            stage_tag="after_merging",
            image_tag=_sanitize_tag(fname, default="image"),
            polygons=merged_polygons,
            raw_distances=None,
        )

    if return_stage_outputs:
        return merged_polygons, orientation_angles, merged_scores, merged_coords, stage_outputs

    return merged_polygons, orientation_angles, merged_scores, merged_coords