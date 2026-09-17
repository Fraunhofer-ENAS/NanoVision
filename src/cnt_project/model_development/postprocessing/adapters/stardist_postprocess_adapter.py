from __future__ import annotations

import numpy as np
from time import time
from cnt_project.model_development.postprocessing.pipeline.cnt_postprocess_pipeline import merge_instances_no_overlap_regions
from .format_conversion import (
    create_polygon_array,
    draw_polygons_on_image,
    extract_centers_probabilities,
)



def postprocess_instances_merge_polygons(
    dist,
    prob,
    points,
    img_shape,
    b=2,
    polygon_nms_overlap_threshold: float = 0.5,
    merge_overlap_threshold: float = 0.0,
    merge_angle_threshold: float = 50.0,
    normalized_centroid_threshold: float = 0.5,
    nms_thresh=None,
    use_bbox=True,
    use_kdtree=True,
    verbose=False,
    fname=None,
    debug_plots=False,
    debug_plot_dir=None,
    debug_run_name=None,
    debug_plot_level="basic",
    return_stage_outputs: bool = False,
    apply_smoothing: bool = True,
):
    """
    Post-process StarDist-style ray-based instance predictions by merging
    overlapping polygon candidates into final instance shapes.

    If return_stage_outputs=True, an additional final item is returned:
        stage_outputs

    Default return remains unchanged for compatibility.
    """
    dist = np.asarray(dist)
    prob = np.asarray(prob)
    points = np.asarray(points)
    n_rays = dist.shape[-1]

    assert dist.ndim == 2 and prob.ndim == 1 and points.ndim == 2 and \
        points.shape[-1] == 2 and len(prob) == len(dist) == len(points)

    verbose and print(
        "predicting instances with nms_thresh = {nms_thresh}".format(
            nms_thresh=nms_thresh
        ),
        flush=True,
    )

    inds_original = np.arange(len(prob))
    _sorted = np.argsort(prob)[::-1]
    probi = prob[_sorted]
    disti = dist[_sorted]
    pointsi = points[_sorted]
    inds_original = inds_original[_sorted]

    if verbose:
        print("non-maximum suppression...")
        t = time()

    result = instances_merge_adapter(
        disti,
        pointsi,
        probi,
        img_shape,
        thresh=nms_thresh,
        use_kdtree=use_kdtree,
        verbose=verbose,
        fname=fname,
        debug_plots=debug_plots,
        polygon_nms_overlap_threshold= polygon_nms_overlap_threshold,
        merge_overlap_threshold = merge_overlap_threshold,
        merge_angle_threshold = merge_angle_threshold,
        normalized_centroid_threshold = normalized_centroid_threshold,
        debug_plot_dir=debug_plot_dir,
        debug_run_name=debug_run_name,
        debug_plot_level=debug_plot_level,
        return_stage_outputs=return_stage_outputs,
        apply_smoothing=apply_smoothing,
    )

    if return_stage_outputs:
        (
            inds,
            new_label,
            new_coord,
            new_points,
            new_prob,
            final_cell_shapes,
            final_scores,
            final_coords,
            stage_outputs,
        ) = result
    else:
        (
            inds,
            new_label,
            new_coord,
            new_points,
            new_prob,
            final_cell_shapes,
            final_scores,
            final_coords,
        ) = result
        stage_outputs = None

    if verbose:
        print("keeping %s/%s polyhedra" % (np.count_nonzero(inds), len(inds)))
        print("NMS took %.4f s" % (time() - t))

    base_result = (
        pointsi[inds],
        probi[inds],
        disti[inds],
        inds_original[inds],
        new_label,
        new_coord,
        new_points,
        new_prob,
        final_cell_shapes,
        final_scores,
        final_coords,
    )

    if return_stage_outputs:
        return base_result + (stage_outputs,)

    return base_result

def instances_merge_adapter(
    dist,
    points,
    scores,
    img_shape,
    thresh=0.5,
    use_bbox=True,
    polygon_nms_overlap_threshold: float = 0.5,
    merge_overlap_threshold: float = 0.0,
    merge_angle_threshold: float = 50.0,
    normalized_centroid_threshold: float = 0.5,
    use_kdtree=True,
    verbose=1,
    fname=None,
    debug_plots=False,
    debug_plot_dir=None,
    debug_run_name=None,
    debug_plot_level="basic",
    return_stage_outputs: bool = False,
    apply_smoothing: bool = True,
):
    """
    Adapter function that converts sorted ray-based instance candidates into
    merged instance representations.

    If return_stage_outputs=True, an additional final item is returned:
        stage_outputs
    """
    assert dist.ndim == 2
    assert points.ndim == 2

    rays = dist.shape[-1]
    n_poly = dist.shape[0]

    if scores is None:
        scores = np.ones(n_poly)

    assert len(scores) == n_poly
    assert points.shape[0] == n_poly

    def _prep(x, dtype):
        return np.ascontiguousarray(x.astype(dtype, copy=False))

    inds = []

    result = build_stardist_outputs_from_merged_polygons(
        _prep(scores, np.float32),
        _prep(dist, np.float32),
        _prep(points, np.float32),
        img_shape,
        rays,
        np.float32(thresh),
        use_kdtree=use_kdtree,
        verbose=verbose,
        fname=fname,
        debug_plots=debug_plots,
        polygon_nms_overlap_threshold= polygon_nms_overlap_threshold,
        merge_overlap_threshold = merge_overlap_threshold,
        merge_angle_threshold = merge_angle_threshold,
        normalized_centroid_threshold = normalized_centroid_threshold,
        debug_plot_dir=debug_plot_dir,
        debug_run_name=debug_run_name,
        debug_plot_level=debug_plot_level,
        return_stage_outputs=return_stage_outputs,
        apply_smoothing=apply_smoothing,
    )

    if return_stage_outputs:
        (
            new_label,
            new_coord,
            new_points,
            new_prob,
            final_cell_shapes,
            final_scores,
            final_coords,
            stage_outputs,
        ) = result

        return (
            inds,
            new_label,
            new_coord,
            new_points,
            new_prob,
            final_cell_shapes,
            final_scores,
            final_coords,
            stage_outputs,
        )

    (
        new_label,
        new_coord,
        new_points,
        new_prob,
        final_cell_shapes,
        final_scores,
        final_coords,
    ) = result

    return (
        inds,
        new_label,
        new_coord,
        new_points,
        new_prob,
        final_cell_shapes,
        final_scores,
        final_coords,
    )

def build_stardist_outputs_from_merged_polygons(
    scores,
    dist,
    points_arr,
    img_shape,
    n_rays,
    thresh,
    polygon_nms_overlap_threshold: float = 0.5,
    merge_overlap_threshold: float = 0.0,
    merge_angle_threshold: float = 50.0,
    normalized_centroid_threshold: float = 0.5,
    use_kdtree=False,
    verbose=False,
    fname=None,
    debug_plots=False,
    debug_plot_dir=None,
    debug_run_name=None,
    debug_plot_level="basic",
    return_stage_outputs: bool = False,
    apply_smoothing: bool = True,
):
    """
    Construct StarDist-compatible output representations from merged polygon
    instances.

    If return_stage_outputs=True, an additional final item is returned:
        stage_outputs
    """
    def _prep(x, dtype):
        return np.ascontiguousarray(x.astype(dtype, copy=False))

    result = merge_instances_no_overlap_regions(
        _prep(scores, np.float32),
        _prep(dist, np.float32),
        _prep(points_arr, np.float32),
        np.float32(thresh),
        int(use_kdtree),
        int(verbose),
        fname=fname,
        debug_plots=debug_plots,
        polygon_nms_overlap_threshold= polygon_nms_overlap_threshold,
        merge_overlap_threshold = merge_overlap_threshold,
        merge_angle_threshold = merge_angle_threshold,
        normalized_centroid_threshold = normalized_centroid_threshold,
        debug_plot_dir=debug_plot_dir,
        debug_run_name=debug_run_name,
        debug_plot_level=debug_plot_level,
        img_shape=img_shape,
        return_stage_outputs=return_stage_outputs,
        apply_smoothing=apply_smoothing,
    )

    if return_stage_outputs:
        (
            final_cell_shapes,
            orientation_angles,
            final_scores,
            final_coords,
            stage_outputs,
        ) = result
    else:
        (
            final_cell_shapes,
            orientation_angles,
            final_scores,
            final_coords,
        ) = result
        stage_outputs = None

    image_size = img_shape
    new_label = np.zeros(image_size, dtype=np.int32)

    draw_polygons_on_image(new_label, final_cell_shapes)

    new_coord = create_polygon_array(final_cell_shapes, n_rays)
    new_points, new_prob = extract_centers_probabilities(
        final_cell_shapes,
        points_arr,
        scores,
    )

    base_result = (
        new_label,
        new_coord,
        new_points,
        new_prob,
        final_cell_shapes,
        final_scores,
        final_coords,
    )

    if return_stage_outputs:
        return base_result + (stage_outputs,)

    return base_result

