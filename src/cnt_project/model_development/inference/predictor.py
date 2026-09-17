from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PredictionBatch:
    """
    Predictions produced for one collection of inference images.

    Attributes
    ----------
    filenames:
        Filename stems corresponding to the predictions.
    images:
        Input images used for inference.
    y_preds:
        Predicted instance-label masks.
    polygons:
        Predicted polygon collections for each image.
    scores:
        Prediction scores corresponding to the polygons.
    coords:
        Predicted coordinate information returned by the patched StarDist model.
    """

    filenames: list[str]
    images: list[np.ndarray]
    y_preds: list[np.ndarray]
    polygons: list[list[Any]]
    scores: list[Any]
    coords: list[Any]


def predict_dataset(
    model: Any,
    images: list[np.ndarray],
    filenames: list[str],
    *,
    prob_thresh: float | None = None,
    apply_smoothing: bool | None = None,
    debug_stage_plots: bool = False,
    debug_plot_dir: str | None = None,
    debug_plot_level: str = "basic",
) -> PredictionBatch:
    """
    Run StarDist instance prediction on a collection of images.

    Parameters
    ----------
    model:
        Initialized StarDist-compatible model exposing ``predict_instances``.
    images:
        Images to predict.
    filenames:
        Filename stems corresponding to ``images``.

    prob_thresh:
        Optional object-probability threshold override. When omitted, the
        model's configured or default probability threshold is used.

    apply_smoothing:
        Optional override controlling whether polygon smoothing is applied
        during model postprocessing. When omitted, the model's default
        behavior is preserved.

    debug_stage_plots:
        Whether to save intermediate postprocessing-stage debug plots.

    debug_plot_dir:
        Output directory for stage-debug plots. Required when
        ``debug_stage_plots=True``.

    debug_plot_level:
        Stage-debug detail level. Supported values are ``"basic"`` and
        ``"full"``.

    Returns
    -------
    PredictionBatch
        Predictions and associated source information.

    Raises
    ------
    ValueError
        If the number of images and filenames differs.
    """
    if len(images) != len(filenames):
        raise ValueError(
            "images and filenames must contain the same number of entries: "
            f"images={len(images)}, filenames={len(filenames)}."
        )
    if prob_thresh is not None and not 0.0 < prob_thresh < 1.0:
        raise ValueError(
            "prob_thresh must be strictly between 0 and 1. "
            f"Got: {prob_thresh}"
        )
    if debug_plot_level not in {"basic", "full"}:
        raise ValueError(
            "debug_plot_level must be one of: 'basic', 'full'. "
            f"Got: {debug_plot_level!r}"
        )

    if debug_stage_plots and debug_plot_dir is None:
        raise ValueError(
            "debug_plot_dir is required when debug_stage_plots=True."
        )

    y_preds: list[np.ndarray] = []
    polygons_all: list[list[Any]] = []
    scores_all: list[Any] = []
    coords_all: list[Any] = []

    for idx, (image, filename) in enumerate(
        zip(images, filenames),
        start=1,
    ):
        print(
            f"Processing {idx}/{len(images)}: {filename}"
        )
        predict_kwargs = { "n_tiles": model._guess_n_tiles(image), "show_tile_progress": False, "fname": filename, }
        if prob_thresh is not None:
            predict_kwargs["prob_thresh"] = prob_thresh
            
        postprocess_kwargs: dict[str, Any] = {}

        if apply_smoothing is not None:
            postprocess_kwargs["apply_smoothing"] = apply_smoothing

        if debug_stage_plots:
            postprocess_kwargs.update(
                {
                    "debug_plots": True,
                    "debug_plot_dir": debug_plot_dir,
                    "debug_run_name": "stage_debug",
                    "debug_plot_level": debug_plot_level,
                }
            )

        if postprocess_kwargs:
            predict_kwargs["nms_kwargs"] = postprocess_kwargs

        (
            y_pred,
            _details,
            polygon_list,
            scores,
            coords,
        ) = model.predict_instances(
            image,
            **predict_kwargs,
        )

        y_preds.append(y_pred)
        polygons_all.append(polygon_list)
        scores_all.append(scores)
        coords_all.append(coords)

    return PredictionBatch(
        filenames=list(filenames),
        images=list(images),
        y_preds=y_preds,
        polygons=polygons_all,
        scores=scores_all,
        coords=coords_all,
    )