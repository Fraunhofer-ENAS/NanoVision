from __future__ import annotations

import pandas as pd


from cnt_project.coco.io import get_annotations_for_image_id

from cnt_project.features.core.cnt_feature_extraction import (
    calculate_cnt_features_from_annotations,
    compute_line_densities_from_polygons,
)
from cnt_project.features.comparison.feature_distribution_metrics import (
    compute_distribution_stats,
)




def build_distribution_inputs_for_image(
    gt_json: dict,
    pred_json: dict,
    image_id: int,
    *,
    image_height: int,
    image_width: int,
) -> dict[str, object]:
    """
    Build the GT/prediction distribution-analysis inputs for one image.
    """
    image_annotations_gt = get_annotations_for_image_id(gt_json, image_id)
    image_annotations_pred = get_annotations_for_image_id(pred_json, image_id)

    line_density_gt = compute_line_densities_from_polygons(
        image_annotations_gt,
        image_height,
        image_width,
    )
    line_density_pred = compute_line_densities_from_polygons(
        image_annotations_pred,
        image_height,
        image_width,
    )

    metrics_gt = calculate_cnt_features_from_annotations(
        image_annotations_gt,
        image_height,
        image_width,
    )
    metrics_pred = calculate_cnt_features_from_annotations(
        image_annotations_pred,
        image_height,
        image_width,
    )

    return {
        "gt_annotations": image_annotations_gt,
        "pred_annotations": image_annotations_pred,
        "line_density_gt": line_density_gt,
        "line_density_pred": line_density_pred,
        "metrics_gt": metrics_gt,
        "metrics_pred": metrics_pred,
    }


def build_distribution_comparison_table(
    data_true,
    data_pred,
    *,
    label: str,
    bins: int = 100,
) -> pd.DataFrame:
    """
    Compute a one-row summary table for one GT vs prediction distribution comparison.
    """
    stats_df = compute_distribution_stats(
        data_true,
        data_pred,
        bins=bins,
    )
    stats_df.insert(0, "metric", label)
    return stats_df


def build_multi_distribution_summary(
    comparisons: dict[str, tuple[object, object]],
    *,
    bins: int = 100,
) -> pd.DataFrame:
    """
    Build a stacked summary dataframe for multiple GT vs prediction comparisons.

    Parameters
    ----------
    comparisons:
        dict like:
            {
                "line_count": (gt_values, pred_values),
                "orientation_angle": (gt_values, pred_values),
                ...
            }
    """
    rows = []

    for label, (data_true, data_pred) in comparisons.items():
        df = build_distribution_comparison_table(
            data_true,
            data_pred,
            label=label,
            bins=bins,
        )
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)



def safe_get_metric_values(
    metrics_dict: dict[str, object],
    key: str,
) -> list[float]:
    """
    Safely extract a list-valued metric from the property dictionary.
    """
    values = metrics_dict.get(key, [])
    if not isinstance(values, list):
        return []
    return values