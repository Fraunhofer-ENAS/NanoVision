from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, ks_2samp, mannwhitneyu, wasserstein_distance


MAX_WASSERSTEIN = 1000.0
MAX_JENSEN_SHANNON = 1.0


def _clean_array(data: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    return arr[~np.isnan(arr)]


def compare_distributions(
    gt_data: list[float] | np.ndarray,
    pred_data: list[float] | np.ndarray,
    metric_name: str,
    bins: int = 60,
) -> dict[str, Any] | None:
    gt_clean = _clean_array(gt_data)
    pred_clean = _clean_array(pred_data)

    if len(gt_clean) == 0 or len(pred_clean) == 0:
        return None

    hist_range = (min(gt_clean.min(), pred_clean.min()), max(gt_clean.max(), pred_clean.max()))
    hist_gt, _ = np.histogram(gt_clean, bins=bins, range=hist_range, density=True)
    hist_pred, _ = np.histogram(pred_clean, bins=bins, range=hist_range, density=True)

    return {
        "metric": metric_name,
        "wasserstein": float(wasserstein_distance(gt_clean, pred_clean)),
        "jensen_shannon": float(jensenshannon(hist_gt, hist_pred, base=2)),
    }


def compare_distributions_enhanced(
    gt_data: list[float] | np.ndarray,
    pred_data: list[float] | np.ndarray,
    metric_name: str,
    bins: int = 50,
    alpha: float = 0.05,
) -> dict[str, Any] | None:
    gt_clean = _clean_array(gt_data)
    pred_clean = _clean_array(pred_data)

    if len(gt_clean) == 0 or len(pred_clean) == 0:
        return None

    results: dict[str, Any] = {
        "metric": metric_name,
        "n_gt": len(gt_clean),
        "n_pred": len(pred_clean),
        "mean_gt": float(np.mean(gt_clean)),
        "mean_pred": float(np.mean(pred_clean)),
        "std_gt": float(np.std(gt_clean)),
        "std_pred": float(np.std(pred_clean)),
        "median_gt": float(np.median(gt_clean)),
        "median_pred": float(np.median(pred_clean)),
        "wasserstein": float(wasserstein_distance(gt_clean, pred_clean)),
    }

    all_data = np.concatenate([gt_clean, pred_clean])
    hist_range = (float(np.min(all_data)), float(np.max(all_data)))

    hist_gt, _ = np.histogram(gt_clean, bins=bins, range=hist_range, density=True)
    hist_pred, _ = np.histogram(pred_clean, bins=bins, range=hist_range, density=True)

    hist_gt = hist_gt / np.sum(hist_gt)
    hist_pred = hist_pred / np.sum(hist_pred)

    results["jensen_shannon"] = float(jensenshannon(hist_gt, hist_pred, base=2))

    epsilon = 1e-10
    hist_gt_smooth = hist_gt + epsilon
    hist_pred_smooth = hist_pred + epsilon
    hist_gt_smooth = hist_gt_smooth / np.sum(hist_gt_smooth)
    hist_pred_smooth = hist_pred_smooth / np.sum(hist_pred_smooth)

    results["kl_divergence"] = float(entropy(hist_gt_smooth, hist_pred_smooth))

    ks_stat, ks_pvalue = ks_2samp(gt_clean, pred_clean)
    results["ks_statistic"] = float(ks_stat)
    results["ks_pvalue"] = float(ks_pvalue)
    results["ks_significant"] = bool(ks_pvalue < alpha)

    if len(gt_clean) > 20 and len(pred_clean) > 20:
        mw_stat, mw_pvalue = mannwhitneyu(gt_clean, pred_clean)
        results["mw_statistic"] = float(mw_stat)
        results["mw_pvalue"] = float(mw_pvalue)
        results["mw_significant"] = bool(mw_pvalue < alpha)

    return results

def compute_distribution_stats(
    data_true,
    data_pred,
    *,
    bins: int = 60,
) -> pd.DataFrame:
    """
    Compute summary statistics between two one-dimensional distributions.

    This is the canonical DataFrame-oriented distribution-comparison API.

    The returned statistics are:

    - mean and standard deviation of the reference distribution,
    - mean and standard deviation of the predicted distribution,
    - Wasserstein distance,
    - Jensen-Shannon distance.

    Parameters
    ----------
    data_true:
        Reference distribution.
    data_pred:
        Predicted distribution.
    bins:
        Number of shared histogram bins used for the Jensen-Shannon
        calculation.

    Returns
    -------
    pandas.DataFrame
        Single-row dataframe containing the distribution statistics.
    """
    data_true = np.asarray( data_true, dtype=float, )
    data_pred = np.asarray( data_pred, dtype=float, )

    results: dict[str, float] = {}

    results["mean_true"] = float( np.mean(data_true) )
    results["std_true"] = float( np.std(data_true) )
    results["mean_pred"] = float( np.mean(data_pred) )
    results["std_pred"] = float( np.std(data_pred) )

    all_data = np.concatenate( [ data_true, data_pred, ] )
    hist_range = ( np.min(all_data), np.max(all_data), )

    hist_true, _ = np.histogram( data_true, bins=bins, range=hist_range, density=True, )

    hist_pred, _ = np.histogram( data_pred, bins=bins, range=hist_range, density=True, )

    if hist_true.sum() > 0:
        hist_true = ( hist_true / hist_true.sum() )

    if hist_pred.sum() > 0:
        hist_pred = ( hist_pred / hist_pred.sum() )

    results["wasserstein_distance"] = float( wasserstein_distance( data_true, data_pred, ) )

    results["jensen_shannon_distance"] = float( jensenshannon( hist_true, hist_pred, base=2, ) )

    return pd.DataFrame( [results] )

def build_penalty_result(metric_name: str, model_name: str, no_predictions: bool = True) -> dict[str, Any]:
    return {
        "metric": metric_name,
        "wasserstein": MAX_WASSERSTEIN,
        "jensen_shannon": MAX_JENSEN_SHANNON,
        "model": model_name,
        "no_predictions": no_predictions,
    }
