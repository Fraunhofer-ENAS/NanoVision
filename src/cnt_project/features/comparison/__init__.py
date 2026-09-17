"""Feature-comparison pipelines and distribution metrics."""

from .comparison_pipeline import (
    build_single_image_results_for_plotting,
    run_model_comparison_pipeline,
)
from .feature_distribution_metrics import (
    MAX_JENSEN_SHANNON,
    MAX_WASSERSTEIN,
    build_penalty_result,
    compare_distributions,
    compare_distributions_enhanced,
    compute_distribution_stats,
)
from .feature_distribution_comparison import (
    build_distribution_comparison_table,
    build_distribution_inputs_for_image,
    build_multi_distribution_summary,
    safe_get_metric_values,
)

__all__ = [
    "MAX_JENSEN_SHANNON",
    "MAX_WASSERSTEIN",
    "build_penalty_result",
    "compare_distributions",
    "compare_distributions_enhanced",
    "run_model_comparison_pipeline",
    "build_single_image_results_for_plotting",
    "compute_distribution_stats",
    "build_distribution_comparison_table",
    "build_distribution_inputs_for_image",
    "build_multi_distribution_summary",
    "safe_get_metric_values",
]
