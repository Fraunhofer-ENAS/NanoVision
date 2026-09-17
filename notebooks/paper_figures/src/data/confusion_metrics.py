from pathlib import Path
from typing import Iterable, Protocol

import pandas as pd


class _ProjectPathsLike(Protocol):
    def run_dir(self, run_name: str) -> Path:
        ...

def load_confusion_metrics(
    model_names: list[str],
    project_paths: _ProjectPathsLike,
    metric_file: str = "dsb_per_image_threshold_metrics.csv",
    metric_subpath: str = "eval/dsb/Diagnostic_eval__gt-old__pred-poly/dsb_per_image_threshold_metrics.csv",
    legacy_metric_candidates: Iterable[str] | None = None,
    require_metric_subpath: bool = False,
) -> pd.DataFrame:
    """
    Load TP/FP/FN and IoU thresholds for all models.
    Returns a combined DataFrame with columns:
    - image_id
    - TP, FP, FN
    - IoU_thresh
    - experiment
    """

    default_legacy_candidates = (
        "eval/dsb/Diagnostic_eval__gt-old__pred-rle/dsb_per_image_threshold_metrics.csv",
        "eval/dsb/Diagnostic_eval__gt-old__pred-mask/dsb_per_image_threshold_metrics.csv",
        "eval/dsb/density_eval__gt-approx-none__pred-poly/dsb_per_image_threshold_metrics.csv",
        "eval/dsb_per_image_threshold_metrics.csv",
        "eval/per_image_threshold_metrics.csv",
    )

    dfs: list[pd.DataFrame] = []
    for name in model_names:
        run_dir = project_paths.run_dir(name)

        candidates: list[Path] = []

        # Highest-priority explicit location.
        explicit_candidate = run_dir / metric_subpath
        if explicit_candidate.exists():
            candidates.append(explicit_candidate)
        elif require_metric_subpath:
            print(f"[warn] Required metric file missing: {explicit_candidate}")
            continue

        # Canonical contract: metrics are under eval/dsb/<eval_tag>/.
        dsb_root = run_dir / "eval" / "dsb"
        if dsb_root.exists():
            for p in sorted(dsb_root.rglob(metric_file)):
                if p not in candidates:
                    candidates.append(p)

        if legacy_metric_candidates is None:
            rel_candidates = default_legacy_candidates
        else:
            rel_candidates = tuple(str(p) for p in legacy_metric_candidates)

        for rel in rel_candidates:
            candidate = run_dir / rel
            if candidate.exists() and candidate not in candidates:
                candidates.append(candidate)

        file = None
        if candidates:
            file = candidates[0]

        if file is None:
            print(
                f"[warn] Missing confusion metric file for run '{name}'. "
                f"Checked explicit and dsb-recursive paths under: {run_dir / 'eval' / 'dsb'}"
            )
            continue

        df = pd.read_csv(file)

        required = {"image_id", "TP", "FP", "FN", "IoU_thresh"}
        if not required.issubset(df.columns):
            print(f"[warn] Missing required columns in: {file}")
            continue

        df["experiment"] = name
        dfs.append(df)

    if not dfs:
        raise ValueError(
            "No valid confusion metric files found under "
            "global_outputs/runs/<RUN>/... for the requested runs."
        )

    return pd.concat(dfs, ignore_index=True)
