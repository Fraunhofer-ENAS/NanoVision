from __future__ import annotations


def display_density_label(density: str) -> str:
    mapping = {
        "LOW_DENSITY": "Low",
        "MID_DENSITY": "Mid",
        "HIGH_DENSITY": "High",
    }
    return mapping.get(density, density)

def infer_smoothing_variant(run_name: str) -> str:
    """
    Infer the smoothing-ablation variant encoded in a run name.
    """
    if run_name.endswith("__with_smoothing"):
        return "with_smoothing"

    if run_name.endswith("__without_smoothing"):
        return "without_smoothing"

    return "unknown"


def infer_smoothing_model_base(run_name: str) -> str:
    """
    Remove the smoothing-ablation suffix from a run name.
    """
    for suffix in (
        "__with_smoothing",
        "__without_smoothing",
    ):
        if run_name.endswith(suffix):
            return run_name[: -len(suffix)]

    return run_name


def default_smoothing_label(run_name: str) -> str:
    """
    Build a human-readable label for a smoothing-ablation run.
    """
    base = infer_smoothing_model_base(run_name)
    variant = infer_smoothing_variant(run_name)

    variant_label = {
        "with_smoothing": "with smoothing",
        "without_smoothing": "without smoothing",
        "unknown": "unknown",
    }[variant]

    return f"{base} {variant_label}"

# -----------------------------------------------------------------------------
# Shared small helpers (no logic changes, only reused)
# -----------------------------------------------------------------------------
def safe_progress_iter(n: int):
    """
    Optional progress bar:
    - If alive_progress is installed, use it.
    - Otherwise: simple range loop.
    """
    try:
        from alive_progress import alive_bar  # type: ignore
    except Exception:
        for _ in range(n):
            yield None
        return

    with alive_bar(n) as bar:
        for _ in range(n):
            yield bar
