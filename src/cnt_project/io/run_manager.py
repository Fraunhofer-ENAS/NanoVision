"""
Canonical experiment run management.

This module should be responsible for all functionality related to experiment runs,
independent of whether the run performs training, inference, evaluation,
dataset preparation, or any future workflow.

Responsibilities
----------------
- Generate canonical run identifiers.
- Generate filesystem-safe run names.
- Create and attach to run directories.
- Define the standard run folder structure.
- Return canonical run-scoped paths.

This module intentionally does not save workflow-specific artifacts.
Training histories, evaluation reports, inference outputs, and similar
artifacts belong to their respective subpackages.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from cnt_project.io.paths import OutputPaths, ProjectPaths


def safe_run_name(
    value: str,
    *,
    max_len: int = 180,
) -> str:
    """
    Return a filesystem-safe run or artifact name.

    The sanitization is compatible with Windows filenames:

    - illegal filesystem characters are replaced with underscores;
    - whitespace sequences are replaced with underscores;
    - leading and trailing underscores are removed;
    - the result is limited to ``max_len`` characters.
    """
    cleaned = str(value)

    cleaned = re.sub(
        r'[<>:"/\\|?*\x00-\x1F]',
        "_",
        cleaned,
    )
    cleaned = re.sub(
        r"\s+",
        "_",
        cleaned,
    ).strip("_")

    return cleaned[:max_len]


def timestamp_now() -> str:
    """Return a local timestamp suitable for run identifiers."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_id(
    *parts: str,
    ts: Optional[str] = None,
) -> str:
    """
    Build a descriptive run identifier from multiple parts and a timestamp.

    Parameters
    ----------
    *parts
        Descriptive run-name components.

        Example:

        ``AM_thesis_model``,
        ``mode-centerness``,
        ``nrays-64``,
        ``grid-2x2``.

    ts
        Optional timestamp string. When omitted, the current local timestamp is
        generated using ``YYYYMMDD_HHMMSS``.

    Returns
    -------
    str
        A filesystem-safe run identifier.

    Examples
    --------
    ``AM_thesis_model__mode-standard__nrays-64__20260805_183000``
    """
    timestamp = ts or timestamp_now()

    cleaned_parts = [
        safe_run_name(part)
        for part in parts
        if str(part).strip()
    ]

    if not cleaned_parts:
        raise ValueError(
            "At least one non-empty run ID component is required."
        )

    base = "__".join(cleaned_parts)

    return safe_run_name(
        f"{base}__{timestamp}"
    )


def build_run_name(
    *,
    model_name: str,
    run_name: str | None = None,
    timestamp: datetime | None = None,
) -> str:
    """
    Resolve a filesystem-safe run identifier.

    If ``run_name`` is supplied, it is sanitized and returned directly.

    Otherwise, the identifier is generated using:

    ``<model_name>_<YYYYMMDD_HHMMSS>``

    Parameters
    ----------
    model_name
        Model name used when automatically generating a run name.

    run_name
        Optional explicit run name.

    timestamp
        Optional timestamp value, primarily useful for deterministic tests.

    Returns
    -------
    str
        Filesystem-safe run identifier.
    """
    if run_name is not None:
        cleaned_run_name = safe_run_name(run_name)

        if not cleaned_run_name:
            raise ValueError(
                "run_name is empty after filesystem sanitization."
            )

        return cleaned_run_name

    cleaned_model_name = safe_run_name(model_name)

    if not cleaned_model_name:
        raise ValueError(
            "model_name is empty after filesystem sanitization."
        )

    resolved_timestamp = timestamp or datetime.now()
    timestamp_part = resolved_timestamp.strftime("%Y%m%d_%H%M%S")

    return safe_run_name(
        f"{cleaned_model_name}_{timestamp_part}"
    )


@dataclass(frozen=True)
class RunPaths:
    """
    Canonical paths associated with one experiment run.
    """

    run_id: str
    run_dir: Path
    train_dir: Path
    inference_dir: Path
    eval_dir: Path
    viz_dir: Path
    figures_dir: Path


def _build_run_paths(
    paths: ProjectPaths | OutputPaths,
    run_id: str,
    *,
    create_dirs: bool,
) -> RunPaths:
    """
    Build canonical run-scoped paths.

    The resulting structure is:

    ``global_outputs/runs/<RUN>/``

    with:

    - ``train/``
    - ``inference/``
    - ``eval/``
    - ``viz/``
    - ``viz/figures/``
    """
    normalized_run_id = safe_run_name(run_id)

    if not normalized_run_id:
        raise ValueError(
            "run_id is empty after filesystem sanitization."
        )

    run_dir = paths.runs_root / normalized_run_id
    train_dir = run_dir / "train"
    inference_dir = run_dir / "inference"
    eval_dir = run_dir / "eval"
    viz_dir = run_dir / "viz"
    figures_dir = viz_dir / "figures"

    if create_dirs:
        for directory in (
            run_dir,
            train_dir,
            inference_dir,
            eval_dir,
            viz_dir,
            figures_dir,
        ):
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    return RunPaths(
        run_id=normalized_run_id,
        run_dir=run_dir,
        train_dir=train_dir,
        inference_dir=inference_dir,
        eval_dir=eval_dir,
        viz_dir=viz_dir,
        figures_dir=figures_dir,
    )


def create_run(
    *,
    project_root: Path,
    run_id: str,
    outputs_root: str | Path = "global_outputs",
    runs_root: str | Path | None = None,
) -> RunPaths:
    """
    Create a canonical run directory and its standard subdirectories.

    Parameters
    ----------
    project_root
        Repository root.

    run_id
        Run identifier.

    outputs_root
        Canonical generated-output root, relative to ``project_root`` unless
        absolute.

    runs_root
        Deprecated compatibility alias for ``outputs_root``.

        When supplied, it takes precedence over ``outputs_root``.

    Returns
    -------
    RunPaths
        Canonical paths for the created run.
    """
    effective_outputs_root = (
        runs_root
        if runs_root is not None
        else outputs_root
    )

    paths = ProjectPaths.from_project_root(
        project_root,
        outputs_root=effective_outputs_root,
    )
    paths.ensure_outputs()

    return _build_run_paths(
        paths,
        run_id,
        create_dirs=True,
    )


def create_run_from_outputs_root(
    *,
    outputs_root: str | Path,
    run_id: str,
) -> RunPaths:
    """
    Create a canonical run below an explicit generated-output root.

    This variant is independent of the CNTLib source-repository layout and is
    therefore suitable for installed-library consumers.

    The resulting structure is:

        <outputs_root>/
            runs/
                <RUN>/
                    train/
                    inference/
                    eval/
                    viz/

            reports/
            misc/

    Parameters
    ----------
    outputs_root
        Root directory under which CNTLib manages runs/, reports/, and misc/.

    run_id
        Run identifier.

    Returns
    -------
    RunPaths
        Canonical paths for the created run.
    """
    paths = OutputPaths.from_root(
        outputs_root
    )

    paths.ensure()

    return _build_run_paths(
        paths,
        run_id,
        create_dirs=True,
    )

def attach_run(
    *,
    project_root: Path,
    run_id: str,
    outputs_root: str | Path = "global_outputs",
    runs_root: str | Path | None = None,
    ensure_exists: bool = True,
    ensure_structure: bool = False,
) -> RunPaths:
    """
    Attach to an existing run.

    Parameters
    ----------
    project_root
        Repository root.

    run_id
        Existing run identifier.

    outputs_root
        Canonical generated-output root.

    runs_root
        Deprecated compatibility alias for ``outputs_root``.

    ensure_exists
        Raise ``FileNotFoundError`` when the run directory does not exist.

    ensure_structure
        Create missing canonical subdirectories.

    Returns
    -------
    RunPaths
        Canonical paths associated with the run.
    """
    effective_outputs_root = (
        runs_root
        if runs_root is not None
        else outputs_root
    )

    paths = ProjectPaths.from_project_root(
        project_root,
        outputs_root=effective_outputs_root,
    )
    paths.ensure_outputs()

    normalized_run_id = safe_run_name(run_id)

    if not normalized_run_id:
        raise ValueError(
            "run_id is empty after filesystem sanitization."
        )

    run_dir = paths.runs_root / normalized_run_id

    if ensure_exists and not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory not found: {run_dir.resolve()}"
        )

    return _build_run_paths(
        paths,
        normalized_run_id,
        create_dirs=ensure_structure,
    )