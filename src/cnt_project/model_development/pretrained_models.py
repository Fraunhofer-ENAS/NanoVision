from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from stardist.models import StarDist2D


def _safe_model_name(name: str, max_len: int = 180) -> str:
    name = str(name)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name)
    name = re.sub(r"\s+", "_", name).strip("_")
    return name[:max_len]


@dataclass(frozen=True)
class ImportedPretrainedModel:
    source_name: str
    source_dir: Path
    target_dir: Path


def list_official_pretrained_models() -> list[str]:
    """Return the names of StarDist 2D pretrained models known to the installed package."""
    model_names = StarDist2D.from_pretrained()
    return list(model_names) if model_names is not None else []


def import_official_pretrained_model(
    pretrained_model_name: str,
    *,
    project_root: str | Path,
    local_model_name: str | None = None,
    models_dir: str | Path = "models",
    overwrite: bool = False,
) -> ImportedPretrainedModel:
    """
    Download an official StarDist pretrained model (if needed) and copy it into the
    repository-local models folder so it can be resumed or fine-tuned like any other
    on-disk model directory in this project.

    Parameters
    ----------
    pretrained_model_name:
        Official StarDist model name, e.g. ``2D_versatile_fluo``.
    project_root:
        Repository root. The target models directory will be resolved relative to this path.
    local_model_name:
        Optional destination folder name. Defaults to ``<pretrained_model_name>_local``.
    models_dir:
        Target models root relative to ``project_root`` or an absolute path.
    overwrite:
        If ``True``, replace any existing local model folder with the imported copy.
    """
    project_root = Path(project_root).resolve()
    models_root = Path(models_dir)
    if not models_root.is_absolute():
        models_root = (project_root / models_root).resolve()
    models_root.mkdir(parents=True, exist_ok=True)

    downloaded_model = StarDist2D.from_pretrained(pretrained_model_name)
    source_dir = Path(downloaded_model.logdir).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Downloaded pretrained model directory does not exist: {source_dir}"
        )

    target_name = _safe_model_name(local_model_name or f"{pretrained_model_name}_local")
    target_dir = models_root / target_name

    if target_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target local model directory already exists: {target_dir}. "
                "Set overwrite=True to replace it."
            )
        shutil.rmtree(target_dir)

    shutil.copytree(source_dir, target_dir)

    metadata_path = target_dir / "pretrained_import_metadata.json"
    metadata = {
        "pretrained_model_name": pretrained_model_name,
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return ImportedPretrainedModel(
        source_name=pretrained_model_name,
        source_dir=source_dir,
        target_dir=target_dir,
    )