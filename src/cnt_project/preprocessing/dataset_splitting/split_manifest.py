from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import yaml

from cnt_project.preprocessing.dataset_splitting.validation import SUPPORTED_SUBSETS
from cnt_project.preprocessing.metadata.schemas import FILENAME_COL
from cnt_project.preprocessing.metadata.validation import DatasetValidationError

SUBSET_COL = "subset"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    _ensure_parent(path)
    with NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding=encoding, newline="") as tf:
        tf.write(text)
        temp_path = Path(tf.name)
    temp_path.replace(path)


def write_split_manifest_csv(df: pd.DataFrame, output_csv: Path, *, overwrite: bool = False) -> None:
    if output_csv.exists() and not overwrite:
        raise DatasetValidationError(
            f"Split manifest already exists: {output_csv}. Use overwrite=True to replace it."
        )

    csv_text = df.to_csv(index=False)
    _atomic_write_text(output_csv, csv_text, encoding="utf-8")


def write_split_metadata_yaml(payload: dict, output_yaml: Path, *, overwrite: bool = False) -> None:
    if output_yaml.exists() and not overwrite:
        raise DatasetValidationError(
            f"Split metadata YAML already exists: {output_yaml}. Use overwrite=True to replace it."
        )

    yaml_text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    _atomic_write_text(output_yaml, yaml_text, encoding="utf-8")


def _validate_split_manifest_structure(df: pd.DataFrame) -> None:
    required = {FILENAME_COL, SUBSET_COL}
    missing = sorted(required - set(df.columns))
    if missing:
        raise DatasetValidationError(f"Split manifest is missing required columns: {missing}")

    if df[FILENAME_COL].duplicated().any():
        dups = df.loc[df[FILENAME_COL].duplicated(), FILENAME_COL].astype(str).tolist()
        raise DatasetValidationError(f"Split manifest has duplicate filenames: {dups[:10]}")

    labels = set(df[SUBSET_COL].astype(str).unique().tolist())
    allowed = set(SUPPORTED_SUBSETS)
    if not labels.issubset(allowed):
        invalid = sorted(labels - allowed)
        raise DatasetValidationError(f"Split manifest has unsupported subset labels: {invalid}")


def validate_split_manifest_df(df: pd.DataFrame, expected_filenames: list[str]) -> None:
    _validate_split_manifest_structure(df)

    got = set(df[FILENAME_COL].astype(str).tolist())
    expected = set(expected_filenames)

    missing_rows = sorted(expected - got)
    extra_rows = sorted(got - expected)

    if missing_rows or extra_rows:
        parts: list[str] = []
        if missing_rows:
            parts.append(f"missing assignments: {missing_rows[:10]}")
        if extra_rows:
            parts.append(f"unexpected assignments: {extra_rows[:10]}")
        raise DatasetValidationError("Split manifest coverage mismatch, " + "; ".join(parts))


def read_split_manifest_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_split_manifest_structure(df)
    return df
