from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import json
import re

import numpy as np
import pandas as pd
import yaml

from cnt_project.coco.convert import (
    convert_tif_annotations_to_json,
    convert_tif_annotations_to_rle_json,
)
from cnt_project.coco.generate import generate_coco_for_subset
from cnt_project.coco.validation import CocoValidationError, validate_coco_json_dict
from cnt_project.preprocessing.dataset_splitting.generate_split import generate_split_manifest
from cnt_project.preprocessing.dataset_splitting.split_manifest import (
	SUBSET_COL,
	read_split_manifest_csv,
	validate_split_manifest_df,
)
from cnt_project.preprocessing.dataset_splitting.validation import SUPPORTED_SUBSETS, validate_dataset_root
from cnt_project.preprocessing.metadata import generate_density_metadata, generate_noise_metadata
from cnt_project.preprocessing.metadata.schemas import (
	ALLOWED_DENSITY_CLASSES,
	ALLOWED_NOISE_CLASSES,
	DENSITY_CLASS_KMEANS_COL,
	DENSITY_CLASS_TERTILE_COL,
	FILENAME_COL,
	GT_OBJECT_COUNT_COL,
	NOISE_CLASS_KMEANS_COL,
	NOISE_CLASS_OTSU_COL,
	NOISE_SIGMA_COL,
)
from cnt_project.preprocessing.metadata.validation import DatasetValidationError, ensure_metadata_coverage


PreparationPolicy = Literal["validate", "regenerate"]

_SPLIT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class PreparationGenerationParameters:
	density_method: Literal["tertile", "kmeans"] = "kmeans"
	density_min_area_px: int = 1
	density_include_comparison_columns: bool = True

	noise_method: Literal["otsu", "kmeans"] = "otsu"
	noise_wavelet: str = "db1"
	noise_expected_shape: tuple[int, int] | None = None
	noise_include_comparison_columns: bool = True

	# fraction of the (dataset - test set) assigned to train
	train_fraction: float = 0.85	
	# what fraction of the (dataset - test set) should be allocated to the val set
	val_fraction: float = 0.15		 
	# fraction of the complete dataset assigned to test: train_fraction + val_fraction must equal 1.0, 
	# 30 is the number of images we want in the test set and 130 is the total number of images we have
	test_fraction: float = 30/130  	
	seed: int = 42
	allow_stratification_fallback: bool = False
	reproduce_manual_split: bool = False
	manual_split_root: Path | None = None
	legacy_val_fraction: float = 0.15
	split_density_column: str = DENSITY_CLASS_TERTILE_COL
	split_noise_column: str = NOISE_CLASS_OTSU_COL

	generate_coco_variants: bool = True
	coco_missing_subset_policy: Literal["skip", "reject"] = "skip"
	coco_subsets: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class PreparedDatasetArtifacts:
	dataset_root: Path
	split_name: str
	policy: PreparationPolicy
	sample_filenames: list[str]
	density_metadata_csv: Path
	noise_metadata_csv: Path
	split_manifest_csv: Path
	split_metadata_yaml: Path
	coco_json_paths: dict[str, dict[str, Path]]


@dataclass(frozen=True)
class _PreparationPaths:
	density_metadata_csv: Path
	noise_metadata_csv: Path
	split_manifest_csv: Path
	split_metadata_yaml: Path


def _validate_split_name(split_name: str) -> None:
	if not split_name:
		raise DatasetValidationError("split_name must not be empty.")
	if not _SPLIT_NAME_RE.match(split_name):
		raise DatasetValidationError(
			"Invalid split_name. Use letters, numbers, underscore, or hyphen, and start with a letter or number."
		)


def _default_density_metadata_path(dataset_root: Path) -> Path:
	return dataset_root / "metadata" / "density_classified_filenames.csv"


def _default_noise_metadata_path(dataset_root: Path) -> Path:
	return dataset_root / "metadata" / "noise_classification.csv"


def _default_split_paths(dataset_root: Path, split_name: str) -> _PreparationPaths:
	splits_root = dataset_root / "splits"
	return _PreparationPaths(
		density_metadata_csv=_default_density_metadata_path(dataset_root),
		noise_metadata_csv=_default_noise_metadata_path(dataset_root),
		split_manifest_csv=splits_root / f"{split_name}.csv",
		split_metadata_yaml=splits_root / f"{split_name}.yaml",
	)


def _read_csv(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise DatasetValidationError(f"Missing artifact: {path}")
	return pd.read_csv(path)


def _validate_metadata_artifact(
	path: Path,
	*,
	sample_filenames: list[str],
	required_columns: set[str],
	class_column: str,
	allowed_labels: tuple[str, ...],
	metadata_name: str,
) -> pd.DataFrame:
	df = _read_csv(path)

	missing_columns = sorted(required_columns - set(df.columns))
	if missing_columns:
		raise DatasetValidationError(f"{metadata_name} is missing required columns: {missing_columns}")

	if df[FILENAME_COL].duplicated().any():
		dups = df.loc[df[FILENAME_COL].duplicated(), FILENAME_COL].astype(str).tolist()
		raise DatasetValidationError(f"Duplicate filename rows in {metadata_name}: {dups[:10]}")

	labels = set(df[class_column].astype(str).unique().tolist())
	invalid = sorted(labels - set(allowed_labels))
	if invalid:
		raise DatasetValidationError(
			f"Unsupported labels in {metadata_name} column '{class_column}': {invalid}. Allowed: {list(allowed_labels)}"
		)

	ensure_metadata_coverage(
		sample_filenames=sample_filenames,
		metadata_filenames=df[FILENAME_COL].astype(str).tolist(),
		metadata_name=metadata_name,
	)

	return df.copy()


def _validate_density_metadata(
	path: Path,
	*,
	sample_filenames: list[str],
	density_column: str,
) -> pd.DataFrame:
	required_columns = {FILENAME_COL, GT_OBJECT_COUNT_COL, density_column}
	return _validate_metadata_artifact(
		path,
		sample_filenames=sample_filenames,
		required_columns=required_columns,
		class_column=density_column,
		allowed_labels=ALLOWED_DENSITY_CLASSES,
		metadata_name="density metadata",
	)


def _validate_noise_metadata(
	path: Path,
	*,
	sample_filenames: list[str],
	noise_column: str,
) -> pd.DataFrame:
	required_columns = {FILENAME_COL, NOISE_SIGMA_COL, noise_column}
	return _validate_metadata_artifact(
		path,
		sample_filenames=sample_filenames,
		required_columns=required_columns,
		class_column=noise_column,
		allowed_labels=ALLOWED_NOISE_CLASSES,
		metadata_name="noise metadata",
	)


def _validate_split_metadata_yaml(
	path: Path,
	*,
	split_name: str,
	split_df: pd.DataFrame,
) -> None:
	if not path.exists():
		raise DatasetValidationError(f"Missing split metadata YAML: {path}")

	with open(path, "r", encoding="utf-8") as handle:
		payload = yaml.safe_load(handle)

	if not isinstance(payload, dict):
		raise DatasetValidationError(f"Split metadata YAML must contain a mapping: {path}")

	if payload.get("split_name") != split_name:
		raise DatasetValidationError(
			f"Split metadata YAML has split_name={payload.get('split_name')!r}, expected {split_name!r}"
		)

	counts = split_df[SUBSET_COL].value_counts().to_dict()
	sample_counts = payload.get("sample_counts")
	if not isinstance(sample_counts, dict):
		raise DatasetValidationError(f"Split metadata YAML missing sample_counts mapping: {path}")

	expected_total = int(len(split_df))
	if int(sample_counts.get("total", -1)) != expected_total:
		raise DatasetValidationError(
			f"Split metadata YAML total does not match manifest rows: {sample_counts.get('total')} != {expected_total}"
		)

	for subset in SUPPORTED_SUBSETS:
		if int(sample_counts.get(subset, -1)) != int(counts.get(subset, 0)):
			raise DatasetValidationError(
				f"Split metadata YAML count mismatch for subset '{subset}': "
				f"{sample_counts.get(subset)} != {counts.get(subset, 0)}"
			)


def _validate_split_manifest_matches_metadata(
	split_df: pd.DataFrame,
	*,
	density_df: pd.DataFrame,
	noise_df: pd.DataFrame,
	density_column: str,
	noise_column: str,
) -> None:
	required_columns = {FILENAME_COL, SUBSET_COL, "gt_object_count", "density_class", "noise_sigma", "noise_class"}
	missing_columns = sorted(required_columns - set(split_df.columns))
	if missing_columns:
		raise DatasetValidationError(f"Split manifest is missing required columns: {missing_columns}")

	density_projection = density_df[[FILENAME_COL, GT_OBJECT_COUNT_COL, density_column]].rename(
		columns={GT_OBJECT_COUNT_COL: "density_gt_object_count", density_column: "expected_density_class"}
	)
	noise_projection = noise_df[[FILENAME_COL, NOISE_SIGMA_COL, noise_column]].rename(
		columns={NOISE_SIGMA_COL: "expected_noise_sigma", noise_column: "expected_noise_class"}
	)

	merged = (
		split_df.merge(density_projection, on=FILENAME_COL, how="left")
		.merge(noise_projection, on=FILENAME_COL, how="left")
	)

	if merged[["density_gt_object_count", "expected_density_class", "expected_noise_sigma", "expected_noise_class"]].isna().any().any():
		raise DatasetValidationError("Split manifest cannot be matched to density/noise metadata by filename.")

	split_density = merged["density_class"].astype(str).tolist()
	expected_density = merged["expected_density_class"].astype(str).tolist()
	if split_density != expected_density:
		raise DatasetValidationError("Split manifest density_class values do not match the current density metadata.")

	split_noise = merged["noise_class"].astype(str).tolist()
	expected_noise = merged["expected_noise_class"].astype(str).tolist()
	if split_noise != expected_noise:
		raise DatasetValidationError("Split manifest noise_class values do not match the current noise metadata.")

	split_counts = pd.to_numeric(merged["gt_object_count"], errors="raise").astype(int)
	expected_counts = pd.to_numeric(merged["density_gt_object_count"], errors="raise").astype(int)
	if not split_counts.equals(expected_counts):
		raise DatasetValidationError("Split manifest gt_object_count values do not match the current density metadata.")

	split_sigma = pd.to_numeric(merged["noise_sigma"], errors="raise").to_numpy(dtype=float)
	expected_sigma = pd.to_numeric(merged["expected_noise_sigma"], errors="raise").to_numpy(dtype=float)
	if not np.allclose(split_sigma, expected_sigma, rtol=1e-9, atol=1e-12):
		raise DatasetValidationError("Split manifest noise_sigma values do not match the current noise metadata.")


def _expected_coco_output_paths(
    dataset_root: Path,
    split_name: str,
    subset: str,
    *,
    generate_variants: bool,
) -> dict[str, Path]:
    base = dataset_root / "COCO_mask" / split_name / subset

    if generate_variants:
        return {
            "annotations_chain_approx_simple": (
                base / "annotations_chain_approx_simple.json"
            ),
            "annotations_chain_approx_none": (
                base / "annotations_chain_approx_none.json"
            ),
            "annotations_rle": (
                base / "annotations_rle.json"
            ),
        }

    return {
        "annotations": base / "annotations.json",
        "annotations_rle": base / "annotations_rle.json",
    }


def _coco_variant_to_contour_approximation(variant_name: str) -> Literal["simple", "none"]:
	if variant_name.endswith("_none"):
		return "none"
	return "simple"


def _validate_existing_coco_json(path: Path, *, expected_filenames: list[str], subset: str) -> None:
	if not path.exists():
		raise DatasetValidationError(f"Missing COCO JSON for subset '{subset}': {path}")

	with open(path, "r", encoding="utf-8") as handle:
		coco = json.load(handle)

	validate_coco_json_dict(coco)

	actual_filenames = [str(image.get("file_name", "")) for image in coco["images"]]
	expected_set = set(expected_filenames)
	actual_set = set(actual_filenames)
	if expected_set != actual_set:
		missing = sorted(expected_set - actual_set)
		extra = sorted(actual_set - expected_set)
		details: list[str] = []
		if missing:
			details.append(f"missing: {missing[:10]}")
		if extra:
			details.append(f"unexpected: {extra[:10]}")
		raise DatasetValidationError(
			f"COCO JSON filenames do not match subset '{subset}' at {path}: " + "; ".join(details)
		)


def _generate_density_metadata(
	dataset_root: Path,
	output_csv: Path,
	generation_parameters: PreparationGenerationParameters,
	*,
	overwrite: bool,
) -> pd.DataFrame:
	if output_csv.exists() and not overwrite:
		raise DatasetValidationError(f"Density metadata already exists: {output_csv}")

	return generate_density_metadata(
		dataset_root=dataset_root,
		output_csv=output_csv,
		method=generation_parameters.density_method,
		random_seed=generation_parameters.seed,
		min_area_px=generation_parameters.density_min_area_px,
		include_comparison_columns=generation_parameters.density_include_comparison_columns,
	)


def _generate_noise_metadata(
	dataset_root: Path,
	output_csv: Path,
	generation_parameters: PreparationGenerationParameters,
	*,
	overwrite: bool,
) -> pd.DataFrame:
	if output_csv.exists() and not overwrite:
		raise DatasetValidationError(f"Noise metadata already exists: {output_csv}")

	return generate_noise_metadata(
		dataset_root=dataset_root,
		output_csv=output_csv,
		method=generation_parameters.noise_method,
		random_seed=generation_parameters.seed,
		wavelet=generation_parameters.noise_wavelet,
		expected_shape=generation_parameters.noise_expected_shape,
		include_comparison_columns=generation_parameters.noise_include_comparison_columns,
	)


def _generate_split_artifact(
	dataset_root: Path,
	split_name: str,
	generation_parameters: PreparationGenerationParameters,
	*,
	density_metadata_path: Path | None = None,
	noise_metadata_path: Path | None = None,
	overwrite: bool,
) -> tuple[pd.DataFrame, Path, Path]:
	result = generate_split_manifest(
		dataset_root=dataset_root,
		split_name=split_name,
		train_fraction=generation_parameters.train_fraction,
		val_fraction=generation_parameters.val_fraction,
		test_fraction=generation_parameters.test_fraction,
		seed=generation_parameters.seed,
		density_metadata_path=density_metadata_path,
		noise_metadata_path=noise_metadata_path,
		density_column=generation_parameters.split_density_column,
		noise_column=generation_parameters.split_noise_column,
		allow_stratification_fallback=generation_parameters.allow_stratification_fallback,
		overwrite=overwrite,
		reproduce_manual_split=generation_parameters.reproduce_manual_split,
		manual_split_root=generation_parameters.manual_split_root,
		legacy_val_fraction=generation_parameters.legacy_val_fraction,
	)
	return result.split_df, result.paths.manifest_csv, result.paths.metadata_yaml


def _generate_or_validate_subset_coco(
	dataset_root: Path,
	split_name: str,
	subset: str,
	subset_filenames: list[str],
	generation_parameters: PreparationGenerationParameters,
	*,
	overwrite: bool,
) -> dict[str, Path]:
	expected_outputs = _expected_coco_output_paths(
		dataset_root,
		split_name,
		subset,
		generate_variants=generation_parameters.generate_coco_variants,
	)

	if overwrite:
		result = generate_coco_for_subset(
			dataset_root=dataset_root,
			split_manifest_path=dataset_root / "splits" / f"{split_name}.csv",
			subset=subset,
			generate_variants=generation_parameters.generate_coco_variants,
			overwrite=True,
		)
		return result.output_json_paths

	missing_paths = {name: path for name, path in expected_outputs.items() if not path.exists()}
	existing_paths = {name: path for name, path in expected_outputs.items() if path.exists()}

	for variant_name, path in existing_paths.items():
		_validate_existing_coco_json(path, expected_filenames=subset_filenames, subset=subset)

	if not missing_paths:
		return expected_outputs

	split_manifest_path = dataset_root / "splits" / f"{split_name}.csv"

	if len(missing_paths) == len(expected_outputs):
		result = generate_coco_for_subset(
			dataset_root=dataset_root,
			split_manifest_path=split_manifest_path,
			subset=subset,
			generate_variants=generation_parameters.generate_coco_variants,
			overwrite=False,
		)
		return result.output_json_paths

	for variant_name, path in missing_paths.items():
		if variant_name == "annotations_rle":
			coco = convert_tif_annotations_to_rle_json(
				dataset_root / "masks",
				path,
				selected_filenames=subset_filenames,
				preserve_tif_filenames=True,
				category_id=1,
			)

		else:
			coco = convert_tif_annotations_to_json(
				dataset_root / "masks",
				path,
				selected_filenames=subset_filenames,
				preserve_tif_filenames=True,
				contour_approximation=(
					_coco_variant_to_contour_approximation(
						variant_name
					)
				),
			)

		validate_coco_json_dict(coco)

		_validate_existing_coco_json(
			path,
			expected_filenames=subset_filenames,
			subset=subset,
		)

	return expected_outputs


def prepare_dataset(
	*,
	dataset_root: str | Path,
	split_name: str,
	policy: PreparationPolicy = "validate",
	generation_parameters: PreparationGenerationParameters | None = None,
) -> PreparedDatasetArtifacts:
	"""
	Orchestrate canonical dataset preparation.

	The orchestration order is:
	  1. Validate images and masks
	  2. Generate or validate density metadata
	  3. Generate or validate noise metadata
	  4. Generate or validate split manifest
	  5. Generate or validate subset COCO JSON files
	  6. Return prepared artifact paths
	"""
	if policy not in {"validate", "regenerate"}:
		raise DatasetValidationError("policy must be either 'validate' or 'regenerate'.")

	params = generation_parameters or PreparationGenerationParameters()
	dataset_root = Path(dataset_root).resolve()
	_validate_split_name(split_name)

	sample_filenames = validate_dataset_root(dataset_root)
	paths = _default_split_paths(dataset_root, split_name)

	density_path = paths.density_metadata_csv
	noise_path = paths.noise_metadata_csv

	if policy == "regenerate" or not density_path.exists():
		density_df = _generate_density_metadata(dataset_root, density_path, params, overwrite=True)
	else:
		density_df = _validate_density_metadata(
			density_path,
			sample_filenames=sample_filenames,
			density_column=params.split_density_column,
		)

	if policy == "regenerate" or not noise_path.exists():
		noise_df = _generate_noise_metadata(dataset_root, noise_path, params, overwrite=True)
	else:
		noise_df = _validate_noise_metadata(
			noise_path,
			sample_filenames=sample_filenames,
			noise_column=params.split_noise_column,
		)

	split_csv = paths.split_manifest_csv
	split_yaml = paths.split_metadata_yaml

	if policy == "validate" and split_csv.exists() != split_yaml.exists():
		raise DatasetValidationError(
			f"Split artifacts are partially present for '{split_name}'. Use policy='regenerate' to recreate them."
		)

	if policy == "regenerate" or not split_csv.exists() or not split_yaml.exists():
		split_df, split_csv, split_yaml = _generate_split_artifact(
			dataset_root,
			split_name,
			params,
			density_metadata_path=density_path,
			noise_metadata_path=noise_path,
			overwrite=(policy == "regenerate"),
		)
	else:
		split_df = read_split_manifest_csv(split_csv)

		validate_split_manifest_df(split_df, expected_filenames=sample_filenames)
		_validate_split_manifest_matches_metadata(
			split_df,
			density_df=density_df,
			noise_df=noise_df,
			density_column=params.split_density_column,
			noise_column=params.split_noise_column,
		)
		_validate_split_metadata_yaml(split_yaml, split_name=split_name, split_df=split_df)

	coco_json_paths: dict[str, dict[str, Path]] = {}
	for subset in params.coco_subsets:
		if subset not in SUPPORTED_SUBSETS:
			raise DatasetValidationError(
				f"Unsupported COCO subset requested in preparation parameters: {subset}. Supported subsets: {list(SUPPORTED_SUBSETS)}"
			)

		subset_rows = split_df.loc[split_df[SUBSET_COL].astype(str) == subset, FILENAME_COL].astype(str).tolist()
		if not subset_rows:
			if params.coco_missing_subset_policy == "skip":
				continue
			raise DatasetValidationError(f"Requested COCO subset '{subset}' is empty in split '{split_name}'.")

		coco_json_paths[subset] = _generate_or_validate_subset_coco(
			dataset_root=dataset_root,
			split_name=split_name,
			subset=subset,
			subset_filenames=subset_rows,
			generation_parameters=params,
			overwrite=(policy == "regenerate"),
		)

	return PreparedDatasetArtifacts(
		dataset_root=dataset_root,
		split_name=split_name,
		policy=policy,
		sample_filenames=sample_filenames,
		density_metadata_csv=density_path,
		noise_metadata_csv=noise_path,
		split_manifest_csv=split_csv,
		split_metadata_yaml=split_yaml,
		coco_json_paths=coco_json_paths,
	)
