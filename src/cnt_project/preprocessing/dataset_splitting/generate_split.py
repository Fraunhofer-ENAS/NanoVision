from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd
import numpy as np

from cnt_project.preprocessing.dataset_splitting.split_manifest import (
    SUBSET_COL,
    validate_split_manifest_df,
    write_split_manifest_csv,
    write_split_metadata_yaml,
)
from cnt_project.preprocessing.dataset_splitting.stratification import (
    StratificationConfig,
    build_combined_labels,
    stratified_assign_subsets,
)
from cnt_project.preprocessing.dataset_splitting.validation import (
    SUPPORTED_SUBSETS,
    validate_dataset_root,
    validate_hierarchical_split_fractions,
)

from cnt_project.preprocessing.metadata.schemas import (
    ALLOWED_DENSITY_CLASSES,
    ALLOWED_NOISE_CLASSES,
    FILENAME_COL,
    GT_OBJECT_COUNT_COL,
    NOISE_SIGMA_COL,
)
from cnt_project.preprocessing.metadata.validation import (
    DatasetValidationError,
    ensure_metadata_coverage,
)


@dataclass(frozen=True)
class SplitPaths:
    manifest_csv: Path
    metadata_yaml: Path


@dataclass(frozen=True)
class SplitGenerationResult:
    split_df: pd.DataFrame
    paths: SplitPaths

@dataclass(frozen=True)
class LegacyManualPools:
    train_pool_filenames: list[str]
    test_filenames: list[str]

_SPLIT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _validate_split_name(split_name: str) -> None:
    if not split_name:
        raise DatasetValidationError("split_name must not be empty.")
    if not _SPLIT_NAME_RE.match(split_name):
        raise DatasetValidationError(
            "Invalid split_name. Use letters, numbers, underscore, or hyphen, "
            "and start with a letter or number."
        )


def _load_metadata(
    path: Path,
    *,
    filename_col: str,
    class_col: str,
    allowed_labels: tuple[str, ...],
    metadata_name: str,
) -> pd.DataFrame:
    if not path.exists():
        raise DatasetValidationError(f"Missing {metadata_name} file: {path}")

    df = pd.read_csv(path)

    required = {filename_col, class_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise DatasetValidationError(
            f"{metadata_name} is missing required columns: {missing}. "
            f"Expected at least: [{filename_col}, {class_col}]"
        )

    if df[filename_col].duplicated().any():
        dups = df.loc[df[filename_col].duplicated(), filename_col].astype(str).tolist()
        raise DatasetValidationError(f"Duplicate filename rows in {metadata_name}: {dups[:10]}")

    labels = set(df[class_col].astype(str).unique().tolist())
    invalid = sorted(labels - set(allowed_labels))
    if invalid:
        raise DatasetValidationError(
            f"Unsupported labels in {metadata_name} column '{class_col}': {invalid}. "
            f"Allowed: {list(allowed_labels)}"
        )

    return df.copy()


def _distribution(df: pd.DataFrame, col: str) -> dict[str, int]:
    counts = df[col].astype(str).value_counts(dropna=False).sort_index()
    return {str(k): int(v) for k, v in counts.to_dict().items()}

def _split_counts(df: pd.DataFrame) -> dict[str, int]:
    """Return total/train/val/test sample counts for an assigned split."""
    subset_counts = df[SUBSET_COL].value_counts().to_dict()

    return {
        "total": int(len(df)),
        "train": int(subset_counts.get("train", 0)),
        "val": int(subset_counts.get("val", 0)),
        "test": int(subset_counts.get("test", 0)),
    }

def _fractions_of_complete_dataset(
    counts: dict[str, int],
) -> dict[str, float]:
    """Convert final subset counts into fractions of the complete dataset."""
    total = int(counts["total"])

    if total == 0:
        return {
            "train": 0.0,
            "val": 0.0,
            "test": 0.0,
        }

    return {
        "train": float(counts["train"] / total),
        "val": float(counts["val"] / total),
        "test": float(counts["test"] / total),
    }

def _build_manifest_projection(
    df: pd.DataFrame,
    *,
    density_column: str,
    noise_column: str,
) -> pd.DataFrame:
    required_cols = [
        FILENAME_COL,
        SUBSET_COL,
        GT_OBJECT_COUNT_COL,
        density_column,
        NOISE_SIGMA_COL,
        noise_column,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise DatasetValidationError(
            "Split manifest projection requires missing columns: "
            f"{missing}. Ensure density/noise metadata include GT count, selected class columns, and noise sigma."
        )

    out = df[required_cols].copy()
    out = out.rename(
        columns={
            GT_OBJECT_COUNT_COL: "gt_object_count",
            density_column: "density_class",
            NOISE_SIGMA_COL: "noise_sigma",
            noise_column: "noise_class",
        }
    )
    return out

def _finalize_split_generation(
    assigned: pd.DataFrame,
    *,
    dataset_root: Path,
    split_name: str,
    sample_filenames: list[str],
    density_column: str,
    noise_column: str,
    yaml_payload: dict,
    overwrite: bool,
) -> SplitGenerationResult:
    """
    Build, validate, and write the final split artifacts.

    `assigned` must contain:
      - filename
      - subset
      - gt_object_count
      - selected density class column
      - noise_sigma
      - selected noise class column

    The function:
      1. removes temporary internal columns;
      2. builds the canonical manifest projection;
      3. validates complete dataset coverage;
      4. writes the CSV and YAML;
      5. returns the generated paths and DataFrame.
    """
    split_df = (
        assigned
        .drop(columns=["_strat_label"], errors="ignore")
        .sort_values(FILENAME_COL)
        .reset_index(drop=True)
    )

    split_df = _build_manifest_projection(
        split_df,
        density_column=density_column,
        noise_column=noise_column,
    )

    validate_split_manifest_df(
        split_df,
        expected_filenames=sample_filenames,
    )

    splits_root = dataset_root / "splits"
    manifest_csv = splits_root / f"{split_name}.csv"
    metadata_yaml = splits_root / f"{split_name}.yaml"

    write_split_manifest_csv(
        split_df,
        manifest_csv,
        overwrite=overwrite,
    )

    write_split_metadata_yaml(
        yaml_payload,
        metadata_yaml,
        overwrite=overwrite,
    )

    return SplitGenerationResult(
        split_df=split_df,
        paths=SplitPaths(
            manifest_csv=manifest_csv,
            metadata_yaml=metadata_yaml,
        ),
    )

def _load_legacy_manual_pools(
    manual_split_root: Path,
    expected_filenames: list[str],
) -> LegacyManualPools:
    """
    Load the original manually separated dataset pools.

    Expected layout:
      - Original training pool:
        <manual_split_root>/images/*.tif

      - Fixed test set:
        <manual_split_root>/test/images/*.tif

    The original training pool will later be divided into a new seeded
    train/validation split.

    An existing legacy val/images directory is rejected because this mode
    defines validation from the original training pool.
    """
    def _scan_tif_filenames(folder: Path) -> set[str]:
        if not folder.exists() or not folder.is_dir():
            return set()

        return {
            path.name
            for path in folder.glob("*.tif")
            if path.is_file()
        }

    train_pool = _scan_tif_filenames(
        manual_split_root / "images"
    )

    legacy_val = _scan_tif_filenames(
        manual_split_root / "val" / "images"
    )

    fixed_test = _scan_tif_filenames(
        manual_split_root / "test" / "images"
    )

    if not train_pool:
        raise DatasetValidationError(
            "Legacy manual split root contains no training images at: "
            f"{manual_split_root / 'images'}"
        )

    if not fixed_test:
        raise DatasetValidationError(
            "Legacy manual split root contains no test images at: "
            f"{manual_split_root / 'test' / 'images'}"
        )

    if legacy_val:
        raise DatasetValidationError(
            "Legacy validation files were found under "
            f"{manual_split_root / 'val' / 'images'}, but this mode "
            "expects validation to be generated from the original "
            "training pool."
        )

    overlap = sorted(train_pool & fixed_test)

    if overlap:
        raise DatasetValidationError(
            "Legacy training and test pools overlap. "
            f"Examples: {overlap[:10]}"
        )

    assigned_filenames = train_pool | fixed_test
    expected_set = set(expected_filenames)

    missing = sorted(expected_set - assigned_filenames)
    extra = sorted(assigned_filenames - expected_set)

    if missing or extra:
        details: list[str] = []

        if missing:
            details.append(
                f"missing from legacy train/test pools: {missing[:10]}"
            )

        if extra:
            details.append(
                f"not present in unified dataset: {extra[:10]}"
            )

        raise DatasetValidationError(
            "Legacy manual split coverage mismatch, "
            + "; ".join(details)
        )

    return LegacyManualPools(
        train_pool_filenames=sorted(train_pool),
        test_filenames=sorted(fixed_test),
    )

def _generate_hierarchical_stratified_assignments(
    metadata_df: pd.DataFrame,
    *,
    density_column: str,
    noise_column: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    allow_stratification_fallback: bool,
) -> pd.DataFrame:
    """
    Generate a two-stage stratified split.

    Stage 1:
      Complete dataset -> development + test

    Stage 2:
      Development pool -> train + val

    `test_fraction` is relative to the complete dataset.
    `train_fraction` and `val_fraction` are relative to the development pool.
    """
    (
        development_test_fractions,
        train_val_fractions,
    ) = validate_hierarchical_split_fractions(
        train=train_fraction,
        val=val_fraction,
        test=test_fraction,
    )

    strat_col = "_strat_label"

    full_df = metadata_df.copy()
    full_df[strat_col] = build_combined_labels(
        full_df,
        density_column,
        noise_column,
    )

    # Stage 1: complete dataset -> development + test.
    development_test_assigned = stratified_assign_subsets(
        full_df,
        filename_col=FILENAME_COL,
        strat_label_col=strat_col,
        config=StratificationConfig(
            fractions=development_test_fractions,
            seed=int(seed),
            allow_fallback=bool(allow_stratification_fallback),
            subsets=("development", "test"),
        ),
    )

    development_df = development_test_assigned[
        development_test_assigned[SUBSET_COL] == "development"
    ].copy()

    test_df = development_test_assigned[
        development_test_assigned[SUBSET_COL] == "test"
    ].copy()

    if development_df.empty:
        raise DatasetValidationError(
            "Hierarchical split produced an empty development pool."
        )

    if test_df.empty:
        raise DatasetValidationError(
            "Hierarchical split produced an empty test subset."
        )

    # Remove the temporary stage-1 assignment before stage 2.
    development_df = development_df.drop(columns=[SUBSET_COL])

    # Stage 2: development -> train + val.
    #
    # Use seed + 1 so test selection and train/val selection are
    # independently deterministic.
    train_val_assigned = stratified_assign_subsets(
        development_df,
        filename_col=FILENAME_COL,
        strat_label_col=strat_col,
        config=StratificationConfig(
            fractions=train_val_fractions,
            seed=int(seed) + 1,
            allow_fallback=bool(allow_stratification_fallback),
            subsets=("train", "val"),
        ),
    )

    # Stage 1 already assigned test correctly.
    test_df[SUBSET_COL] = "test"

    assigned = pd.concat(
        [
            train_val_assigned,
            test_df,
        ],
        ignore_index=True,
    )

    if len(assigned) != len(metadata_df):
        raise DatasetValidationError(
            "Hierarchical split assignment count mismatch: "
            f"assigned {len(assigned)} samples for "
            f"{len(metadata_df)} input samples."
        )

    if assigned[FILENAME_COL].duplicated().any():
        duplicates = (
            assigned.loc[
                assigned[FILENAME_COL].duplicated(),
                FILENAME_COL,
            ]
            .astype(str)
            .tolist()
        )

        raise DatasetValidationError(
            "Hierarchical split produced duplicate filename assignments: "
            f"{duplicates[:10]}"
        )

    return assigned


def generate_split_manifest(
    dataset_root: Path,
    *,
    split_name: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    density_metadata_path: Path | None = None,
    noise_metadata_path: Path | None = None,
    density_column: str = "density_class_tertile",
    noise_column: str = "noise_class_otsu",
    allow_stratification_fallback: bool = False,
    overwrite: bool = False,
    reproduce_manual_split: bool = False,
    reproduce_legacy_dataloader_split: bool = False,   
    manual_split_root: Path | None = None,
    legacy_val_fraction: float = 0.15,
) -> SplitGenerationResult:
    dataset_root = dataset_root.resolve()
    _validate_split_name(split_name)

    sample_filenames = validate_dataset_root(dataset_root)


    if reproduce_manual_split and reproduce_legacy_dataloader_split:
        raise DatasetValidationError(
            "Choose only one legacy split mode: "
            "reproduce_manual_split or reproduce_legacy_dataloader_split."
        )

    density_path = (density_metadata_path or (dataset_root / "metadata" / "density_classified_filenames.csv")).resolve()
    noise_path = (noise_metadata_path or (dataset_root / "metadata" / "noise_classification.csv")).resolve()

    density_df = _load_metadata(
        density_path,
        filename_col=FILENAME_COL,
        class_col=density_column,
        allowed_labels=ALLOWED_DENSITY_CLASSES,
        metadata_name="density metadata",
    )
    noise_df = _load_metadata(
        noise_path,
        filename_col=FILENAME_COL,
        class_col=noise_column,
        allowed_labels=ALLOWED_NOISE_CLASSES,
        metadata_name="noise metadata",
    )

    ensure_metadata_coverage(
        sample_filenames=sample_filenames,
        metadata_filenames=density_df[FILENAME_COL].astype(str).tolist(),
        metadata_name="density metadata",
    )
    ensure_metadata_coverage(
        sample_filenames=sample_filenames,
        metadata_filenames=noise_df[FILENAME_COL].astype(str).tolist(),
        metadata_name="noise metadata",
    )

    base = pd.DataFrame({FILENAME_COL: sample_filenames})
    metadata_merged = base.merge(density_df, on=FILENAME_COL, how="left").merge(noise_df, on=FILENAME_COL, how="left")

    if metadata_merged[density_column].isna().any() or metadata_merged[noise_column].isna().any():
        raise DatasetValidationError("Merged metadata has missing density/noise labels after coverage validation.")

    if reproduce_manual_split:
        manual_root = (manual_split_root or (dataset_root.parents[0] / "annotations_uniques")).resolve()

        legacy_pools = _load_legacy_manual_pools( manual_split_root=manual_root, expected_filenames=sample_filenames,)

        train_pool_set = set(legacy_pools.train_pool_filenames)
        test_set = set(legacy_pools.test_filenames)

        legacy_train_pool_df = metadata_merged[metadata_merged[FILENAME_COL].isin(train_pool_set)].copy()

        if len(legacy_train_pool_df) != len(train_pool_set):
            raise DatasetValidationError(
                "Legacy training pool metadata coverage mismatch: "
                f"expected {len(train_pool_set)} rows, got "
                f"{len(legacy_train_pool_df)}."
            )

        strat_col = "_strat_label"

        legacy_train_pool_df[strat_col] = build_combined_labels(
            legacy_train_pool_df,
            density_column,
            noise_column,
        )

        train_val_fractions = {
            "train": 1.0 - float(legacy_val_fraction),
            "val": float(legacy_val_fraction),
        }

        train_val_assigned = stratified_assign_subsets(
            legacy_train_pool_df,
            filename_col=FILENAME_COL,
            strat_label_col=strat_col,
            config=StratificationConfig(
                fractions=train_val_fractions,
                seed=int(seed),
                allow_fallback=bool(
                    allow_stratification_fallback
                ),
                subsets=("train", "val"),
            ),
        )

        fixed_test_df = metadata_merged[metadata_merged[FILENAME_COL].isin(test_set)].copy()

        if len(fixed_test_df) != len(test_set):
            raise DatasetValidationError(
                "Legacy test metadata coverage mismatch: "
                f"expected {len(test_set)} rows, got "
                f"{len(fixed_test_df)}."
            )

        fixed_test_df[SUBSET_COL] = "test"

        fixed_test_df[strat_col] = build_combined_labels(
            fixed_test_df,
            density_column,
            noise_column,
        )

        assigned = pd.concat(
            [
                train_val_assigned,
                fixed_test_df,
            ],
            ignore_index=True,
        )

        legacy_split_preview = (
            assigned
            .drop(columns=[strat_col], errors="ignore")
            .sort_values(FILENAME_COL)
            .reset_index(drop=True)
        )

        legacy_split_preview = _build_manifest_projection(
            legacy_split_preview,
            density_column=density_column,
            noise_column=noise_column,
        )

        validate_split_manifest_df(
            legacy_split_preview,
            expected_filenames=sample_filenames,
        )

        actual_test_filenames = set(
            legacy_split_preview.loc[
                legacy_split_preview[SUBSET_COL] == "test",
                FILENAME_COL,
            ].astype(str)
        )

        if actual_test_filenames != test_set:
            missing_test = sorted(
                test_set - actual_test_filenames
            )
            unexpected_test = sorted(
                actual_test_filenames - test_set
            )

            raise DatasetValidationError(
                "Generated legacy-fixed split does not preserve the "
                "manual test set exactly. "
                f"Missing: {missing_test[:10]}; "
                f"Unexpected: {unexpected_test[:10]}"
            )

        final_train_val = set(
            legacy_split_preview.loc[
                legacy_split_preview[SUBSET_COL].isin(["train", "val"]),
                FILENAME_COL,
            ].astype(str)
        )

        if final_train_val != train_pool_set:
            raise DatasetValidationError(
                "Final train/val membership does not exactly cover the "
                "legacy training pool."
        )

        legacy_counts = _split_counts(assigned)

        yaml_payload = {
            "split_name": split_name,
            "seed": int(seed),
            "sample_identifier": FILENAME_COL,
            "image_extension": ".tif",
            "mask_pairing": "exact_filename",
            "source": {
                "type": (
                    "legacy_test_fixed_with_"
                    "stratified_train_validation"
                ),
                "manual_split_root": str(manual_root),
            },
            "legacy_partition": {
                "original_train_pool_count": int(
                    len(train_pool_set)
                ),
                "fixed_test_count": int(
                    len(test_set)
                ),
                "test_membership_preserved_exactly": True,
            },
            "train_val_split": {
                "validation_fraction_of_legacy_train_pool": float(
                    legacy_val_fraction
                ),
                "train_fraction_of_legacy_train_pool": float(
                    1.0 - legacy_val_fraction
                ),
                "stratification": {
                    "columns": [
                        density_column,
                        noise_column,
                    ],
                    "combined_labels": True,
                    "allow_fallback": bool(
                        allow_stratification_fallback
                    ),
                },
            },
            "fractions_of_complete_dataset": (
                _fractions_of_complete_dataset(legacy_counts)
            ),
            "sample_counts": legacy_counts,
            "distributions": {
                "legacy_train_pool": {
                    density_column: _distribution(
                        legacy_train_pool_df,
                        density_column,
                    ),
                    noise_column: _distribution(
                        legacy_train_pool_df,
                        noise_column,
                    ),
                    "combined": _distribution(
                        legacy_train_pool_df,
                        strat_col,
                    ),
                },
                "subsets": {
                    subset: {
                        density_column: _distribution(
                            assigned[
                                assigned[SUBSET_COL] == subset
                            ],
                            density_column,
                        ),
                        noise_column: _distribution(
                            assigned[
                                assigned[SUBSET_COL] == subset
                            ],
                            noise_column,
                        ),
                        "combined": _distribution(
                            assigned[
                                assigned[SUBSET_COL] == subset
                            ],
                            strat_col,
                        ),
                    }
                    for subset in SUPPORTED_SUBSETS
                },
            },
        }

        return _finalize_split_generation(
            assigned,
            dataset_root=dataset_root,
            split_name=split_name,
            sample_filenames=sample_filenames,
            density_column=density_column,
            noise_column=noise_column,
            yaml_payload=yaml_payload,
            overwrite=overwrite,
        )


    if reproduce_legacy_dataloader_split:
        manual_root = (
            manual_split_root
            or (dataset_root.parents[0] / "annotations_uniques")
        ).resolve()

        legacy_pools = _load_legacy_manual_pools(
            manual_split_root=manual_root,
            expected_filenames=sample_filenames,
        )

        train_pool_set = set(legacy_pools.train_pool_filenames)
        test_set = set(legacy_pools.test_filenames)

        legacy_train_pool_df = metadata_merged[
            metadata_merged[FILENAME_COL].isin(train_pool_set)
        ].copy()

        if len(legacy_train_pool_df) != len(train_pool_set):
            raise DatasetValidationError(
                "Legacy training pool metadata coverage mismatch: "
                f"expected {len(train_pool_set)} rows, got "
                f"{len(legacy_train_pool_df)}."
            )

        # Reproduce the historical Modified_Stardist DataLoader:
        #
        #   sorted filenames
        #       -> np.random.RandomState(seed).permutation(...)
        #       -> final round(val_fraction * N) samples = validation
        #
        # No density/noise stratification is applied.
        train_val_assigned = _assign_legacy_dataloader_train_val(
            legacy_train_pool_df,
            filename_col=FILENAME_COL,
            val_fraction=legacy_val_fraction,
            seed=seed,
        )

        fixed_test_df = metadata_merged[
            metadata_merged[FILENAME_COL].isin(test_set)
        ].copy()

        if len(fixed_test_df) != len(test_set):
            raise DatasetValidationError(
                "Legacy test metadata coverage mismatch: "
                f"expected {len(test_set)} rows, got "
                f"{len(fixed_test_df)}."
            )

        fixed_test_df[SUBSET_COL] = "test"

        assigned = pd.concat(
            [
                train_val_assigned,
                fixed_test_df,
            ],
            ignore_index=True,
        )

        # Verify that test membership was preserved exactly.
        actual_test = set(
            assigned.loc[
                assigned[SUBSET_COL] == "test",
                FILENAME_COL,
            ].astype(str)
        )

        if actual_test != test_set:
            raise DatasetValidationError(
                "Historical DataLoader reproduction did not preserve "
                "the legacy test set exactly."
            )

        # Verify that train + val still exactly cover the original
        # historical development pool.
        actual_train_val = set(
            assigned.loc[
                assigned[SUBSET_COL].isin(["train", "val"]),
                FILENAME_COL,
            ].astype(str)
        )

        if actual_train_val != train_pool_set:
            raise DatasetValidationError(
                "Historical DataLoader reproduction does not exactly "
                "cover the legacy training pool."
            )

        legacy_counts = _split_counts(assigned)

        yaml_payload = {
            "split_name": split_name,
            "seed": int(seed),
            "sample_identifier": FILENAME_COL,
            "image_extension": ".tif",
            "mask_pairing": "exact_filename",

            "source": {
                "type": "legacy_dataloader_exact_reproduction",
                "manual_split_root": str(manual_root),
            },

            "legacy_partition": {
                "original_train_pool_count": int(len(train_pool_set)),
                "fixed_test_count": int(len(test_set)),
                "test_membership_preserved_exactly": True,
            },

            "train_val_split": {
                "algorithm": "legacy_modified_stardist_dataloader",
                "validation_fraction_of_legacy_train_pool": float(
                    legacy_val_fraction
                ),
                "train_fraction_of_legacy_train_pool": float(
                    1.0 - legacy_val_fraction
                ),
                "seed": int(seed),
                "rng": "numpy.random.RandomState",
                "input_order": "filename_sorted",
                "assignment": (
                    "permutation[:-n_val] -> train; "
                    "permutation[-n_val:] -> val"
                ),
                "stratified": False,
            },

            "fractions_of_complete_dataset": (
                _fractions_of_complete_dataset(legacy_counts)
            ),

            "sample_counts": legacy_counts,
        }

        return _finalize_split_generation(
            assigned,
            dataset_root=dataset_root,
            split_name=split_name,
            sample_filenames=sample_filenames,
            density_column=density_column,
            noise_column=noise_column,
            yaml_payload=yaml_payload,
            overwrite=overwrite,
        )
    # begining of the normal branch
    merged = metadata_merged.copy()

    assigned = _generate_hierarchical_stratified_assignments(
        merged,
        density_column=density_column,
        noise_column=noise_column,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        allow_stratification_fallback=allow_stratification_fallback,
    )

    strat_col = "_strat_label"
    merged[strat_col] = build_combined_labels(merged, density_column, noise_column)

    assigned[strat_col] = build_combined_labels(
        assigned,
        density_column,
        noise_column,
    )   

    normal_counts = _split_counts(assigned)

    yaml_payload = {
        "split_name": split_name,
        "seed": int(seed),
        "sample_identifier": FILENAME_COL,
        "image_extension": ".tif",
        "mask_pairing": "exact_filename",

        "source": {
            "type": "hierarchical_stratified_split",
        },

        "split_strategy": {
            "stage_1": {
                "description": (
                    "Split the complete dataset into a development pool "
                    "and a test subset."
                ),
                "development_fraction_of_complete_dataset": float(
                    1.0 - test_fraction
                ),
                "test_fraction_of_complete_dataset": float(
                    test_fraction
                ),
                "seed": int(seed),
            },

            "stage_2": {
                "description": (
                    "Split the development pool into train and validation subsets."
                ),
                "train_fraction_of_development_pool": float(
                    train_fraction
                ),
                "val_fraction_of_development_pool": float(
                    val_fraction
                ),
                "seed": int(seed) + 1,
            },

            "stratification": {
                "columns": [
                    density_column,
                    noise_column,
                ],
                "combined_labels": True,
                "allow_fallback": bool(
                    allow_stratification_fallback
                ),
            },
        },

        "fractions_of_complete_dataset": (
            _fractions_of_complete_dataset(normal_counts)
        ),

        "sample_counts": normal_counts,

        "distributions": {
            "full": {
                density_column: _distribution(
                    merged,
                    density_column,
                ),
                noise_column: _distribution(
                    merged,
                    noise_column,
                ),
                "combined": _distribution(
                    merged,
                    strat_col,
                ),
            },

            "subsets": {
                subset: {
                    density_column: _distribution(
                        assigned[
                            assigned[SUBSET_COL] == subset
                        ],
                        density_column,
                    ),
                    noise_column: _distribution(
                        assigned[
                            assigned[SUBSET_COL] == subset
                        ],
                        noise_column,
                    ),
                    "combined": _distribution(
                        assigned[
                            assigned[SUBSET_COL] == subset
                        ],
                        strat_col,
                    ),
                }
                for subset in SUPPORTED_SUBSETS
            },
        },
    }

    return _finalize_split_generation(
        assigned,
        dataset_root=dataset_root,
        split_name=split_name,
        sample_filenames=sample_filenames,
        density_column=density_column,
        noise_column=noise_column,
        yaml_payload=yaml_payload,
        overwrite=overwrite,
    )



def _assign_legacy_dataloader_train_val(
    train_pool_df: pd.DataFrame,
    *,
    filename_col: str,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Reproduce the historical Modified_Stardist DataLoader split exactly.

    Historical behavior:
      1. filenames were sorted before DataSample creation;
      2. np.random.RandomState(seed) generated one permutation over the
         complete development pool;
      3. n_val = max(1, round(val_fraction * N));
      4. the last n_val permuted samples became validation;
      5. all preceding samples became training.

    No density/noise stratification is applied.
    """
    if not 0.0 < float(val_fraction) < 1.0:
        raise DatasetValidationError(
            f"val_fraction must be between 0 and 1, got {val_fraction}"
        )

    ordered = (
        train_pool_df
        .sort_values(filename_col)
        .reset_index(drop=True)
        .copy()
    )

    rng = np.random.RandomState(int(seed))
    indices = rng.permutation(len(ordered))

    n_val = max(
        1,
        int(round(float(val_fraction) * len(indices))),
    )

    train_indices = indices[:-n_val]
    val_indices = indices[-n_val:]

    train_df = ordered.iloc[train_indices].copy()
    train_df[SUBSET_COL] = "train"

    val_df = ordered.iloc[val_indices].copy()
    val_df[SUBSET_COL] = "val"

    return pd.concat(
        [train_df, val_df],
        ignore_index=True,
    )
