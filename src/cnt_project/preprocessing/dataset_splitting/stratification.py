from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cnt_project.preprocessing.metadata.validation import DatasetValidationError


@dataclass(frozen=True)
class StratificationConfig:
    fractions: dict[str, float]
    seed: int
    allow_fallback: bool = False
    subsets: tuple[str, ...] = ("train", "val", "test")


def _apportion_counts(
    total: int,
    fractions: dict[str, float],
    *,
    subsets: tuple[str, ...],
) -> dict[str, int]:
    """
    Convert fractional subset targets into exact integer counts.

    The returned counts always sum exactly to `total`.
    """
    if not subsets:
        raise DatasetValidationError("At least one subset is required.")

    missing_fraction_keys = sorted(set(subsets) - set(fractions))
    if missing_fraction_keys:
        raise DatasetValidationError(
            f"Missing fractions for subsets: {missing_fraction_keys}"
        )

    total_fraction = sum(float(fractions[s]) for s in subsets)
    if abs(total_fraction - 1.0) > 1e-9:
        raise DatasetValidationError(
            f"Subset fractions must sum to 1.0, got {total_fraction:.12f}"
        )

    for subset in subsets:
        value = float(fractions[subset])
        if value <= 0.0:
            raise DatasetValidationError(
                f"Fraction for subset '{subset}' must be > 0, got {value}"
            )

    quotas = {
        subset: total * float(fractions[subset])
        for subset in subsets
    }

    counts = {
        subset: int(np.floor(quotas[subset]))
        for subset in subsets
    }

    remainder = int(total - sum(counts.values()))

    ranking = sorted(
        subsets,
        key=lambda subset: (
            quotas[subset] - counts[subset],
            quotas[subset],
        ),
        reverse=True,
    )

    for subset in ranking[:remainder]:
        counts[subset] += 1

    if sum(counts.values()) != total:
        raise DatasetValidationError(
            "Internal stratification error: subset counts do not sum "
            "to the requested total."
        )

    return counts


def _allocate_group_counts_exact(
    group_sizes: dict[str, int],
    fractions: dict[str, float],
    *,
    subsets: tuple[str, ...],
    require_all_subsets_per_group: bool,
) -> dict[str, dict[str, int]]:
    """
    Allocate every stratification group across the requested subsets.

    Guarantees:
    1. Every group is fully allocated.
    2. Global subset counts match the requested fractions exactly.
    3. In strict mode, each group appears at least once in every subset.
    """
    if not subsets:
        raise DatasetValidationError("At least one subset is required.")

    total_n = int(sum(group_sizes.values()))

    global_target = _apportion_counts(
        total=total_n,
        fractions=fractions,
        subsets=subsets,
    )

    group_counts: dict[str, dict[str, int]] = {
        group_name: {
            subset: 0
            for subset in subsets
        }
        for group_name in group_sizes
    }

    global_remaining = dict(global_target)
    group_remaining: dict[str, int] = {}

    for group_name, group_size in group_sizes.items():
        if require_all_subsets_per_group:
            if group_size < len(subsets):
                raise DatasetValidationError(
                    "Exact stratification is not possible for combined group "
                    f"'{group_name}' (size={group_size}, requires at least "
                    f"{len(subsets)} samples)."
                )

            for subset in subsets:
                group_counts[group_name][subset] += 1
                global_remaining[subset] -= 1

                if global_remaining[subset] < 0:
                    raise DatasetValidationError(
                        "Requested global fractions are incompatible with "
                        "strict per-group representation across all subsets."
                    )

            group_remaining[group_name] = group_size - len(subsets)
        else:
            group_remaining[group_name] = group_size

    quotas: dict[tuple[str, str], float] = {}

    for group_name, remaining_count in group_remaining.items():
        for subset in subsets:
            quotas[(group_name, subset)] = (
                remaining_count * float(fractions[subset])
            )

    for group_name in group_remaining:
        for subset in subsets:
            base = int(np.floor(quotas[(group_name, subset)]))

            if base <= 0:
                continue

            take = min(base, global_remaining[subset])

            group_counts[group_name][subset] += take
            group_remaining[group_name] -= take
            global_remaining[subset] -= take

    if any(value < 0 for value in global_remaining.values()):
        raise DatasetValidationError(
            "Requested global fractions are incompatible with "
            "stratification constraints."
        )

    for group_name in sorted(group_remaining):
        remaining_count = int(group_remaining[group_name])

        if remaining_count <= 0:
            continue

        fractional_priority = sorted(
            subsets,
            key=lambda subset: (
                quotas[(group_name, subset)]
                - np.floor(quotas[(group_name, subset)]),
                quotas[(group_name, subset)],
            ),
            reverse=True,
        )

        for _ in range(remaining_count):
            candidates = [
                subset
                for subset in fractional_priority
                if global_remaining[subset] > 0
            ]

            if not candidates:
                raise DatasetValidationError(
                    "Requested global fractions are incompatible with "
                    f"stratification constraints for group '{group_name}'."
                )

            selected_subset = max(
                candidates,
                key=lambda subset: (
                    global_remaining[subset],
                    quotas[(group_name, subset)]
                    - np.floor(quotas[(group_name, subset)]),
                ),
            )

            group_counts[group_name][selected_subset] += 1
            global_remaining[selected_subset] -= 1

    if any(value != 0 for value in global_remaining.values()):
        raise DatasetValidationError(
            "Internal stratification error: final global subset counts "
            "do not match the requested targets."
        )

    for group_name, group_size in group_sizes.items():
        assigned_count = sum(group_counts[group_name].values())

        if assigned_count != group_size:
            raise DatasetValidationError(
                f"Internal stratification error: group '{group_name}' "
                f"assigned {assigned_count} samples, expected {group_size}."
            )

    return group_counts


def build_combined_labels(df: pd.DataFrame, density_col: str, noise_col: str) -> pd.Series:
    return df[density_col].astype(str) + " + " + df[noise_col].astype(str)


def stratified_assign_subsets(
    df: pd.DataFrame,
    *,
    filename_col: str,
    strat_label_col: str,
    config: StratificationConfig,
) -> pd.DataFrame:
    """
    Assign subset membership using seeded group-wise stratification.

    The supported subset names are supplied through `config.subsets`.

    Examples:
      - Full split: subsets=("train", "val", "test")
      - Legacy train pool split: subsets=("train", "val")
    """
    if df.empty:
        raise DatasetValidationError(
            "No samples available for split assignment."
        )

    subsets = tuple(config.subsets)

    if not subsets:
        raise DatasetValidationError(
            "StratificationConfig.subsets must not be empty."
        )

    if len(set(subsets)) != len(subsets):
        raise DatasetValidationError(
            f"Duplicate subset names are not allowed: {subsets}"
        )

    missing_fraction_keys = sorted(
        set(subsets) - set(config.fractions)
    )
    if missing_fraction_keys:
        raise DatasetValidationError(
            f"Missing fractions for subsets: {missing_fraction_keys}"
        )

    rng = np.random.default_rng(config.seed)

    group_sizes = {
        str(label): int(len(group))
        for label, group in df.groupby(strat_label_col, sort=True)
    }

    counts_by_group = _allocate_group_counts_exact(
        group_sizes=group_sizes,
        fractions=config.fractions,
        subsets=subsets,
        require_all_subsets_per_group=not config.allow_fallback,
    )

    assigned_parts: list[pd.DataFrame] = []

    for label, group in df.groupby(strat_label_col, sort=True):
        label_key = str(label)
        counts = counts_by_group[label_key]

        shuffled_indices = rng.permutation(group.index.to_numpy())
        shuffled_group = group.loc[shuffled_indices].reset_index(drop=True)

        start = 0
        subset_parts: list[pd.DataFrame] = []

        for subset in subsets:
            take = int(counts[subset])

            chunk = shuffled_group.iloc[
                start : start + take
            ].copy()

            chunk["subset"] = subset
            subset_parts.append(chunk)

            start += take

        assigned_parts.append(
            pd.concat(subset_parts, ignore_index=True)
        )

    assigned = pd.concat(assigned_parts, ignore_index=True)

    if len(assigned) != len(df):
        raise DatasetValidationError(
            "Internal stratification error: assigned "
            f"{len(assigned)} rows for {len(df)} input samples."
        )

    if assigned[filename_col].duplicated().any():
        duplicates = (
            assigned.loc[
                assigned[filename_col].duplicated(),
                filename_col,
            ]
            .astype(str)
            .tolist()
        )

        raise DatasetValidationError(
            f"Duplicate filename assignment detected: {duplicates[:10]}"
        )

    return assigned