from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal

from cnt_project.coco.convert import (
    convert_tif_annotations_to_json,
    convert_tif_annotations_to_rle_json,
)
from cnt_project.coco.validation import (
    CocoValidationError,
    validate_coco_dataset_structure,
    validate_coco_json_dict,
    validate_split_manifest_for_coco,
)
from cnt_project.preprocessing.dataset_splitting.split_manifest import SUBSET_COL, read_split_manifest_csv
from cnt_project.preprocessing.dataset_splitting.validation import SUPPORTED_SUBSETS
from cnt_project.preprocessing.metadata.schemas import FILENAME_COL


@dataclass(frozen=True)
class CocoGenerationResult:
    subset: str
    output_json_paths: dict[str, Path]
    filenames: list[str]


def _default_output_json_path(dataset_root: Path, split_manifest_path: Path, subset: str, filename: str) -> Path:
    split_name = split_manifest_path.stem
    return dataset_root / "COCO_mask" / split_name / subset / filename


def _validate_subset_value(subset: str) -> None:
    if subset not in SUPPORTED_SUBSETS:
        raise CocoValidationError(
            f"Unsupported subset '{subset}'. Supported subsets: {list(SUPPORTED_SUBSETS)}"
        )


def _validate_requested_subsets(subsets: tuple[str, ...]) -> None:
    invalid = sorted({subset for subset in subsets if subset not in SUPPORTED_SUBSETS})
    if invalid:
        raise CocoValidationError(
            f"Unsupported subset values requested: {invalid}. Supported subsets: {list(SUPPORTED_SUBSETS)}"
        )


def _preflight_output_paths(variants: list[tuple[str, str, Path]], *, overwrite: bool) -> None:
    duplicate_paths = sorted(
        str(path)
        for path in {variant_path for _, _, variant_path in variants}
        if sum(1 for _, _, candidate in variants if candidate == path) > 1
    )
    if duplicate_paths:
        raise CocoValidationError(f"Duplicate output paths detected for COCO variants: {duplicate_paths}")

    for _, _, output_path in variants:
        if output_path.exists() and not overwrite:
            raise CocoValidationError(
                f"Output COCO JSON already exists: {output_path}. Use overwrite=True to replace it."
            )


def _validate_coco_filenames_match_subset(
    coco: dict[str, Any],
    *,
    expected_filenames: list[str],
    subset: str,
    output_path: Path,
) -> None:
    image_entries = coco.get("images", [])
    actual_filenames = [str(entry.get("file_name", "")) for entry in image_entries]

    duplicate_actual = sorted(
        {name for name in actual_filenames if actual_filenames.count(name) > 1 and name}
    )
    if duplicate_actual:
        raise CocoValidationError(
            f"Generated COCO for subset '{subset}' has duplicate image filenames: {duplicate_actual[:10]}"
        )

    expected_set = set(expected_filenames)
    actual_set = set(actual_filenames)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing: {missing[:10]}")
        if extra:
            details.append(f"unexpected: {extra[:10]}")
        raise CocoValidationError(
            f"Generated COCO image filenames do not match manifest subset '{subset}' at {output_path}, "
            + "; ".join(details)
        )


def generate_coco_for_subset(
    dataset_root: str | Path,
    split_manifest_path: str | Path,
    subset: str,
    *,
    output_json_path: str | Path | None = None,
    generate_variants: bool = True,
    overwrite: bool = False,
) -> CocoGenerationResult:
    dataset_root = Path(dataset_root).resolve()
    split_manifest_path = Path(split_manifest_path).resolve()

    _validate_subset_value(subset)

    dataset_filenames = validate_coco_dataset_structure(
        dataset_root
    )

    manifest_df = read_split_manifest_csv(
        split_manifest_path
    )

    validate_split_manifest_for_coco(
        manifest_df,
        dataset_filenames=dataset_filenames,
    )

    subset_df = manifest_df[
        manifest_df[SUBSET_COL].astype(str) == subset
    ].copy()

    if subset_df.empty:
        raise CocoValidationError(
            f"Requested subset '{subset}' is empty in manifest: "
            f"{split_manifest_path}"
        )

    filenames = (
        subset_df[FILENAME_COL]
        .astype(str)
        .tolist()
    )

    output_paths: dict[str, Path] = {}

    if output_json_path is not None and generate_variants:
        raise CocoValidationError(
            "output_json_path cannot be combined with "
            "generate_variants=True because multiple files "
            "are produced."
        )

    # --------------------------------------------------------------
    # Define polygon COCO outputs
    # --------------------------------------------------------------

    polygon_variants = (
        [
            (
                "annotations",
                "simple",
                Path(output_json_path).resolve(),
            )
        ]
        if output_json_path is not None
        else (
            [
                (
                    "annotations_chain_approx_simple",
                    "simple",
                    _default_output_json_path(
                        dataset_root,
                        split_manifest_path,
                        subset,
                        "annotations_chain_approx_simple.json",
                    ),
                ),
                (
                    "annotations_chain_approx_none",
                    "none",
                    _default_output_json_path(
                        dataset_root,
                        split_manifest_path,
                        subset,
                        "annotations_chain_approx_none.json",
                    ),
                ),
            ]
            if generate_variants
            else [
                (
                    "annotations",
                    "simple",
                    _default_output_json_path(
                        dataset_root,
                        split_manifest_path,
                        subset,
                        "annotations.json",
                    ),
                ),
            ]
        )
    )

    # A custom output_json_path means that the caller explicitly
    # requested one polygon JSON. RLE is generated only for the
    # canonical dataset output layout.
    generate_rle = output_json_path is None

    rle_output_path = _default_output_json_path(
        dataset_root,
        split_manifest_path,
        subset,
        "annotations_rle.json",
    )

    # --------------------------------------------------------------
    # Validate output paths before writing
    # --------------------------------------------------------------

    _preflight_output_paths(
        polygon_variants,
        overwrite=overwrite,
    )

    if (
        generate_rle
        and rle_output_path.exists()
        and not overwrite
    ):
        raise CocoValidationError(
            f"Output COCO JSON already exists: "
            f"{rle_output_path}. "
            "Use overwrite=True to replace it."
        )

    # --------------------------------------------------------------
    # Generate polygon COCO annotations
    # --------------------------------------------------------------

    for (
        variant_name,
        contour_approximation,
        output_path,
    ) in polygon_variants:

        coco = convert_tif_annotations_to_json(
            dataset_root / "masks",
            output_path,
            selected_filenames=filenames,
            preserve_tif_filenames=True,
            contour_approximation=contour_approximation,
        )

        validate_coco_json_dict(coco)

        _validate_coco_filenames_match_subset(
            coco,
            expected_filenames=filenames,
            subset=subset,
            output_path=output_path,
        )

        output_paths[variant_name] = output_path

    # --------------------------------------------------------------
    # Generate exact-mask COCO RLE annotations
    # --------------------------------------------------------------

    if generate_rle:
        rle_coco = convert_tif_annotations_to_rle_json(
            dataset_root / "masks",
            rle_output_path,
            selected_filenames=filenames,
            preserve_tif_filenames=True,
            category_id=1,
        )

        validate_coco_json_dict(rle_coco)

        _validate_coco_filenames_match_subset(
            rle_coco,
            expected_filenames=filenames,
            subset=subset,
            output_path=rle_output_path,
        )

        output_paths["annotations_rle"] = (
            rle_output_path
        )

    return CocoGenerationResult(
        subset=subset,
        output_json_paths=output_paths,
        filenames=filenames,
    )


def generate_coco_for_split(
    dataset_root: str | Path,
    split_manifest_path: str | Path,
    *,
    subsets: tuple[str, ...] = ("train", "val", "test"),
    missing_subset_policy: Literal["skip", "reject"] = "reject",
    generate_variants: bool = True,
    overwrite: bool = False,
) -> dict[str, dict[str, Path]]:
    dataset_root = Path(dataset_root).resolve()
    split_manifest_path = Path(split_manifest_path).resolve()

    _validate_requested_subsets(subsets)

    dataset_filenames = validate_coco_dataset_structure(dataset_root)

    manifest_df = read_split_manifest_csv(split_manifest_path)
    validate_split_manifest_for_coco(manifest_df, dataset_filenames=dataset_filenames)

    available_subsets = set(manifest_df[SUBSET_COL].astype(str).unique().tolist())
    expected_subsets = set(subsets)
    missing_subsets = sorted(expected_subsets - available_subsets)
    if missing_subsets and missing_subset_policy == "reject":
        raise CocoValidationError(
            f"Split manifest is missing requested subsets: {missing_subsets}. "
            "Use missing_subset_policy='skip' to ignore them."
        )

    outputs: dict[str, dict[str, Path]] = {}
    for subset in subsets:
        if subset not in available_subsets:
            continue
        result = generate_coco_for_subset(
            dataset_root=dataset_root,
            split_manifest_path=split_manifest_path,
            subset=subset,
            generate_variants=generate_variants,
            overwrite=overwrite,
        )
        outputs[subset] = result.output_json_paths

    return outputs
