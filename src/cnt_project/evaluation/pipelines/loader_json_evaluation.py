from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from cnt_project.coco.masks import coco_polygons_json_to_instance_masks
from cnt_project.evaluation.inputs.evaluation_grouping import (
    load_density_groups,
    load_length_groups,
    normalize_evaluation_filename,
)

from cnt_project.evaluation.pipelines.loader_mask_evaluation import run_pixel_metrics_only_from_loader
from cnt_project.io.paths import ProjectPaths
from cnt_project.io.run_manager import create_run
from cnt_project.preprocessing.dataset.dataset_loader import CNTDataset


class FilteredLoader:
    def __init__(self, samples):
        self._samples = samples

    def get_test_data(self):
        return self._samples


def _filter_samples_by_filenames(samples, filenames: Sequence[str]):
    wanted_keys = {
        normalize_evaluation_filename(name)
        for name in filenames
    }

    return [
        sample
        for sample in samples
        if normalize_evaluation_filename(sample.X_fn) in wanted_keys
    ]


def _pred_tag_from_filename(pred_filename: str) -> str:
    stem = Path(pred_filename).stem

    if stem.startswith("predicted_annotations_"):
        return "pred-" + stem.replace("predicted_annotations_", "")

    return stem


def _write_metadata(
    *,
    out_dir: Path,
    run_name: str,
    pred_json_path: Path,
    pred_filename: str,
    grayscale: bool,
    mode: str,
    metadata_extra: dict | None = None,
) -> None:
    metadata = {
        "run_name": run_name,
        "pred_json_path": str(pred_json_path),
        "pred_filename": pred_filename,
        "prediction_source": "COCO JSON reconstructed to loader instance masks",
        "gt_source": "CNTDataset TIFF masks",
        "grayscale": grayscale,
        "mode": mode,
        "output_dir": str(out_dir),
    }

    if metadata_extra:
        metadata.update(metadata_extra)

    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "loader_from_json_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _run_loader_eval_on_subset(
    *,
    samples,
    out_dir: Path,
    run_name: str,
    pred_json_path: Path,
    pred_filename: str,
    grayscale: bool,
    rewrite: bool,
    fig: bool,
    mode: str,
    metadata_extra: dict | None = None,
) -> None:
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_metadata(
        out_dir=out_dir,
        run_name=run_name,
        pred_json_path=pred_json_path,
        pred_filename=pred_filename,
        grayscale=grayscale,
        mode=mode,
        metadata_extra=metadata_extra,
    )

    run_pixel_metrics_only_from_loader(
        FilteredLoader(samples),
        str(out_dir),
        rewrite=rewrite,
        fig=fig,
    )

    print(
        f"Loader-mask metrics saved to: {out_dir}"
    )

def run_loader_metrics_from_json(
    *,
    run_name: str,
    dataset_root: str | Path,
    split_manifest_path: str | Path,
    subset: str = "test",
    density_source: str = "manifest",
    density_column: str = "density_class",
    legacy_density_csv: str | Path | None = None,
    length_source: str = "canonical_metadata",
    length_measurement: str = "skeleton",
    length_percentile: float = 98.5,
    legacy_length_csv: str | Path | None = None,
    pred_filename: str = "predicted_annotations_poly.json",
    mode: str = "full",
    grayscale: bool = True,
    rewrite: bool = False,
    fig: bool = True,
) -> None:
    """
    Reconstruct instance masks from a COCO prediction JSON and run
    loader-based pixel/feature evaluation outputs against TIFF GT masks.

    Modes:
      - full: evaluate all test samples
      - density: evaluate LOW/MID/HIGH density subsets
      - length: evaluate long/short object subsets
      - all: run full + density + length
    """
    mode = str(mode).strip().lower()
    if mode not in {"full", "density", "length", "all"}:
        raise ValueError(
            f"Unsupported mode='{mode}'. Use one of: full, density, length, all."
        )

    P = ProjectPaths.from_here(__file__)
    create_run(project_root=P.project_root, run_id=run_name)

    pred_json_path = Path(P.predicted_poly_json(run_name, filename=pred_filename))
    if not pred_json_path.exists():
        raise FileNotFoundError(
            f"Prediction JSON not found: {pred_json_path}. "
            f"Expected under global_outputs/runs/{run_name}/inference/{pred_filename}."
        )

    dataset = CNTDataset(
        dataset_root=dataset_root,
        split_manifest_path=split_manifest_path,
        subset=subset,
        grayscale=grayscale,
        require_masks=True,
    )

    samples = dataset.get_test_data()

    image_stems = [
        sample.X_fn
        for sample in samples
    ]

    image_shapes = [
        sample.X.shape
        for sample in samples
    ]

    pred_masks = coco_polygons_json_to_instance_masks(
        str(pred_json_path),
        image_stems=image_stems,
        image_shapes=image_shapes,
    )

    dataset.set_predictions(pred_masks)

    samples = dataset.get_test_data()

    pred_tag = _pred_tag_from_filename(pred_filename)

    if mode in {"full", "all"}:
        out_dir = Path(
            P.eval_subdir(
                run_name,
                "loader_masks_from_json",
                f"full__{pred_tag}__gt-tif",
            )
        )

        _run_loader_eval_on_subset(
            samples=samples,
            out_dir=out_dir,
            run_name=run_name,
            pred_json_path=pred_json_path,
            pred_filename=pred_filename,
            grayscale=grayscale,
            rewrite=rewrite,
            fig=fig,
            mode="full",
            metadata_extra={
                "num_subset_images": len(samples),
            },
        )

    if mode in {"density", "all"}:
        density_groups = load_density_groups(
            density_source=density_source,
            split_manifest_path=split_manifest_path,
            subset=subset,
            density_column=density_column,
            legacy_density_csv=legacy_density_csv,
        )

        for density_name, filenames in density_groups.items(): 
            subset_samples = _filter_samples_by_filenames( samples, filenames, )
            out_dir = Path(
                P.eval_subdir(
                    run_name,
                    "loader_masks_from_json",
                    f"density_{density_name}__{pred_tag}__gt-tif",
                )
            )

            _run_loader_eval_on_subset(
                samples=subset_samples,
                out_dir=out_dir,
                run_name=run_name,
                pred_json_path=pred_json_path,
                pred_filename=pred_filename,
                grayscale=grayscale,
                rewrite=rewrite,
                fig=fig,
                mode="density",
                metadata_extra={
                    "density_group": density_name,
                    "density_source": density_source,
                    "density_column": density_column,
                    "legacy_density_csv": (
                        str(legacy_density_csv)
                        if legacy_density_csv is not None
                        else None
                    ),
                    "split_manifest_path": str(split_manifest_path),
                    "subset": subset,
                    "num_subset_images": len(subset_samples),
                    "requested_filenames": sorted(filenames),
                },
            )

    if mode in {"length", "all"}:
        object_lengths_csv = ( Path(dataset_root) / "metadata" / "object_lengths.csv" )

        image_lengths_csv = ( Path(dataset_root) / "metadata" / "image_lengths.csv" )

        length_groups, length_threshold = load_length_groups(
            length_source=length_source,
            object_lengths_csv=object_lengths_csv,
            image_lengths_csv=image_lengths_csv,
            split_manifest_path=split_manifest_path,
            subset=subset,
            measurement=length_measurement,
            percentile=length_percentile,
            legacy_length_csv=legacy_length_csv,
        )

        for length_group, filenames in length_groups.items():
            subset_samples = _filter_samples_by_filenames(
                samples,
                filenames,
            )

            out_dir = P.eval_subdir(
                run_name,
                "loader_masks_from_json",
                f"length_{length_group}__{pred_tag}__gt-tif",
            )

            _run_loader_eval_on_subset(
                samples=subset_samples,
                out_dir=out_dir,
                run_name=run_name,
                pred_json_path=pred_json_path,
                pred_filename=pred_filename,
                grayscale=grayscale,
                rewrite=rewrite,
                fig=fig,
                mode="length",
                metadata_extra={
                    "length_group": length_group,
                    "length_source": length_source,
                    "length_measurement": length_measurement,
                    "length_percentile": length_percentile,
                    "length_threshold": length_threshold,
                    "legacy_length_csv": (
                        str(legacy_length_csv)
                        if legacy_length_csv is not None
                        else None
                    ),
                    "object_lengths_csv": str(object_lengths_csv),
                    "image_lengths_csv": str(image_lengths_csv),
                    "split_manifest_path": str(split_manifest_path),
                    "subset": subset,
                    "num_subset_images": len(subset_samples),
                    "requested_filenames": sorted(filenames),
                },
            )

