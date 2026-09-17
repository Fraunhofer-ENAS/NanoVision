from __future__ import annotations

import argparse
from pathlib import Path

from cnt_project.preprocessing.dataset.dataset_loader import CNTDataset
from cnt_project.model_development.stardist_patched.stardist_model_configuration import (
    MyStarDist2D,
)
from cnt_project.model_development.inference.predictor import predict_dataset
from cnt_project.model_development.postprocessing.visualization.polygon_plot_util import (
    plot_polygons,
)
from cnt_project.model_development.inference.export_predictions import (
    export_label_predictions_to_coco_json,
    export_predictions_to_coco_json,
)
from cnt_project.model_development.model_compat import (
    check_model_compatibility,
    infer_model_input_channels,
    parse_grayscale_arg,
    recommended_grayscale_setting,
)
from cnt_project.io.run_manager import build_run_name
from cnt_project.io.paths import OutputPaths, ProjectPaths


def _parse_optional_bool(value: str) -> bool:
    value_norm = str(value).strip().lower()

    if value_norm in {"true", "1", "yes", "y"}:
        return True

    if value_norm in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(
        "Value must be one of: true, false"
    )

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
        "Run StarDist inference on either a canonical dataset subset "
        "or an arbitrary image folder.")
    )

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Canonical dataset root. Use together with "
            "--split-manifest-path and --subset."
        ),
    )

    input_group.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Arbitrary dataset/image-folder root containing images/. "
            "This mode does not use a split manifest."
        ),
    )
    parser.add_argument(
        "--split-manifest-path",
        type=Path,
        default=None,
        help=(
            "Canonical split manifest CSV. Required when "
            "--dataset-root is used."
        ),
    )

    parser.add_argument(
        "--subset",
        choices=("train", "val", "test"),
        default="test",
        help=(
            "Subset to load in canonical dataset mode. "
            "Default: test."
        ),
    )

    parser.add_argument(
        "--model-basedir",
        type=str,
        default=None,
        help=(
            "Base directory containing StarDist model folders. "
            "When omitted in a source-repository workflow, defaults to "
            "<project_root>/models. External library consumers should "
            "provide this path explicitly."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model folder name under --model-basedir.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output folder.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Optional run identifier under global_outputs/runs/. "
            "When omitted, a name is generated from the model name and timestamp."
        ),
    )
    parser.add_argument(
        "--global-outputs-root",
        type=Path,
        default=None,
        help=(
            "Root directory for CNTLib generated outputs. "
            "CNTLib manages runs/, reports/, and misc/ beneath this root. "
            "When omitted in a source-repository workflow, defaults to "
            "<project_root>/global_outputs."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Optional maximum number of inference samples to load. "
            "When omitted, all matching images are loaded."
        ),
    )

    parser.add_argument(
        "--apply-smoothing",
        type=_parse_optional_bool,
        default=None,
        help=(
            "Optionally override polygon smoothing during NMS/postprocessing. "
            "Use true or false. When omitted, the model's default behavior "
            "is preserved."
        ),
    )
    parser.add_argument(
        "--debug-stage-plots",
        action="store_true",
        help=(
            "Save intermediate postprocessing-stage debug plots "
            "during inference."
        ),
    )

    parser.add_argument(
        "--debug-plot-level",
        choices=("basic", "full"),
        default="basic",
        help=(
            "Detail level for postprocessing-stage debug plots. "
            "Default: basic."
        ),
    )
    parser.add_argument(
        "--file-extension",
        type=str,
        default="tif",
        help=(
            "Image extension to load from <data-dir>/images. "
            "Examples: tif, tiff, png, jpg. Default: tif."
        ),
    )
    parser.add_argument(
        "--grayscale",
        nargs="?",
        const=True,
        default=None,
        type=parse_grayscale_arg,
        help=(
            "Image channel mode: auto, true, or false. "
            "When passed without a value, '--grayscale' means true. "
            "When omitted, the setting is inferred from the model."
        ),
    )
    parser.add_argument(
        "--do-plot",
        action="store_true",
        help="Save polygon plots.",
    )
    parser.add_argument(
        "--plot-index",
        type=int,
        default=None,
        help=(
            "Optional zero-based prediction index to plot. "
            "When omitted with --do-plot, all predictions are plotted."
        ),
    )
    parser.add_argument(
        "--export-coco",
        action="store_true",
        help="Export predictions to COCO JSON.",
    )
    parser.add_argument(
        "--export-mask-coco",
        action="store_true",
        help=(
            "Export instance-label predictions to "
            "predicted_annotations_mask.json."
        ),
    )
    parser.add_argument(
        "--category-id",
        type=int,
        default=1,
        help="COCO category ID.",
    )

    return parser

def _sanitize_plot_name(
    value: str,
    max_len: int = 150,
) -> str:
    """
    Return a filesystem-safe plot filename component.
    """
    bad_chars = '<>:"/\\|?*\n\r\t'

    sanitized = "".join(
        "_"
        if char in bad_chars
        else char
        for char in str(value)
    )

    sanitized = (
        sanitized
        .strip()
        .strip(".")
    )

    return sanitized[:max_len]

def main() -> None:
    args = build_argparser().parse_args()
    if ( args.out_dir is not None and args.global_outputs_root is not None ):
        raise ValueError(
            "--out-dir and --global-outputs-root cannot be used together. "
            "Use --global-outputs-root for the canonical CNTLib output "
            "structure, or --out-dir for an explicit run directory."
        )
    if ( args.plot_index is not None and not args.do_plot ):
        raise ValueError(
            "--plot-index requires --do-plot."
        )

    project_paths: ProjectPaths | None = None

    if args.global_outputs_root is not None:
        output_paths = OutputPaths.from_root(
            args.global_outputs_root
        )
    else:
        project_paths = ProjectPaths.from_here(__file__)

        output_paths = OutputPaths.from_root(
            project_paths.outputs_root
        )

    output_paths.ensure()

    run_name = build_run_name(
        model_name=args.model_name,
        run_name=args.run_name,
    )

    if args.out_dir is not None:
        run_dir = Path(args.out_dir).resolve()
        inference_dir = run_dir / "inference"
        viz_dir = run_dir / "viz" / "inference"
        inference_dir.mkdir( parents=True, exist_ok=True, )
        viz_dir.mkdir( parents=True, exist_ok=True, )
    else:
        run_dir = output_paths.run_dir( run_name )
        inference_dir = output_paths.inference_dir( run_name )
        viz_dir = output_paths.viz_dir( run_name, "inference", )

    print(f"Run name: {run_name}")
    print(f"Run directory: {run_dir}")
    print(f"Inference outputs: {inference_dir}")
    print(f"Visualization outputs: {viz_dir}")

    if args.model_basedir is not None:
        model_basedir = str( Path(args.model_basedir).resolve() )

    elif project_paths is not None:
        model_basedir = str( project_paths.models_root )

    else:
        raise ValueError(
            "--model-basedir is required when "
            "--global-outputs-root is supplied."
        )
    
    model_n_channels_hint = infer_model_input_channels(
        model_basedir,
        args.model_name,
    )

    if args.grayscale is None:
        resolved_grayscale = recommended_grayscale_setting(
            current_grayscale=False,
            model_name=args.model_name,
            model_n_channels=model_n_channels_hint,
        )
    else:
        resolved_grayscale = bool(args.grayscale)

    print(
        f"Grayscale mode: "
        f"{'auto' if args.grayscale is None else args.grayscale}"
    )
    print(
        f"Resolved grayscale: {resolved_grayscale}"
    )
    print(
        f"Saved model input channels: {model_n_channels_hint}"
    )
    check_model_compatibility(
        grayscale=resolved_grayscale,
        model_name=args.model_name,
        model_n_channels=model_n_channels_hint,
    )

    # 1) Load inference dataset.
    if args.dataset_root is not None:
        if args.split_manifest_path is None:
            raise ValueError(
                "--split-manifest-path is required when "
                "--dataset-root is used."
            )

        dataset = CNTDataset(
            dataset_root=args.dataset_root,
            split_manifest_path=args.split_manifest_path,
            subset=args.subset,
            grayscale=resolved_grayscale,
            require_masks=False,
            max_samples=args.max_samples,
        )

        source_description = (
            f"canonical dataset subset '{args.subset}' "
            f"from {Path(args.dataset_root).resolve()}"
        )

    else:
        image_dir = Path(args.data_dir).resolve() / "images"

        dataset = CNTDataset.from_image_folder(
            image_dir=image_dir,
            grayscale=resolved_grayscale,
            file_extensions=(args.file_extension,),
            max_samples=args.max_samples,
        )

        source_description = (
            f"image folder {image_dir}"
        )

    samples = dataset.get_test_data()

    if not samples:
        raise RuntimeError(
            f"No inference samples found in {source_description}."
        )
    
    images = [ sample.X for sample in samples ]
    filenames = [sample.filename for sample in samples]

    # 2) Load model.
    model = MyStarDist2D(
        None,
        name=args.model_name,
        basedir=model_basedir,
    )

    dataset_n_channels = int(
        samples[0].n_channel
    )

    model_n_channels = int(
        model.config.n_channel_in
    )

    check_model_compatibility(
        grayscale=resolved_grayscale,
        model_name=args.model_name,
        dataset_n_channels=dataset_n_channels,
        model_n_channels=model_n_channels,
    )

    # 3) Predict.
    batch = predict_dataset(
        model=model,
        images=images,
        filenames=filenames,
        apply_smoothing=args.apply_smoothing,
        debug_stage_plots=args.debug_stage_plots,
        debug_plot_dir=str(viz_dir),
        debug_plot_level=args.debug_plot_level,
    )
    if args.apply_smoothing is None:
        print("Smoothing override: model default")
    else:
        print(f"Apply smoothing: {args.apply_smoothing}")

    if args.do_plot:
        if args.plot_index is None:
            plot_indices = range( len(batch.filenames) )
        else:
            if not ( 0 <= args.plot_index < len(batch.filenames) ):
                raise ValueError(
                    "--plot-index is outside the available prediction range: "
                    f"requested={args.plot_index}, "
                    f"prediction_count={len(batch.filenames)}."
                )

            plot_indices = [ args.plot_index ]

        for index in plot_indices:
            filename = batch.filenames[index]
            polygons = batch.polygons[index]
            scores = batch.scores[index]
            coords = batch.coords[index]

            plot_name = _sanitize_plot_name( f"Masked_CNTs_{filename}" )

            plot_polygons(
                polygons,
                title=plot_name,
                scores=scores,
                points_arr=coords,
                save_path=str(viz_dir),
                flip_vertical=True,
            )

    # 4) Export COCO predictions.
    if args.export_coco:
        polygon_json_path = ( inference_dir / "predicted_annotations_poly.json" )

        rle_json_path = ( inference_dir / "predicted_annotations_rle.json" )

        coco_export = export_predictions_to_coco_json(
            images=batch.images,
            filenames=batch.filenames,
            polygons=batch.polygons,
            scores=batch.scores,
            out_json_path=polygon_json_path,
            out_rle_json_path=rle_json_path,
            category_id=args.category_id,
        )

        dataset.register_prediction_coco_paths(
            {
                "polygon": coco_export.polygon_json_path,
                "rle": coco_export.rle_json_path,
            }
        )

        print(
            "Saved polygon COCO predictions: "
            f"{coco_export.polygon_json_path}"
        )
        print(
            "Saved RLE COCO predictions: "
            f"{coco_export.rle_json_path}"
        )
    if args.export_mask_coco:
        mask_json_path = (
            inference_dir
            / "predicted_annotations_mask.json"
        )

        exported_mask_path = export_label_predictions_to_coco_json(
            y_preds=batch.y_preds,
            filenames=batch.filenames,
            scores=batch.scores,
            out_json_path=mask_json_path,
            category_id=args.category_id,
        )

        dataset.register_prediction_coco_paths(
            {
                "mask": exported_mask_path,
            }
        )

        print(
            "Saved mask COCO predictions: "
            f"{exported_mask_path}"
        )


if __name__ == "__main__":
    main()