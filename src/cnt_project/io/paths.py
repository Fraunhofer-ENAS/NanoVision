from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetPaths:
    """
    Canonical filesystem paths for one prepared CNT dataset.

    Expected dataset layout
    -----------------------
    data/<dataset_name>/
        images/
        masks/
        metadata/
        COCO_mask/
        splits/

    Images and masks are not physically separated into train/val/test
    directories. Dataset membership is defined by a split manifest.

    Split-specific COCO annotations are stored under:

        COCO_mask/<split_name>/<subset>/annotations.json

    where ``subset`` is typically ``train``, ``val``, or ``test``.
    """

    name: str
    root: Path

    images_root: Path
    masks_root: Path
    metadata_root: Path
    coco_root: Path
    splits_root: Path

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        *,
        name: str | None = None,
    ) -> "DatasetPaths":
        """
        Construct canonical dataset paths from a dataset root.

        This allows the dataset path abstraction to be used independently
        of the repository layout.
        """
        root = Path(root).resolve()

        dataset_name = name or root.name

        return cls(
            name=dataset_name,
            root=root,
            images_root=root / "images",
            masks_root=root / "masks",
            metadata_root=root / "metadata",
            coco_root=root / "COCO_mask",
            splits_root=root / "splits",
        )

    def image_file(self, filename: str) -> Path:
        """Return the path of one image file."""
        return self.images_root / filename

    def mask_file(self, filename: str) -> Path:
        """Return the path of one mask file."""
        return self.masks_root / filename

    def metadata_file(self, filename: str) -> Path:
        """Return the path of one dataset metadata file."""
        return self.metadata_root / filename

    def split_manifest_csv(
        self,
        split_name: str = "default_split",
    ) -> Path:
        """
        Return the CSV split manifest path.

        Example:
            splits/default_split.csv
        """
        return self.splits_root / f"{split_name}.csv"

    def split_metadata_yaml(
        self,
        split_name: str = "default_split",
    ) -> Path:
        """
        Return the YAML metadata/configuration file associated with a split.

        Example:
            splits/default_split.yaml
        """
        return self.splits_root / f"{split_name}.yaml"

    def coco_split_root(
        self,
        split_name: str,
        subset: str,
    ) -> Path:
        """
        Return the directory containing COCO annotations for one subset.

        Example:
            COCO_mask/default_split/test/
        """
        return self.coco_root / split_name / subset

    def coco_json(
        self,
        split_name: str,
        subset: str,
        filename: str = "annotations.json",
    ) -> Path:
        """
        Return the COCO annotation JSON for one dataset subset.

        Example:
            COCO_mask/default_split/test/annotations.json
        """
        return self.coco_split_root(
            split_name,
            subset,
        ) / filename


@dataclass(frozen=True)
class OutputPaths:
    """
    Canonical CNTLib generated-output paths.

    Expected layout
    ---------------
    <outputs_root>/
        runs/
            <run_name>/
                train/
                inference/
                eval/
                viz/
        reports/
        misc/

    Unlike ProjectPaths, this abstraction is independent of the CNTLib
    source-repository layout and can therefore be used by installed-library
    consumers.
    """

    root: Path
    runs_root: Path
    reports_root: Path
    misc_root: Path

    @classmethod
    def from_root(
        cls,
        root: str | Path,
    ) -> "OutputPaths":
        root = Path(root).resolve()

        return cls(
            root=root,
            runs_root=root / "runs",
            reports_root=root / "reports",
            misc_root=root / "misc",
        )

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.reports_root.mkdir(parents=True, exist_ok=True)
        self.misc_root.mkdir(parents=True, exist_ok=True)

    def run_dir(
        self,
        run_name: str,
    ) -> Path:
        directory = self.runs_root / run_name
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def train_dir(
        self,
        run_name: str,
        *parts: str,
    ) -> Path:
        directory = self.run_dir(run_name) / "train"

        for part in parts:
            directory = directory / part

        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def inference_dir(
        self,
        run_name: str,
        *parts: str,
    ) -> Path:
        directory = self.run_dir(run_name) / "inference"

        for part in parts:
            directory = directory / part

        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def eval_dir(
        self,
        run_name: str,
        evaluator: str,
    ) -> Path:
        directory = (
            self.run_dir(run_name)
            / "eval"
            / evaluator
        )

        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def eval_subdir(
        self,
        run_name: str,
        evaluator: str,
        *parts: str,
    ) -> Path:
        directory = self.eval_dir(
            run_name,
            evaluator,
        )

        for part in parts:
            directory = directory / part

        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def viz_dir(
        self,
        run_name: str,
        *parts: str,
    ) -> Path:
        directory = self.run_dir(run_name) / "viz"

        for part in parts:
            directory = directory / part

        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def report_dir(
        self,
        report_name: str,
        *parts: str,
    ) -> Path:
        directory = self.reports_root / report_name

        for part in parts:
            directory = directory / part

        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def misc_dir(
        self,
        *parts: str,
    ) -> Path:
        directory = self.misc_root

        for part in parts:
            directory = directory / part

        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def predicted_poly_json(
        self,
        run_name: str,
        filename: str = "predicted_annotations_poly.json",
    ) -> Path:
        return self.inference_dir(run_name) / filename

    def predicted_mask_json(
        self,
        run_name: str,
        filename: str = "predicted_annotations_mask.json",
    ) -> Path:
        return self.inference_dir(run_name) / filename

    def predicted_rle_json(
        self,
        run_name: str,
        filename: str = "predicted_annotations_rle.json",
    ) -> Path:
        return self.inference_dir(run_name) / filename

@dataclass(frozen=True)
class ProjectPaths:
    """
    Centralized repository filesystem contract.

    Repository layout
    -----------------
    repo/
        data/
            <dataset_name>/

        models/

        global_outputs/
            runs/
                <run_name>/
                    train/
                    inference/
                    eval/
                    viz/
            reports/
            misc/

        src/
            cnt_project/

    Notes
    -----
    ``annotations_uniques`` is a legacy dataset containing the historical
    manually created train/test split.

    It is currently retained as a reproducibility reference for generating
    the canonical default split of the new unsplit CNT dataset.

    The legacy train/test path attributes and helpers below remain temporarily
    available because older evaluation and visualization code still depends
    on them. New code should use ``DatasetPaths`` instead.
    """

    project_root: Path

    # ------------------------------------------------------------------
    # Repository roots
    # ------------------------------------------------------------------

    src_root: Path
    package_root: Path
    data_root: Path
    models_root: Path

    # ------------------------------------------------------------------
    # Generated output roots
    # ------------------------------------------------------------------

    output_paths: OutputPaths

    # ------------------------------------------------------------------
    # Legacy annotations_uniques compatibility
    #
    # Do not use these fields in new code.
    # They preserve the historical manually split dataset layout until
    # remaining legacy consumers have been migrated.
    # ------------------------------------------------------------------

    annotations_root: Path

    train_root: Path
    test_root: Path

    train_coco_root: Path
    train_images_root: Path
    train_masks_root: Path
    train_metadata_root: Path

    test_coco_root: Path
    test_images_root: Path
    test_masks_root: Path
    test_metadata_root: Path

    train_coco_json: Path
    train_description_txt: Path

    test_coco_json: Path
    test_description_txt: Path

    train_density_csv: Path
    test_density_csv: Path


    @property
    def outputs_root(self) -> Path:
        """
        Canonical generated-output root.

        Compatibility property delegating to ``OutputPaths``.
        """
        return self.output_paths.root


    @property
    def runs_root(self) -> Path:
        """
        Canonical run root.

        Compatibility property delegating to ``OutputPaths``.
        """
        return self.output_paths.runs_root


    @property
    def reports_root(self) -> Path:
        """
        Canonical report root.

        Compatibility property delegating to ``OutputPaths``.
        """
        return self.output_paths.reports_root


    @property
    def misc_root(self) -> Path:
        """
        Canonical miscellaneous-output root.

        Compatibility property delegating to ``OutputPaths``.
        """
        return self.output_paths.misc_root
    
    @property
    def legacy_manual_split_reference_root(self) -> Path:
        """
        Historical manually split CNT dataset.

        This dataset is retained as the reference for reproducing the
        historical/default dataset split.

        New dataset workflows should not treat this as the canonical
        prepared dataset.
        """
        return self.annotations_root

    def ensure_outputs(self) -> None:
        """
        Create the canonical generated-output roots.
        """
        self.output_paths.ensure()

    @staticmethod
    def _infer_project_root(
        file: str | Path,
    ) -> tuple[Path, Path, Path]:
        """
        Infer repository roots from a file located below ``src/``.

        Returns
        -------
        project_root
            Repository root.

        src_root
            ``<repo>/src``.

        package_root
            ``<repo>/src/cnt_project``.
        """
        here = Path(file).resolve()

        for parent in [here, *here.parents]:
            if parent.name == "src":
                src_root = parent
                project_root = src_root.parent
                package_root = src_root / "cnt_project"

                return (
                    project_root,
                    src_root,
                    package_root,
                )

        raise ValueError(
            f"Could not infer project root from path '{here}'. "
            "Expected the path to be inside a repository using "
            "a 'src/' layout."
        )


    @classmethod
    def from_project_root(
        cls,
        project_root: str | Path,
        *,
        outputs_root: str | Path = "global_outputs",
    ) -> "ProjectPaths":
        """
        Construct repository paths from an explicit project root.

        The project is expected to use the standard src-layout:

            <project_root>/
                src/
                    cnt_project/
        """
        project_root = Path(project_root).resolve()

        src_root = project_root / "src"
        package_root = src_root / "cnt_project"

        if not package_root.exists():
            raise FileNotFoundError(
                "Could not locate cnt_project package at: "
                f"{package_root}"
            )

        data_root = project_root / "data"
        models_root = project_root / "models"

        outputs_root_path = Path(outputs_root)

        if not outputs_root_path.is_absolute():
            outputs_root_path = project_root / outputs_root_path

        output_paths = OutputPaths.from_root(
            outputs_root_path
        )

        # Legacy annotations_uniques compatibility
        annotations_root = data_root / "annotations_uniques"

        train_root = annotations_root
        test_root = annotations_root / "test"

        train_coco_root = train_root / "COCO_mask"
        train_images_root = train_root / "images"
        train_masks_root = train_root / "masks"
        train_metadata_root = train_root / "metadata"

        test_coco_root = test_root / "COCO_mask"
        test_images_root = test_root / "images"
        test_masks_root = test_root / "masks"
        test_metadata_root = test_root / "metadata"

        train_coco_json = train_coco_root / "annotations.json"
        train_description_txt = train_metadata_root / "description.txt"

        test_coco_json = test_coco_root / "annotations.json"
        test_description_txt = test_metadata_root / "description.txt"

        train_density_csv = (
            train_metadata_root
            / "density_classified_filenames.csv"
        )

        test_density_csv = (
            test_metadata_root
            / "density_classified_filenames.csv"
        )

        return cls(
            project_root=project_root,
            src_root=src_root,
            package_root=package_root,
            data_root=data_root,
            models_root=models_root,
            output_paths=output_paths,
            annotations_root=annotations_root,
            train_root=train_root,
            test_root=test_root,
            train_coco_root=train_coco_root,
            train_images_root=train_images_root,
            train_masks_root=train_masks_root,
            train_metadata_root=train_metadata_root,
            test_coco_root=test_coco_root,
            test_images_root=test_images_root,
            test_masks_root=test_masks_root,
            test_metadata_root=test_metadata_root,
            train_coco_json=train_coco_json,
            train_description_txt=train_description_txt,
            test_coco_json=test_coco_json,
            test_description_txt=test_description_txt,
            train_density_csv=train_density_csv,
            test_density_csv=test_density_csv,
        )

    @classmethod
    def from_here(
        cls,
        file: str | Path,
        *,
        outputs_root: str | Path = "global_outputs",
    ) -> "ProjectPaths":
        """
        Discover the repository root from a file inside the src-layout project.
        """
        project_root, _, _ = cls._infer_project_root(file)

        return cls.from_project_root(
            project_root,
            outputs_root=outputs_root,
        )

    # ------------------------------------------------------------------
    # Canonical dataset helpers
    # ------------------------------------------------------------------

    def dataset(
        self,
        dataset_name: str,
    ) -> DatasetPaths:
        """
        Return canonical paths for a dataset stored under ``data/``.

        Example
        -------
        ``P.dataset("cnt_segmentation")`` resolves to:

            data/cnt_segmentation/
        """
        return DatasetPaths.from_root(
            self.data_root / dataset_name,
            name=dataset_name,
        )

    def dataset_from_root(
        self,
        dataset_root: str | Path,
    ) -> DatasetPaths:
        """
        Construct DatasetPaths for an explicitly supplied dataset root.

        Relative paths are interpreted relative to the repository root.
        Absolute paths are preserved.

        This helper is useful for CLI workflows that accept ``--dataset-root``.
        """
        root = Path(dataset_root)

        if not root.is_absolute():
            root = self.project_root / root

        return DatasetPaths.from_root(root)

    # ------------------------------------------------------------------
    # Core output directory helpers
    # ------------------------------------------------------------------

    def run_dir( self, run_name: str, ) -> Path:
        """
        global_outputs/runs/<RUN>/
        """
        return self.output_paths.run_dir(
            run_name
        )

    def train_dir( self, run_name: str, *parts: str, ) -> Path:
        """
        global_outputs/runs/<RUN>/train/<parts...>/
        """
        return self.output_paths.train_dir(
            run_name,
            *parts,
        )

    def eval_dir( self, run_name: str, evaluator: str, ) -> Path:
        """
        global_outputs/runs/<RUN>/eval/<evaluator>/
        """
        return self.output_paths.eval_dir(
            run_name,
            evaluator,
        )

    def eval_subdir( self, run_name: str, evaluator: str, *parts: str, ) -> Path:
        """
        global_outputs/runs/<RUN>/eval/<evaluator>/<parts...>/
        """
        return self.output_paths.eval_subdir(
            run_name,
            evaluator,
            *parts,
        )

    def inference_dir( self, run_name: str, *parts: str, ) -> Path:
        """
        global_outputs/runs/<RUN>/inference/<parts...>/
        """
        return self.output_paths.inference_dir(
            run_name,
            *parts,
        )

    def viz_dir( self, run_name: str, *parts: str, ) -> Path:
        """
        global_outputs/runs/<RUN>/viz/<parts...>/
        """
        return self.output_paths.viz_dir(
            run_name,
            *parts,
        )

    def report_dir( self, report_name: str, *parts: str, ) -> Path:
        """
        global_outputs/reports/<REPORT>/<parts...>/
        """
        return self.output_paths.report_dir(
            report_name,
            *parts,
        )

    def misc_dir( self, *parts: str, ) -> Path:
        """
        global_outputs/misc/<parts...>/
        """
        return self.output_paths.misc_dir(
            *parts
        )

    # ------------------------------------------------------------------
    # Prediction file helpers
    # ------------------------------------------------------------------

    def predicted_poly_json( self, pred_run: str, filename: str = "predicted_annotations_poly.json", ) -> Path:
        """
        Canonical predicted polygon JSON path.
        """
        return self.output_paths.predicted_poly_json(
            pred_run,
            filename=filename,
        )

    def predicted_mask_json( self, pred_run: str, filename: str = "predicted_annotations_mask.json", ) -> Path:
        """
        Canonical predicted mask JSON path.
        """
        return self.output_paths.predicted_mask_json(
            pred_run,
            filename=filename,
        )

    def predicted_rle_json( self, pred_run: str, filename: str = "predicted_annotations_rle.json", ) -> Path:
        """
        Canonical predicted RLE JSON path.
        """
        return self.output_paths.predicted_rle_json(
            pred_run,
            filename=filename,
        )

    # ==================================================================
    # LEGACY DATASET API
    #
    # Temporary compatibility layer for code that still expects the
    # annotations_uniques train/test directory layout.
    #
    # Do not use these methods in new code.
    # ==================================================================

    def split_root(
        self,
        split: str,
    ) -> Path:
        split_norm = split.lower()

        if split_norm == "train":
            return self.train_root

        if split_norm == "test":
            return self.test_root

        raise ValueError(
            f"Unsupported legacy split '{split}'. "
            "Expected 'train' or 'test'."
        )

    def coco_root(
        self,
        split: str,
    ) -> Path:
        split_norm = split.lower()

        if split_norm == "train":
            return self.train_coco_root

        if split_norm == "test":
            return self.test_coco_root

        raise ValueError(
            f"Unsupported legacy split '{split}'. "
            "Expected 'train' or 'test'."
        )

    def images_root(
        self,
        split: str,
    ) -> Path:
        split_norm = split.lower()

        if split_norm == "train":
            return self.train_images_root

        if split_norm == "test":
            return self.test_images_root

        raise ValueError(
            f"Unsupported legacy split '{split}'. "
            "Expected 'train' or 'test'."
        )

    def masks_root(
        self,
        split: str,
    ) -> Path:
        split_norm = split.lower()

        if split_norm == "train":
            return self.train_masks_root

        if split_norm == "test":
            return self.test_masks_root

        raise ValueError(
            f"Unsupported legacy split '{split}'. "
            "Expected 'train' or 'test'."
        )

    def metadata_root(
        self,
        split: str,
    ) -> Path:
        split_norm = split.lower()

        if split_norm == "train":
            return self.train_metadata_root

        if split_norm == "test":
            return self.test_metadata_root

        raise ValueError(
            f"Unsupported legacy split '{split}'. "
            "Expected 'train' or 'test'."
        )

    def coco_json(
        self,
        split: str,
    ) -> Path:
        """
        Legacy annotations_uniques COCO path.
        """
        return (
            self.coco_root(split)
            / "annotations.json"
        )

    def description_txt(
        self,
        split: str,
    ) -> Path:
        """
        Legacy annotations_uniques description path.
        """
        return (
            self.metadata_root(split)
            / "description.txt"
        )

    def metadata_file(
        self,
        split: str,
        filename: str,
    ) -> Path:
        """
        Legacy annotations_uniques metadata path.
        """
        return (
            self.metadata_root(split)
            / filename
        )

    def density_csv(
        self,
        split: str,
    ) -> Path:
        """
        Legacy annotations_uniques density metadata path.
        """
        return (
            self.metadata_root(split)
            / "density_classified_filenames.csv"
        )

    def image_file(
        self,
        split: str,
        filename: str,
    ) -> Path:
        """
        Legacy annotations_uniques image path.
        """
        return (
            self.images_root(split)
            / filename
        )

    def mask_file(
        self,
        split: str,
        filename: str,
    ) -> Path:
        """
        Legacy annotations_uniques mask path.
        """
        return (
            self.masks_root(split)
            / filename
        )