from glob import glob
from tqdm import tqdm
import numpy as np

import shapely
from tifffile import imread as tiff_imread
from pathlib import Path
from typing import List, Optional
import os
import tifffile as tiff
import matplotlib.pyplot as plt
from .dataset_visualization import plot_img_label  

from dataclasses import dataclass, field
from collections.abc import Iterator, Sequence
from typing import Literal

import pandas as pd

from cnt_project.preprocessing.dataset.validation import (
    DatasetValidationError,
)
from cnt_project.preprocessing.dataset_splitting.split_manifest import (
    SUBSET_COL,
    read_split_manifest_csv,
)
from cnt_project.preprocessing.dataset_splitting.validation import (
    SUPPORTED_SUBSETS,
)
from cnt_project.preprocessing.metadata.schemas import FILENAME_COL

# subset type alias prepared split mode accepts only standardized subset names
PreparedSubset = Literal["train", "val", "test"]
DatasetSourceMode = Literal[ "prepared_split", "image_folder", ]

@dataclass
class DataSample:
    """
    One loaded CNT image and its associated ground truth and predictions.

    Attributes
    ----------
    grayscale
        Whether the input image should be represented as a single channel.

    X_fn
        Legacy sample identifier without the file extension.

        This is retained because existing training, inference,
        evaluation, and COCO-export code depends on it. 

    X
        Loaded and normalized input image.

    Y
        Optional ground-truth instance mask.

    Y_pred
        Optional predicted instance mask.

    filename
        sample filename including the file extension, for example
        ``sample_001.tif``.

        This matches the identifier used by the split manifest. It is optional
        for compatibility with existing in-memory DataSample construction.

    image_path
        Optional path to the source image.

    mask_path
        Optional path to the corresponding ground-truth mask.

    polygons
        Optional predicted polygons for this sample.

    scores
        Optional prediction scores corresponding to the predicted instances.
    """

    grayscale: bool
    X_fn: str
    X: np.ndarray

    Y: Optional[np.ndarray] = None
    Y_pred: Optional[np.ndarray] = None

    filename: Optional[str] = None
    image_path: Optional[Path] = None
    mask_path: Optional[Path] = None

    polygons: Optional[List[shapely.geometry.Polygon]] = None
    scores: Optional[np.ndarray | List[float]] = None

    n_channel: int = field(init=False)

    def __post_init__(self) -> None:
        """
        Normalize sample identifiers and preprocess the loaded image and mask.
        """
        original_identifier = str(self.X_fn)

        # Preserve the current legacy contract:
        # X_fn is always a sanitized filename stem without an extension.
        self.X_fn = Path(original_identifier).stem.replace(",", "_")

        # When no canonical filename is provided, retain the supplied identifier.
        #
        # Existing callers often construct DataSample using only a stem, so we
        # must not invent a file extension.
        if self.filename is None:
            self.filename = Path(original_identifier).name
        else:
            self.filename = Path(str(self.filename)).name

        if self.image_path is not None:
            self.image_path = Path(self.image_path)

        if self.mask_path is not None:
            self.mask_path = Path(self.mask_path)

        if self.grayscale:
            if self.X.ndim == 3:
                # Preserve the current compatibility behavior by selecting
                # the first channel from multi-channel input.
                self.X = self.X[..., 0]
            elif self.X.ndim != 2:
                raise ValueError(
                    "Unsupported image shape for grayscale conversion: "
                    f"{self.X.shape}"
                )

            self.n_channel = 1
        else:
            self.n_channel = ( 1 if self.X.ndim == 2 else int(self.X.shape[-1]) )

        self._normalize()

    def _normalize(self) -> None:
        axis_norm = ( (0, 1) if self.n_channel == 1 else (0, 1, 2) )

        self.X = self._normalize_percentile( self.X, 1, 99.8, axis_norm, )

        if self.Y is not None:
            self.Y = self._fill_label_holes(self.Y)

    @staticmethod
    def _normalize_percentile(
        image: np.ndarray,
        pmin: float,
        pmax: float,
        axis: tuple[int, ...],
    ) -> np.ndarray:
        mi, ma = np.percentile( image, (pmin, pmax), axis=axis, keepdims=True, )

        return np.clip( (image - mi) / (ma - mi + 1e-10), 0, 1, )

    @staticmethod
    def _fill_label_holes(mask: np.ndarray) -> np.ndarray:
        # Placeholder for label hole-filling logic.
        return mask

    def plot_img_label( self, lbl_title: str, save_folder: str | None = None,):
        return plot_img_label(self, lbl_title=lbl_title, save_folder=save_folder,)

class CNTDataset:
    """
    Load one manifest-defined subset from a prepared CNT dataset.

    A CNTDataset instance represents exactly one subset: train, val, or test.
    Split membership is read from the split manifest and is never generated
    or modified by this class.

    This first implementation supports prepared datasets with the structure:

        dataset_root/
        ├── images/
        ├── masks/
        └── splits/
            └── <split_name>.csv

    External unlabeled image-folder loading will be added separately after
    prepared split loading has been validated.
    """

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        split_manifest_path: str | Path,
        subset: PreparedSubset,
        grayscale: bool,
        require_masks: bool = True,
        max_samples: int | None = None,
        selected_filenames: Sequence[str] | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.split_manifest_path = Path( split_manifest_path ).resolve()
        self.source_mode: DatasetSourceMode = "prepared_split"

        self.subset = str(subset)
        self.grayscale = bool(grayscale)
        self.require_masks = bool(require_masks)

        self.image_dir = self.dataset_root / "images"
        self.mask_dir = self.dataset_root / "masks"

        self.max_samples = max_samples
        self.selected_filenames = ( tuple(str(name) for name in selected_filenames) if selected_filenames is not None else None )

        self.gt_coco_paths: dict[str, Path] = {}
        self.prediction_coco_paths: dict[str, Path] = {}

        self._validate_initialization()

        self.samples = self._load_prepared_subset()

        # Temporary compatibility with existing code.
        self.data_samples = self.samples

    @classmethod
    def _create_uninitialized(cls) -> "CNTDataset":
        """
        Create a CNTDataset instance without invoking the prepared-split
        constructor.

        This is used internally by alternative constructors such as
        from_image_folder().
        """
        return cls.__new__(cls)

    @classmethod
    def from_image_folder(
        cls,
        *,
        image_dir: str | Path,
        grayscale: bool,
        file_extensions: Sequence[str] = (
            ".tif",
            ".tiff",
            ".png",
            ".jpg",
            ".jpeg",
        ),
        max_samples: int | None = None,
        selected_filenames: Sequence[str] | None = None,
    ) -> "CNTDataset":
        """
        Load an arbitrary folder of images for inference.

        This mode does not require:

        - a canonical dataset root;
        - ground-truth masks;
        - metadata;
        - a split manifest;
        - ground-truth COCO files.

        Every loaded DataSample has Y=None and mask_path=None.
        """
        dataset = cls._create_uninitialized()

        dataset.source_mode = "image_folder"

        dataset.dataset_root = None
        dataset.split_manifest_path = None
        dataset.subset = None

        dataset.grayscale = bool(grayscale)
        dataset.require_masks = False

        dataset.image_dir = Path(image_dir).resolve()
        dataset.mask_dir = None

        dataset.max_samples = max_samples
        dataset.selected_filenames = (
            tuple(str(name) for name in selected_filenames)
            if selected_filenames is not None
            else None
        )

        dataset.file_extensions = cls._normalize_extensions(
            file_extensions
        )

        dataset.gt_coco_paths = {}
        dataset.prediction_coco_paths = {}

        dataset._validate_image_folder_initialization()

        dataset.samples = dataset._load_image_folder()

        # Temporary compatibility with existing code.
        dataset.data_samples = dataset.samples

        return dataset

    def _validate_initialization(self) -> None:
        """Validate paths and constructor arguments before loading data."""
        if self.subset not in SUPPORTED_SUBSETS:
            raise DatasetValidationError(
                f"Unsupported subset '{self.subset}'. "
                f"Supported subsets: {list(SUPPORTED_SUBSETS)}"
            )

        if not self.dataset_root.exists():
            raise DatasetValidationError(
                f"Dataset root does not exist: {self.dataset_root}"
            )

        if not self.dataset_root.is_dir():
            raise DatasetValidationError(
                f"Dataset root is not a directory: {self.dataset_root}"
            )

        if not self.image_dir.exists() or not self.image_dir.is_dir():
            raise DatasetValidationError(
                f"Missing prepared dataset images directory: "
                f"{self.image_dir}"
            )

        if self.require_masks:
            if not self.mask_dir.exists() or not self.mask_dir.is_dir():
                raise DatasetValidationError(
                    f"Missing prepared dataset masks directory: "
                    f"{self.mask_dir}"
                )

        if not self.split_manifest_path.exists():
            raise DatasetValidationError(
                f"Split manifest does not exist: "
                f"{self.split_manifest_path}"
            )

        if not self.split_manifest_path.is_file():
            raise DatasetValidationError(
                f"Split manifest path is not a file: "
                f"{self.split_manifest_path}"
            )

        if self.max_samples is not None:
            if not isinstance(self.max_samples, int):
                raise TypeError(
                    "max_samples must be an integer or None."
                )

            if self.max_samples <= 0:
                raise ValueError(
                    "max_samples must be greater than zero."
                )

    def _validate_image_folder_initialization(self) -> None:
        """Validate arbitrary image-folder loading arguments."""
        if not self.image_dir.exists():
            raise DatasetValidationError(
                f"Image directory does not exist: {self.image_dir}"
            )

        if not self.image_dir.is_dir():
            raise DatasetValidationError(
                f"Image path is not a directory: {self.image_dir}"
            )

        if self.max_samples is not None:
            if not isinstance(self.max_samples, int):
                raise TypeError(
                    "max_samples must be an integer or None."
                )

            if self.max_samples <= 0:
                raise ValueError(
                    "max_samples must be greater than zero."
                )

    def _select_subset_filenames(self) -> list[str]:
        """
        Read the manifest and return filenames assigned to this subset.
        """
        manifest_df = read_split_manifest_csv( self.split_manifest_path )

        required_columns = { FILENAME_COL, SUBSET_COL, }

        missing_columns = sorted( required_columns - set(manifest_df.columns) )

        if missing_columns:
            raise DatasetValidationError(
                "Split manifest is missing columns required by "
                f"CNTDataset: {missing_columns}"
            )

        subset_df = manifest_df.loc[ manifest_df[SUBSET_COL].astype(str) == self.subset ].copy()

        if subset_df.empty:
            raise DatasetValidationError(
                f"Subset '{self.subset}' is empty in split manifest: "
                f"{self.split_manifest_path}"
            )

        if subset_df[FILENAME_COL].duplicated().any():
            duplicates = (
                subset_df.loc[
                    subset_df[FILENAME_COL].duplicated(),
                    FILENAME_COL,
                ]
                .astype(str)
                .tolist()
            )

            raise DatasetValidationError(
                f"Subset '{self.subset}' contains duplicate filenames: "
                f"{duplicates[:10]}"
            )

        filenames = sorted( subset_df[FILENAME_COL].astype(str).tolist() )

        if self.selected_filenames is not None:
            requested = { Path(name).name for name in self.selected_filenames }

            available = set(filenames)

            unknown = sorted(requested - available)
            if unknown:
                raise DatasetValidationError(
                    "selected_filenames contains files that are not "
                    f"members of subset '{self.subset}': {unknown[:10]}"
                )

            filenames = [ filename for filename in filenames if filename in requested ]

        if self.max_samples is not None:
            filenames = filenames[: self.max_samples]

        if not filenames:
            raise DatasetValidationError(
                "No filenames remain after applying subset and debug "
                "selection."
            )

        return filenames

    def _load_prepared_subset(self) -> list[DataSample]:
        """
        Load images and optional masks for the requested manifest subset.
        """
        filenames = self._select_subset_filenames()

        samples: list[DataSample] = []

        for filename in tqdm(
            filenames,
            desc=f"Loading CNT subset '{self.subset}'",
        ):
            image_path = self.image_dir / filename

            if not image_path.exists() or not image_path.is_file():
                raise DatasetValidationError(
                    f"Missing image for manifest sample: {image_path}"
                )

            image = self.read_image(image_path)

            mask_path: Path | None = None
            mask: np.ndarray | None = None

            if self.require_masks:
                mask_path = self.mask_dir / filename

                if not mask_path.exists() or not mask_path.is_file():
                    raise DatasetValidationError(
                        f"Missing mask for manifest sample: {mask_path}"
                    )

                mask = self.read_image(mask_path)

            sample = DataSample(
                X_fn=Path(filename).stem,
                filename=filename,
                image_path=image_path,
                mask_path=mask_path,
                X=image,
                Y=mask,
                grayscale=self.grayscale,
            )

            samples.append(sample)

        return samples

    def _load_image_folder(self) -> list[DataSample]:
        """
        Load arbitrary inference images without ground-truth masks.
        """
        filenames = self._discover_image_folder_filenames()

        samples: list[DataSample] = []

        for filename in tqdm(
            filenames,
            desc="Loading CNT inference images",
        ):
            image_path = self.image_dir / filename
            image = self.read_image(image_path)

            sample = DataSample(
                X_fn=Path(filename).stem,
                filename=filename,
                image_path=image_path,
                mask_path=None,
                X=image,
                Y=None,
                grayscale=self.grayscale,
            )

            samples.append(sample)

        return samples

    @staticmethod
    def read_image(path: str | Path) -> np.ndarray:
        """Read a supported image file into a NumPy array."""
        path = Path(path)
        extension = path.suffix.lower()

        if extension in {".tif", ".tiff"}:
            return tiff_imread(path)

        if extension in {".png", ".jpg", ".jpeg"}:
            return np.asarray(plt.imread(path))

        raise DatasetValidationError(
            f"Unsupported image extension '{extension}' for: {path}"
        )

    @staticmethod
    def _normalize_extensions(
        extensions: Sequence[str],
    ) -> tuple[str, ...]:
        """
        Normalize file extensions to lowercase values beginning with a dot.
        """
        normalized: list[str] = []

        for extension in extensions:
            value = str(extension).strip().lower()

            if not value:
                continue

            if not value.startswith("."):
                value = f".{value}"

            normalized.append(value)

        normalized = sorted(set(normalized))

        if not normalized:
            raise ValueError(
                "file_extensions must include at least one extension."
            )

        return tuple(normalized)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[DataSample]:
        return iter(self.samples)

    def __getitem__(self, index: int) -> DataSample:
        return self.samples[index]

    def get_data(self) -> list[DataSample]:
        """Return all samples loaded by this dataset instance."""
        return self.samples

    def get_test_data(self) -> list[DataSample]:
        """
        Compatibility alias for existing inference and export code.

        Despite the historical method name, this returns all samples loaded by
        the current CNTDataset instance, which may represent train, val, or
        test.
        """
        return self.samples

    def _discover_image_folder_filenames(self) -> list[str]:
        """
        Discover supported image files in deterministic filename order.
        """
        filenames = sorted(
            path.name
            for path in self.image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in self.file_extensions
        )

        if not filenames:
            raise DatasetValidationError(
                "No supported image files were found in "
                f"{self.image_dir}. Supported extensions: "
                f"{list(self.file_extensions)}"
            )

        if len(filenames) != len(set(filenames)):
            raise DatasetValidationError(
                "Duplicate image filenames were detected in the "
                f"inference folder: {self.image_dir}"
            )

        if self.selected_filenames is not None:
            requested = {
                Path(name).name
                for name in self.selected_filenames
            }

            available = set(filenames)
            unknown = sorted(requested - available)

            if unknown:
                raise DatasetValidationError(
                    "selected_filenames contains files that do not exist "
                    f"in the image folder: {unknown[:10]}"
                )

            filenames = [
                filename
                for filename in filenames
                if filename in requested
            ]

        if self.max_samples is not None:
            filenames = filenames[: self.max_samples]

        if not filenames:
            raise DatasetValidationError(
                "No image files remain after applying the requested "
                "selection."
            )

        return filenames

    def filenames(self) -> list[str]:
        """Return canonical filenames including extensions."""
        return [ str(sample.filename) for sample in self.samples ]

    def set_predictions(
        self,
        predictions: Sequence[np.ndarray],
    ) -> None:
        """Attach one predicted mask to each loaded sample."""
        if len(predictions) != len(self.samples):
            raise ValueError(
                "Number of predicted masks must match the number of "
                f"dataset samples: {len(predictions)} != "
                f"{len(self.samples)}"
            )

        for sample, prediction in zip(
            self.samples,
            predictions,
        ):
            sample.Y_pred = prediction

    def set_polygon_predictions(
        self,
        polygon_predictions: Sequence[
            list[shapely.geometry.Polygon]
        ],
    ) -> None:
        """Attach predicted polygons to each loaded sample."""
        if len(polygon_predictions) != len(self.samples):
            raise ValueError(
                "Number of polygon prediction lists must match the "
                f"number of dataset samples: "
                f"{len(polygon_predictions)} != {len(self.samples)}"
            )

        for sample, polygons in zip(
            self.samples,
            polygon_predictions,
        ):
            sample.polygons = list(polygons)

    def set_prediction_scores(
        self,
        prediction_scores: Sequence[
            np.ndarray | list[float]
        ],
    ) -> None:
        """Attach per-instance prediction scores to each sample."""
        if len(prediction_scores) != len(self.samples):
            raise ValueError(
                "Number of score collections must match the number "
                f"of dataset samples: {len(prediction_scores)} != "
                f"{len(self.samples)}"
            )

        for sample, scores in zip(
            self.samples,
            prediction_scores,
        ):
            sample.scores = scores

    def register_gt_coco_paths(
        self,
        paths: dict[str, str | Path],
    ) -> None:
        """Register ground-truth COCO artifacts associated with this subset."""
        self.gt_coco_paths = {
            str(name): Path(path).resolve()
            for name, path in paths.items()
        }

    def register_prediction_coco_paths(
        self,
        paths: dict[str, str | Path],
    ) -> None:
        """Register exported prediction COCO artifacts."""
        self.prediction_coco_paths = {
            str(name): Path(path).resolve()
            for name, path in paths.items()
        }

class DataLoader:
    def __init__(self, data_dir: str, grayscale: bool, val_split: float = 0.15, random_seed: int = 42, debug: bool = False, max_images: Optional[int] = None, file_extension: str = "tif", inference=True):
        self.inference = inference
        self.file_extension = file_extension
        self.image_dir = os.path.join(data_dir, "images")
        self.label_dir = os.path.join(data_dir, "masks")
        self.grayscale = grayscale
        self.val_split = val_split
        self.random_seed = random_seed
        self.debug = debug
        self.max_images = max_images
        self.data_samples = self._load_data()
        self.train_samples = []
        self.val_samples = []
        self.split_data()

    def _load_data(self) -> List[DataSample]:
        image_files = {Path(f).stem: f for f in glob(os.path.join(self.image_dir, f"*.{self.file_extension}"))}

        has_labels = not self.inference
        label_files = {Path(f).stem: f for f in glob(os.path.join(self.label_dir, f"*.tif"))} if has_labels else {}
        if len(label_files)==0:
            self.inference = True
            print(f"No labels found. Inference mode will be used. Check that self.file_extension {self.file_extension} match the expected format.")
        common_files = sorted(image_files.keys() & label_files.keys()) if has_labels else sorted(image_files.keys())

        if self.debug:
            if isinstance(self.max_images, int):
                self.max_images = min(self.max_images, len(common_files))
                print(f"Debug mode enabled: Loading only {self.max_images} images.")
                common_files = common_files[:self.max_images]
            elif isinstance(self.max_images, list):
                if all(isinstance(i, int) for i in self.max_images):
                    common_files = [common_files[i] for i in self.max_images if i < len(common_files)]
                elif all(isinstance(i, str) for i in self.max_images):
                    common_files = [Path(f).stem for f in self.max_images if Path(f).stem in common_files]
            elif isinstance(self.max_images, str):
                if Path(self.max_images).stem in common_files:
                    common_files = [Path(self.max_images).stem]
                else:
                    common_files = []

        data_samples = []
        for filename in tqdm(common_files, desc="Loading Data"):
            X = self.read_image(image_files[filename])
            Y = self.read_image(label_files[filename]) if has_labels and filename in label_files else None
            data_samples.append(DataSample(X_fn=filename, X=X, Y=Y, grayscale=self.grayscale))

        return data_samples
    
    def read_image(self, filename):
        file_extension = filename.split('.')[-1].lower()
        if file_extension in ['png', 'jpg', 'jpeg']:
            image = plt.imread(filename)
        elif file_extension in ['tif', 'tiff']:
            image = tiff_imread(filename)
        return image
    
    def split_data(self):
        rng = np.random.RandomState(self.random_seed)
        indices = rng.permutation(len(self.data_samples))
        n_val = max(1, int(round(self.val_split * len(indices))))
        
        self.train_samples = [self.data_samples[i] for i in indices[:-n_val]]
        self.val_samples = [self.data_samples[i] for i in indices[-n_val:]]

    def get_train_data(self):
        return self.train_samples

    def get_val_data(self):
        return self.val_samples

    def get_test_data(self):
        print('Loads all data from the specified path.')
        return self.data_samples

    def set_predictions(self, y_preds: List[np.ndarray]):

        if len(y_preds) != len(self.data_samples):
            raise ValueError("Number of predictions must match test samples.")
        for sample, y_pred in zip(self.data_samples, y_preds):
            sample.Y_pred = y_pred
        print("Predictions stored successfully.")
    
    def set_polygon_predictions(self, all_polygons: List[List[shapely.geometry.Polygon]]):
        """
        Store a list of Shapely Polygon lists for each sample.

        Args:
            all_polygons: outer list length == number of samples;
                          each element is the list of Polygon objects for that image.
        """
        if len(all_polygons) != len(self.data_samples):
            raise ValueError("Number of polygon lists must match number of samples.")
        for sample, poly_list in zip(self.data_samples, all_polygons):
            sample.polygons = poly_list
        print("Polygon predictions stored successfully.")

class DataLoaderWithoutLabel(DataLoader):
    """
    A specialized DataLoader for datasets without GroundTruth labels.

    This class loads images but does not require corresponding mask/label files.
    It should be used in cases where only input images are available.
    """

    def _load_data(self) -> List[DataSample]:
        """
        Load image data without associated GroundTruth labels.

        Returns:
            List[DataSample]: A list of DataSample objects containing image data.
        """
        image_files = {Path(f).stem: f for f in glob(os.path.join(self.image_dir, f"*.{self.file_extension}"))}    

        data_samples = []
        for filename in tqdm(image_files, desc="Loading Data"):
            X = tiff.imread(image_files[filename])
            data_samples.append(DataSample(X_fn=filename, X=X, grayscale=self.grayscale))
        
        return data_samples
    
