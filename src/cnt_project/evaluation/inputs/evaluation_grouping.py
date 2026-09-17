from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np


def parse_density_description_file(description_path: str) -> dict[str, list[str]]:
    """
    Parses the description.txt format:

      **LOW_DENSITY**
      name1
      name2

      **MID_DENSITY**
      ...

    Returns:
      {
        "LOW_DENSITY": [...],
        "MID_DENSITY": [...],
        "HIGH_DENSITY": [...],
      }
    """
    groups: dict[str, list[str]] = {}
    current: str | None = None

    with open(description_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("**") and line.endswith("**"):
                key = line.strip("*").strip()
                current = key
                groups.setdefault(current, [])
                continue

            if current is None:
                continue

            groups[current].append(line)

    return groups


def load_length_classification_csv(
    classification_csv: str,
    long_label: str = "contains_long_objects",
    short_label: str = "mostly_short_objects",
    add_ext: str = ".jpg",
) -> dict[str, list[str]]:
    """
    Reads the classification CSV and returns filename lists per subset.

    Expected columns:
      - 'category'  (values include long_label / short_label)
      - 'file_name' (stem without extension)

    Returns:
      { "long_objects": [...], "short_objects": [...] } with extension appended.
    """
    df = pd.read_csv(classification_csv)

    required = {"category", "file_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"classification CSV missing columns: {sorted(missing)}")

    long_files = df.loc[df["category"] == long_label, "file_name"].astype(str).tolist()
    short_files = df.loc[df["category"] == short_label, "file_name"].astype(str).tolist()

    if add_ext:
        long_files = [f + add_ext for f in long_files]
        short_files = [f + add_ext for f in short_files]

    return {
        "long_objects": long_files,
        "short_objects": short_files,
    }


def load_length_groups_from_canonical_metadata(
    *,
    object_lengths_csv: str | Path,
    image_lengths_csv: str | Path,
    split_manifest_path: str | Path,
    subset: str = "test",
    measurement: str = "skeleton",
    percentile: float = 98.5,
) -> tuple[dict[str, set[str]], float]:
    """
    Build image-level length groups from canonical length metadata.

    The classification reproduces the historical rule:

        threshold = percentile(object lengths, 98.5)

        contains_long_objects:
            image maximum length > threshold

        mostly_short_objects:
            image maximum length <= threshold

    The percentile is calculated only from objects belonging to the
    requested manifest subset.

    Parameters
    ----------
    object_lengths_csv
        Canonical object-level length metadata.

    image_lengths_csv
        Canonical image-level aggregated length metadata.

    split_manifest_path
        Canonical dataset split manifest.

    subset
        Manifest subset used for classification.

    measurement
        Length representation:

        - ``skeleton``:
          object ``skeleton_pixel_count`` and image
          ``max_skeleton_pixel_count``.

        - ``geodesic_px``:
          object ``geodesic_length_px`` and image
          ``max_geodesic_length_px``.

        - ``geodesic_um``:
          object ``geodesic_length_um`` and image
          ``max_geodesic_length_um``.

    percentile
        Object-length percentile defining the long-object threshold.
        The historical value is 98.5.

    Returns
    -------
    tuple[dict[str, set[str]], float]
        Length groups and the calculated threshold.
    """
    measurement_columns = {
        "skeleton": (
            "skeleton_pixel_count",
            "max_skeleton_pixel_count",
        ),
        "geodesic_px": (
            "geodesic_length_px",
            "max_geodesic_length_px",
        ),
        "geodesic_um": (
            "geodesic_length_um",
            "max_geodesic_length_um",
        ),
    }

    measurement_key = str(measurement).strip().lower()

    if measurement_key not in measurement_columns:
        raise ValueError(
            "measurement must be one of: "
            "'skeleton', 'geodesic_px', 'geodesic_um'. "
            f"Got: {measurement!r}"
        )

    object_length_column, image_max_column = (
        measurement_columns[measurement_key]
    )

    object_df = pd.read_csv(object_lengths_csv)
    image_df = pd.read_csv(image_lengths_csv)
    manifest_df = pd.read_csv(split_manifest_path)

    required_object_columns = {
        "filename",
        object_length_column,
    }
    missing = required_object_columns - set(object_df.columns)

    if missing:
        raise ValueError(
            "Object length metadata is missing required column(s): "
            f"{sorted(missing)}"
        )

    required_image_columns = {
        "filename",
        image_max_column,
    }
    missing = required_image_columns - set(image_df.columns)

    if missing:
        raise ValueError(
            "Image length metadata is missing required column(s): "
            f"{sorted(missing)}"
        )

    required_manifest_columns = {
        "filename",
        "subset",
    }
    missing = required_manifest_columns - set(manifest_df.columns)

    if missing:
        raise ValueError(
            "Split manifest is missing required column(s): "
            f"{sorted(missing)}"
        )

    subset_manifest = manifest_df[
        manifest_df["subset"].astype(str).str.lower()
        == str(subset).strip().lower()
    ].copy()

    subset_keys = {
        normalize_evaluation_filename(filename)
        for filename in subset_manifest["filename"]
    }

    object_df["_image_key"] = object_df["filename"].map(
        normalize_evaluation_filename
    )

    image_df["_image_key"] = image_df["filename"].map(
        normalize_evaluation_filename
    )

    subset_object_df = object_df[
        object_df["_image_key"].isin(subset_keys)
    ].copy()

    subset_image_df = image_df[
        image_df["_image_key"].isin(subset_keys)
    ].copy()

    if subset_object_df.empty:
        raise ValueError(
            f"No object length metadata found for subset {subset!r}."
        )

    if subset_image_df.empty:
        raise ValueError(
            f"No image length metadata found for subset {subset!r}."
        )

    object_lengths = pd.to_numeric(
        subset_object_df[object_length_column],
        errors="coerce",
    ).dropna()

    if object_lengths.empty:
        raise ValueError(
            f"No valid values found in {object_length_column!r} "
            f"for subset {subset!r}."
        )

    threshold = float(
        np.percentile(
            object_lengths.to_numpy(dtype=float),
            percentile,
        )
    )

    groups: dict[str, set[str]] = {
        "long_objects": set(),
        "short_objects": set(),
    }

    for _, row in subset_image_df.iterrows():
        max_length = float(row[image_max_column])

        image_key = normalize_evaluation_filename(
            row["filename"]
        )

        # Preserve historical strict comparison:
        # max_length > threshold
        if max_length > threshold:
            groups["long_objects"].add(image_key)
        else:
            groups["short_objects"].add(image_key)

    return groups, threshold

def load_length_groups(
    *,
    length_source: str,
    object_lengths_csv: str | Path,
    image_lengths_csv: str | Path,
    split_manifest_path: str | Path,
    subset: str = "test",
    measurement: str = "skeleton",
    percentile: float = 98.5,
    legacy_length_csv: str | Path | None = None,
) -> tuple[dict[str, set[str]], float | None]:
    source = str(length_source).strip().lower()

    if source == "canonical_metadata":
        return load_length_groups_from_canonical_metadata(
            object_lengths_csv=object_lengths_csv,
            image_lengths_csv=image_lengths_csv,
            split_manifest_path=split_manifest_path,
            subset=subset,
            measurement=measurement,
            percentile=percentile,
        )

    if source == "legacy_metadata":
        if legacy_length_csv is None:
            raise ValueError(
                "legacy_length_csv is required when "
                "length_source='legacy_metadata'."
            )

        groups = load_length_classification_csv(
            str(legacy_length_csv),
            add_ext="",
        )

        return {
            key: {
                normalize_evaluation_filename(name)
                for name in values
            }
            for key, values in groups.items()
        }, None

    raise ValueError(
        "length_source must be one of: "
        "'canonical_metadata', 'legacy_metadata'. "
        f"Got: {length_source!r}"
    )


def load_noise_classification_csv(
    classification_csv: str,
    *,
    label_column: str = "Noise_KMeans",
    filename_column: str = "filename",
) -> dict[str, list[str]]:
    """
    Reads noise-classification CSV and returns filenames grouped by normalized class key.

    Returned keys are normalized uppercase with spaces replaced by underscores.
    Example: "Clean" -> "CLEAN", "very noisy" -> "VERY_NOISY".

    Filenames are normalized to stem-only values so matching is extension-agnostic
    (e.g., CSV .tif names still match COCO .jpg names).
    """
    df = pd.read_csv(classification_csv)

    required = {label_column, filename_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"noise classification CSV missing columns: {sorted(missing)}")

    grouped: dict[str, list[str]] = {}
    for raw_label, sub_df in df.groupby(label_column):
        label = str(raw_label).strip()
        if not label:
            continue

        norm_label = normalize_noise_selection(label)
        if norm_label == "ALL":
            continue

        files = [
            Path(name).stem
            for name in sub_df[filename_column].dropna().astype(str).str.strip().tolist()
            if str(name).strip()
        ]
        grouped[norm_label] = sorted(set(files))

    return grouped

def normalize_density_selection(selected_density: str | None) -> str:
    """
    Normalize user input for density selection.

    Accepted:
    - None -> "ALL" (default: evaluate all density groups)
      - "low" / "LOW" / "LOW_DENSITY"
      - "mid" / "MID" / "MID_DENSITY"
      - "high" / "HIGH" / "HIGH_DENSITY"
      - "all" / "ALL"

    Returns one of:
      "LOW_DENSITY", "MID_DENSITY", "HIGH_DENSITY", "ALL"
    """
    if selected_density is None:
                return "ALL"

    s = str(selected_density).strip().upper()

    aliases = {
        "LOW": "LOW_DENSITY",
        "LOW_DENSITY": "LOW_DENSITY",
        "MID": "MID_DENSITY",
        "MIDDLE": "MID_DENSITY",
        "MID_DENSITY": "MID_DENSITY",
        "HIGH": "HIGH_DENSITY",
        "HIGH_DENSITY": "HIGH_DENSITY",
        "ALL": "ALL",
    }
    return aliases.get(s, s)


def normalize_noise_selection(selected_noise: str | None) -> str:
    """
    Normalize user input for noise-class selection.

    Accepted:
      - None -> "ALL"
      - "all" / "ALL"
      - any class label, normalized to uppercase with spaces as underscores
    """
    if selected_noise is None:
        return "ALL"

    s = str(selected_noise).strip()
    if not s:
        return "ALL"

    if s.upper() == "ALL":
        return "ALL"

    return s.upper().replace(" ", "_")


def normalize_evaluation_filename(value: str) -> str:
    """
    Normalize an image filename for cross-source matching.

    The extension is removed and matching is case-insensitive.
    """
    return Path(str(value).strip()).stem.lower()


def load_density_groups_from_split_manifest(
    split_manifest_path: str | Path,
    *,
    subset: str = "test",
    density_column: str = "density_class",
) -> dict[str, set[str]]:
    """
    Load density groups from the canonical split manifest.
    """
    manifest_df = pd.read_csv(split_manifest_path)

    required_columns = {
        "filename",
        "subset",
        density_column,
    }
    missing = required_columns - set(manifest_df.columns)

    if missing:
        raise ValueError(
            "Split manifest is missing required column(s): "
            f"{sorted(missing)}"
        )

    subset_df = manifest_df[
        manifest_df["subset"].astype(str).str.lower()
        == str(subset).lower()
    ].copy()

    groups: dict[str, set[str]] = {
        "LOW_DENSITY": set(),
        "MID_DENSITY": set(),
        "HIGH_DENSITY": set(),
    }

    density_mapping = {
        "low": "LOW_DENSITY",
        "low_density": "LOW_DENSITY",
        "mid": "MID_DENSITY",
        "middle": "MID_DENSITY",
        "mid_density": "MID_DENSITY",
        "high": "HIGH_DENSITY",
        "high_density": "HIGH_DENSITY",
    }

    for _, row in subset_df.iterrows():
        raw_density = str(row[density_column]).strip().lower()
        density_key = density_mapping.get(raw_density)

        if density_key is None:
            raise ValueError(
                f"Unsupported density label {row[density_column]!r} "
                f"for filename {row['filename']!r}."
            )

        groups[density_key].add(
            normalize_evaluation_filename(row["filename"])
        )

    return groups


def load_density_groups_from_legacy_csv(
    legacy_density_csv: str | Path,
) -> dict[str, set[str]]:
    """
    Load the frozen historical density classification.

    Expected columns:
        filename
        density_class
    """
    density_df = pd.read_csv(legacy_density_csv)

    required_columns = {"filename", "density_class"}
    missing = required_columns - set(density_df.columns)

    if missing:
        raise ValueError(
            "Legacy density CSV is missing required column(s): "
            f"{sorted(missing)}"
        )

    groups: dict[str, set[str]] = {
        "LOW_DENSITY": set(),
        "MID_DENSITY": set(),
        "HIGH_DENSITY": set(),
    }

    density_mapping = {
        "low": "LOW_DENSITY",
        "low_density": "LOW_DENSITY",
        "mid": "MID_DENSITY",
        "middle": "MID_DENSITY",
        "mid_density": "MID_DENSITY",
        "high": "HIGH_DENSITY",
        "high_density": "HIGH_DENSITY",
    }

    for _, row in density_df.iterrows():
        raw_density = str(row["density_class"]).strip().lower()
        density_key = density_mapping.get(raw_density)

        if density_key is None:
            raise ValueError(
                f"Unsupported legacy density label "
                f"{row['density_class']!r} for "
                f"{row['filename']!r}."
            )

        groups[density_key].add(
            normalize_evaluation_filename(row["filename"])
        )

    return groups


def load_density_groups(
    *,
    density_source: str,
    split_manifest_path: str | Path,
    subset: str = "test",
    density_column: str = "density_class",
    legacy_density_csv: str | Path | None = None,
) -> dict[str, set[str]]:
    source = str(density_source).strip().lower()

    if source == "manifest":
        return load_density_groups_from_split_manifest(
            split_manifest_path,
            subset=subset,
            density_column=density_column,
        )

    if source == "legacy_metadata":
        if legacy_density_csv is None:
            raise ValueError(
                "legacy_density_csv is required when "
                "density_source='legacy_metadata'."
            )

        return load_density_groups_from_legacy_csv(
            legacy_density_csv,
        )

    raise ValueError(
        "density_source must be one of: "
        "'manifest', 'legacy_metadata'. "
        f"Got: {density_source!r}"
    )