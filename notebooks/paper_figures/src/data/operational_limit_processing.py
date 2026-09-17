from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Polygon as ShapelyPolygon, LineString

from ..utils.constants import MICRONS_PER_PIXEL, IMAGE_HEIGHT, IMAGE_WIDTH
from ..data.loading_json import read_json
from typing import Iterable, Dict, Any, List, Optional
import re


# ---------- Core helpers ----------

def _rows_to_check(rows: Iterable[int] | None) -> tuple[int, ...]:
    return tuple(rows) if rows else (58, 117, 187)

def _norm_stem(name: str) -> str:
    # strip extension and lowercase so .jpg/.png match
    return re.sub(r"\.(png|jpg|jpeg)$", "", name, flags=re.IGNORECASE)

# ---------- GT: count polygon intersections per selected rows ----------

def compute_gt_row_intersections(
    gt_annotations_json: str | Path,
    filename: str,
    rows: Iterable[int] | None = None,
) -> Dict[str, Any]:
    """
    For one image (by filename) in GT COCO polygons, count how many objects intersect
    each requested horizontal row.
    Returns: {"file_name": <str>, "row_58_masks": int, "row_117_masks": int, "row_187_masks": int}
    """
    rows = _rows_to_check(rows)
    data = read_json(gt_annotations_json)

    # Find image entry
    images = data.get("images", [])
    anns = data.get("annotations", [])

    image = next((im for im in images if im.get("file_name") == filename), None)
    if image is None:
        raise ValueError(f"Filename '{filename}' not found in GT images.")
    img_id = image["id"]
    W = image.get("width", IMAGE_WIDTH)
    H = image.get("height", IMAGE_HEIGHT)

    # Annotations for that image
    image_anns = [a for a in anns if a.get("image_id") == img_id]

    # Build shapely polygons
    polys: List[ShapelyPolygon] = []
    for ann in image_anns:
        for seg in ann.get("segmentation", []):
            arr = np.array(seg).reshape(-1, 2)
            polys.append(ShapelyPolygon(arr))

    out = {"file_name": filename}
    for r in rows:
        line = LineString([(0, r), (W, r)])
        count = sum(p.intersects(line) for p in polys)
        out[f"row_{r}_masks"] = int(count)

    return out

# ---------- Predictions: load line-density per row (AI tool CSV) ----------

def load_predicted_densities(csv_path: str | Path, rows: Iterable[int]) -> pd.DataFrame:
    """
    Load line_density_per_row.csv, keep angle_degrees == 0, pivot to get the three rows
    into columns named row_<r>_density. This matches the original notebook.
    """
    rows = list(rows)
    df = pd.read_csv(csv_path)

    # filter angle 0 (exactly as in the notebook)
    if "angle_degrees" not in df.columns:
        raise ValueError("CSV must contain 'angle_degrees' column.")
    df = df[df["angle_degrees"] == 0].copy()

    # pivot to columns = line_position_index
    if {"filename", "line_position_index", "line_density"} - set(df.columns):
        missing = {"filename", "line_position_index", "line_density"} - set(df.columns)
        raise ValueError(f"CSV is missing required columns: {missing}")

    pivot = df.pivot_table(
        index="filename",
        columns="line_position_index",
        values="line_density",
        aggfunc="first"
    )

    # make sure requested rows exist
    missing_rows = [r for r in rows if r not in pivot.columns]
    if missing_rows:
        raise ValueError(f"Missing requested rows in CSV: {missing_rows}. "
                         f"Available: {sorted(map(int, pivot.columns))}")

    out = pivot[rows].reset_index()
    out.columns = ["filename"] + [f"row_{r}_density" for r in rows]
    return out

# newer runner based feature extraction
def load_image_level_predicted_density(
    csv_path: str | Path,
) -> pd.DataFrame:
    """
    Load image-level predicted line density.

    Expected columns:
      - image_file
      - line_density_mean

    line_density_mean is the mean number of CNT intersections
    across image rows. For a 5 µm image width, divide by 5 to
    obtain CNT/µm.
    """
    df = pd.read_csv(csv_path)

    required = {
        "image_file",
        "line_density_mean",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    out = pd.DataFrame(
        {
            "filename": df["image_file"].astype(str),
            "mean_line_density_image": pd.to_numeric(
                df["line_density_mean"],
                errors="raise",
            ),
        }
    )

    out["line_density_per_micrometer"] = (
        out["mean_line_density_image"] / 5.0
    )

    return out
# ---------- Build test-set structured table (GT vs predictions, three rows) ----------
def build_test_set_structured(
    gt_json: str | Path,
    line_density_csv: str | Path,
    rows: Iterable[int] | None = None,
) -> pd.DataFrame:
    """
    Build the structured test-set table used for operational-limit analysis.

    For every requested row:
      - mean_<row>: predicted line density
      - manual_mean<row>: GT-derived line density

    Image-level quantities are calculated as the mean across all requested rows.

    Examples
    --------
    rows=(58, 117, 187)
        Reproduces the original 3-line analysis.

    rows=range(256)
        Uses all 256 image rows.
    """
    from cnt_project.features.core.cnt_feature_extraction import (
    compute_line_densities_from_polygons,
    )

    rows = tuple(_rows_to_check(rows))

    # ---------------- Load GT JSON ----------------
    gt = read_json(gt_json)
    images = gt.get("images", [])
    anns = gt.get("annotations", [])

    # Group annotations by image_id
    anns_by_image: Dict[int, List[Dict[str, Any]]] = {}

    for a in anns:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    # ---------------- GT extraction ----------------
    gt_rows: List[Dict[str, Any]] = []

    for im in images:
        fname = im["file_name"]
        img_id = im["id"]

        W = int(im.get("width", IMAGE_WIDTH))
        H = int(im.get("height", IMAGE_HEIGHT))

        line_densities = compute_line_densities_from_polygons(
            annotations=anns_by_image.get(img_id, []),
            image_height=H,
            image_width=W,
            # min_object_size=2,
        )

        rec: Dict[str, Any] = {
            "filename": fname,
        }

        for r in rows:
            rec[f"manual_mean{r}"] = line_densities[r]

        gt_rows.append(rec)

    test_set_gt = pd.DataFrame(gt_rows)
    test_set_gt["file_base"] = test_set_gt["filename"].apply(_norm_stem)

    # ---------------- Model predictions ----------------
    pred = load_predicted_densities(
        line_density_csv,
        rows=rows,
    )

    pred["file_base"] = pred["filename"].apply(_norm_stem)

    # ---------------- Merge GT + prediction ----------------
    m = pd.merge(
        test_set_gt,
        pred,
        on="file_base",
        how="inner",
    )

    # ---------------- Build output efficiently ----------------
    data: Dict[str, Any] = {
        "filename": m["file_base"] + ".png",
        "run_nbr": 99,
    }

    for r in rows:
        data[f"mean_{r}"] = m[f"row_{r}_density"].to_numpy()
        data[f"manual_mean{r}"] = m[f"manual_mean{r}"].to_numpy()

    out = pd.DataFrame(data)

    model_columns = [
        f"mean_{r}"
        for r in rows
    ]

    manual_columns = [
        f"manual_mean{r}"
        for r in rows
    ]

    out["mean_line_density_image"] = (
        out[model_columns].mean(axis=1)
    )

    out["manual_mean_line_density_image"] = (
        out[manual_columns].mean(axis=1)
    )

    out["line_density_per_micrometer"] = (
        out["mean_line_density_image"] / 5.0
    )

    out["manual_line_density_per_micrometer"] = (
        out["manual_mean_line_density_image"] / 5.0
    )

    return out

def build_gt_image_level_density(
    gt_json: str | Path,
    rows: Iterable[int] | None = None,
) -> pd.DataFrame:
    """
    Compute image-level GT line density from COCO annotations.

    When rows=range(256), the result is the mean GT line density
    across the complete image height.
    """
    from cnt_project.features.core.cnt_feature_extraction import (
        compute_line_densities_from_polygons,
    )

    rows = tuple(_rows_to_check(rows))

    gt = read_json(gt_json)
    images = gt.get("images", [])
    anns = gt.get("annotations", [])

    anns_by_image: Dict[int, List[Dict[str, Any]]] = {}

    for ann in anns:
        anns_by_image.setdefault(
            ann["image_id"],
            [],
        ).append(ann)

    records = []

    for im in images:
        filename = str(im["file_name"])
        image_id = im["id"]

        width = int(
            im.get("width", IMAGE_WIDTH)
        )
        height = int(
            im.get("height", IMAGE_HEIGHT)
        )

        line_densities = compute_line_densities_from_polygons(
            annotations=anns_by_image.get(image_id, []),
            image_height=height,
            image_width=width,
            # min_object_size=2,
        )

        selected_values = [
            line_densities[r]
            for r in rows
        ]

        mean_density = float(
            np.mean(selected_values)
        )

        records.append(
            {
                "filename": filename,
                "manual_mean_line_density_image": mean_density,
                "manual_line_density_per_micrometer": mean_density / 5.0,
            }
        )

    return pd.DataFrame(records)
# ---------- plug-in high-density sources (Martin Excel + KI CSV) ----------
# ----------------------------------------------------------------------
# 1) Load Martin’s manual Excel (HIGH-DENSITY manual annotations)
# ----------------------------------------------------------------------
def load_high_density_martin_excel(excel_path: str | Path) -> pd.DataFrame:
    """
    Load Martin’s Excel file and format it exactly like the original notebook.
    Builds correct filename, renames columns, and returns clean manual subset.
    """

    excel_path = Path(excel_path)

    df = pd.read_excel(excel_path, header=1)

    # The filename is split across 3 columns → must rebuild
    df = df.rename(columns={"Unnamed: 2": "basename"})

    df = df[[
        "basename", "column", "row",
        "CNT amount 1", "CNT amount 2", "CNT amount 3",
        "crop pic average"
    ]].dropna(subset=["basename", "column", "row"])

    # Build filename EXACTLY like original:
    df["filename"] = (
        df["basename"].astype(str)
        + "_"
        + df["column"].astype(str)
        + "_"
        + df["row"].astype(str)
        + ".png"
    )

    # Compute mean CNT count
    df["manual_cnt_mean"] = df[["CNT amount 1", "CNT amount 2", "CNT amount 3"]].mean(axis=1)

    # Rename columns to unified names
    df = df.rename(columns={
        "CNT amount 1": "manual_mean58",
        "CNT amount 2": "manual_mean117",
        "CNT amount 3": "manual_mean187",
        "crop pic average": "manual_line_density_per_micrometer",
        "manual_cnt_mean": "manual_mean_line_density_image"
    })

    # Return final cleaned columns
    manual_subset = df[[
        "filename",
        "manual_mean58",
        "manual_mean117",
        "manual_mean187",
        "manual_line_density_per_micrometer",
        "manual_mean_line_density_image"
    ]]

    return manual_subset


# ----------------------------------------------------------------------
# 2) Load KI Tool image-level CSV
# ----------------------------------------------------------------------
def load_high_density_pred_rows(
    csv_path: str | Path,
    rows: Iterable[int] = (58, 117, 187)
) -> pd.DataFrame:
    """
    Load high-density model predictions from line_density_per_row.csv.
    This mirrors exactly the logic used for the test set:
    - Keep only angle_degrees == 0
    - Extract densities at specific rows (58,117,187)
    - Compute mean_line_density_image = mean of these 3 rows
    - Compute line_density_per_micrometer = /5
    """
    import pandas as pd
    import numpy as np

    rows = list(rows)
    csv_path = Path(csv_path)

    # Load CSV
    df = pd.read_csv(csv_path)

    # Keep only angle == 0 (same filter as original notebook)
    if "angle_degrees" in df.columns:
        df = df[df["angle_degrees"] == 0].copy()

    # Pivot (filename, row_index) -> line_density
    pivot = (
        df.pivot_table(
            index="filename",
            columns="line_position_index",
            values="line_density",
            aggfunc="first"
        )
        .reset_index()
    )

    # Ensure requested rows exist
    missing = [r for r in rows if r not in pivot.columns]
    if missing:
        raise ValueError(
            f"Missing row(s) in high-density CSV: {missing}. "
            f"Available columns: {list(pivot.columns)}"
        )

    # Rename: row 58 → mean_58 etc.
    for r in rows:
        pivot.rename(columns={r: f"mean_{r}"}, inplace=True)

    # Compute mean of the 3 rows
    pivot["mean_line_density_image"] = pivot[[f"mean_{r}" for r in rows]].mean(axis=1)

    # CNT/µm conversion
    pivot["line_density_per_micrometer"] = pivot["mean_line_density_image"] / 5.0

    # Add run_nbr to match schema
    pivot["run_nbr"] = 99

    # Keep only final needed columns
    ordered_cols = (
        ["filename", "run_nbr"]
        + [f"mean_{r}" for r in rows]
        + ["mean_line_density_image", "line_density_per_micrometer"]
    )

    return pivot[ordered_cols].copy()


# ----------------------------------------------------------------------
# 3) Merge everything
# ----------------------------------------------------------------------

def combine_test_and_high_density(
    test_set_structured: pd.DataFrame,
    high_density_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine:
    - Test set structured data (using rows 58/117/187)
    - High-density dataset (manual + KI row-based predictions)

    Expected final unified columns:
        filename
        run_nbr
        mean_58, mean_117, mean_187
        manual_mean58, manual_mean117, manual_mean187
        mean_line_density_image
        manual_mean_line_density_image
        line_density_per_micrometer
        manual_line_density_per_micrometer
        dataset_type
    """

    # -------------------------------
    # 1. Select columns from TEST set
    # -------------------------------
    test_cols = [
        "filename",
        "run_nbr",
        "mean_58",
        "mean_117",
        "mean_187",
        "manual_mean58",
        "manual_mean117",
        "manual_mean187",
        "mean_line_density_image",
        "manual_mean_line_density_image",
        "line_density_per_micrometer",
        "manual_line_density_per_micrometer",
        "dataset_type",
    ]

    df_test = test_set_structured[test_cols].copy()

    # ---------------------------------------
    # 2. Select columns from HIGH-DENSITY set
    # ---------------------------------------
    high_cols = [
        "filename",
        "run_nbr",
        "mean_58",
        "mean_117",
        "mean_187",
        "manual_mean58",
        "manual_mean117",
        "manual_mean187",
        "mean_line_density_image",
        "manual_mean_line_density_image",
        "line_density_per_micrometer",
        "manual_line_density_per_micrometer",
        "dataset_type",
    ]

    df_high = high_density_df[high_cols].copy()

    # -----------------------
    # 3. Concatenate datasets
    # -----------------------
    df_all = pd.concat([df_test, df_high], ignore_index=True)

    # Confirm correct dtypes
    df_all["dataset_type"] = df_all["dataset_type"].astype("string")

    return df_all


def prepare_operational_limit_dataframe(df_all):
    df = df_all.copy()

    # CNT per 256px → CNT per 5 µm
    df["manual_density_um"] = df["manual_mean_line_density_image"] / 5.0

    # predictions are already per µm
    df["predicted_density_um"] = df["line_density_per_micrometer"]

    df["density_error_um"] = (df["manual_density_um"] - df["predicted_density_um"]).abs()

    return df