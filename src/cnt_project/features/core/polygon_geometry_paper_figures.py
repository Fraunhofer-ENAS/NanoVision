from pathlib import Path
import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

MICRONS_PER_PIXEL = 5.0 / 256.0

def get_length_width_elongation(polygon_coords: List[float]) -> tuple[float, float, float]:
    """Estimate length, width and elongation using minimum rotated rectangle."""
    coords = np.array(polygon_coords).reshape(-1, 2)
    if len(coords) < 3:
        return 0.0, 0.0, 0.0
    poly = Polygon(coords)
    if (not poly.is_valid) or poly.is_empty:
        return 0.0, 0.0, 0.0

    min_rect = poly.minimum_rotated_rectangle
    rect = np.array(min_rect.exterior.coords)[:-1]
    edges = [np.linalg.norm(rect[i] - rect[(i + 1) % 4]) for i in range(4)]
    length, width = max(edges), min(edges)
    elong = (length / width) if width > 0 else 0.0
    return float(length), float(width), float(elong)

# TODO this is the function used originally to extract the width of the polygon for the precision vs width 
# noise vs width ...etc we need to update it to use agreed upon width definition
def extract_polygon_geometry_stats_from_pred(pred_json_path: str | Path) -> "pd.DataFrame":
    """
    - Parse COCO predicted polygons
    - Compute length/width via minimum rotated rectangle (get_length_width_elongation)
    - Aggregate per image: mean_width_px, mean_length_px, n_objects
    """
    import pandas as pd
    from pathlib import Path as _Path
    p = _Path(pred_json_path)
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {p}")

    with open(p, "r") as f:
        data = json.load(f)

    id_to_file = {im["id"]: im["file_name"] for im in data.get("images", [])}
    per_image: Dict[int, Dict[str, list]] = {}

    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        for seg in ann.get("segmentation", []):
            L, W, _ = get_length_width_elongation(seg)
            if img_id not in per_image:
                per_image[img_id] = {"widths": [], "lengths": []}
            per_image[img_id]["widths"].append(W)
            per_image[img_id]["lengths"].append(L)

    rows: List[Dict[str, Any]] = []
    for img_id, d in per_image.items():
        fn = id_to_file.get(img_id, "unknown")
        rows.append({
            "image_id": fn.split(".")[0].lower(),
            "mean_width_px": float(np.mean(d["widths"])) if d["widths"] else 0.0,
            "mean_length_px": float(np.mean(d["lengths"])) if d["lengths"] else 0.0,
            "n_objects": int(len(d["widths"]))
        })

    return pd.DataFrame(rows)