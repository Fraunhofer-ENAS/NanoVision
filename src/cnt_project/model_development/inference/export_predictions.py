from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

from cnt_project.coco.convert import (
    build_coco_from_stardist_label_predictions,
    export_polygon_predictions_to_coco_json,
    export_polygon_predictions_to_coco_rle_json,
)


@dataclass(frozen=True)
class CocoPredictionExport:
    """
    Paths produced when exporting inference predictions to COCO.
    """

    polygon_json_path: Path
    rle_json_path: Path

def export_label_predictions_to_coco_json(
    *,
    y_preds: list[Any],
    filenames: list[str],
    scores: list[Any],
    out_json_path: str | Path,
    category_id: int = 1,
    score_thresh: float | None = None,
) -> Path:
    """
    Export instance-label predictions to polygon-style COCO JSON.
    """
    output_path = Path(out_json_path)

    coco_data = build_coco_from_stardist_label_predictions(
        y_preds=y_preds,
        filenames=filenames,
        scores=scores,
        category_id=category_id,
        score_thresh=score_thresh,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            coco_data,
            indent=2,
        ),
        encoding="utf-8",
    )

    return output_path


def export_predictions_to_coco_json(
    *,
    images: list[Any],
    filenames: list[str],
    polygons: list[list[Any]],
    scores: list[Any],
    out_json_path: str | Path,
    out_rle_json_path: str | Path | None = None,
    category_id: int = 1,
) -> CocoPredictionExport:
    """
    Export polygon predictions to polygon and RLE COCO JSON files.
    """
    polygon_json_path = Path(out_json_path)

    rle_json_path = (
        Path(out_rle_json_path)
        if out_rle_json_path is not None
        else polygon_json_path.with_name(
            f"{polygon_json_path.stem}_rle.json"
        )
    )

    export_polygon_predictions_to_coco_json(
        images=images,
        filenames=filenames,
        polygons=polygons,
        scores=scores,
        output_json_path=polygon_json_path,
        category_id=category_id,
    )

    export_polygon_predictions_to_coco_rle_json(
        images=images,
        filenames=filenames,
        polygons=polygons,
        scores=scores,
        output_json_path=rle_json_path,
        category_id=category_id,
    )

    return CocoPredictionExport(
        polygon_json_path=polygon_json_path,
        rle_json_path=rle_json_path,
    )