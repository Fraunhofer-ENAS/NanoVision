from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any


def load_coco(json_path: str | os.PathLike) -> dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def index_coco_by_image(
    coco: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    images_by_id = {int(image["id"]): image for image in coco.get("images", [])}
    annotations_by_image_id: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for annotation in coco.get("annotations", []):
        annotations_by_image_id[int(annotation["image_id"])] .append(annotation)

    return images_by_id, annotations_by_image_id


def get_annotations_for_image_id(coco: dict[str, Any], image_id: int) -> list[dict[str, Any]]:
    return [
        annotation
        for annotation in coco.get("annotations", [])
        if int(annotation.get("image_id", -1)) == int(image_id)
    ]