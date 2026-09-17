from __future__ import annotations

from pathlib import Path

from pycocotools.coco import COCO


def filter_predictions_by_selected_annotations(predicted_data: COCO, selected_annotations: COCO):
    selected_ann_ids = selected_annotations.getAnnIds()
    selected_anns = selected_annotations.loadAnns(selected_ann_ids)
    selected_img_ids = list({ann["image_id"] for ann in selected_anns})

    predicted_anns = predicted_data.loadAnns(selected_ann_ids)
    predicted_imgs = predicted_data.loadImgs(selected_img_ids)

    return {
        "images": predicted_imgs,
        "annotations": predicted_anns,
        "categories": predicted_data.dataset.get("categories", []),
    }


def subselect_coco_by_filenames(coco_obj, target_filenames):
    target_values = [str(name).strip() for name in target_filenames if str(name).strip()]
    target_name_set = {Path(name).name.lower() for name in target_values}
    target_stem_set = {Path(name).stem.lower() for name in target_values}
    target_contains = [name.lower() for name in target_values]

    def _matches_target(file_name: str) -> bool:
        file_name_l = str(file_name).strip().lower()
        file_stem_l = Path(file_name_l).stem

        if file_name_l in target_name_set or file_stem_l in target_stem_set:
            return True

        # Backward-compatible fallback for legacy partial filename selectors.
        return any(token in file_name_l for token in target_contains)

    filtered_img_ids = [
        img["id"]
        for img in coco_obj.dataset["images"]
        if _matches_target(img["file_name"])
    ]

    filtered_images = coco_obj.loadImgs(filtered_img_ids)
    filtered_annotations = coco_obj.loadAnns(coco_obj.getAnnIds(imgIds=filtered_img_ids))
    filtered_categories = coco_obj.loadCats(coco_obj.getCatIds())

    return {
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": filtered_categories,
    }