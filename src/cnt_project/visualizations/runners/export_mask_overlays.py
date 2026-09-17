from __future__ import annotations

import os
from glob import glob
from tifffile import imread

from cnt_project.io.paths import ProjectPaths
from cnt_project.visualizations.overlays.masks import save_image_with_mask



def main() -> None:
    paths = ProjectPaths.from_here(__file__)
    paths.ensure_outputs()

    image_dir = str(paths.test_images_root)
    mask_dir = str(paths.test_masks_root)
    save_dir = str(paths.misc_dir("mask_overlays"))
    image_ext = ".tif"  # same ext for images + masks in this dataset
    alpha = 0.5

    os.makedirs(save_dir, exist_ok=True)

    # --- LOAD FILES ---
    image_files = sorted(glob(os.path.join(image_dir, f"*{image_ext}")))
    mask_files = {
        os.path.splitext(os.path.basename(f))[0]: f
        for f in glob(os.path.join(mask_dir, f"*{image_ext}"))
    }

    # --- MAIN LOOP ---
    for image_path in image_files:
        fname = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = mask_files.get(fname)

        if not mask_path or not os.path.exists(mask_path):
            print(f"[!] No matching mask for image: {fname}")
            continue

        image = imread(image_path)
        mask = imread(mask_path)

        if image.shape[:2] != mask.shape[:2]:
            print(f"[!] Shape mismatch: {fname} image={image.shape} mask={mask.shape}")
            continue

        out_path = os.path.join(save_dir, f"{fname}.png")
        save_image_with_mask(image, mask, save_path=out_path, alpha=alpha)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
