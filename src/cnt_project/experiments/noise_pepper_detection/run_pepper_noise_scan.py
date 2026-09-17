from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cnt_project.io.paths import ProjectPaths


def detect_pepper_noise(gray: np.ndarray) -> np.ndarray:
    """
    Detect 'pepper' noise candidates via:
      - median blur
      - abs diff
      - Otsu threshold
      - keep only dark pixels (gray < 50)
    Returns uint8 mask in {0,255}.
    """
    median = cv2.medianBlur(gray, 3)
    diff = cv2.absdiff(gray, median)

    _thr, noise_mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    pepper_mask = np.logical_and(noise_mask == 255, gray < 50)
    return (pepper_mask.astype(np.uint8) * 255)


def process_image(image_path: str, output_folder: str) -> float | None:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise_mask = detect_pepper_noise(gray)

    noise_pixels = int(np.sum(noise_mask == 255))
    total_pixels = int(gray.size)
    noise_percent = (noise_pixels / total_pixels) * 100.0

    marked_img = img.copy()
    marked_img[noise_mask == 255] = [0, 0, 255]  # BGR red

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2.imshow(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"Pepper Noise: {noise_percent:.4f}%")
    ax2.axis("off")

    plt.tight_layout()

    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_folder, f"{stem}_pepper.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    return float(noise_percent)


def process_folder(input_folder: str, output_folder: str) -> pd.DataFrame:
    os.makedirs(output_folder, exist_ok=True)

    rows = []
    for filename in sorted(os.listdir(input_folder)):
        if not filename.lower().endswith(".jpg"):
            continue

        image_path = os.path.join(input_folder, filename)
        noise_level = process_image(image_path, output_folder)
        if noise_level is None:
            continue

        rows.append({"filename": filename, "pepper_noise_percent": noise_level})

    df = pd.DataFrame(rows)

    # summary txt (legacy-style)
    summary_path = os.path.join(output_folder, "noise_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Image Noise Summary (Pepper Noise Percentage)\n")
        f.write("=" * 50 + "\n")
        for r in rows:
            f.write(f"{r['filename']}: {r['pepper_noise_percent']:.4f}%\n")

        avg_noise = float(df["pepper_noise_percent"].mean()) if len(df) else 0.0
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"AVERAGE NOISE LEVEL: {avg_noise:.4f}%\n")

    # also save CSV (non-invasive, useful artifact)
    csv_path = os.path.join(output_folder, "noise_summary.csv")
    df.to_csv(csv_path, index=False)

    print(f"Saved: {summary_path}")
    print(f"Saved: {csv_path}")
    return df


def main() -> None:
    paths = ProjectPaths.from_here(__file__)
    paths.ensure_outputs()

    input_folder = str(paths.test_images_root)
    output_folder = str(paths.misc_dir("noise_pepper_detection"))

    df = process_folder(input_folder, output_folder)
    print(f"Processed {len(df)} images. Results saved to {output_folder}")


if __name__ == "__main__":
    main()
