from __future__ import annotations

# from visualize_dsb_ap_lenght_based.py
import ast
import os
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# ----------------------------
# NEW: Path resolver 
# ----------------------------
def _resolve_length_eval_csv(model_path: str, length_class: str) -> str | None:
    """
    Resolve the per-image metrics CSV for a given length class.

    Legacy (old):
      <model_path>/ap_dice_results_poly_lenght_class_<lc>.csv

    Option 1 (new standard):
      <model_path>/length_eval/<lc>/dsb_per_image_metrics.csv

    Returns:
      CSV path if it exists, else None.
    """
    legacy = os.path.join(model_path, f"ap_dice_results_poly_lenght_class_{length_class}.csv")
    if os.path.exists(legacy):
        return legacy

    option1 = os.path.join(model_path, "length_eval", length_class, "dsb_per_image_metrics.csv")
    if os.path.exists(option1):
        return option1

    return None


# ----------------------------
# Style + naming
# ----------------------------
def apply_science_style() -> None:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])


def safe_filename(name: str) -> str:
    return re.sub(r"[^\w_.-]", "_", name)

def generate_blue_palette(n: int, *, start: float = 0.35, end: float = 0.90) -> list[str]:
    """
    Generate n distinct blue shades (hex) from matplotlib's 'Blues' colormap.

    start/end control how light/dark the palette is (0..1).
    We avoid too-light colors so bars stay visible on white background.
    """
    if n <= 0:
        return []

    cmap = cm.get_cmap("Blues")
    if n == 1:
        vals = [(start + end) / 2.0]
    else:
        vals = [start + (end - start) * i / (n - 1) for i in range(n)]
    return [mcolors.to_hex(cmap(v)) for v in vals]


def resolve_model_colors(model_keys: list[str], spec: "ModelDisplaySpec") -> list[str]:
    """
    Keep legacy colors when available; for missing colors, assign additional blue shades.
    Deterministic with respect to model_keys order.
    """
    base_colors: list[str | None] = [spec.color(k) for k in model_keys]
    missing_idx = [i for i, c in enumerate(base_colors) if c is None]

    if not missing_idx:
        return [c for c in base_colors if c is not None]  # type: ignore[return-value]

    extra = generate_blue_palette(len(missing_idx))

    out: list[str] = []
    extra_it = iter(extra)
    for c in base_colors:
        out.append(c if c is not None else next(extra_it))
    return out



@dataclass(frozen=True)
class ModelDisplaySpec:
    model_name_map: dict[str, str]
    color_map: dict[str, str]

    def display_name(self, model_key: str) -> str:
        return self.model_name_map.get(model_key, model_key)

    def color(self, model_key: str) -> str | None:
        return self.color_map.get(self.display_name(model_key), None)


# ----------------------------
# Plot 1: dataset AP vs IoU
# ----------------------------
def plot_ap_vs_iou(
    csv_paths: dict[str, str],
    *,
    spec: ModelDisplaySpec,
    output_path: str | None = None,
    fig_size=(8, 6),
) -> None:
    plt.figure(figsize=fig_size)
    model_keys = list(csv_paths.keys())
    colors = resolve_model_colors(model_keys, spec)

    for i, (model_key, csv_file) in enumerate(csv_paths.items()):
        df = pd.read_csv(csv_file).sort_values("IoU_thresh")
        plt.plot(df["IoU_thresh"], df["average_ap"], marker="o", label=model_key, color=colors[i])

    plt.xlabel("IoU Threshold")
    plt.ylabel("Average AP")
    plt.legend()
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()


# ----------------------------
# Plot 2: per-image AP curves
# ----------------------------
def plot_per_image_ap_curve(
    models: dict[str, str],
    *,
    image_filename: str,
    length_class: str,
    spec: ModelDisplaySpec,
    output_dir: str | None = None,
    fig_size=(10, 7),
) -> None:
    plt.figure(figsize=fig_size)

    for model_key, model_path in models.items():
        csv_path = _resolve_length_eval_csv(model_path, length_class)
        if csv_path is None:
            # keep legacy-style message
            legacy_path = os.path.join(model_path, f"ap_dice_results_poly_lenght_class_{length_class}.csv")
            print(f"CSV not found for {model_key}: {legacy_path}")
            continue

        df = pd.read_csv(csv_path)
        row = df[df["filename"] == image_filename]
        if row.empty:
            print(f"Image {image_filename} not found in {model_key}")
            continue

        # IMPORTANT:
        # This plot requires an "APs" column (dict-string).
        # Option 1 dsb_per_image_metrics.csv currently does NOT include APs.
        if "APs" not in df.columns:
            print(f"Column 'APs' not found in {model_key} ({csv_path}). "
                  f"This plot requires legacy per-image APs output.")
            continue

        aps_dict = ast.literal_eval(row["APs"].iloc[0])

        iou_thresholds = sorted(float(k.split("@")[1]) for k in aps_dict.keys())
        ap_values = [aps_dict[f"IoU@{iou:.2f}"] for iou in iou_thresholds]

        color = spec.color(model_key)
        plt.plot(
            iou_thresholds,
            ap_values,
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=model_key,
            color=color,
        )

    plt.xlabel("IoU Threshold", fontsize=12)
    plt.ylabel("Average Precision (AP)", fontsize=12)
    plt.legend(fontsize=10)
    plt.ylim(0, 1.05)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out = os.path.join(output_dir, f"{safe_filename(image_filename)}_AP_curves_{length_class}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved AP curve: {out}")
    else:
        plt.show()


# ----------------------------
# Plot 3: per-image Dice bars
# ----------------------------
def plot_per_image_mean_dice(
    models: dict[str, str],
    *,
    image_filename: str,
    length_class: str,
    spec: ModelDisplaySpec,
    output_dir: str | None = None,
    fig_size=(10, 6),
) -> None:
    model_keys: list[str] = []
    dice_scores: list[float] = []

    for model_key, model_path in models.items():
        csv_path = _resolve_length_eval_csv(model_path, length_class)
        if csv_path is None:
            legacy_path = os.path.join(model_path, f"ap_dice_results_poly_lenght_class_{length_class}.csv")
            print(f"CSV not found for {model_key}: {legacy_path}")
            continue

        df = pd.read_csv(csv_path)
        row = df[df["filename"] == image_filename]
        if row.empty:
            print(f"Image {image_filename} not found in {model_key}")
            continue

        model_keys.append(model_key)
        dice_scores.append(float(row["mean_dice"].iloc[0]))

    if not model_keys:
        print(f"No dice data found for image: {image_filename}")
        return

    plt.figure(figsize=fig_size)

    display_names = [spec.display_name(k) for k in model_keys]
    colors = resolve_model_colors(model_keys, spec)

    bars = plt.bar(model_keys, dice_scores, color=colors)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Mean Dice Coefficient", fontsize=12)
    plt.ylim(0, 1.15)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out = os.path.join(output_dir, f"{safe_filename(image_filename)}_Dice_{length_class}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved Dice comparison: {out}")
    else:
        plt.show()


# ----------------------------
# Plot 4: avg Dice across models
# ----------------------------
def plot_avg_dice_across_models(
    models: dict[str, str],
    *,
    length_class: str,
    spec: ModelDisplaySpec,
    output_dir: str | None = None,
    fig_size=(12, 6),
) -> None:
    model_keys: list[str] = []
    means: list[float] = []
    stds: list[float] = []

    for model_key, model_path in models.items():
        csv_path = _resolve_length_eval_csv(model_path, length_class)
        if csv_path is None:
            legacy_path = os.path.join(model_path, f"ap_dice_results_poly_lenght_class_{length_class}.csv")
            print(f"CSV not found for {model_key}: {legacy_path}")
            continue

        df = pd.read_csv(csv_path)
        dice = df["mean_dice"].dropna()
        if len(dice) == 0:
            print(f"No dice data for {model_key}")
            continue

        model_keys.append(model_key)
        means.append(float(dice.mean()))
        stds.append(float(dice.std()))

    if not model_keys:
        print(f"No dice data found for any model in {length_class}")
        return

    fig, ax = plt.subplots(figsize=fig_size)

    display_names = [spec.display_name(k) for k in model_keys]
    colors = resolve_model_colors(model_keys, spec)

    x = np.arange(len(display_names))

    ax.bar(x, means, yerr=stds, capsize=6, color=colors, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, ha="center", fontsize=20, fontweight="bold")
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylabel("Average Dice Coefficient", fontsize=20, fontweight="bold")

    min_val = min(m - s for m, s in zip(means, stds))
    max_val = max(m + s for m, s in zip(means, stds))
    pad = 0.05 * (max_val - min_val) if max_val > min_val else 0.05
    ax.set_ylim(max(0.0, min_val - pad), min(1.0, max_val + pad))

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.01, f"{m:.3f} ± {s:.3f}", ha="center", va="bottom", fontsize=16, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(False)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out = os.path.join(output_dir, f"Avg_Dice_{safe_filename(length_class)}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved average Dice plot: {out}")
    else:
        plt.show()


# ----------------------------
# Plot 5: noise vs AP
# ----------------------------
def plot_ap_vs_noise(
    models: dict[str, str],
    *,
    noise_csv: str,
    length_classes: Iterable[str],
    spec: ModelDisplaySpec,
    output_dir: str | None = None,
    fig_size=(8, 6),
) -> None:
    noise_df = pd.read_csv(noise_csv)
    noise_df["filename"] = noise_df["filename"].astype(str).str.replace(".tif", ".jpg", regex=False)

    plt.figure(figsize=fig_size)

    for model_key, model_path in models.items():
        parts = []
        for lc in length_classes:
            csv_path = _resolve_length_eval_csv(model_path, lc)
            if csv_path is None:
                legacy_path = os.path.join(model_path, f"ap_dice_results_poly_lenght_class_{lc}.csv")
                print(f"Missing CSV for {model_key} - {lc}: {legacy_path}")
                continue

            df = pd.read_csv(csv_path)[["filename", "mAP"]]
            df["filename"] = df["filename"].astype(str)
            parts.append(df)

        if not parts:
            print(f"No AP data found for {model_key}")
            continue

        combined = pd.concat(parts).drop_duplicates("filename")
        merged = pd.merge(noise_df, combined, on="filename", how="inner")
        if merged.empty:
            print(f"No matching images for noise vs AP for {model_key}")
            continue

        plt.scatter(merged["noise_std"], merged["mAP"], label=model_key, alpha=0.7)

    plt.xlabel("Image Noise (std)")
    plt.ylabel("AP")
    plt.legend()
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out = os.path.join(output_dir, "Noise_vs_mAP.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved noise vs mAP plot: {out}")
    else:
        plt.show()
