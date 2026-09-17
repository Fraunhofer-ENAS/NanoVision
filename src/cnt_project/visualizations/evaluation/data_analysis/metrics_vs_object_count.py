import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_metrics_vs_object_count(csv_path: str, save_dir: str, tag: str = "") -> tuple[str, str]:
    df = pd.read_csv(csv_path)

    required = ["num_gt_objects", "mAP", "mean_dice"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    os.makedirs(save_dir, exist_ok=True)
    sns.set(style="white")

    suffix = f"_{tag}" if tag else ""

    # mAP
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="num_gt_objects", y="mAP")
    sns.lineplot(data=df, x="num_gt_objects", y="mAP", linewidth=1, estimator="mean", ci=None)
    plt.xlabel("Number of Objects")
    plt.ylabel("mAP")
    plt.title("mAP vs Number of Objects")
    plt.tight_layout()
    out_map = os.path.join(save_dir, f"map_vs_num_objects{suffix}.png")
    plt.savefig(out_map)
    plt.close()

    # Dice
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="num_gt_objects", y="mean_dice")
    sns.lineplot(data=df, x="num_gt_objects", y="mean_dice", linewidth=1, estimator="mean", ci=None)
    plt.xlabel("Number of Objects")
    plt.ylabel("Mean Dice Coefficient")
    plt.title("Dice vs Number of Objects")
    plt.tight_layout()
    out_dice = os.path.join(save_dir, f"dice_vs_num_objects{suffix}.png")
    plt.savefig(out_dice)
    plt.close()

    return out_map, out_dice
