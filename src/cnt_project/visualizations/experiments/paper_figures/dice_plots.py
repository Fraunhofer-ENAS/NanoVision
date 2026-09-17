import seaborn as sns
import matplotlib.pyplot as plt

def plot_dice_boxstrip(
    df,
    color_map: dict,
    order: list[str],
    figsize=(3.3, 2.5),
    ylim=(0.25, 0.8),
    ylabel="Average DICE Coefficient",
    save_path=None
):
    """
    EXACT reproduction of the DICE box+strip plot from the original notebook:
    - SciencePlots style must already be active
    - boxplot background with alpha=0.4
    - black box edges
    - stripplot foreground
    - exact colors and order
    - y-grid dashed
    - identical vector rendering behavior
    """
    plt.figure(figsize=figsize)

    # BOX PLOT (background layer)
    ax = sns.boxplot(
        data=df,
        x='experiment',
        y='avg_dice',
        order=order,
        palette=color_map,
        width=0.6,
        showfliers=False,
        boxprops={'alpha': 0.4},
    )

    # Make box edges black
    for patch in ax.artists:
        patch.set_edgecolor('black')
        patch.set_linewidth(1)

    # STRIP PLOT (foreground layer)
    sns.stripplot(
        data=df,
        x='experiment',
        y='avg_dice',
        order=order,
        hue='experiment',
        dodge=False,
        jitter=False,
        size=5,
        alpha=0.5,
        palette=color_map,
        legend=False,
        edgecolor="gray",
        linewidth=0.1,
    )

    # Labeling and style
    plt.xlabel("")
    plt.ylabel(ylabel, labelpad=10, weight='bold')
    plt.ylim(*ylim)
    plt.tick_params(axis='x', pad=10)
    plt.grid(axis='y', linestyle='--', alpha=0.2)

    # Force vector rendering for swarm dots (same as original)
    for coll in plt.gca().collections:
        coll.set_rasterized(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.02)

    plt.show()
