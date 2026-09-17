import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

color_map = {
    'fluo_new_edt': '#4169e1',        # NanoVision Royal Blue
    'detectron2_trial_1': '#e34234',  # Vermilion
    'nano1D': '#50c878',              # Emerald Green
    'wormswin': '#9966cc'             # Amethyst Purple
}
gt = "#f0e68c"
orig_gt = "#706A6A"

def plot_combined_histogram_temp(data_true=None, data_pred=None, title="Histogram", xlabel="Value", ylabel="Density",
                            filename="histogram.png", save_folder=".", 
                            color_true="#706A6A", color_pred="#4169e1"):
    """
    Plot a combined histogram of ground truth and predicted data with scientific styling.
    Uses LaTeX fonts and scienceplots style for research publications.

    Args:
    - data_true: Ground truth data (optional).
    - data_pred: Predicted data (required).
    - title: Title of the plot (default: "Histogram").
    - xlabel: Label for the x-axis (default: "Value").
    - ylabel: Label for the y-axis (default: "Density").
    - filename: Name of the output file (default: "histogram.png").
    - save_folder: Destination folder to save the image (default: current directory).
    - color_true: Color for ground truth histogram (default: '#4d4d4d')
    - color_pred: Color for prediction histogram (default: '#9966cc')

    Returns:
    - A dictionary containing the mean and standard deviation of the input data.
    """
    # Ensure predictions are provided
    if data_pred is None:
        raise ValueError("data_pred is required to plot the histogram.")

    # Ensure the destination folder exists
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, filename)

    # Set up scientific plotting style
    plt.style.use(['science', 'ieee'])
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'text.usetex': True,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 300,
        'figure.figsize': (3.5, 2.5),
    })

    # Create figure with constrained layout
    fig, ax = plt.subplots(constrained_layout=True)
    results = {}
    
    # Calculate common bin edges based on combined data range
    all_data = data_pred if data_true is None else np.concatenate([data_true, data_pred])
    data_min = np.nanmin(all_data)
    data_max = np.nanmax(all_data)
    
    # Handle case where all values are identical
    if data_max - data_min < 1e-10:
        data_min -= 0.5
        data_max += 0.5
    
    # Use 60 bins with consistent edges across both distributions
    bin_edges = np.linspace(data_min, data_max, 61)  

    # Prepare legend handles and labels
    legend_handles = []
    legend_labels = []

    # Plot Ground Truth with common bins
    if data_true is not None:
        # Use density normalization for histogram
        hist_true = sns.histplot(
            data_true, bins=bin_edges, color=color_true,
            stat="density", alpha=0.7, ax=ax,  # Changed to density
            edgecolor='black', linewidth=0.5
        )
        
        # Calculate statistics
        mean_true = np.mean(data_true)
        std_true = np.std(data_true)
        
        # Add to results
        results["mean_true"] = mean_true
        results["std_true"] = std_true
        
        # Create legend entry
        true_label = f'Ground Truth\n$\mu={mean_true:.2f}$, $\sigma={std_true:.2f}$'
        legend_handles.append(plt.Rectangle((0,0), 1, 1, fc=color_true, alpha=0.7, 
                                           edgecolor='black', linewidth=0.5))
        legend_labels.append(true_label)

    # Plot Prediction with same bins
    # Use density normalization for histogram
    hist_pred = sns.histplot(
        data_pred, bins=bin_edges, color=color_pred,
        stat="density", alpha=0.7, ax=ax,  # Changed to density
        edgecolor='black', linewidth=0.5
    )
    
    # Calculate statistics
    mean_pred = np.mean(data_pred)
    std_pred = np.std(data_pred)
    
    # Add to results
    results["mean_pred"] = mean_pred
    results["std_pred"] = std_pred
    
    # Create legend entry
    pred_label = f'Prediction\n$\mu={mean_pred:.2f}$, $\sigma={std_pred:.2f}$'
    legend_handles.append(plt.Rectangle((0,0), 1, 1, fc=color_pred, alpha=0.7, 
                                       edgecolor='black', linewidth=0.5))
    legend_labels.append(pred_label)

    # Add KDE lines - use common bandwidth for consistent scaling
    bw_adjust = 0.8  # Bandwidth adjustment factor
    if data_true is not None:
        sns.kdeplot(
            data_true, color=color_true, ax=ax, 
            linewidth=1.0, alpha=1.0, bw_adjust=bw_adjust
        )
    sns.kdeplot(
        data_pred, color=color_pred, ax=ax, 
        linewidth=1.0, alpha=1.0, bw_adjust=bw_adjust
    )

    # Add vertical lines for means
    if data_true is not None:
        ax.axvline(mean_true, color=color_true, linestyle='--', linewidth=1.2)
    ax.axvline(mean_pred, color=color_pred, linestyle='--', linewidth=1.2)

    # Set axis labels and styling
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)  # Added title
    
    # Create custom legend with statistics
    ax.legend(legend_handles, legend_labels, frameon=False, 
              loc='best', handlelength=1.5, handleheight=1.5)
    
    sns.despine(ax=ax)
    ax.grid(False)

    # Save high-resolution output
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)

    return results


def plot_combined_histogram(data_true=None, data_pred=None, title="Histogram", xlabel="Value",
                             filename_base="histogram", save_folder=".", 
                             color_true="#f0e68c", color_pred="#4169e1"):
    """
    Plot two separate histograms:
    1. Density normalization with KDE
    2. Probability normalization without KDE
    
    Returns:
    - A dictionary containing the mean and standard deviation of the input data.
    """
    # Ensure predictions are provided
    if data_pred is None:
        raise ValueError("data_pred is required to plot the histogram.")

    # Ensure the destination folder exists
    os.makedirs(save_folder, exist_ok=True)
    results = {}
    
    # Calculate common bin edges based on combined data range
    all_data = data_pred if data_true is None else np.concatenate([data_true, data_pred])
    data_min = np.nanmin(all_data)
    data_max = np.nanmax(all_data)
    
    # Handle case where all values are identical
    if data_max - data_min < 1e-10:
        data_min -= 0.5
        data_max += 0.5
    
    # Use 60 bins with consistent edges across both distributions
    bin_edges = np.linspace(data_min, data_max, 61)
    bin_width = bin_edges[1] - bin_edges[0]

    # Scientific plotting style setup (do this once)
    plt.style.use(['science', 'ieee'])
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'text.latex.preamble': r'\usepackage{amsmath}',  # Add amsmath package
        'text.usetex': True,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 300,
        'figure.figsize': (3.5, 2.5),
    })
    
    # ========================================================================
    # Plot 1: Density with KDE
    # ========================================================================
    fig1, ax1 = plt.subplots(constrained_layout=True)
    
    # Plot Ground Truth
    if data_true is not None:
        sns.histplot(
            data_true, bins=bin_edges, color=color_true,
            stat="density", alpha=1, ax=ax1,
            edgecolor='black', linewidth=0.5
        )
        # Calculate statistics
        mean_true = np.mean(data_true)
        std_true = np.std(data_true)
        results["mean_true"] = mean_true
        results["std_true"] = std_true
        
        # Add KDE and vertical line
        sns.kdeplot(data_true, color=color_true, ax=ax1, linewidth=1.0, alpha=1.0, bw_adjust=0.8)
        ax1.axvline(mean_true, color=color_true, linestyle='--', linewidth=1.2)

    # Plot Predictions
    sns.histplot(
        data_pred, bins=bin_edges, color=color_pred,
        stat="density", alpha=1, ax=ax1,
        edgecolor='black', linewidth=0.5
    )
    mean_pred = np.mean(data_pred)
    std_pred = np.std(data_pred)
    results["mean_pred"] = mean_pred
    results["std_pred"] = std_pred
    sns.kdeplot(data_pred, color=color_pred, ax=ax1, linewidth=1.0, alpha=1.0, bw_adjust=0.8)
    ax1.axvline(mean_pred, color=color_pred, linestyle='--', linewidth=1.2)

    # Finalize plot 1
    ax1.set_xlabel(xlabel, fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    # ax1.set_title(f"{title} (Density)", fontsize=11)
    # true_str = r'Ground Truth: $\mu={:.2f}$, $\sigma={:.2f}$'.format(mean_true, std_true)
    # pred_str = r'Prediction: $\mu={:.2f}$, $\sigma={:.2f}$'.format(mean_pred, std_pred)
    # 
    # from matplotlib.lines import Line2D
    # legend_handles = [
    # Line2D([0], [0], color=color_true, linewidth=2, label=true_str),
    # Line2D([0], [0], color=color_pred, linewidth=2, label=pred_str)
    # ]
# 
    # ax1.legend(handles=legend_handles, frameon=False, loc='best')
    # sns.despine(ax=ax1)
    density_path = os.path.join(save_folder, f"{filename_base}_density.svg")
    plt.savefig(density_path, format='svg', bbox_inches='tight')  # specify format explicitly
    plt.close(fig1)
    
    # ========================================================================
    # Plot 2: Probability without KDE
    # ========================================================================
    fig2, ax2 = plt.subplots(constrained_layout=True)
    
    # Plot Ground Truth
    if data_true is not None:
        sns.histplot(
            data_true, bins=bin_edges, color=color_true,
            stat="probability", alpha=0.7, ax=ax2,
            edgecolor='black', linewidth=0.5
        )

    # Plot Predictions
    sns.histplot(
        data_pred, bins=bin_edges, color=color_pred,
        stat="probability", alpha=0.7, ax=ax2,
        edgecolor='black', linewidth=0.5
    )

    # Finalize plot 2
    ax2.set_xlabel(xlabel, fontsize=10)
    ax2.set_ylabel("Probability", fontsize=10)
    # ax2.set_title(f"{title} (Probability)", fontsize=11)
    # ax2.legend(['Ground Truth', 'Prediction'], frameon=False, loc='best')
    sns.despine(ax=ax2)
    probability_path = os.path.join(save_folder, f"{filename_base}_probability.svg")  # change extension to .svg
    plt.savefig(probability_path, format='svg', bbox_inches='tight')  # specify format explicitly
    plt.close(fig2)

    return results
