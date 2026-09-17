
# src/utils/styles.py
import matplotlib.pyplot as plt

def set_figure_style():
    """Apply a consistent publication-ready style across all figures."""
    try:
        import scienceplots
        plt.style.use(['science', 'no-latex'])
    except Exception:
        plt.style.use('default')
    print("Current font sizes:")
    print(f"Font size: {plt.rcParams['font.size']}")
    print(f"Axes labelsize: {plt.rcParams['axes.labelsize']}")
    print(f"XTick labelsize: {plt.rcParams['xtick.labelsize']}")
    print(f"YTick labelsize: {plt.rcParams['ytick.labelsize']}")
