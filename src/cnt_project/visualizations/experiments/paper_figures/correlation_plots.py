import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.linear_model import RANSACRegressor, LinearRegression
from scipy.stats import pearsonr

def precision_vs_width(
    mean_width_px,
    precision,
    noise,
    pixel_to_micron=5/256.0,
    save_path: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    # Enforce style here so figures always match the paper style
    try:
        import scienceplots  
        plt.style.use(['science', 'no-latex'])
    except Exception:
        pass

    x = np.asarray(mean_width_px) * pixel_to_micron
    y = np.asarray(precision)
    cmap = mcolors.LinearSegmentedColormap.from_list("emerald_to_vermillion", ["#4cec81", "#fc1803"])
    norm = mcolors.Normalize(vmin=float(np.min(noise)), vmax=float(np.max(noise)))

    X = x.reshape(-1, 1)
    ransac = RANSACRegressor(LinearRegression(), random_state=42).fit(X, y)
    line_X = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    line_y = ransac.predict(line_X)
    a = float(ransac.estimator_.coef_[0]); b = float(ransac.estimator_.intercept_)

    inliers = ransac.inlier_mask_
    r, p = pearsonr(X[inliers, 0], y[inliers])

    plot_figsize = figsize if figsize is not None else (4.5, 2.8)
    fig, ax = plt.subplots(figsize=plot_figsize)
    sc = ax.scatter(x, y, s=80, c=noise, cmap=cmap, norm=norm, alpha=0.5, edgecolors="black", linewidth=0.5)
    ax.plot(line_X, line_y, color="black", linestyle="--", linewidth=1.0)
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r"Noise $\sigma$", labelpad=4)
    ax.set_xlabel(r"Mean width ($\mu$m)")
    ax.set_ylabel("Precision")
    ax.set_ylim(0.1, 1.1)
    ax.tick_params(direction="in")
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path) + ".svg", format='svg', bbox_inches='tight', pad_inches=0.02)
        fig.savefig(str(save_path) + ".png", dpi=300, bbox_inches='tight')
    return {'slope': a, 'intercept': b, 'pearson_r': float(r), 'pvalue': float(p)}

def noise_vs_width(
    mean_width_px, noise, pixel_to_micron=5/256.0,
    save_path: str | None = None
):
    """
    Plot: Noise σ  →  Mean width (µm)
    Includes RANSAC regression and Pearson correlation.
    """

    # Enforce scienceplots style
    try:
        import scienceplots
        plt.style.use(["science", "no-latex"])
    except Exception:
        pass

    x = np.asarray(noise)
    y = np.asarray(mean_width_px) * pixel_to_micron

    # Regression
    X = x.reshape(-1, 1)
    ransac = RANSACRegressor(LinearRegression(), random_state=42).fit(X, y)
    line_X = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    line_y = ransac.predict(line_X)

    a = float(ransac.estimator_.coef_[0])
    b = float(ransac.estimator_.intercept_)

    inliers = ransac.inlier_mask_
    r, p = pearsonr(X[inliers, 0], y[inliers])

    # Plot
    fig, ax = plt.subplots(figsize=(4.5, 2.8))

    norm = mcolors.Normalize(vmin=x.min(),
                             vmax=x.max())
    ax.scatter(
    x,y,
    s=80, alpha=0.5,
    edgecolors="black", 
    color="#4169e1",
    norm=norm
    )

    ax.plot(line_X, line_y, color="black", linestyle="--", linewidth=1.0)

    ax.set_xlabel(r"Noise $\sigma$")
    ax.set_ylabel(r"Mean width ($\mu$m)")

    ax.tick_params(direction="in")
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")

    plt.tight_layout()

    # Save
    if save_path:
        fig.savefig(str(save_path) + ".svg", format="svg",
                    bbox_inches="tight", pad_inches=0.02)
        fig.savefig(str(save_path) + ".png", dpi=300,
                    bbox_inches="tight")

    return {
        "slope": a,
        "intercept": b,
        "pearson_r": float(r),
        "pvalue": float(p)
    }

def precision_vs_noise(
    precision, noise, save_path: str | None = None
):
    """
    Plot: Noise σ  →  Precision
    Includes RANSAC regression and Pearson correlation.
    """

    # Enforce scienceplots style
    try:
        import scienceplots
        plt.style.use(["science", "no-latex"])
    except Exception:
        pass

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import RANSACRegressor, LinearRegression
    from scipy.stats import pearsonr

    x = np.asarray(noise)
    y = np.asarray(precision)

    # Regression
    X = x.reshape(-1, 1)
    ransac = RANSACRegressor(LinearRegression(), random_state=42).fit(X, y)
    line_X = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    line_y = ransac.predict(line_X)

    a = float(ransac.estimator_.coef_[0])
    b = float(ransac.estimator_.intercept_)

    inliers = ransac.inlier_mask_
    r, p = pearsonr(X[inliers, 0], y[inliers])

    # Plot
    fig, ax = plt.subplots(figsize=(4.5, 2.8))

    ax.scatter(
        x, y,
        s=80,
        alpha=0.5,
        edgecolors="black",
        color="#4169e1",
    )

    ax.plot(line_X, line_y, color="black", linestyle="--", linewidth=1.0)

    ax.set_xlabel(r"Noise $\sigma$")
    ax.set_ylabel("Precision")
    # ax.set_ylim(0, 1)

    ax.tick_params(direction="in")
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")

    plt.tight_layout()

    # Save files
    if save_path:
        fig.savefig(str(save_path) + ".svg", format='svg',
                    bbox_inches='tight', pad_inches=0.02)
        fig.savefig(str(save_path) + ".png", dpi=300,
                    bbox_inches='tight')

    return {
        "slope": a,
        "intercept": b,
        "pearson_r": float(r),
        "pvalue": float(p)
    }