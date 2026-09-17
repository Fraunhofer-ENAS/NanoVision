import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
from shapely.geometry import MultiPolygon
import os

# makes each polygon in the image with different color
def generate_gradient_colors(num_colors):
    """Generates a list of distinct colors using ColorBrewer."""
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10)[:3] for i in range(num_colors)]  # Take only RGB values
    return colors

def plot_polygon(polygon, ax, color):
    """Plots an individual polygon or MultiPolygon with a specified color."""
    if polygon.is_empty:
        return

    # Simplify the polygon for faster rendering
    polygon = polygon.simplify(0.1, preserve_topology=True)

    if isinstance(polygon, MultiPolygon):
        for poly in polygon.geoms:
            x, y = poly.exterior.xy
            ax.fill(x, y, facecolor=color, edgecolor='black', alpha=0.3)
            for interior in poly.interiors:
                x, y = interior.xy
                ax.fill(x, y, facecolor="white", edgecolor='black', alpha=1.0)
    else:
        x, y = polygon.exterior.xy
        ax.fill(x, y, facecolor=color, edgecolor='black', alpha=0.3)
        for interior in polygon.interiors:
            x, y = interior.xy
            ax.fill(x, y, facecolor="white", edgecolor='black', alpha=1.0)

# draws distance lines from the centeroid of the polygon based on number of rays
def plot_orientation_vectors(polygon, angle, ax):
    """Draws a small line indicating the orientation vector of the polygon."""
    centroid = polygon.centroid
    angle_rad = np.radians(angle)
    length = 5  # Length of the orientation vector

    end_x = centroid.x + length * np.cos(angle_rad)
    end_y = centroid.y + length * np.sin(angle_rad)

    ax.plot([centroid.x, end_x], [centroid.y, end_y], 'g-', lw=1)


def plot_polygons(polygons, title="Polygons", n_rays=8, dist=None, points_arr=None, top_n=1, orientation_angles=None,
                  plot_rays=True, save_path=None, flip_vertical=False, scores=None):
    """
    Plots multiple polygons with gradient colors, centroids, and rays extending from centroids.

    Parameters:
        polygons (list): List of polygons to plot.
        title (str): Title of the plot.
        n_rays (int): Number of rays to plot from each centroid.
        dist (list): List of distances for rays.
        points_arr (list): List of centroid points.
        top_n (int): Number of top polygons to highlight.
        orientation_angles (list): List of orientation angles for polygons.
        plot_rays (bool): Whether to plot rays from centroids.
        save_path (str): Path to save the plot. If a directory, the plot will be saved as "{title}_all.png".
                         If a full path (including filename), the plot will be saved with that name.
        flip_vertical (bool): Whether to flip the image vertically before saving.
    """
    if len(polygons) == 0:
        print("No polygons to plot.")
        return

    num_polygons = len(polygons)
    colors = generate_gradient_colors(num_polygons)  # Generate colors
    lengths = [polygon.length for polygon in polygons if not polygon.is_empty]

    if len(lengths) == 0:
        print("No valid polygons found.")
        return

    max_length_idx = np.argmax(lengths)
    average_length = np.mean(lengths)
    max_length = lengths[max_length_idx]

    print(f"Average length of polygons: {average_length}")
    print(f"Maximum length of polygons: {max_length}")
    print("################################################################")

    # Compute plot limits using bounding boxes
    min_x, min_y, max_x, max_y = polygons[0].bounds
    for polygon in polygons[1:]:
        x_min, y_min, x_max, y_max = polygon.bounds
        min_x, min_y = min(min_x, x_min), min(min_y, y_min)
        max_x, max_y = max(max_x, x_max), max(max_y, y_max)

    fig_all, ax_all = plt.subplots(figsize=(12, 8))

    # Plot polygons asynchronously for faster performance
    with ThreadPoolExecutor() as executor:
        futures = []
        for i, polygon in enumerate(polygons):
            color = colors[i]
            futures.append(executor.submit(plot_polygon, polygon, ax_all, color))

            # Calculate and plot centroids
            if points_arr is not None and dist is not None:
                centroid = points_arr[polygons.index(polygon)]
                x, y = centroid[1], centroid[0]
                ax_all.plot(x, y, 'ko', markersize=1)
                # Add score text (if scores provided)
                if scores is not None and i < len(scores):
                    ax_all.text(
                        x, y + 0.5,  # Offset y-position
                        f"{scores[i]:.2f}", 
                        fontsize=8, 
                        color='black', 
                        ha='center', 
                        va='bottom',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.1)
                    )
                if plot_rays:
                    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
                    ray_starts_x, ray_starts_y, ray_ends_x, ray_ends_y = [], [], [], []
                    for j, angle in enumerate(angles):
                        ray_dist = dist[i][j]
                        end_x = centroid[1] + np.cos(angle) * ray_dist
                        end_y = centroid[0] + np.sin(angle) * ray_dist
                        ray_starts_x.append(centroid[1])
                        ray_starts_y.append(centroid[0])
                        ray_ends_x.append(end_x)
                        ray_ends_y.append(end_y)
                    ax_all.plot([ray_starts_x, ray_ends_x], [ray_starts_y, ray_ends_y], 'r-', lw=0.2, alpha=0.5)
            else:
                centroid = polygon.centroid
                x, y = centroid.x, centroid.y
                ax_all.plot(x, y, 'ko', markersize=1)
                # Add score text (if scores provided)
                if scores is not None and i < len(scores):
                    ax_all.text(
                        x, y + 0.5, 
                        f"{scores[i]:.2f}", 
                        fontsize=8, 
                        color='black', 
                        ha='center', 
                        va='bottom',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.1)
                    )
                if orientation_angles is not None:
                    angle = orientation_angles[i]
                    plot_orientation_vectors(polygon, angle, ax_all)

    # Wait for all plots to complete
    [f.result() for f in futures]

    # Set plot limits
    ax_all.set_xlim(min_x - 1, max_x + 1)
    ax_all.set_ylim(min_y - 1, max_y + 1)
    ax_all.set_aspect('equal')
    ax_all.set_title(f"{title} - All Polygons")
    ax_all.grid(False)

    # Flip the image vertically if requested
    if flip_vertical:
        ax_all.invert_yaxis()

    # Determine the save path
    if save_path:
        # If save_path is a directory, append the default filename
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, f"{title}_all.png")
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save the plot in both PNG and SVG formats
        base, ext = os.path.splitext(save_path)

        # Always save as PNG (bitmap, for quick viewing)
        fig_all.savefig(base + ".png", dpi=300, bbox_inches="tight")

        # Also save as SVG (vector, for publication)
        fig_all.savefig(base + ".svg", format="svg", bbox_inches="tight")

        print(f"Plots saved to: {base}.png and {base}.svg")
    else:
        # Default behavior: save to current working directory
        default_save_path = f"{title}_all.png"

        base, ext = os.path.splitext(default_save_path)

        # Always save as PNG (bitmap, for quick viewing)
        fig_all.savefig(base + ".png", dpi=300, bbox_inches="tight")

        # Also save as SVG (vector, for publication)
        fig_all.savefig(base + ".svg", format="svg", bbox_inches="tight")

        print(f"Plots saved to: {base}.png and {base}.svg")

    plt.close(fig_all)


# for debug purposes plotting the merged polygon to check the shape
def plot_merged_polygon(merged_polygon, merged_id):
    fig, ax = plt.subplots()

    # Plot Merged Polygon
    x_merged, y_merged = merged_polygon.exterior.xy
    ax.fill(x_merged, y_merged, label='Merged Polygon', color='orange', alpha=0.5)

    ax.set_title('Merged Polygon Visualization')
    ax.legend()
    plt.axis('equal')
    plt.grid(False)
    plt.savefig(f"merged_polygon_{merged_id}.png")
    plt.close(fig)
    # plt.show()  # Show the merged polygon plot
