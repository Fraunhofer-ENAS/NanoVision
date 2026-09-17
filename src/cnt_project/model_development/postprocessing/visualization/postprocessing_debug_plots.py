import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.draw import polygon as draw_polygon

try:
    import scienceplots  # noqa: F401
    _SCIENCE_STYLE = ["science", "no-latex"]
except Exception:
    _SCIENCE_STYLE = None

from cnt_project.io.paths import ProjectPaths

# Process-local cache used to accumulate histogram values across images
# during one postprocessing debug run. Retained as-is for reproducibility.
_GLOBAL_HIST_CACHE: dict[tuple[str, str, str], np.ndarray] = {}




def _sanitize_tag(value: str | None, default: str = "image") -> str:
    if value is None:
        return default
    s = str(value).strip()
    if not s:
        return default
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)
    return safe[:120] or default


def _plot_hist(values, title: str, xlabel: str, out_path: Path, xlim: tuple[float, float] | None = None, ylim: tuple[float, float] | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    style_ctx = plt.style.context(_SCIENCE_STYLE) if _SCIENCE_STYLE else plt.style.context("default")
    with style_ctx:
        fig, ax = plt.subplots(figsize=(8, 5))
        if values.size:
            n_bins = int(min(60, max(10, np.sqrt(values.size))))
            ax.hist(values, bins=n_bins, color="steelblue", edgecolor="black", alpha=0.85)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count")
            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(alpha=0.25)
        else:
            ax.text(0.5, 0.5, "No values available", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def _collect_polygon_area_values(polygons) -> np.ndarray:
    areas = [float(p.area) for p in polygons if p is not None and not p.is_empty]
    return np.asarray(areas, dtype=np.float32)


def _flatten_raw_ray_distances(raw_distances) -> np.ndarray:
    if raw_distances is None:
        return np.asarray([], dtype=np.float32)
    arr = np.asarray(raw_distances, dtype=np.float32)
    if arr.size == 0:
        return np.asarray([], dtype=np.float32)
    return arr.reshape(-1)


def _append_and_plot_global(
    values: np.ndarray,
    cache_key: tuple[str, str, str],
    title: str,
    xlabel: str,
    out_path: Path,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    prev = _GLOBAL_HIST_CACHE.get(cache_key)
    if prev is None:
        merged = values.astype(np.float32, copy=False)
    else:
        merged = np.concatenate([prev, values.astype(np.float32, copy=False)])

    _GLOBAL_HIST_CACHE[cache_key] = merged
    _plot_hist(merged, title=title, xlabel=xlabel, out_path=out_path, xlim=xlim, ylim=ylim)


def save_postprocessing_stage_histograms(
    debug_root: Path,
    stage_tag: str,
    image_tag: str,
    polygons,
    raw_distances=None,
) -> None:
    area_values = _collect_polygon_area_values(polygons)
    ray_dist_values = _flatten_raw_ray_distances(raw_distances)
    should_plot_ray_dist = stage_tag in ("before_nms", "after_nms")

    stage_root = debug_root / "histograms" / stage_tag
    per_image_area_dir = stage_root / "per_image" / "all_polygon_area_histograms"
    per_image_pixel_dir = stage_root / "per_image" / "all_pixel_distance_histograms"
    global_area_dir = stage_root / "across_all_images" / "all_polygon_area_histograms"
    global_pixel_dir = stage_root / "across_all_images" / "all_pixel_distance_histograms"

    per_image_area_path = per_image_area_dir / f"{image_tag}_{stage_tag}_all_polygon_area_hist.png"
    per_image_pixel_path = per_image_pixel_dir / f"{image_tag}_{stage_tag}_all_pixel_distance_hist.png"
    global_area_plot = global_area_dir / f"global_{stage_tag}_all_polygon_area_hist.png"
    global_pixel_plot = global_pixel_dir / f"global_{stage_tag}_all_pixel_distance_hist.png"
    cache_root = str(debug_root.resolve())

    _plot_hist(
        area_values,
        title=f"All Polygon Area Distribution ({stage_tag}) - {image_tag}",
        xlabel="Polygon area (pixels)",
        out_path=per_image_area_path,
        xlim=(0, 500) if stage_tag in ("before_nms", "after_nms") else ((0, 1500) if stage_tag == "after_merging" else None),
        ylim=(0, 8500) if stage_tag == "before_nms" else ((0, 3000) if stage_tag == "after_nms" else ((0, 1200) if stage_tag == "after_merging" else None)),
    )
    if should_plot_ray_dist:
        _plot_hist(
            ray_dist_values,
            title=f"Raw Ray Distance Distribution ({stage_tag}) - {image_tag}",
            xlabel="Raw ray distance",
            out_path=per_image_pixel_path,
            xlim=(0, 50),
            ylim=(0, 4e4),
        )

    _append_and_plot_global(
        area_values,
        cache_key=(cache_root, stage_tag, "all_polygon_area"),
        title=f"All Polygon Area Distribution Across All Images ({stage_tag})",
        xlabel="Polygon area (pixels)",
        out_path=global_area_plot,
        xlim=(0, 500) if stage_tag in ("before_nms", "after_nms") else ((0, 1500) if stage_tag == "after_merging" else None),
        ylim=(0, 8500) if stage_tag == "before_nms" else ((0, 3000) if stage_tag == "after_nms" else ((0, 1200) if stage_tag == "after_merging" else None)),
    )
    if should_plot_ray_dist:
        _append_and_plot_global(
            ray_dist_values,
            cache_key=(cache_root, stage_tag, "all_pixel_distance"),
            title=f"Raw Ray Distance Distribution Across All Images ({stage_tag})",
            xlabel="Raw ray distance",
            out_path=global_pixel_plot,
            xlim=(0, 60),
            ylim=(0, 8e5),
        )


def save_before_nms_score_peak_plot(
    *,
    debug_root: Path,
    image_tag: str,
    points: np.ndarray,
    scores: np.ndarray,
    polygons,
    img_shape: tuple[int, int] | None = None,
) -> None:
    if points is None or scores is None:
        return

    points = np.asarray(points)
    scores = np.asarray(scores)
    if points.size == 0 or scores.size == 0:
        return

    if img_shape is None:
        y_max = int(np.ceil(np.max(points[:, 0]))) if points.ndim == 2 else 0
        x_max = int(np.ceil(np.max(points[:, 1]))) if points.ndim == 2 else 0
        img_shape = (max(1, y_max + 1), max(1, x_max + 1))

    h, w = int(img_shape[0]), int(img_shape[1])
    if h <= 0 or w <= 0:
        return

    score_map = np.zeros((h, w), dtype=np.float32)
    ys = np.clip(np.rint(points[:, 0]).astype(int), 0, h - 1)
    xs = np.clip(np.rint(points[:, 1]).astype(int), 0, w - 1)
    for y, x, s in zip(ys, xs, scores):
        if s > score_map[y, x]:
            score_map[y, x] = float(s)

    # Treat all polygons together as one object mask for peak detection.
    union_mask = np.zeros((h, w), dtype=bool)
    for poly in polygons:
        if poly is None or poly.is_empty:
            continue

        x_coords, y_coords = poly.exterior.xy
        x = np.asarray(x_coords, dtype=np.float32)
        y = np.asarray(y_coords, dtype=np.float32)
        if x.size < 3 or y.size < 3:
            continue

        rr, cc = draw_polygon(y, x, shape=(h, w))
        union_mask[rr, cc] = True

    masked_scores = np.array(score_map, copy=True)
    masked_scores[~union_mask] = 0

    candidate_points = []
    strongest_point = None
    strongest_value = None

    if np.any(union_mask):
        peak_radius_px = 10
        peak_filter_size = 2 * peak_radius_px + 1
        local_max = ndimage.maximum_filter(masked_scores, size=peak_filter_size, mode="constant")
        peak_mask = (masked_scores > 0) & (masked_scores == local_max) & union_mask
        peak_coords = np.argwhere(peak_mask)
        candidate_points = [(int(py), int(px)) for py, px in peak_coords]

        if np.any(masked_scores > 0):
            best_y, best_x = np.unravel_index(int(np.argmax(masked_scores)), masked_scores.shape)
            strongest_point = (int(best_y), int(best_x))
            strongest_value = float(masked_scores[best_y, best_x])
            if not candidate_points:
                candidate_points = [strongest_point]

    out_dir = debug_root / "score_peak_maps" / "before_nms"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_tag}_before_nms_score_peak_map.png"
    out_path_3d = out_dir / f"{image_tag}_before_nms_score_peak_map_3d.png"

    style_ctx = plt.style.context(_SCIENCE_STYLE) if _SCIENCE_STYLE else plt.style.context("default")
    with style_ctx:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(score_map, cmap="viridis")
        fig.colorbar(im, ax=ax, label="Score Map Value")

        if candidate_points:
            py, px = zip(*candidate_points)
            ax.scatter(px, py, s=26, c="yellow", edgecolors="black", linewidths=0.4, alpha=0.35, label="Local Peak Candidates")

        if strongest_point is not None and strongest_value is not None:
            py, px = strongest_point
            ax.scatter([px], [py], s=44, c="red", marker="x", linewidths=1.3, alpha=0.45, label="Overall Maximum")
            ax.text(
                px + 2,
                py - 2,
                f"{strongest_value:.3f}",
                color="red",
                fontsize=7,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.15"),
            )

        ax.set_title("Score Map - before_nms (local-max radius=10px)")
        ax.axis("off")
        if strongest_point is not None or candidate_points:
            ax.legend(loc="upper right", framealpha=0.85)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    style_ctx = plt.style.context(_SCIENCE_STYLE) if _SCIENCE_STYLE else plt.style.context("default")
    with style_ctx:
        fig3d = plt.figure(figsize=(10, 7))
        ax3d = fig3d.add_subplot(111, projection="3d")

        # 3D readability controls: tune these when adjusting the figure.
        z_percentile_clip = 99.9
        z_exaggeration = 0.25
        surface_alpha = 0.98
        contour_levels = 10
        surface_linewidth = 0.38
        smoothing_sigma = 0.85

        max_dim = max(h, w)
        stride = 1 if max_dim <= 256 else (2 if max_dim <= 512 else 4)
        y_coords = np.arange(0, h, stride)
        x_coords = np.arange(0, w, stride)
        X, Y = np.meshgrid(x_coords, y_coords)
        z_source = masked_scores if np.any(union_mask) else score_map
        Z = z_source[::stride, ::stride]

        positive = Z[Z > 0]
        if positive.size:
            zmax = float(np.percentile(positive, z_percentile_clip))
            zmax = max(zmax, float(np.max(positive))) if np.isclose(zmax, 0.0) else zmax
        else:
            zmax = max(1e-6, float(np.max(Z)))
        Z_plot = np.clip(Z, 0.0, zmax)
        if np.any(Z_plot > 0):
            # Smooth local spikes so the surface reads as connected neighborhoods.
            Z_plot = ndimage.gaussian_filter(Z_plot, sigma=smoothing_sigma)

        surface = ax3d.plot_surface(
            X,
            Y,
            Z_plot,
            cmap="viridis",
            linewidth=surface_linewidth,
            edgecolor=(0.08, 0.08, 0.08, 0.35),
            antialiased=False,
            alpha=surface_alpha,
            vmin=0.0,
            vmax=zmax,
        )

        if np.any(Z_plot > 0):
            ax3d.contour(
                X,
                Y,
                Z_plot,
                zdir="z",
                offset=0.0,
                levels=contour_levels,
                cmap="viridis",
                linewidths=0.7,
                alpha=0.75,
            )

        fig3d.colorbar(surface, ax=ax3d, shrink=0.62, pad=0.08, label="Score Map Value")

        if candidate_points:
            cand = np.asarray(candidate_points, dtype=np.int32)
            cy = np.clip(cand[:, 0], 0, h - 1)
            cx = np.clip(cand[:, 1], 0, w - 1)
            cz = score_map[cy, cx]
            ax3d.scatter(cx, cy, cz, c="yellow", s=10, alpha=0.45, depthshade=False)

        if strongest_point is not None:
            py, px = strongest_point
            pz = score_map[py, px]
            ax3d.scatter([px], [py], [pz], c="red", s=48, marker="x", depthshade=False)

        ax3d.set_title("Score Map 3D Surface - before_nms (local-max radius=10px)")
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Score")
        ax3d.set_xlim(0, w - 1)
        ax3d.set_ylim(h - 1, 0)
        ax3d.set_zlim(0, max(1e-6, zmax * 1.02))
        ax3d.set_box_aspect((w, h, max(1.0, max_dim * z_exaggeration)))
        ax3d.view_init(elev=35, azim=-128)
        ax3d.grid(True, alpha=0.3)

        fig3d.tight_layout()
        fig3d.savefig(out_path_3d, dpi=300, bbox_inches="tight")
        plt.close(fig3d)


def resolve_postprocessing_debug_plot_dir(
    *,
    run_name: str | None = None,
    explicit_dir: str | Path | None = None,
    approach_tag: str,
) -> Path:
    """Resolve debug plot output directory using ProjectPaths contracts."""
    if explicit_dir is not None:
        out_dir = Path(explicit_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    paths = ProjectPaths.from_here(__file__)
    if run_name:
        return paths.viz_dir(run_name, "postprocessing", "merging_debug", approach_tag)
    return paths.misc_dir("postprocessing", "merging_debug", approach_tag)
