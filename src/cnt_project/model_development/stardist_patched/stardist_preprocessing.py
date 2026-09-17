from __future__ import print_function, unicode_literals, absolute_import, division

from csbdeep.utils.tf import keras_import, IS_TF_1
import numpy as np
import os, re
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
from skimage.draw import polygon as draw_polygon
from scipy.ndimage import zoom
from scipy.ndimage import find_objects
from scipy import ndimage
from csbdeep.utils.tf import keras_import, IS_TF_1, BACKEND as K
from stardist.models import StarDistData2D
from stardist.sample_patches import sample_patches
from stardist.utils import mask_to_categorical 
from stardist.geometry import star_dist
from cnt_project.io.paths import ProjectPaths
from cnt_project.model_development.stardist_patched.scoring_functions import edt_prob

try:
    import scienceplots  # noqa: F401
    _SCIENCE_STYLE = ["science", "no-latex"]
except Exception:
    _SCIENCE_STYLE = None


Sequence = keras_import('utils', 'Sequence')
Adam = keras_import('optimizers', 'Adam')
ReduceLROnPlateau, TensorBoard = keras_import('callbacks', 'ReduceLROnPlateau', 'TensorBoard')
keras = keras_import()
Input, Conv2D, MaxPooling2D = keras_import('layers', 'Input', 'Conv2D', 'MaxPooling2D')
Model = keras_import('models', 'Model')

_gen_rtype = list if IS_TF_1 else tuple


def _resolve_debug_plot_dir(
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
        return paths.viz_dir(run_name, "preprocessing", "stardist", approach_tag)
    return paths.misc_dir("preprocessing", "stardist", approach_tag)


def _safe_name(s: str) -> str:
    # remove extension if present + replace illegal filesystem characters
    s = os.path.splitext(str(s))[0]
    s = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s[:180] 

# Custom StarDist Data Model (Proprocessing)
class MyStarDistData2D(StarDistData2D):
    def __init__(
        self,
        X,
        Y,
        batch_size,
        n_rays,
        length,
        n_classes=None,
        classes=None,
        filenames=None,
        patch_size=(256, 256),
        b=32,
        grid=(1, 1),
        shape_completion=False,
        augmenter=None,
        foreground_prob=0,
        edt_mode="standard",
        debug_plots=False,
        debug_plot_dir=None,
        debug_run_name=None,
        debug_dir=None,
        scoring_debug_dir=None,
        **kwargs,
    ):
        # Defensive: if debug keys are passed through **kwargs (e.g., from notebook state),
        # consume them here so they never reach StarDistDataBase.__init__.
        debug_plots = kwargs.pop("debug_plots", debug_plots)
        debug_plot_dir = kwargs.pop("debug_plot_dir", debug_plot_dir)
        debug_run_name = kwargs.pop("debug_run_name", debug_run_name)
        debug_dir = kwargs.pop("debug_dir", debug_dir)
        scoring_debug_dir = kwargs.pop("scoring_debug_dir", scoring_debug_dir)

        self.edt_mode = edt_mode
        self.debug_plots = bool(debug_plots)
        self.debug_run_name = debug_run_name

        # Keep backward compatibility with legacy args while preferring contract-based paths.
        polygon_debug_explicit = debug_plot_dir if debug_plot_dir is not None else debug_dir
        scoring_debug_explicit = scoring_debug_dir

        self.debug_dir = (
            _resolve_debug_plot_dir(
                run_name=self.debug_run_name,
                explicit_dir=polygon_debug_explicit,
                approach_tag="polygon_seed_debug",
            )
            if self.debug_plots or polygon_debug_explicit is not None
            else None
        )
        self.scoring_debug_dir = (
            _resolve_debug_plot_dir(
                run_name=self.debug_run_name,
                explicit_dir=scoring_debug_explicit,
                approach_tag="scoring_maps",
            )
            if self.debug_plots or scoring_debug_explicit is not None
            else None
        )
        if filenames is None:
            self.filenames = None
        else:
            if len(filenames) != len(X):
                raise ValueError(f"`filenames` length ({len(filenames)}) must match X length ({len(X)})")
            self.filenames = list(filenames)
        super().__init__(X=X, Y=Y, n_rays=n_rays, grid=grid,
                         n_classes=n_classes, classes=classes,
                         batch_size=batch_size, patch_size=patch_size, length=length,
                         augmenter=augmenter, foreground_prob=foreground_prob, **kwargs)

        self.shape_completion = bool(shape_completion)
          
        if self.shape_completion and b > 0:
            if not all(b % g == 0 for g in self.grid):
                raise ValueError(f"'shape_completion' requires that crop size {b} ('train_completion_crop' in config) is evenly divisible by all grid values {self.grid}")
            self.b = slice(b,-b),slice(b,-b)
        else:
            self.b = slice(None),slice(None)

        self.sd_mode = 'opencl' if self.use_gpu else 'cpp'


    def __getitem__(self, i):
        def polygon_from_rays(dist_map, cy, cx, scale=0.5):
            H, W, R = dist_map.shape
            r = dist_map[cy, cx].astype(np.float32) * scale
            theta = np.linspace(0, 2*np.pi, R, endpoint=False, dtype=np.float32)

            xs = cx + r * np.cos(theta)
            ys = cy + r * np.sin(theta)

            rr, cc = draw_polygon(ys, xs, shape=(H, W))
            mask = np.zeros((H, W), dtype=np.uint8)
            mask[rr, cc] = 1
            return mask, xs, ys

        def object_local_peak_centroids(label_img, score_map, peak_radius=1, peak_rel_threshold=0.9):
            """Match scoring_functions centroid logic: local maxima + per-object relative threshold."""
            peak_filter_size = 2 * int(peak_radius) + 1

            strongest_points = []
            strongest_values = []
            candidate_points = []

            for obj_id, sl in enumerate(find_objects(label_img), 1):
                if sl is None:
                    continue

                obj_mask = (label_img[sl] == obj_id)
                if not np.any(obj_mask):
                    continue

                patch_scores = np.array(score_map[sl], copy=True)
                patch_scores[~obj_mask] = 0
                local_scores = patch_scores[obj_mask]

                if local_scores.size == 0:
                    continue

                obj_max = float(np.max(local_scores))
                if obj_max <= 0:
                    continue

                local_max = ndimage.maximum_filter(patch_scores, size=peak_filter_size, mode='constant')
                peak_mask = (patch_scores == local_max) & obj_mask & (patch_scores >= peak_rel_threshold * obj_max)
                local_peak_coords = np.argwhere(peak_mask)

                if local_peak_coords.size == 0:
                    local_peak_coords = np.argwhere(obj_mask)

                peak_scores = patch_scores[local_peak_coords[:, 0], local_peak_coords[:, 1]]
                best_idx = int(np.argmax(peak_scores))
                best_y, best_x = local_peak_coords[best_idx]

                strongest_points.append((sl[0].start + best_y, sl[1].start + best_x))
                strongest_values.append(float(patch_scores[best_y, best_x]))

                for py, px in local_peak_coords:
                    candidate_points.append((sl[0].start + py, sl[1].start + px))

            return strongest_points, strongest_values, candidate_points

        def visualize_polygons_rays_with_scores(
            label_img,
            dist_map,
            score_map,
            scale=0.5,
            rays_every=1,
            figsize=(12, 12),
            background="scores",
            peak_radius=1,
            peak_rel_threshold=0.9,
            output_stem="debug",
            save_dir=None,
        ):
            assert dist_map.shape[:2] == score_map.shape, "dist_map and score_map spatial shapes must match."

            if save_dir is None:
                return None

            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            per_image_hist_root = save_dir / "histograms" / "per_image"
            selected_centroid_area_hist_dir = per_image_hist_root / "selected_centroid_polygon_areas"
            all_object_area_hist_dir = per_image_hist_root / "all_object_pixel_polygon_areas"
            score_map_hist_dir = per_image_hist_root / "score_map_values"
            ray_distance_hist_dir = per_image_hist_root / "ray_distance_values"
            selected_centroid_area_hist_dir.mkdir(parents=True, exist_ok=True)
            all_object_area_hist_dir.mkdir(parents=True, exist_ok=True)
            score_map_hist_dir.mkdir(parents=True, exist_ok=True)
            ray_distance_hist_dir.mkdir(parents=True, exist_ok=True)

            filename = (
                f"{output_stem}_"
                f"bg-{background}_radius-{peak_radius}_relthr-{peak_rel_threshold}_"
                f"scale-{scale}_raysEvery-{rays_every}.png"
            )
            save_path = save_dir / filename
            selected_centroid_area_hist_path = selected_centroid_area_hist_dir / (
                f"{output_stem}_"
                f"bg-{background}_radius-{peak_radius}_relthr-{peak_rel_threshold}_"
                f"scale-{scale}_raysEvery-{rays_every}_selected_centroid_polygon_area_hist.png"
            )
            all_object_area_hist_path = all_object_area_hist_dir / (
                f"{output_stem}_"
                f"bg-{background}_radius-{peak_radius}_relthr-{peak_rel_threshold}_"
                f"scale-{scale}_raysEvery-{rays_every}_all_object_pixel_polygon_area_hist.png"
            )
            score_map_hist_path = score_map_hist_dir / (
                f"{output_stem}_"
                f"bg-{background}_radius-{peak_radius}_relthr-{peak_rel_threshold}_"
                f"scale-{scale}_raysEvery-{rays_every}_score_map_hist.png"
            )
            ray_distance_hist_path = ray_distance_hist_dir / (
                f"{output_stem}_"
                f"bg-{background}_radius-{peak_radius}_relthr-{peak_rel_threshold}_"
                f"scale-{scale}_raysEvery-{rays_every}_ray_distance_hist.png"
            )

            maxr = dist_map.max(axis=-1)
            strongest_points, strongest_values, candidate_points = object_local_peak_centroids(
                label_img=label_img,
                score_map=score_map,
                peak_radius=peak_radius,
                peak_rel_threshold=peak_rel_threshold,
            )

            fig, ax = plt.subplots(figsize=figsize)

            if background == "scores":
                ax.imshow(score_map, cmap="gray")
            else:
                ax.imshow(maxr, cmap="gray")

            if candidate_points:
                cys, cxs = zip(*candidate_points)
                ax.scatter(
                    cxs,
                    cys,
                    s=25,
                    c='yellow',
                    edgecolors='black',
                    linewidths=0.4,
                    alpha=0.35,
                    label='Local Peak Candidates',
                )

            polygon_areas = []
            for (cy, cx), s in zip(strongest_points, strongest_values):
                poly_mask, xs, ys = polygon_from_rays(dist_map, cy, cx, scale=scale)
                polygon_areas.append(float(np.count_nonzero(poly_mask)))

                for k in range(0, len(xs), rays_every):
                    ax.plot([cx, xs[k]], [cy, ys[k]], linewidth=0.8, alpha=0.35, color="white")

                x_closed = np.r_[xs, xs[0]]
                y_closed = np.r_[ys, ys[0]]
                ax.plot(x_closed, y_closed, linewidth=3.5, alpha=0.9, color="black")
                ax.plot(x_closed, y_closed, linewidth=1.8, alpha=0.95, color="white")

                ax.scatter(cx, cy, s=45, c="red", edgecolors="black", linewidths=0.6, zorder=5)
                ax.text(
                    cx + 2,
                    cy - 2,
                    f"{s:.3f}",
                    color="white",
                    fontsize=9,
                    bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.2"),
                )

            ax.set_title(
                f"Object local-max centroids (radius={peak_radius}, rel-thr={peak_rel_threshold}) with star-polygons",
                fontsize=14,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if strongest_points or candidate_points:
                ax.legend(loc="upper right", framealpha=0.85)
            plt.tight_layout()

            fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)

            # Per-object area from all labeled object-pixel polygons.
            all_object_areas = []
            for obj_id, sl in enumerate(find_objects(label_img), 1):
                if sl is None:
                    continue
                obj_mask = (label_img[sl] == obj_id)
                area = int(np.count_nonzero(obj_mask))
                if area > 0:
                    all_object_areas.append(float(area))

            # Score-map scalar values for all object pixels.
            score_map_values = np.asarray(score_map[label_img > 0], dtype=np.float32).ravel()

            # Ray distances for all object pixels across all rays.
            ray_distance_values = np.asarray(dist_map[label_img > 0], dtype=np.float32).ravel()

            style_ctx = plt.style.context(_SCIENCE_STYLE) if _SCIENCE_STYLE else plt.style.context("default")
            with style_ctx:
                hist_fig, hist_ax = plt.subplots(figsize=(8, 5))
                if polygon_areas:
                    n_bins = int(min(30, max(5, np.sqrt(len(polygon_areas)))))
                    hist_ax.hist(polygon_areas, bins=n_bins, color="steelblue", edgecolor="black", alpha=0.85)
                    hist_ax.set_title("Selected Centroid Polygon Area Distribution", fontsize=12)
                    hist_ax.set_xlabel("Polygon area (pixels)")
                    hist_ax.set_ylabel("Count")
                    hist_ax.set_xlim(0, 200)
                    hist_ax.set_ylim(0, 700)
                    hist_ax.grid(alpha=0.25)
                else:
                    hist_ax.text(0.5, 0.5, "No polygons available", ha="center", va="center", transform=hist_ax.transAxes)
                    hist_ax.set_title("Selected Centroid Polygon Area Distribution", fontsize=12)
                    hist_ax.set_xticks([])
                    hist_ax.set_yticks([])

                hist_fig.tight_layout()
                hist_fig.savefig(selected_centroid_area_hist_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
                plt.close(hist_fig)

            style_ctx = plt.style.context(_SCIENCE_STYLE) if _SCIENCE_STYLE else plt.style.context("default")
            with style_ctx:
                area_fig, area_ax = plt.subplots(figsize=(8, 5))
                if all_object_areas:
                    n_bins = int(min(40, max(8, np.sqrt(len(all_object_areas)))))
                    area_ax.hist(all_object_areas, bins=n_bins, color="darkorange", edgecolor="black", alpha=0.85)
                    area_ax.set_title("All Object-Pixel Polygon Area Distribution", fontsize=12)
                    area_ax.set_xlabel("Polygon area (pixels)")
                    area_ax.set_ylabel("Count")
                    area_ax.set_xlim(0, 200)
                    area_ax.set_ylim(0, 700)
                    area_ax.grid(alpha=0.25)
                else:
                    area_ax.text(0.5, 0.5, "No object areas available", ha="center", va="center", transform=area_ax.transAxes)
                    area_ax.set_title("All Object-Pixel Polygon Area Distribution", fontsize=12)
                    area_ax.set_xticks([])
                    area_ax.set_yticks([])

                area_fig.tight_layout()
                area_fig.savefig(all_object_area_hist_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
                plt.close(area_fig)

            style_ctx = plt.style.context(_SCIENCE_STYLE) if _SCIENCE_STYLE else plt.style.context("default")
            with style_ctx:
                score_fig, score_ax = plt.subplots(figsize=(8, 5))
                if score_map_values.size:
                    n_bins = int(min(60, max(10, np.sqrt(score_map_values.size))))
                    score_ax.hist(score_map_values, bins=n_bins, color="mediumpurple", edgecolor="black", alpha=0.85)
                    score_ax.set_title("Score Map Value Distribution", fontsize=12)
                    score_ax.set_xlabel("Score value")
                    score_ax.set_ylabel("Count")
                    score_ax.set_xlim(0, 1)
                    score_ax.set_ylim(0, 6000)
                    score_ax.grid(alpha=0.25)
                else:
                    score_ax.text(0.5, 0.5, "No score values available", ha="center", va="center", transform=score_ax.transAxes)
                    score_ax.set_title("Score Map Value Distribution", fontsize=12)
                    score_ax.set_xticks([])
                    score_ax.set_yticks([])

                score_fig.tight_layout()
                score_fig.savefig(score_map_hist_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
                plt.close(score_fig)

            style_ctx = plt.style.context(_SCIENCE_STYLE) if _SCIENCE_STYLE else plt.style.context("default")
            with style_ctx:
                ray_fig, ray_ax = plt.subplots(figsize=(8, 5))
                if ray_distance_values.size:
                    n_bins = int(min(80, max(20, np.sqrt(ray_distance_values.size))))
                    ray_ax.hist(ray_distance_values, bins=n_bins, color="seagreen", edgecolor="black", alpha=0.85)
                    ray_ax.set_title("Ray Distance Distribution (All Object Pixels)", fontsize=12)
                    ray_ax.set_xlabel("Ray distance")
                    ray_ax.set_ylabel("Count")
                    ray_ax.set_xlim(0, 150)
                    ray_ax.set_ylim(0, 2.5e6)
                    ray_ax.grid(alpha=0.25)
                else:
                    ray_ax.text(0.5, 0.5, "No ray distances available", ha="center", va="center", transform=ray_ax.transAxes)
                    ray_ax.set_title("Ray Distance Distribution (All Object Pixels)", fontsize=12)
                    ray_ax.set_xticks([])
                    ray_ax.set_yticks([])

                ray_fig.tight_layout()
                ray_fig.savefig(ray_distance_hist_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
                plt.close(ray_fig)

            return {
                "selected_centroid_polygon_areas": np.asarray(polygon_areas, dtype=np.float32),
                "all_object_pixel_polygon_areas": np.asarray(all_object_areas, dtype=np.float32),
                "score_map_values": score_map_values,
                "ray_distance_values": ray_distance_values,
            }

        idx = self.batch(i)
        arrays = [sample_patches((self.Y[k],) + self.channels_as_tuple(self.X[k]),
                                 patch_size=self.patch_size, n_samples=1,
                                 valid_inds=self.get_valid_inds(k)) for k in idx]

        if self.n_channel is None:
            X, Y = list(zip(*[(x[0][self.b],y[0]) for y,x in arrays]))
        else:
            X, Y = list(zip(*[(np.stack([_x[0] for _x in x],axis=-1)[self.b], y[0]) for y,*x in arrays]))

        X, Y = tuple(zip(*tuple(self.augmenter(_x, _y) for _x, _y in zip(X,Y))))

        mask_neg_labels = tuple(y[self.b][self.ss_grid[1:3]] < 0 for y in Y)
        has_neg_labels = any(m.any() for m in mask_neg_labels)
        if has_neg_labels:
            mask_neg_labels = np.stack(mask_neg_labels)
            # set negative label pixels to 0 (background)
            Y = tuple(np.maximum(y, 0) for y in Y)

        if self.shape_completion:
            Y_cleared = [clear_border(lbl) for lbl in Y]
            _dist     = np.stack([star_dist(lbl,self.n_rays,mode=self.sd_mode)[self.b+(slice(None),)] for lbl in Y_cleared])
            dist      = _dist[self.ss_grid]

            # Compute EDT based on the cleared borders
            prob = np.stack([
                edt_prob(
                    lbl[self.b][self.ss_grid[1:3]],
                    dist,
                    debug_dir=self.scoring_debug_dir,
                )
                for lbl in Y_cleared
            ])

            dist_mask = np.stack([
                edt_prob(
                    lbl[self.b][self.ss_grid[1:3]],
                    debug_dir=self.scoring_debug_dir,
                )
                for lbl in Y_cleared
            ])
        else:
            # Initialize lists to store the results
            dist_list = []
            prob_list = []
            all_selected_centroid_polygon_areas = []
            all_object_pixel_polygon_areas = []
            all_score_map_values = []
            all_ray_distance_values = []

            # Iterate over each labeled image in Y
            for j, (lbl, k) in enumerate(zip(Y, idx)):
                # Compute the star distance transform for the current labeled image
                dist = star_dist(lbl, self.n_rays, mode=self.sd_mode, grid=self.grid)
                dist_list.append(dist)

                    # --- use actual filename if available ---
                if self.filenames is not None:
                    base = _safe_name(self.filenames[k])   # k is global index into X/Y
                else:
                    base = f"image_{k:04d}"
                # Compute EDT after the star distance transform
                unique_id = f"{base}_mode_{self.edt_mode}"
                prob = edt_prob(lbl[self.b][self.ss_grid[1:3]], dist, mode=self.edt_mode, output_prefix=unique_id, debug_dir=self.scoring_debug_dir,)

                if self.debug_plots:
                    lbl_patch = lbl[self.b][self.ss_grid[1:3]]
                    metrics = visualize_polygons_rays_with_scores(
                        label_img=lbl_patch,
                        dist_map=dist,
                        score_map=prob,
                        scale=0.5,
                        rays_every=1,
                        peak_radius=1,
                        peak_rel_threshold=0.9,
                        output_stem=unique_id,
                        save_dir=self.debug_dir,
                    )
                    all_selected_centroid_polygon_areas.extend(metrics["selected_centroid_polygon_areas"].tolist())
                    all_object_pixel_polygon_areas.extend(metrics["all_object_pixel_polygon_areas"].tolist())
                    all_score_map_values.extend(metrics["score_map_values"].tolist())
                    all_ray_distance_values.extend(metrics["ray_distance_values"].tolist())
                prob_list.append(prob)

            if self.debug_plots and self.debug_dir is not None:
                debug_root = Path(self.debug_dir)
                global_hist_root = debug_root / "histograms" / "across_all_images"
                global_selected_centroid_area_dir = global_hist_root / "selected_centroid_polygon_areas"
                global_all_object_area_dir = global_hist_root / "all_object_pixel_polygon_areas"
                global_score_map_dir = global_hist_root / "score_map_values"
                global_ray_distance_dir = global_hist_root / "ray_distance_values"
                global_selected_centroid_area_dir.mkdir(parents=True, exist_ok=True)
                global_all_object_area_dir.mkdir(parents=True, exist_ok=True)
                global_score_map_dir.mkdir(parents=True, exist_ok=True)
                global_ray_distance_dir.mkdir(parents=True, exist_ok=True)

                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                global_selected_centroid_hist_path = global_selected_centroid_area_dir / f"batch_{i:04d}_mode_{self.edt_mode}_hist_{ts}.png"
                global_all_object_area_hist_path = global_all_object_area_dir / f"batch_{i:04d}_mode_{self.edt_mode}_hist_{ts}.png"
                global_score_map_hist_path = global_score_map_dir / f"batch_{i:04d}_mode_{self.edt_mode}_hist_{ts}.png"
                global_ray_distance_hist_path = global_ray_distance_dir / f"batch_{i:04d}_mode_{self.edt_mode}_hist_{ts}.png"

                style_ctx = plt.style.context(_SCIENCE_STYLE) if _SCIENCE_STYLE else plt.style.context("default")
                with style_ctx:
                    fig, ax = plt.subplots(figsize=(9, 5))
                    if all_selected_centroid_polygon_areas:
                        n_bins = int(min(40, max(8, np.sqrt(len(all_selected_centroid_polygon_areas)))))
                        ax.hist(all_selected_centroid_polygon_areas, bins=n_bins, color="steelblue", edgecolor="black", alpha=0.85)
                        ax.set_title("Selected Centroid Polygon Area Distribution Across All Images", fontsize=12)
                        ax.set_xlabel("Polygon area (pixels)")
                        ax.set_ylabel("Count")
                        ax.set_xlim(0, 200)
                        ax.set_ylim(0, 700)
                        ax.grid(alpha=0.25)
                    else:
                        ax.text(0.5, 0.5, "No polygons available", ha="center", va="center", transform=ax.transAxes)
                        ax.set_title("Selected Centroid Polygon Area Distribution Across All Images", fontsize=12)
                        ax.set_xticks([])
                        ax.set_yticks([])

                    fig.tight_layout()
                    fig.savefig(global_selected_centroid_hist_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
                    plt.close(fig)

                style_ctx = plt.style.context(_SCIENCE_STYLE) if _SCIENCE_STYLE else plt.style.context("default")
                with style_ctx:
                    area_fig, area_ax = plt.subplots(figsize=(9, 5))
                    if all_object_pixel_polygon_areas:
                        all_object_areas_arr = np.asarray(all_object_pixel_polygon_areas, dtype=np.float32)
                        n_bins = int(min(40, max(8, np.sqrt(all_object_areas_arr.size))))
                        area_ax.hist(all_object_areas_arr, bins=n_bins, color="darkorange", edgecolor="black", alpha=0.85)
                        area_ax.set_title("All Object-Pixel Polygon Area Distribution Across All Images", fontsize=12)
                        area_ax.set_xlabel("Polygon area (pixels)")
                        area_ax.set_ylabel("Count")
                        area_ax.set_xlim(0, 200)
                        area_ax.set_ylim(0, 700)
                        area_ax.grid(alpha=0.25)
                    else:
                        area_ax.text(0.5, 0.5, "No object areas available", ha="center", va="center", transform=area_ax.transAxes)
                        area_ax.set_title("All Object-Pixel Polygon Area Distribution Across All Images", fontsize=12)
                        area_ax.set_xticks([])
                        area_ax.set_yticks([])

                    area_fig.tight_layout()
                    area_fig.savefig(global_all_object_area_hist_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
                    plt.close(area_fig)

                style_ctx = plt.style.context(_SCIENCE_STYLE) if _SCIENCE_STYLE else plt.style.context("default")
                with style_ctx:
                    score_fig, score_ax = plt.subplots(figsize=(9, 5))
                    if all_score_map_values:
                        all_score_map_values_arr = np.asarray(all_score_map_values, dtype=np.float32)
                        n_bins = int(min(60, max(10, np.sqrt(all_score_map_values_arr.size))))
                        score_ax.hist(all_score_map_values_arr, bins=n_bins, color="mediumpurple", edgecolor="black", alpha=0.85)
                        score_ax.set_title("Score Map Value Distribution Across All Images", fontsize=12)
                        score_ax.set_xlabel("Score value")
                        score_ax.set_ylabel("Count")
                        score_ax.set_xlim(0, 1)
                        score_ax.set_ylim(0, 6000)
                        score_ax.grid(alpha=0.25)
                    else:
                        score_ax.text(0.5, 0.5, "No score values available", ha="center", va="center", transform=score_ax.transAxes)
                        score_ax.set_title("Score Map Value Distribution Across All Images", fontsize=12)
                        score_ax.set_xticks([])
                        score_ax.set_yticks([])

                    score_fig.tight_layout()
                    score_fig.savefig(global_score_map_hist_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
                    plt.close(score_fig)

                style_ctx = plt.style.context(_SCIENCE_STYLE) if _SCIENCE_STYLE else plt.style.context("default")
                with style_ctx:
                    ray_fig, ray_ax = plt.subplots(figsize=(9, 5))
                    if all_ray_distance_values:
                        all_ray_distance_values_arr = np.asarray(all_ray_distance_values, dtype=np.float32)
                        n_bins = int(min(80, max(20, np.sqrt(all_ray_distance_values_arr.size))))
                        ray_ax.hist(all_ray_distance_values_arr, bins=n_bins, color="seagreen", edgecolor="black", alpha=0.85)
                        ray_ax.set_title("Ray Distance Distribution Across All Images", fontsize=12)
                        ray_ax.set_xlabel("Ray distance")
                        ray_ax.set_ylabel("Count")
                        ray_ax.set_xlim(0, 150)
                        ray_ax.set_ylim(0, 2.5e6)
                        ray_ax.grid(alpha=0.25)
                    else:
                        ray_ax.text(0.5, 0.5, "No ray distances available", ha="center", va="center", transform=ray_ax.transAxes)
                        ray_ax.set_title("Ray Distance Distribution Across All Images", fontsize=12)
                        ray_ax.set_xticks([])
                        ray_ax.set_yticks([])

                    ray_fig.tight_layout()
                    ray_fig.savefig(global_ray_distance_hist_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
                    plt.close(ray_fig)

            # Stack the results into numpy arrays
            dist = np.stack(dist_list)
            prob = np.stack(prob_list)
            dist_mask = prob

        X = np.stack(X)
        if X.ndim == 3: # input image has no channel axis
            X = np.expand_dims(X,-1)
        prob = np.expand_dims(prob,-1)
        dist_mask = np.expand_dims(dist_mask,-1)

        # append dist_mask to dist as additional channel
        # dist_and_mask = np.concatenate([dist,dist_mask],axis=-1)
        # faster than concatenate
        dist_and_mask = np.empty(dist.shape[:-1]+(self.n_rays+1,), np.float32)
        dist_and_mask[...,:-1] = dist
        dist_and_mask[...,-1:] = dist_mask

        if has_neg_labels:
            prob[mask_neg_labels] = -1  # set to -1 to disable loss

        # note: must return tuples in keras 3 (cf. https://stackoverflow.com/a/78158487)
        if self.n_classes is None:
            return _gen_rtype((X,)), _gen_rtype((prob,dist_and_mask))
        else:
            prob_class = np.stack(tuple((mask_to_categorical(y[self.b], self.n_classes, self.classes[k]) for y,k in zip(Y, idx))))

            # TODO: investigate downsampling via simple indexing vs. using 'zoom'
            # prob_class = prob_class[self.ss_grid]
            # 'zoom' might lead to better registered maps (especially if upscaled later)
            prob_class = zoom(prob_class, (1,)+tuple(1/g for g in self.grid)+(1,), order=0)

            if has_neg_labels:
                prob_class[mask_neg_labels] = -1  # set to -1 to disable loss

            return _gen_rtype((X,)), _gen_rtype((prob,dist_and_mask, prob_class))

