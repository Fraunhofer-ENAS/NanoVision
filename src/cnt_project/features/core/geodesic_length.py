from scipy.sparse.csgraph import shortest_path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.ndimage import distance_transform_edt

def trim(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    Crop a boolean mask to its tight bounding box.

    Returns:
        tm: cropped mask
        r0, c0: top-left offset (so that (r, c) in tm maps to (r+r0, c+c0) in original mask)
    """
    mask = mask.astype(bool)
    if not np.any(mask):
        return mask.copy(), 0, 0

    rr, cc = np.nonzero(mask)
    r0, r1 = int(rr.min()), int(rr.max())
    c0, c1 = int(cc.min()), int(cc.max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), r0, c0


def start_pixels(mask: np.ndarray) -> np.ndarray:
    """
    Notebook 'shell' pixels: keep only top row and left column of the trimmed mask,
    but only where mask==True.
    """
    shell = mask.copy()
    shell[1:, 1:] = False
    return shell


def offset_to_slice(offset: int) -> slice:
    """
    Helper for vectorized neighbor pairing.

    For an offset (dy or dx), returns the slice that aligns a shifted view of an array
    with its opposite-shifted view (same trick as notebook).
    """
    if offset == 0:
        return slice(None)
    if offset > 0:
        return slice(offset, None)
    return slice(None, offset)


def adjacency_matrix_center_weighted(
    mask: np.ndarray,
    alpha: float = 2.0,
    eps: float = 1e-3,
):
    """
    Build a weighted 8-connected pixel graph on True pixels of `mask`.

    Nodes:
        One node per True pixel, indexed by `idx` in [0..N_true-1].

    Edges:
        Between 8-neighbors that are both True.
        Base step length is 1 (orth) or sqrt(2) (diag).

    Weights (costs):
        w = base_step * (1 + alpha / (d + eps))
        where d is the average distance-transform value of the two pixels.
        -> penalizes boundary pixels (small d), encourages central paths.

    Returns:
        A: CSR sparse adjacency (N_true x N_true)
        true_flat: flat indices into mask.ravel() for True pixels (node-id order)
        idx: int array same shape as mask, idx[r,c] gives node id or -1
    """
    mask = mask.astype(bool)

    idx = np.full(mask.shape, -1, dtype=int)
    true_flat = np.flatnonzero(mask)
    idx.flat[true_flat] = np.arange(true_flat.size)
    N_true = true_flat.size

    if N_true == 0:
        return coo_matrix((0, 0)).tocsr(), true_flat, idx

    D = distance_transform_edt(mask).astype(np.float32)

    rows, cols, weights = [], [], []

    # 8-neighborhood offsets (exclude center), same spirit as notebook
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dy == 0 and dx == 0)]

    for dy, dx in offsets:
        sv1 = offset_to_slice(dy)
        sv2 = offset_to_slice(-dy)
        sh1 = offset_to_slice(dx)
        sh2 = offset_to_slice(-dx)

        m = mask[sv1, sh1] & mask[sv2, sh2]
        if not np.any(m):
            continue

        r = idx[sv1, sh1][m]
        c = idx[sv2, sh2][m]

        base = float(np.sqrt(dy * dy + dx * dx))  # 1 or sqrt(2)

        d1 = D[sv1, sh1][m]
        d2 = D[sv2, sh2][m]
        d = 0.5 * (d1 + d2)

        penalty = 1.0 + alpha * (1.0 / (d + eps))
        w = (base * penalty).astype(np.float32)

        rows.append(r)
        cols.append(c)
        weights.append(w)

    if len(rows) == 0:
        # isolated pixels (no edges)
        return coo_matrix((N_true, N_true)).tocsr(), true_flat, idx

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    weights = np.concatenate(weights)

    A = coo_matrix((weights, (rows, cols)), shape=(N_true, N_true))
    A = (A + A.T).tocsr()
    return A, true_flat, idx


def path_step_stats(path_rc: np.ndarray) -> dict:
    """
    Compute geometric step counts and geometric path length in pixel units.

    For a valid 8-neighbor path:
        orth step length = 1
        diag step length = sqrt(2)

    Returns:
        dict with n_steps, n_orth, n_diag, n_other, length_px
    """
    if path_rc is None or path_rc.shape[0] < 2:
        return {"n_steps": 0, "n_orth": 0, "n_diag": 0, "n_other": 0, "length_px": 0.0}

    diffs = path_rc[1:] - path_rc[:-1]
    dr = np.abs(diffs[:, 0])
    dc = np.abs(diffs[:, 1])

    n_diag = int(np.sum((dr == 1) & (dc == 1)))
    n_orth = int(np.sum(((dr == 1) & (dc == 0)) | ((dr == 0) & (dc == 1))))
    n_other = int(np.sum(~(((dr == 1) & (dc == 1)) | ((dr == 1) & (dc == 0)) | ((dr == 0) & (dc == 1)))))

    length_px = float(n_orth * 1.0 + n_diag * np.sqrt(2))

    # If anything weird appears (shouldn't), fall back to Euclidean sum
    if n_other > 0:
        step = np.sqrt((diffs[:, 0].astype(float) ** 2) + (diffs[:, 1].astype(float) ** 2))
        length_px = float(np.sum(step))

    return {
        "n_steps": int(path_rc.shape[0] - 1),
        "n_orth": n_orth,
        "n_diag": n_diag,
        "n_other": n_other,
        "length_px": length_px,
    }


def longest_shortest_path_weighted(A, allowed: np.ndarray, return_path: bool = True):
    """
    Notebook-like objective on a (possibly weighted) graph:

        max_{s in allowed}  max_{v in V}  dist(s, v)

    Implementation:
        For memory safety we run shortest_path once per s in allowed
        (equivalent maximum distance; tie-breaking may differ from the notebook's
        multi-source argmax, but distance value is the same).

    Returns:
        best_cost: float
        best_path_nodes: list[int] of node ids (only if return_path=True)
        best_start: int | None
        best_goal: int | None
    """
    allowed = np.asarray(allowed, dtype=int)
    if allowed.size == 0:
        return 0.0, [], None, None

    best_cost = -np.inf
    best_start = None
    best_goal = None
    best_pred = None

    for s in allowed:
        dist, pred = shortest_path(A, directed=False, return_predecessors=True, indices=int(s))

        finite = np.isfinite(dist)
        if not np.any(finite):
            continue

        g = int(np.argmax(np.where(finite, dist, -np.inf)))
        d = float(dist[g])

        if d > best_cost:
            best_cost = d
            best_start = int(s)
            best_goal = g
            best_pred = pred

    if best_start is None or not np.isfinite(best_cost):
        return 0.0, [], None, None

    if not return_path:
        return float(best_cost), [], best_start, best_goal

    # Reconstruct node-id path from goal back to start using predecessor chain
    path_nodes: list[int] = []
    cur = int(best_goal)
    while cur != -9999:
        path_nodes.append(int(cur))
        if cur == best_start:
            break
        cur = int(best_pred[cur])

    path_nodes.reverse()
    return float(best_cost), path_nodes, best_start, best_goal


def polygon_mask_geodesic_length(
    poly_mask: np.ndarray,
    microns_per_pixel: float,
    alpha: float = 2.0,
    eps: float = 1e-3,
    return_path: bool = False,
    return_debug: bool = False,
):
    """
    Compute CNT length from a filled polygon mask using the "weighted notebook" approach.

    Steps (mirrors notebook flow):
        1) trim mask to bounding box
        2) build adjacency graph on True pixels (8-connected)
        3) define allowed starts using notebook's shell (top row + left col)
        4) run shortest paths from allowed starts and take the maximum distance
        5) reconstruct the corresponding path
        6) report geometric length of that path (orth=1, diag=sqrt(2)), and convert to µm

    Returns (depending on flags):
        if not return_path and not return_debug:
            (length_px, length_um)
        if return_path and not return_debug:
            (length_px, length_um, path_rc_full)
        if return_debug:
            adds debug dict containing weighted_cost, step stats, endpoints, allowed coords
    """
    mask_bool = poly_mask.astype(bool)
    tm, r0, c0 = trim(mask_bool)
    if tm.size == 0 or not np.any(tm):
        empty_path = np.zeros((0, 2), dtype=int)
        if return_debug:
            dbg = {"alpha": float(alpha), "eps": float(eps), "weighted_cost": 0.0, "step_stats": path_step_stats(empty_path)}
            if return_path:
                return 0.0, 0.0, empty_path, dbg
            return 0.0, 0.0, dbg
        if return_path:
            return 0.0, 0.0, empty_path
        return 0.0, 0.0

    A, true_flat, idx = adjacency_matrix_center_weighted(tm, alpha=alpha, eps=eps)

    sm = start_pixels(tm)

    # Notebook-style allowed: mark shell membership for each True pixel in node-id order
    allowed = np.flatnonzero(sm.flat[true_flat]).astype(int)

    if allowed.size == 0:
        empty_path = np.zeros((0, 2), dtype=int)
        if return_debug:
            dbg = {"alpha": float(alpha), "eps": float(eps), "weighted_cost": 0.0, "step_stats": path_step_stats(empty_path)}
            if return_path:
                return 0.0, 0.0, empty_path, dbg
            return 0.0, 0.0, dbg
        if return_path:
            return 0.0, 0.0, empty_path
        return 0.0, 0.0

    weighted_cost, path_nodes, best_start, best_goal = longest_shortest_path_weighted(
        A, allowed, return_path=True
    )

    if len(path_nodes) == 0:
        empty_path = np.zeros((0, 2), dtype=int)
        if return_debug:
            dbg = {"alpha": float(alpha), "eps": float(eps), "weighted_cost": float(weighted_cost), "step_stats": path_step_stats(empty_path)}
            if return_path:
                return 0.0, 0.0, empty_path, dbg
            return 0.0, 0.0, dbg
        if return_path:
            return 0.0, 0.0, empty_path
        return 0.0, 0.0

    # Map node ids -> tm (row,col) -> full-image (row,col)
    flat_idxs = true_flat[np.array(path_nodes, dtype=int)]
    rr, cc = np.unravel_index(flat_idxs, tm.shape)
    path_rc_full = np.column_stack([rr + r0, cc + c0]).astype(int)

    # Final reported length = geometric length along the chosen path
    stats = path_step_stats(path_rc_full)
    length_px = float(stats["length_px"])
    length_um = float(length_px * microns_per_pixel)

    if not return_debug:
        if return_path:
            return length_px, length_um, path_rc_full
        return length_px, length_um

    # Optional debug info for plots
    allowed_flat = true_flat[allowed]
    ar, ac = np.unravel_index(allowed_flat, tm.shape)
    allowed_rc_full = np.column_stack([ar + r0, ac + c0]).astype(int)

    # Start/goal in full coords
    start_rc = None
    goal_rc = None
    if best_start is not None:
        sflat = true_flat[int(best_start)]
        sr, sc = np.unravel_index(sflat, tm.shape)
        start_rc = (int(sr + r0), int(sc + c0))
    if best_goal is not None:
        gflat = true_flat[int(best_goal)]
        gr, gc = np.unravel_index(gflat, tm.shape)
        goal_rc = (int(gr + r0), int(gc + c0))

    debug = {
        "alpha": float(alpha),
        "eps": float(eps),
        "weighted_cost": float(weighted_cost),
        "allowed_rc_full": allowed_rc_full,
        "start_rc": start_rc,
        "goal_rc": goal_rc,
        "step_stats": stats,
    }

    if return_path:
        return length_px, length_um, path_rc_full, debug
    return length_px, length_um, debug
