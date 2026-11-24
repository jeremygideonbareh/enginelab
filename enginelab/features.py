"""Geometric and assembly-context descriptors for each part.

Shape features describe the part on its own (size, proportions, holes, how
round it is, its radial profile, teeth...). Context features describe how it
sits in the assembly (how many identical copies, whether they line up, which
parts share its axis...). Both feed the classifier.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, cKDTree

N_SAMPLES = 1500
PROFILE_BINS = 8
D2_BINS = 8


def winding_number(mesh: trimesh.Trimesh, points: np.ndarray, chunk: int = 64) -> np.ndarray:
    """Generalised winding number: ~1 inside a closed surface, ~0 outside.

    Works on meshes with small holes too, unlike ray casting.
    """
    tri = mesh.triangles  # (F,3,3)
    out = np.zeros(len(points))
    for s in range(0, len(points), chunk):
        p = points[s:s + chunk]
        a = tri[None, :, 0, :] - p[:, None, :]
        b = tri[None, :, 1, :] - p[:, None, :]
        c = tri[None, :, 2, :] - p[:, None, :]
        la, lb, lc = (np.linalg.norm(v, axis=2) for v in (a, b, c))
        det = np.einsum("pfi,pfi->pf", a, np.cross(b, c))
        div = (la * lb * lc + np.einsum("pfi,pfi->pf", a, b) * lc
               + np.einsum("pfi,pfi->pf", b, c) * la + np.einsum("pfi,pfi->pf", c, a) * lb)
        out[s:s + chunk] = np.arctan2(det, div).sum(1) / (2 * np.pi)
    return out


@dataclass
class ShapeInfo:
    """Per-part geometry kept for context features, specs and the UI."""
    center: np.ndarray
    axes: np.ndarray            # rows: principal directions, longest extent first
    extents: np.ndarray         # sizes along `axes`
    sym_axis: np.ndarray
    sym_score: float
    sym_center: np.ndarray      # a point on the symmetry axis
    sym_length: float           # length along the symmetry axis
    sym_diameter: float         # max diameter around the symmetry axis
    volume: float
    area: float
    watertight: bool
    genus: int
    features: dict = field(default_factory=dict)


def _principal_frame(points: np.ndarray, weights: np.ndarray | None = None):
    c = np.average(points, axis=0, weights=weights)
    X = points - c
    if weights is not None:
        X = X * np.sqrt(weights / weights.mean())[:, None]
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    proj = (points - c) @ vt.T
    lo, hi = proj.min(0), proj.max(0)
    ext = hi - lo
    order = np.argsort(-ext)
    axes = vt[order]
    ext = ext[order]
    center = c + ((lo + hi) / 2)[order] @ axes
    return center, axes, ext


def _axis_symmetry(pts: np.ndarray, center: np.ndarray, axis: np.ndarray, n_t: int = 12, n_sec: int = 8):
    """How close the point set is to a body of revolution around `axis`."""
    u = np.cross(axis, [1.0, 0, 0])
    if np.linalg.norm(u) < 0.1:
        u = np.cross(axis, [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    d = pts - center
    t = d @ axis
    x, y = d @ u, d @ v
    r = np.hypot(x, y)
    ang = np.arctan2(y, x)
    t_lo, t_hi = t.min(), t.max()
    tb = np.clip(((t - t_lo) / (t_hi - t_lo + 1e-12) * n_t).astype(int), 0, n_t - 1)
    sb = np.clip(((ang + np.pi) / (2 * np.pi) * n_sec).astype(int), 0, n_sec - 1)
    sums = np.zeros((n_t, n_sec))
    cnt = np.zeros((n_t, n_sec))
    np.add.at(sums, (tb, sb), r)
    np.add.at(cnt, (tb, sb), 1)
    occupied = cnt.sum(1) > 4
    mean_r = np.where(cnt > 0, sums / np.maximum(cnt, 1), 0.0)
    rmax = r.max() + 1e-12
    if not occupied.any():
        return 0.0, r, t, ang
    spread = mean_r[occupied].std(1) / (mean_r[occupied].mean(1) + 0.05 * rmax)
    score = float(np.clip(1 - 2.5 * spread.mean(), 0, 1))
    return score, r, t, ang


def shape_features(mesh: trimesh.Trimesh, seed: int = 0) -> ShapeInfo:
    rng = np.random.default_rng(seed)
    pts, fidx = trimesh.sample.sample_surface(mesh, N_SAMPLES, seed=int(rng.integers(1 << 31)))
    normals = mesh.face_normals[fidx]
    watertight = bool(mesh.is_watertight)
    pca_center, pca_axes, _ = _principal_frame(pts)
    frame, sym_center, candidates = pca_axes, pca_center, list(pca_axes)
    if watertight:
        # The mass-inertia frame of a solid is exact; point-cloud PCA wobbles when
        # two dimensions are similar (pistons, springs, gears).
        try:
            iv = np.asarray(mesh.principal_inertia_vectors, float)
            com = np.asarray(mesh.center_mass, float)
            if np.all(np.isfinite(iv)) and np.all(np.isfinite(com)) and abs(np.linalg.det(iv)) > 0.5:
                frame, sym_center = iv, com
                candidates = list(iv) + candidates
        except Exception:
            pass
    proj = pts @ frame.T
    lo, hi = proj.min(0), proj.max(0)
    order = np.argsort(-(hi - lo))
    axes = frame[order]
    ext = np.maximum((hi - lo)[order], 1e-6)
    center = ((lo + hi) / 2)[order] @ axes
    L1, L2, L3 = ext
    area = float(mesh.area)
    # Signed volume is still accurate when only a few small holes remain.
    edge_groups = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=None)
    open_edges = sum(1 for g in edge_groups if len(g) == 1)
    closed_frac = 1.0 - open_edges / max(len(edge_groups), 1)
    volume = float(abs(mesh.volume)) if (watertight or closed_frac > 0.98) else 0.0
    try:
        hull = ConvexHull(mesh.vertices if len(mesh.vertices) < 20000 else pts)
        hull_vol = float(hull.volume)
    except Exception:
        hull_vol = float(L1 * L2 * L3)
    if volume <= 0:
        volume = 0.5 * hull_vol  # rough guess for open meshes
    genus = 0
    if watertight:
        chi = len(mesh.vertices) - len(mesh.edges_unique) + len(mesh.faces)
        genus = int(max(0, round(1 - chi / 2)))

    # Symmetry: test candidate axes, then refine the best one with a local search.
    best_s, sym_axis = -1.0, axes[0]
    for a in candidates:
        s = _axis_symmetry(pts, sym_center, a)[0]
        if s > best_s + 0.02:
            best_s, sym_axis = s, a
    for step in (np.radians(6), np.radians(2)):
        for _ in range(10):
            jitter = rng.normal(size=3)
            jitter -= (jitter @ sym_axis) * sym_axis
            trial = sym_axis + np.tan(step) * jitter / (np.linalg.norm(jitter) + 1e-12)
            trial /= np.linalg.norm(trial)
            s = _axis_symmetry(pts, sym_center, trial)[0]
            if s > best_s + 0.01:  # ignore sampling noise
                best_s, sym_axis = s, trial
    sym_score, r, t, ang = _axis_symmetry(pts, sym_center, sym_axis)
    sym_k = int(np.argmax(np.abs(axes @ sym_axis)))
    sym_len = float(t.max() - t.min())
    sym_diam = float(2 * r.max())
    if abs(axes[sym_k] @ sym_axis) > 0.995:
        # Axis matches the frame: the perpendicular box sides give a noise-free diameter.
        others = [k for k in range(3) if k != sym_k]
        sym_diam = float(max(ext[others[0]], ext[others[1]]))
        sym_len = float(ext[sym_k])

    # Radial profile along the symmetry axis (max radius per slice), heavier end first.
    tb = np.clip(((t - t.min()) / (sym_len + 1e-12) * PROFILE_BINS).astype(int), 0, PROFILE_BINS - 1)
    prof = np.zeros(PROFILE_BINS)
    np.maximum.at(prof, tb, r)
    prof /= prof.max() + 1e-12
    if prof[: PROFILE_BINS // 2].sum() < prof[PROFILE_BINS // 2:].sum():
        prof = prof[::-1]

    # Is the material on the axis? (solid shaft = 1, tube/ring = 0, cup = partial)
    def axis_fill(axis, length, origin=center):
        q = origin + np.linspace(-0.4, 0.4, 9)[:, None] * length * axis
        return float((winding_number(mesh, q) > 0.5).mean())

    mid_t = 0.5 * (t.max() + t.min())
    fill_sym = axis_fill(sym_axis, sym_len, sym_center + mid_t * sym_axis)
    fill_long = axis_fill(axes[0], L1)
    fill_short = axis_fill(axes[2], L3)
    center_inside = float(winding_number(mesh, center[None])[0] > 0.5)

    # Surface orientation.
    flat = [float(((np.abs(normals @ axes[k]) > 0.95)).mean()) for k in range(3)]
    d = pts - sym_center
    radial = d - np.outer(d @ sym_axis, sym_axis)
    radial /= np.linalg.norm(radial, axis=1, keepdims=True) + 1e-12
    cyl_frac = float((np.abs(np.einsum("ij,ij->i", normals, radial)) > 0.95).mean())
    end_frac = float((np.abs(normals @ sym_axis) > 0.95).mean())

    # Teeth / hex detection: angular spectrum of the outer rim.
    rim = r > 0.85 * r.max()
    feats_ang = np.zeros(2)
    if rim.sum() > 32 and sym_score > 0.3:
        bins = 96
        ab = np.clip(((ang[rim] + np.pi) / (2 * np.pi) * bins).astype(int), 0, bins - 1)
        rr = np.zeros(bins)
        np.maximum.at(rr, ab, r[rim])
        rr[rr == 0] = np.median(rr[rr > 0])
        spec = np.abs(np.fft.rfft(rr - rr.mean()))
        amp = 2 * spec / bins / (rr.mean() + 1e-12)   # harmonic amplitude relative to radius
        feats_ang[0] = min(float(amp[10:].max()) * 10, 1.0)   # many teeth
        feats_ang[1] = min(float(amp[6]) * 10, 1.0)            # hexagon
    # Shape distribution D2.
    i = rng.integers(0, len(pts), 600)
    j = rng.integers(0, len(pts), 600)
    dist = np.linalg.norm(pts[i] - pts[j], axis=1)
    d2, _ = np.histogram(dist / (dist.max() + 1e-12), bins=D2_BINS, range=(0, 1))
    d2 = d2 / d2.sum()

    wall = (2 * volume / area) / L3 if area > 0 else 0
    f = {
        "log_L1": np.log10(L1),
        "ar2": L2 / L1,
        "ar3": L3 / L1,
        "log_volume": np.log10(volume + 1e-9),
        "sphericity": (36 * np.pi * volume ** 2) ** (1 / 3) / area if area > 0 else 0,
        "convexity": min(volume / hull_vol, 1.0) if hull_vol > 0 else 0,
        "box_fill": min(volume / (L1 * L2 * L3), 1.0),
        "wall_ratio": min(wall, 2.0),
        "genus": min(genus, 30),
        "log_genus": np.log1p(genus),
        "watertight": float(watertight),
        "sym_score": sym_score,
        "sym_is_long": float(sym_k == 0),
        "sym_is_mid": float(sym_k == 1),
        "sym_is_short": float(sym_k == 2),
        "slender": sym_len / (sym_diam + 1e-9),
        "profile_std": float(prof.std()),
        "fill_sym": fill_sym,
        "fill_long": fill_long,
        "fill_short": fill_short,
        "center_inside": center_inside,
        "flat1": flat[0], "flat2": flat[1], "flat3": flat[2],
        "cyl_frac": cyl_frac,
        "end_frac": end_frac,
        "teeth": feats_ang[0],
        "hex": feats_ang[1],
    }
    for k in range(PROFILE_BINS):
        f[f"prof{k}"] = prof[k]
    for k in range(D2_BINS):
        f[f"d2_{k}"] = d2[k]
    return ShapeInfo(center=center, axes=axes, extents=ext, sym_axis=sym_axis, sym_score=sym_score,
                     sym_center=sym_center + mid_t * sym_axis,
                     sym_length=sym_len, sym_diameter=sym_diam, volume=volume, area=area,
                     watertight=watertight, genus=genus, features={k: float(v) for k, v in f.items()})


def _similar(a: ShapeInfo, b: ShapeInfo, tol: float = 0.04) -> bool:
    """Identical copies: same volume, surface area, size and shape signature."""
    if abs(a.volume - b.volume) > tol * max(a.volume, b.volume):
        return False
    if abs(a.area - b.area) > tol * max(a.area, b.area):
        return False
    if abs(a.genus - b.genus) > 0 or abs(a.sym_score - b.sym_score) > 0.15:
        return False
    return bool(np.all(np.abs(a.extents - b.extents) <= tol * np.maximum(a.extents, b.extents) + 1e-6))


def copy_groups(infos: list[ShapeInfo]) -> list[int]:
    """Group index per part; parts in the same group are identical copies."""
    group = [-1] * len(infos)
    g = 0
    for i in range(len(infos)):
        if group[i] >= 0:
            continue
        group[i] = g
        for j in range(i + 1, len(infos)):
            if group[j] < 0 and _similar(infos[i], infos[j]):
                group[j] = g
        g += 1
    return group


def context_features(infos: list[ShapeInfo]) -> list[dict]:
    n = len(infos)
    centers = np.array([s.center for s in infos])
    vols = np.array([max(s.volume, 1e-9) for s in infos])
    lo = np.array([s.center - 0.5 * np.abs(s.axes.T) @ s.extents for s in infos])
    hi = np.array([s.center + 0.5 * np.abs(s.axes.T) @ s.extents for s in infos])
    all_lo, all_hi = lo.min(0), hi.max(0)
    diag = float(np.linalg.norm(all_hi - all_lo)) or 1.0
    vmax = vols.max()

    # Main axis of the assembly from the volume-weighted spread of part centres.
    if n >= 3:
        _, main_axes, _ = _principal_frame(np.vstack([lo, hi]))
        main = main_axes[0]
    else:
        main = infos[0].axes[0]

    groups = copy_groups(infos)
    gid = np.array(groups)
    rank = np.argsort(np.argsort(-vols)) / max(n - 1, 1)
    tol = 0.01 * diag
    overlap = np.all((lo[:, None, :] <= hi[None, :, :] + tol) & (hi[:, None, :] >= lo[None, :, :] - tol), axis=2)
    np.fill_diagonal(overlap, False)
    inside = np.all((centers[:, None, :] >= lo[None, :, :]) & (centers[:, None, :] <= hi[None, :, :]), axis=2)
    np.fill_diagonal(inside, False)

    out = []
    for i, s in enumerate(infos):
        members = np.nonzero(gid == gid[i])[0]
        ncopy = len(members)
        line_lin, axis_vs_line, long_vs_line, spacing = 0.0, 0.0, 0.0, 0.0
        if ncopy >= 2:
            c = centers[members]
            cc = c - c.mean(0)
            _, sv, vt = np.linalg.svd(cc, full_matrices=False)
            line_lin = float(sv[0] / (sv.sum() + 1e-12))
            ld = vt[0]
            axis_vs_line = abs(float(s.sym_axis @ ld))
            long_vs_line = abs(float(s.axes[0] @ ld))
            tree = cKDTree(c)
            dd, _ = tree.query(c, k=2)
            spacing = float(np.median(dd[:, 1]) / (s.extents[0] + 1e-9))
        coax = 0
        if s.sym_score > 0.55:
            for j, o in enumerate(infos):
                if j == i or o.sym_score < 0.55 or abs(float(s.sym_axis @ o.sym_axis)) < 0.97:
                    continue
                dvec = o.center - s.center
                off = dvec - (dvec @ s.sym_axis) * s.sym_axis
                if np.linalg.norm(off) < 0.08 * max(s.sym_diameter, o.sym_diameter):
                    coax += 1
        bigger = vols > vols[i]
        out.append({
            "rel_L1": s.extents[0] / diag,
            "rel_vol": np.log10(vols[i] / vmax),
            "vol_rank": float(rank[i]),
            "copies": np.log2(ncopy),
            "copy_line": line_lin,
            "axis_vs_line": axis_vs_line,
            "long_vs_line": long_vs_line,
            "copy_spacing": min(spacing, 5.0),
            "neighbors": np.log1p(overlap[i].sum()),
            "inside_bigger": np.log1p((inside[i] & bigger).sum()),
            "encloses": np.log1p(inside[:, i].sum()),
            "sym_vs_main": abs(float(s.sym_axis @ main)),
            "long_vs_main": abs(float(s.axes[0] @ main)),
            "coaxial": np.log1p(coax),
        })
    return [{k: float(v) for k, v in d.items()} for d in out]


def feature_vector(shape: dict, ctx: dict, names: list[str]) -> np.ndarray:
    merged = {**shape, **ctx}
    return np.array([merged[k] for k in names], dtype=np.float64)


def all_feature_names() -> list[str]:
    import trimesh as _t
    box = _t.creation.box((3, 2, 1))
    s = shape_features(box)
    c = context_features([s, s])[0]
    return list(s.features) + list(c)
