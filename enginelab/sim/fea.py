"""Voxel finite-element analysis for any part (linear elastic, small strain).

The part is voxelised along a chosen load axis, every voxel becomes an
8-node hexahedral element (2x2x2 Gauss integration), and the system is solved
with a sparse direct solver. Load cases:

  axial     fix the -axis end, push/pull the +axis end with a total force
  bending   fix the -axis end, transverse force on the +axis end (cantilever)
  torsion   fix the -axis end, torque about the axis on the +axis end
  pressure  fix the -axis end, pressure on the +axis facing surface

Results: von Mises stress per voxel, peak stress and where it is, maximum
displacement, safety factor against yield at the given temperature, and a
stress value for every mesh vertex so the viewer can paint a heat map.
Voxel FEA is coarse at sharp corners; treat peaks as estimates (+-15%).
"""
from __future__ import annotations

import math
import time

import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from ..materials import Material

_GP = np.array([-1, 1]) / math.sqrt(3)
_NODES = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                   [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)


def _D(E: float, nu: float) -> np.ndarray:
    c = E / ((1 + nu) * (1 - 2 * nu))
    D = np.zeros((6, 6))
    D[:3, :3] = nu
    np.fill_diagonal(D[:3, :3], 1 - nu)
    D[3:, 3:] = np.eye(3) * (1 - 2 * nu) / 2
    return c * D


def _B(xi, eta, zeta, h):
    """Strain-displacement matrix of a cube of side h at a natural coordinate."""
    dN = np.zeros((8, 3))
    for i, (a, b, c) in enumerate(_NODES):
        dN[i] = [a * (1 + b * eta) * (1 + c * zeta), b * (1 + a * xi) * (1 + c * zeta),
                 c * (1 + a * xi) * (1 + b * eta)]
    dN = dN / 8 * (2 / h)
    B = np.zeros((6, 24))
    for i in range(8):
        x, y, z = dN[i]
        B[0, 3 * i] = x
        B[1, 3 * i + 1] = y
        B[2, 3 * i + 2] = z
        B[3, 3 * i] = y
        B[3, 3 * i + 1] = x
        B[4, 3 * i + 1] = z
        B[4, 3 * i + 2] = y
        B[5, 3 * i] = z
        B[5, 3 * i + 2] = x
    return B


def element_stiffness(E: float, nu: float, h: float) -> np.ndarray:
    D = _D(E, nu)
    K = np.zeros((24, 24))
    w = (h / 2) ** 3
    for xi in _GP:
        for eta in _GP:
            for zeta in _GP:
                B = _B(xi, eta, zeta, h)
                K += B.T @ D @ B * w
    return K


def _frame(axis: np.ndarray):
    z = axis / np.linalg.norm(axis)
    x = np.cross(z, [1.0, 0, 0])
    if np.linalg.norm(x) < 0.1:
        x = np.cross(z, [0, 1.0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.vstack([x, y, z])  # rows: local axes


def voxelize(mesh: trimesh.Trimesh, R: np.ndarray, n_long: int, min_across: int = 6, target: int = 9000):
    """Inside/outside of every voxel by casting one ray per voxel column (parity rule).
    Voxel size: about `target` elements for the part's volume, at least `n_long` along the
    longest side and `min_across` through the thinnest side."""
    local = mesh.vertices @ R.T
    lo, hi = local.min(0), local.max(0)
    vol = abs(float(mesh.volume)) if mesh.is_watertight else float(np.prod(hi - lo)) * 0.3
    h = min(float((hi - lo).max() / n_long), (vol / target) ** (1 / 3))
    thin = float(np.sort(hi - lo)[0])
    h = min(h, thin / min_across)
    h = thin / math.ceil(thin / h - 1e-9)   # whole number of voxels through the thinnest side
    dims = np.maximum(np.ceil((hi - lo) / h - 1e-9).astype(int), 1)
    lo = (lo + hi) / 2 - dims * h / 2       # centre the grid so any overhang is symmetric
    lm = trimesh.Trimesh(local, mesh.faces, process=False)
    ii, jj = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    # Tiny irrational offsets keep rays off edges and vertices.
    ox = lo[0] + (ii + 0.5) * h + h * 1.37e-4
    oy = lo[1] + (jj + 0.5) * h + h * 2.71e-4
    origins = np.c_[ox, oy, np.full(len(ox), lo[2] - h)]
    dirs = np.tile([0.0, 0.0, 1.0], (len(ox), 1))
    zc = lo[2] + (np.arange(dims[2]) + 0.5) * h
    try:
        locs, ray_idx, _ = lm.ray.intersects_location(origins, dirs, multiple_hits=True)
        hits = [[] for _ in range(len(ox))]
        for z, r in zip(locs[:, 2], ray_idx):
            hits[r].append(z)
        cells = []
        for r, hz in enumerate(hits):
            if len(hz) < 2:
                continue
            hz = np.unique(np.round(np.sort(hz), 9))
            inside = (np.searchsorted(hz, zc) % 2) == 1
            for k in np.nonzero(inside)[0]:
                cells.append((ii[r], jj[r], k))
        grid = np.asarray(cells, dtype=int).reshape(-1, 3)
    except Exception:
        from ..features import winding_number
        grid = np.stack(np.meshgrid(*[np.arange(d) for d in dims], indexing="ij"), -1).reshape(-1, 3)
        centers = (lo + (grid + 0.5) * h) @ R
        grid = grid[winding_number(mesh, centers, chunk=256) > 0.5]
    return grid, lo, h, dims


def _solve_linear(K, f, coords=None):
    """Smoothed-aggregation AMG (with the 6 rigid-body modes, as elasticity needs) + CG;
    sparse direct solve as a fallback."""
    try:
        import pyamg
        from scipy.sparse.linalg import cg
        B = None
        if coords is not None:
            x, y, z = (coords - coords.mean(0)).T
            n = len(x)
            B = np.zeros((3 * n, 6))
            B[0::3, 0] = 1
            B[1::3, 1] = 1
            B[2::3, 2] = 1
            B[1::3, 3], B[2::3, 3] = -z, y
            B[0::3, 4], B[2::3, 4] = z, -x
            B[0::3, 5], B[1::3, 5] = -y, x
        A = K.tobsr(blocksize=(3, 3)) if B is not None else K.tocsr()
        ml = pyamg.smoothed_aggregation_solver(A, B=B, max_coarse=500)
        x, info = cg(K, f, M=ml.aspreconditioner(), rtol=1e-8, maxiter=400)
        if info == 0:
            return x
    except Exception:
        pass
    return spsolve(K.tocsc(), f)


def solve(mesh: trimesh.Trimesh, material: Material, axis, case: str = "axial", magnitude: float = 10000.0,
          temp_c: float = 20.0, n_long: int = 36, max_elements: int = 14000, fix_frac: float = 0.08,
          load_frac: float = 0.08) -> dict:
    """magnitude: N (axial, bending), N-m (torsion) or MPa (pressure)."""
    t0 = time.time()
    axis = np.asarray(axis, float)
    R = _frame(axis)
    vox, lo, h, dims = voxelize(mesh, R, n_long)
    while len(vox) > max_elements and n_long > 12:
        n_long = int(n_long * 0.85)
        vox, lo, h, dims = voxelize(mesh, R, n_long)
    if len(vox) < 8:
        raise ValueError("Part is too thin to voxelise at this resolution.")
    ne = len(vox)
    E = material.E * 1e3  # MPa (N/mm^2), lengths in mm
    nu = material.poisson
    Ke = element_stiffness(E, nu, h)

    # Nodes: 8 corners per voxel.
    offs = ((_NODES + 1) / 2).astype(int)
    corner = (vox[:, None, :] + offs[None, :, :]).reshape(-1, 3)
    key = (corner[:, 0] * (dims[1] + 1) + corner[:, 1]) * (dims[2] + 1) + corner[:, 2]
    uniq, node_of = np.unique(key, return_inverse=True)
    conn = node_of.reshape(ne, 8)
    nn = len(uniq)
    nz_ = uniq % (dims[2] + 1)
    ny_ = (uniq // (dims[2] + 1)) % (dims[1] + 1)
    nx_ = uniq // ((dims[2] + 1) * (dims[1] + 1))
    node_local = lo + np.c_[nx_, ny_, nz_] * h

    dofs = (conn[:, :, None] * 3 + np.arange(3)[None, None, :]).reshape(ne, 24)
    rows = np.repeat(dofs, 24, axis=1).ravel()
    cols = np.tile(dofs, (1, 24)).ravel()
    vals = np.tile(Ke.ravel(), ne)
    K = sparse.coo_matrix((vals, (rows, cols)), shape=(3 * nn, 3 * nn)).tocsr()

    z = node_local[:, 2]
    zmin, zmax = z.min(), z.max()
    L = zmax - zmin
    fixed_nodes = np.nonzero(z <= zmin + fix_frac * L + 1e-9)[0]
    load_nodes = np.nonzero(z >= zmax - load_frac * L - 1e-9)[0]
    F = np.zeros(3 * nn)
    cxy = node_local[load_nodes, :2].mean(0)
    if case == "axial":
        F[3 * load_nodes + 2] = magnitude / len(load_nodes)
    elif case == "bending":
        F[3 * load_nodes + 0] = magnitude / len(load_nodes)
    elif case == "torsion":
        rel = node_local[load_nodes, :2] - cxy
        r2 = (rel ** 2).sum(1)
        scale = magnitude * 1000 / max(r2.sum(), 1e-9)   # N-m -> N-mm, F_i = T r_i / sum r^2
        F[3 * load_nodes + 0] = -rel[:, 1] * scale
        F[3 * load_nodes + 1] = rel[:, 0] * scale
    elif case == "pressure":
        # Pressure on the +z facing top surface of each voxel column.
        top = {}
        for e, (i, j, k) in enumerate(vox):
            if (i, j) not in top or k > vox[top[(i, j)]][2]:
                top[(i, j)] = e
        area_force = magnitude * h * h / 4
        for e in top.values():
            for c in (4, 5, 6, 7):
                F[3 * conn[e, c] + 2] -= area_force
    else:
        raise ValueError(f"Unknown load case '{case}'")

    fixed = np.zeros(3 * nn, bool)
    fixed[(3 * fixed_nodes[:, None] + np.arange(3)).ravel()] = True
    free = ~fixed
    Kff = K[free][:, free]
    u = np.zeros(3 * nn)
    free_nodes = np.nonzero(free[0::3])[0]
    u[free] = _solve_linear(Kff, F[free], coords=node_local[free_nodes])

    # Stresses recovered at element corners and averaged per node (exact for linear fields,
    # so surface peaks in bending/torsion are not under-read like centre values are).
    D = _D(E, nu)
    ue = u[dofs]
    node_sig = np.zeros((nn, 6))
    node_cnt = np.zeros(nn)
    for c, (a, b, cc) in enumerate(_NODES):
        sig_c = (ue @ _B(a, b, cc, h).T) @ D.T
        np.add.at(node_sig, conn[:, c], sig_c)
        np.add.at(node_cnt, conn[:, c], 1)
    node_sig /= np.maximum(node_cnt, 1)[:, None]
    sx, sy, sz, txy, tyz, tzx = node_sig.T
    vm = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (txy ** 2 + tyz ** 2 + tzx ** 2))

    # Ignore the clamped and loaded slabs for the peak (artificial stress concentrations).
    interior = (z > zmin + (fix_frac + 0.06) * L) & (z < zmax - (load_frac + 0.06) * L)
    vm_eval = vm[interior] if interior.any() else vm
    idx_eval = np.nonzero(interior)[0] if interior.any() else np.arange(nn)
    k_peak = idx_eval[int(np.argmax(vm_eval))]
    peak = float(vm[k_peak])
    peak_99 = float(np.percentile(vm_eval, 99))
    centers_world = node_local @ R
    sy_t = material.yield_strength * material.strength_factor(temp_c)
    disp = np.linalg.norm(u.reshape(-1, 3), axis=1)

    tree = cKDTree(centers_world)
    _, nearest = tree.query(mesh.vertices, k=1)
    vertex_stress = vm[nearest]
    vertex_stress = np.where(np.isin(nearest, np.nonzero(interior)[0]) | ~interior.any(), vertex_stress,
                             np.minimum(vertex_stress, peak))
    return {
        "case": case, "magnitude": magnitude, "temp_c": temp_c,
        "elements": int(ne), "nodes": int(nn), "voxel_mm": round(h, 3),
        "peak_von_mises_mpa": round(peak, 2), "p99_von_mises_mpa": round(peak_99, 2),
        "peak_location": [round(float(x), 2) for x in centers_world[k_peak]],
        "max_displacement_mm": round(float(disp.max()), 5),
        "yield_at_temp_mpa": round(sy_t, 1),
        "safety_factor_yield": round(sy_t / peak, 3) if peak > 0 else None,
        "vertex_stress": [round(float(x), 2) for x in vertex_stress],
        "elapsed_s": round(time.time() - t0, 2),
        "note": "Linear-elastic voxel FEA; peaks at sharp corners are approximate (+-15%).",
    }
