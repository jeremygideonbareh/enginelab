"""Turn raw geometry into individual parts.

1. Every node/body is split into closed shells (connected through shared edges).
2. Shells that are cavities of another shell (inward-facing, enclosed) are merged
   back into their owner, and tiny fragments are merged into their nearest part.
3. If a mesh is one fused shell, it can be cut along concave creases with
   spectral clustering of the face graph (the "minima rule": boundaries
   between parts lie along concave edges).
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

from .io import RawPart


def _split_edges_only(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    """Split by faces sharing a manifold edge.

    Edges used by more than two faces are where separate bodies touch with
    coincident triangles, so they are not used to join faces.
    """
    if len(mesh.faces) == 0:
        return []
    edges = mesh.edges_sorted
    groups = trimesh.grouping.group_rows(edges, require_count=None)
    pairs = []
    for g in groups:
        if len(g) == 2:
            pairs.append((g[0] // 3, g[1] // 3))
    n = len(mesh.faces)
    if pairs:
        p = np.asarray(pairs)
        adj = sparse.coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(n, n))
    else:
        adj = sparse.coo_matrix((n, n))
    count, labels = connected_components(adj, directed=False)
    out = []
    for c in range(count):
        faces = np.nonzero(labels == c)[0]
        sub = mesh.submesh([faces], append=True, repair=False)
        out.append(sub)
    return out


def _bbox_contains(outer: np.ndarray, inner: np.ndarray, tol: float) -> bool:
    return bool(np.all(inner[0] >= outer[0] - tol) and np.all(inner[1] <= outer[1] + tol))


def split_shells(raw: RawPart, min_area_frac: float = 2e-4) -> list[RawPart]:
    shells = _split_edges_only(raw.mesh)
    if len(shells) <= 1:
        return [raw]
    total_area = sum(s.area for s in shells) or 1.0
    diag = float(np.linalg.norm(raw.mesh.extents)) or 1.0

    keep, small, cavities = [], [], []
    for s in shells:
        if s.area / total_area < min_area_frac or len(s.faces) < 4:
            small.append(s)
        elif s.is_watertight and s.volume < 0:
            cavities.append(s)
        else:
            keep.append(s)

    def owner_for(s: trimesh.Trimesh, candidates: list[trimesh.Trimesh]) -> int | None:
        best, best_d = None, np.inf
        c = s.bounds.mean(0)
        for i, k in enumerate(candidates):
            if _bbox_contains(k.bounds, s.bounds, 0.01 * diag):
                d = np.linalg.norm(k.bounds.mean(0) - c)
                if d < best_d:
                    best, best_d = i, d
        return best

    merged: list[list[trimesh.Trimesh]] = [[k] for k in keep]
    for cav in cavities:
        i = owner_for(cav, keep)
        if i is None:
            cav.invert()  # it was a body with flipped normals, not a cavity
            keep.append(cav)
            merged.append([cav])
        else:
            merged[i].append(cav)
    for frag in small:
        if not keep:
            keep.append(frag)
            merged.append([frag])
            continue
        c = frag.bounds.mean(0)
        i = int(np.argmin([np.linalg.norm(k.bounds.mean(0) - c) for k in keep]))
        merged[i].append(frag)

    out = []
    multi = len(merged) > 1
    for idx, group in enumerate(merged):
        mesh = trimesh.util.concatenate(group) if len(group) > 1 else group[0]
        name = raw.name
        if name and multi:
            name = f"{raw.name}_{idx + 1}"
        out.append(RawPart(mesh=mesh, name=name, source=raw.source if raw.source == "cad-body" else
                           ("named-node" if raw.name else "shell")))
    return out


def segment_fused(mesh: trimesh.Trimesh, max_parts: int = 12, seed: int = 0) -> list[trimesh.Trimesh]:
    """Cut one fused shell into parts with concave-weighted spectral clustering.

    Faces joined by a concave edge get a weak link, so clusters split along
    the creases where parts meet. The number of parts comes from the
    eigengap of the graph Laplacian.
    """
    work = mesh
    adj_faces = work.face_adjacency
    if len(adj_faces) == 0:
        return [mesh]
    angles = work.face_adjacency_angles
    convex = work.face_adjacency_convex
    eta = np.where(convex, 0.1, 1.0)  # concave creases count fully
    d_geo = eta * (1 - np.cos(angles))
    avg = d_geo.mean() if d_geo.mean() > 0 else 1.0
    w = np.exp(-d_geo / avg)
    n = len(work.faces)
    W = sparse.coo_matrix((np.r_[w, w], (np.r_[adj_faces[:, 0], adj_faces[:, 1]],
                                          np.r_[adj_faces[:, 1], adj_faces[:, 0]])), shape=(n, n)).tocsr()
    deg = np.asarray(W.sum(1)).ravel()
    deg[deg == 0] = 1.0
    dinv = sparse.diags(1.0 / np.sqrt(deg))
    L = sparse.identity(n) - dinv @ W @ dinv
    k = min(max_parts + 1, n - 2)
    vals, vecs = eigsh(L.tocsc(), k=k, sigma=-1e-3, which="LM")
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    gaps = np.diff(vals[1:])
    n_parts = int(np.argmax(gaps) + 2) if len(gaps) else 2
    n_parts = int(np.clip(n_parts, 2, max_parts))
    emb = vecs[:, :n_parts]
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    labels = KMeans(n_clusters=n_parts, n_init=4, random_state=seed).fit_predict(emb)

    # Clusters must be connected; split any that are not.
    out = []
    for c in range(n_parts):
        faces = np.nonzero(labels == c)[0]
        if len(faces) == 0:
            continue
        sub = work.submesh([faces], append=True, repair=False)
        for piece in sub.split(only_watertight=False):
            if len(piece.faces) >= 4:
                out.append(piece)
    return out or [mesh]


def repair(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Close the small holes that exporters and vertex welding leave behind."""
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_winding(mesh)
    if mesh.is_watertight and mesh.volume < 0:
        mesh.invert()
    return mesh


def extract_parts(raws: list[RawPart], fused: str = "auto") -> list[RawPart]:
    """All raw nodes/bodies → individual parts."""
    parts: list[RawPart] = []
    for raw in raws:
        if raw.source == "cad-body":
            parts.append(raw)  # a CAD solid is already exactly one part
        else:
            parts.extend(split_shells(raw))
    should_cut = fused == "always" or (
        fused == "auto" and len(parts) == 1 and len(parts[0].mesh.faces) > 3000
        and parts[0].source != "cad-body")   # a CAD solid is exactly one part: never cut it
    if should_cut:
        pieces = segment_fused(parts[0].mesh)
        parts = [RawPart(mesh=m, name=None, source="segment") for m in pieces]
    for p in parts:
        if p.source != "cad-body":   # CAD solids carry exact volume/topology; meshes only for display
            repair(p.mesh)
    return parts
