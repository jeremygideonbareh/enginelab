"""Specifications measured from geometry.

General specs for every part (size, volume, mass, area, holes), part-specific
measurements taken from mesh cross-sections (crank throw, rod centre
distance, gear teeth, spring wire, bolt thread...), and engine-level specs
(bore, stroke, cylinders, displacement) inferred by combining parts.
"""
from __future__ import annotations

import math

import numpy as np
import trimesh

from .features import ShapeInfo


def sections(mesh: trimesh.Trimesh, axis: np.ndarray, n: int = 40, origin: np.ndarray | None = None,
             margin: float = 0.01):
    """Slice perpendicular to `axis`. Returns [(t, [(area, centroid3d, polygon), ...])]."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    origin = mesh.centroid if origin is None else origin
    t = (mesh.vertices - origin) @ axis
    lo, hi = t.min(), t.max()
    span = hi - lo
    hs = np.linspace(lo + margin * span, hi - margin * span, n)
    try:
        secs = mesh.section_multiplane(plane_origin=origin, plane_normal=axis, heights=list(hs))
    except Exception:
        return []
    out = []
    for h, s in zip(hs, secs):
        polys = []
        if s is not None:
            T = s.metadata["to_3D"]
            for p in s.polygons_full:
                c = (T @ np.r_[p.centroid.x, p.centroid.y, 0.0, 1.0])[:3]
                c = c + (h - (c - origin) @ axis) * axis  # put it on its slicing plane
                polys.append((float(p.area), c, p))
        out.append((float(h), polys))
    return out


def _round(x, nd=2):
    return float(round(float(x), nd))


def crank_measurements(mesh: trimesh.Trimesh, info: ShapeInfo, axis: np.ndarray | None = None,
                       _pass: int = 0, details: dict | None = None) -> dict:
    """Throw (half the stroke), main journal and crankpin diameters.

    Pass 1 slices along the longest axis to find the main journals; pass 2
    re-slices along the line fitted through the journal centres, because an
    unbalanced crank's inertia axis is slightly tilted from its real axis.
    """
    axis = info.axes[0] if axis is None else axis
    secs = sections(mesh, axis, n=160)
    # The main axis passes through the journal sections: the biggest round-ish
    # single sections whose centroids lie on one line.
    single = [(t, polys[0]) for t, polys in secs if len(polys) == 1]
    if not single:
        return {}
    areas = np.array([a for _, (a, c, p) in single])
    cents = np.array([c for _, (a, c, p) in single])
    # Journals: sections whose shape is a circle (area ~ pi r^2 with r from bounds).
    circ = []
    for (t, (a, c, p)) in single:
        minx, miny, maxx, maxy = p.bounds
        d = 0.5 * ((maxx - minx) + (maxy - miny))
        circ.append(a / (math.pi * d * d / 4 + 1e-9))
    circ = np.array(circ)
    round_idx = np.nonzero(circ > 0.9)[0]
    if len(round_idx) < 2:
        return {}
    # Main axis = line through the round sections' centroids that has most support.
    # Journals are the coaxial round sections of ONE diameter seen in the most slices. Scoring by
    # count (not count x diameter) keeps big round webs / counterweights of heavy cranks from
    # winning; the coaxial tolerance allows for the first-guess axis being a little tilted.
    dia = np.sqrt(4 * areas / math.pi)
    best = None
    for i in round_idx:
        d_i = dia[i]
        off = cents[round_idx] - cents[i]
        off -= np.outer(off @ axis, axis)
        same = np.abs(dia[round_idx] - d_i) < 0.03 * d_i
        support = round_idx[(np.linalg.norm(off, axis=1) < max(0.05 * d_i, 1.0)) & same]
        score = len(support) - 1e-4 * d_i
        if best is None or score > best[0]:
            best = (score, support)
    mains = best[1]
    main_pt = cents[mains].mean(0)
    if _pass == 0 and len(mains) >= 2:
        pts = cents[mains]
        for _ in range(3):  # robust line fit: drop sections that sit off the line
            cen = pts.mean(0)
            fitted = np.linalg.svd(pts - cen, full_matrices=False)[2][0]
            d = pts - cen
            dist = np.linalg.norm(d - np.outer(d @ fitted, fitted), axis=1)
            keep = dist <= 3 * np.median(dist) + 0.05
            if keep.all() or keep.sum() < 2:
                break
            pts = pts[keep]
        if fitted @ axis < 0:
            fitted = -fitted
        if np.degrees(np.arccos(min(1.0, abs(float(fitted @ axis))))) > 0.05:
            return crank_measurements(mesh, info, fitted, _pass=1, details=details)
    main_d = float(np.median(np.sqrt(4 * areas[mains] / math.pi)))
    # Crankpins are no bigger than the journals; larger round sections are webs.
    pins = [i for i in round_idx if i not in set(mains) and np.sqrt(4 * areas[i] / math.pi) < 1.1 * main_d]
    throws, pin_d, pin_pts = [], [], []
    for i in pins:
        off = cents[i] - main_pt
        off -= (off @ axis) * axis
        dist = np.linalg.norm(off)
        if dist > 0.15 * main_d:
            throws.append(dist)
            pin_d.append(np.sqrt(4 * areas[i] / math.pi))
            pin_pts.append(cents[i])
    if details is not None:   # for balance / vibration analysis: the crank axis and every crankpin slice
        details.update(axis=np.asarray(axis, float), point=np.asarray(main_pt, float),
                       pin_points=np.asarray(pin_pts, float), main_d=main_d,
                       pin_d=float(np.median(pin_d)) if pin_d else None)
    out = {"main_journal_diameter_mm": _round(main_d)}
    if throws:
        throw = float(np.median(throws))
        out.update({"crank_throw_mm": _round(throw), "stroke_mm": _round(2 * throw),
                    "crankpin_diameter_mm": _round(np.median(pin_d))})
    return out


def rod_measurements(mesh: trimesh.Trimesh, info: ShapeInfo) -> dict:
    """Centre-to-centre length and the two eye bores, from a mid-thickness slice."""
    thick_axis = info.axes[2]
    secs = sections(mesh, thick_axis, n=3, origin=info.center, margin=0.45)
    for t, polys in secs:
        if not polys:
            continue
        a, c, p = max(polys, key=lambda q: q[0])
        holes = sorted(p.interiors, key=lambda r: -abs(_ring_area(r)))
        if len(holes) >= 2:
            T = None
            (h1, h2) = holes[:2]
            c1, c2 = np.mean(np.asarray(h1.coords), 0), np.mean(np.asarray(h2.coords), 0)
            d1, d2 = (2 * math.sqrt(abs(_ring_area(h)) / math.pi) for h in (h1, h2))
            big, small = (d1, d2) if d1 >= d2 else (d2, d1)
            min_area = _min_section_area(mesh, info.axes[0])
            return {"centre_distance_mm": _round(np.linalg.norm(c1 - c2)),
                    "big_end_bore_mm": _round(big), "small_end_bore_mm": _round(small),
                    "thickness_mm": _round(info.extents[2]),
                    "min_shank_area_mm2": _round(min_area, 1) if min_area else None}
    return {}


def _round_axis(info: ShapeInfo) -> tuple[int, float, float]:
    """For a round part (piston, liner): the principal axis whose two perpendicular extents are
    the most equal is the axis of revolution. Returns (axis index, diameter, length along axis).
    Unlike the symmetry estimate this does not depend on how the part is proportioned
    (a piston taller than its bore, a liner longer than its diameter)."""
    e = np.asarray(info.extents, float)
    best = min(range(3), key=lambda k: abs(np.subtract(*np.delete(e, k))) / max(np.delete(e, k).max(), 1e-9))
    others = np.delete(e, best)
    return best, float(others.mean()), float(e[best])


def _tube_measurements(mesh: trimesh.Trimesh, info: ShapeInfo) -> dict:
    """Bore, outside diameter and wall of a tube (cylinder liner) from a mid-length cross-section:
    the bore is the largest hole, measured from its area."""
    k, d_out, _ = _round_axis(info)
    ax = np.asarray(info.axes[k], float)
    sec = mesh.section(plane_origin=info.center, plane_normal=ax)
    if sec is not None:
        polys = list(sec.to_planar()[0].polygons_full)
        if polys:
            p = max(polys, key=lambda q: q.area)
            if p.interiors:
                hole = max(p.interiors, key=lambda r: abs(_ring_area(r)))
                bore = 2 * math.sqrt(abs(_ring_area(hole)) / math.pi)
                outer = 2 * math.sqrt((p.area + abs(_ring_area(hole))) / math.pi)
                return {"bore_mm": _round(bore), "outer_diameter_mm": _round(outer), "wall_mm": _round((outer - bore) / 2)}
    v = mesh.vertices - info.center
    r = np.linalg.norm(v - np.outer(v @ ax, ax), axis=1)
    return {"bore_mm": _round(2 * r.min()), "outer_diameter_mm": _round(2 * r.max()), "wall_mm": _round(r.max() - r.min())}


def rod_cap_measurements(rod: trimesh.Trimesh, info: ShapeInfo, cap: trimesh.Trimesh) -> dict:
    """Rod and bolted cap as separate parts: each holds half of the big-end bore, so slice
    both at the rod's mid-thickness, join the outlines, then measure the two eyes."""
    from shapely.ops import unary_union
    n = np.asarray(info.axes[2], float)
    o = np.asarray(info.center, float)
    sr = rod.section(plane_origin=o, plane_normal=n)
    sc = cap.section(plane_origin=o, plane_normal=n)
    if sr is None or sc is None:
        return {}
    pr, to3d = sr.to_planar()
    pc, _ = sc.to_planar(to_2D=np.linalg.inv(to3d), check=False)
    shape = unary_union([p.buffer(0.3) for p in list(pr.polygons_full) + list(pc.polygons_full)]).buffer(-0.3)
    polys = list(getattr(shape, "geoms", [shape]))
    p = max(polys, key=lambda q: q.area)
    holes = sorted(p.interiors, key=lambda r: -abs(_ring_area(r)))
    if len(holes) < 2:
        return {}
    c1, c2 = (np.mean(np.asarray(h.coords), 0) for h in holes[:2])
    d1, d2 = (2 * math.sqrt(abs(_ring_area(h)) / math.pi) for h in holes[:2])
    min_area = _min_section_area(rod, info.axes[0])
    return {"centre_distance_mm": _round(np.linalg.norm(c1 - c2)), "big_end_bore_mm": _round(max(d1, d2)),
            "small_end_bore_mm": _round(min(d1, d2)), "thickness_mm": _round(info.extents[2]),
            "min_shank_area_mm2": _round(min_area, 1) if min_area else None,
            "measured_with_cap": True}


def _ring_area(ring) -> float:
    xy = np.asarray(ring.coords)
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _min_section_area(mesh, axis, lo_frac=0.3, hi_frac=0.7) -> float | None:
    secs = sections(mesh, axis, n=21)
    if not secs:
        return None
    n = len(secs)
    vals = [sum(a for a, _, _ in polys) for t, polys in secs[int(lo_frac * n):int(hi_frac * n) + 1] if polys]
    return float(min(vals)) if vals else None


def gear_teeth(mesh: trimesh.Trimesh, info: ShapeInfo) -> dict:
    axis = info.sym_axis
    v = mesh.vertices - info.sym_center
    u = np.cross(axis, [1.0, 0, 0])
    if np.linalg.norm(u) < 0.1:
        u = np.cross(axis, [0, 1.0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(axis, u)
    x, y = v @ u, v @ w
    r = np.hypot(x, y)
    rim = r > 0.8 * r.max()
    bins = 2048
    ab = ((np.arctan2(y[rim], x[rim]) + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    rr = np.full(bins, np.nan)
    for b, val in zip(ab, r[rim]):
        if np.isnan(rr[b]) or val > rr[b]:
            rr[b] = val
    ok = ~np.isnan(rr)
    if ok.sum() < 32:
        return {}
    idx = np.arange(bins)
    rr = np.interp(idx, idx[ok], rr[ok], period=bins)
    spec = np.abs(np.fft.rfft(rr - rr.mean()))
    k = int(np.argmax(spec[6:400]) + 6)
    tip = float(2 * r.max())
    root = float(2 * np.percentile(r[rim], 3))
    if (tip - root) / tip < 0.02:
        return {}
    module = tip / (k + 2)
    return {"teeth": k, "tip_diameter_mm": _round(tip), "root_diameter_mm": _round(root),
            "module_mm": _round(module, 3), "face_width_mm": _round(info.sym_length)}


def _circle_fit(xy: np.ndarray):
    A = np.c_[2 * xy, np.ones(len(xy))]
    b = (xy ** 2).sum(1)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    c = sol[:2]
    return c, math.sqrt(max(sol[2] + c @ c, 0))


def _coil_cv(vertices: np.ndarray, ax: np.ndarray):
    ax = ax / np.linalg.norm(ax)
    u = np.cross(ax, [1.0, 0, 0])
    if np.linalg.norm(u) < 0.1:
        u = np.cross(ax, [0, 1.0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(ax, u)
    xy = np.c_[vertices @ u, vertices @ w]
    c, _ = _circle_fit(xy)
    rr = np.linalg.norm(xy - c, axis=1)
    return rr.std() / (rr.mean() + 1e-12), rr, vertices @ ax


def spring_measurements(mesh: trimesh.Trimesh, info: ShapeInfo, volume: float) -> dict:
    """Coil axis = the direction around which every vertex sits at nearly the same radius."""
    from scipy.optimize import minimize

    v = mesh.vertices - mesh.vertices.mean(0)
    cands = [info.sym_axis] + list(np.asarray(mesh.principal_inertia_vectors)) + list(info.axes)
    start = min(cands, key=lambda a: _coil_cv(v, a)[0])
    start = start / np.linalg.norm(start)
    th0, ph0 = math.acos(np.clip(start[2], -1, 1)), math.atan2(start[1], start[0])
    to_vec = lambda a: np.array([math.sin(a[0]) * math.cos(a[1]), math.sin(a[0]) * math.sin(a[1]), math.cos(a[0])])
    res = minimize(lambda a: _coil_cv(v, to_vec(a))[0], [th0, ph0], method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 300})
    axis = to_vec(res.x)
    _, r, tt = _coil_cv(v, axis)
    free_len = float(tt.max() - tt.min())
    od, idd = 2 * r.max(), 2 * r.min()
    d = (od - idd) / 2
    dm = (od + idd) / 2
    wire_len = volume / (math.pi * d * d / 4) if d > 0 else 0
    coils = wire_len / (math.pi * dm) if dm > 0 else 0
    return {"outer_diameter_mm": _round(od), "wire_diameter_mm": _round(d), "mean_coil_diameter_mm": _round(dm),
            "free_length_mm": _round(free_len), "total_coils": _round(coils, 1),
            "active_coils": _round(max(coils - 2, 1), 1), "spring_index": _round(dm / d if d else 0)}


def crown_thickness(mesh, info) -> float | None:
    """Longest solid run along the piston axis = crown thickness."""
    from .features import winding_number
    ts = np.linspace(-0.5, 0.5, 241) * info.sym_length * 1.02
    pts = info.sym_center + ts[:, None] * info.sym_axis
    inside = winding_number(mesh, pts) > 0.5
    best, run = 0, 0
    for v in inside:
        run = run + 1 if v else 0
        best = max(best, run)
    return best * (ts[1] - ts[0]) if best else None


def shaft_min_diameter(mesh, info) -> float | None:
    secs = sections(mesh, info.sym_axis, n=30)
    ds = [math.sqrt(4 * sum(a for a, _, _ in polys) / math.pi) for _, polys in secs if polys]
    return float(min(ds)) if ds else None


def part_specs(label: str, mesh: trimesh.Trimesh, info: ShapeInfo, density: float | None = None) -> dict:
    vol = info.volume
    s = {
        "length_mm": _round(info.extents[0]), "width_mm": _round(info.extents[1]),
        "height_mm": _round(info.extents[2]), "volume_cm3": _round(vol / 1000, 3),
        "surface_area_cm2": _round(info.area / 100, 2), "through_holes": int(info.genus),
        "watertight": bool(info.watertight), "triangles": int(len(mesh.faces)),
    }
    if density:
        s["mass_kg"] = _round(vol * 1e-9 * density, 4)
    if info.sym_score > 0.6:
        s["axisymmetric"] = True
        s["diameter_mm"] = _round(info.sym_diameter)
        s["axial_length_mm"] = _round(info.sym_length)
    try:
        if label == "crankshaft":
            s.update(crank_measurements(mesh, info))
        elif label == "connecting_rod":
            s.update(rod_measurements(mesh, info))
        elif label in ("timing_gear", "flywheel"):
            s.update(gear_teeth(mesh, info))
        elif label == "valve_spring":
            s.update(spring_measurements(mesh, info, vol))
        elif label == "piston":
            k, d_round, length = _round_axis(info)
            s["piston_diameter_mm"] = _round(d_round)
            s["piston_height_mm"] = _round(length)
            t = crown_thickness(mesh, info)
            if t:
                s["crown_thickness_mm"] = _round(t)
        elif label == "cylinder_liner":
            s.update(_tube_measurements(mesh, info))
        elif label == "valve":
            s["head_diameter_mm"] = _round(info.sym_diameter)
            d = shaft_min_diameter(mesh, info)
            if d:
                s["stem_diameter_mm"] = _round(d)
        elif label == "piston_pin":
            v = mesh.vertices - info.sym_center
            r = np.linalg.norm(v - np.outer(v @ info.sym_axis, info.sym_axis), axis=1)
            s["outer_diameter_mm"] = _round(info.sym_diameter)
            s["inner_diameter_mm"] = _round(2 * r.min()) if r.min() > 0.05 * r.max() else 0.0
            s["pin_length_mm"] = _round(info.sym_length)
        elif label == "bolt":
            d = shaft_min_diameter(mesh, info)
            if d:
                s["shank_diameter_mm"] = _round(d)
                s["thread"] = f"M{int(round(d))}"
    except Exception as exc:  # measurement is best-effort; never block the analysis
        s["measurement_note"] = f"detailed measurement failed: {exc.__class__.__name__}"
    return s


def engine_specs(labels: list[str], specs: list[dict], infos: list[ShapeInfo]) -> dict:
    """Combine parts into engine-level specs."""
    idx = lambda lab: [i for i, l in enumerate(labels) if l == lab]
    out: dict = {}
    pistons = idx("piston")
    liners = idx("cylinder_liner")
    if pistons:
        out["cylinders"] = len(pistons)
        out["piston_diameter_mm"] = _round(np.median([specs[i]["piston_diameter_mm"] for i in pistons]))
    if liners:
        bores = [specs[i]["bore_mm"] for i in liners if specs[i].get("bore_mm")]
        if pistons:   # a liner bore must fit its piston (a few tenths of clearance)
            fit = [b for b in bores if abs(b - out["piston_diameter_mm"]) / out["piston_diameter_mm"] < 0.03]
            bores = fit or bores
        out["bore_mm"] = _round(np.median(bores))
    elif pistons:
        out["bore_mm"] = _round(out["piston_diameter_mm"] + 0.24)  # typical cold running clearance
        out["bore_note"] = "estimated from piston diameter plus typical clearance"
    cranks = idx("crankshaft")
    if cranks and "stroke_mm" in specs[cranks[0]]:
        out["stroke_mm"] = specs[cranks[0]]["stroke_mm"]
        out["main_journal_diameter_mm"] = specs[cranks[0]].get("main_journal_diameter_mm")
        out["crankpin_diameter_mm"] = specs[cranks[0]].get("crankpin_diameter_mm")
    rods = [specs[i] for i in idx("connecting_rod") if "centre_distance_mm" in specs[i]]
    if rods:
        out["rod_length_mm"] = _round(np.median([r["centre_distance_mm"] for r in rods]))
    if "bore_mm" in out and "stroke_mm" in out and "cylinders" in out:
        B, S = out["bore_mm"], out["stroke_mm"]
        out["displacement_cc"] = _round(math.pi / 4 * B * B * S * out["cylinders"] / 1000, 1)
        out["bore_stroke_ratio"] = _round(B / S, 3)
        if "rod_length_mm" in out:
            out["rod_ratio"] = _round(out["rod_length_mm"] / (S / 2), 2)
    # Layout: are all piston axes parallel (inline) or in two banks (V)?
    if len(pistons) >= 2:
        ax = np.array([infos[i].sym_axis for i in pistons])
        cosines = np.abs(ax @ ax[0])
        if np.all(cosines > 0.98):
            out["layout"] = f"inline-{len(pistons)}" if len(pistons) > 1 else "single"
        else:
            other = ax[np.argmin(cosines)]
            angle = math.degrees(math.acos(min(1.0, abs(float(other @ ax[0])))))
            out["layout"] = f"V{len(pistons)}"
            out["bank_angle_deg"] = _round(angle, 1)
    elif len(pistons) == 1:
        out["layout"] = "single-cylinder"
    valves = idx("valve")
    if valves and pistons:
        out["valves_per_cylinder"] = int(round(len(valves) / len(pistons)))
    out["camshafts"] = len(idx("camshaft"))
    return out
