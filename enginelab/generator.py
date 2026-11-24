"""Parametric engine generator.

Builds complete, labelled piston engines (inline 1-6, V6/V8) from exact solid
booleans. Every dimension is drawn from realistic engineering ratios, so the
output is useful as:
  * test models with known answers (labels, bore, stroke, ...),
  * training data for the part classifier,
  * demo uploads.

Frame: crankshaft along +X, cylinders point up (+Z) for an inline engine.
Units: millimetres.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import manifold3d as mf
import numpy as np
import trimesh

LABELS = [
    "engine_block", "cylinder_head", "cylinder_liner", "piston", "piston_ring",
    "piston_pin", "connecting_rod", "crankshaft", "camshaft", "valve", "valve_spring",
    "main_bearing", "flywheel", "timing_gear", "bolt", "oil_pan", "spark_plug",
    "head_gasket", "valve_cover", "other",
]

Man = mf.Manifold
CS = mf.CrossSection

# 2D section (a, b) + extrusion e  ->  world (x, y, z) = (e, a, b)
_YZ_EXTRUDE_X = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)


# ---------------------------------------------------------------- primitives
_RES = [1.0]  # tessellation multiplier, varied per engine so features stay mesh-independent


def _seg(seg: int) -> int:
    return seg if seg <= 8 else max(10, int(round(seg * _RES[0])))


def cyl_z(r: float, h: float, seg: int = 32, center: bool = True) -> Man:
    return Man.cylinder(h, r, -1.0, _seg(seg), center)


def cyl_x(r: float, h: float, seg: int = 32) -> Man:
    return cyl_z(r, h, seg).rotate((0, 90, 0))


def box(sx, sy, sz, center=(0, 0, 0)) -> Man:
    return Man.cube((sx, sy, sz), True).translate(tuple(center))


def tube_z(r_in: float, r_out: float, h: float, seg: int = 40) -> Man:
    return cyl_z(r_out, h, seg) - cyl_z(r_in, h * 1.01 + 0.2, seg)


def revolve(profile: list[tuple[float, float]], seg: int = 40) -> Man:
    pts = np.asarray(profile, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    if np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y) < 0:  # make counter-clockwise
        pts = pts[::-1]
    return Man.revolve(CS([pts]), _seg(seg))


def extrude_yz(section: CS, thickness: float) -> Man:
    """Extrude a 2D section drawn in the (y, z) plane along X, centred on x=0."""
    return Man.extrude(section, thickness).transform(_YZ_EXTRUDE_X).translate((-thickness / 2, 0, 0))


def gear_section(n_teeth: int, r_root: float, r_tip: float) -> CS:
    pts = []
    for k in range(n_teeth):
        a0 = 2 * math.pi * k / n_teeth
        da = 2 * math.pi / n_teeth
        for frac, rr in ((0.0, r_root), (0.18, r_tip), (0.5, r_tip), (0.68, r_root)):
            a = a0 + frac * da
            pts.append((rr * math.cos(a), rr * math.sin(a)))
    return CS([np.asarray(pts)])


def to_trimesh(m: Man) -> trimesh.Trimesh:
    mesh = m.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3].astype(np.float64)
    f = np.asarray(mesh.tri_verts).astype(np.int64)
    # Manifold output is already welded; trimesh's own merge can collapse
    # near-coincident vertices into degenerate faces, so keep its topology.
    out = trimesh.Trimesh(v, f, process=False)
    if not out.is_watertight:
        # Booleans on coplanar faces can leave zero-thickness slivers; drop them.
        pieces = out.split(only_watertight=False)
        keep = [q for q in pieces if abs(q.volume) > 1e-6 * abs(out.volume) and len(q.faces) > 3]
        if keep:
            out = trimesh.util.concatenate(keep)
            out.merge_vertices()
    return out


def helix_spring(r_coil: float, r_wire: float, length: float, turns: float,
                 seg_path: int = 18, seg_wire: int = 8) -> trimesh.Trimesh:
    """Closed coil spring: a circle swept along a helix with capped ends."""
    n = max(int(turns * seg_path), 8)
    t = np.linspace(0, 2 * math.pi * turns, n)
    path = np.c_[r_coil * np.cos(t), r_coil * np.sin(t), np.linspace(0, length, n)]
    tang = np.gradient(path, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    radial = np.c_[np.cos(t), np.sin(t), np.zeros(n)]
    b1 = radial - (radial * tang).sum(1, keepdims=True) * tang
    b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
    b2 = np.cross(tang, b1)
    ang = np.linspace(0, 2 * math.pi, seg_wire, endpoint=False)
    ring = (np.cos(ang)[None, :, None] * b1[:, None, :] + np.sin(ang)[None, :, None] * b2[:, None, :])
    verts = (path[:, None, :] + r_wire * ring).reshape(-1, 3)
    faces = []
    for i in range(n - 1):
        for j in range(seg_wire):
            a, b = i * seg_wire + j, i * seg_wire + (j + 1) % seg_wire
            c, d = a + seg_wire, b + seg_wire
            faces += [(a, c, b), (b, c, d)]
    s, e = len(verts), len(verts) + 1
    verts = np.vstack([verts, path[0], path[-1]])
    last = (n - 1) * seg_wire
    for j in range(seg_wire):
        faces.append((s, j, (j + 1) % seg_wire))
        faces.append((e, last + (j + 1) % seg_wire, last + j))
    mesh = trimesh.Trimesh(verts, np.asarray(faces), process=True)
    trimesh.repair.fix_normals(mesh)
    return mesh


def rot_x(deg: float) -> np.ndarray:
    return trimesh.transformations.rotation_matrix(math.radians(deg), (1, 0, 0))


def translate(v) -> np.ndarray:
    return trimesh.transformations.translation_matrix(v)


# ---------------------------------------------------------------- engine
@dataclass
class EngineParams:
    layout: str = "inline"      # inline | V
    n_cyl: int = 4
    bank_angle: float = 0.0     # degrees between banks for V engines
    bore: float = 86.0
    stroke: float = 86.0
    rod_length: float = 145.0
    pitch: float = 100.0        # cylinder spacing along the crank
    valves_per_cyl: int = 4
    crank_angle: float = 0.0    # current crank position, degrees
    options: dict = field(default_factory=dict)

    @property
    def displacement_cc(self) -> float:
        return math.pi / 4 * self.bore ** 2 * self.stroke * self.n_cyl / 1000


def random_params(rng: np.random.Generator) -> EngineParams:
    layout = rng.choice(["inline", "inline", "inline", "V"])
    if layout == "inline":
        n = int(rng.choice([1, 2, 3, 4, 4, 4, 5, 6, 6]))
        bank = 0.0
    else:
        n = int(rng.choice([6, 8]))
        bank = float(rng.choice([60.0, 90.0]))
    # Mostly car-sized, with motorcycle/small-engine and truck-sized tails.
    band = rng.random()
    bore = float(rng.uniform(38, 62) if band < 0.15 else rng.uniform(108, 155) if band > 0.85 else rng.uniform(62, 108))
    stroke = float(bore * rng.uniform(0.78, 1.25))
    # "wide" engines cover unusual proportions seen in real CAD (student / custom designs):
    # long rods, thick rods, big solid pins, tall pistons, big journals.
    wide = bool(rng.random() < 0.4)
    rod = float(stroke * (rng.uniform(1.45, 2.45) if wide else rng.uniform(1.5, 1.95)))
    pitch = float(bore * (rng.uniform(1.14, 1.28) if layout == "inline" else rng.uniform(1.32, 1.42)))
    opts = {k: bool(rng.random() < p) for k, p in {
        "valve_cover": 0.75, "oil_pan": 0.8, "gasket": 0.7, "rings": 0.85, "bearings": 0.75,
        "springs": 0.85, "head_bolts": 0.8, "flywheel": 0.85, "timing": 0.75, "plugs": 0.8,
        "liners": 0.7, "dohc": 0.6, "head": 0.92,
    }.items()}
    opts["resolution"] = float(rng.uniform(0.55, 1.8))
    opts["wide"] = wide
    opts["split_rod"] = bool(rng.random() < 0.45)      # rod and bolted cap as separate parts
    opts["alt_phasing"] = bool(rng.random() < 0.15)    # e.g. 0-180-0-180 four-cylinder cranks
    return EngineParams(layout=layout, n_cyl=n, bank_angle=bank, bore=bore, stroke=stroke,
                        rod_length=rod, pitch=pitch, valves_per_cyl=int(rng.choice([2, 4, 4])),
                        crank_angle=float(rng.uniform(0, 360)), options=opts)


@dataclass
class GeneratedEngine:
    params: EngineParams
    parts: list[tuple[str, trimesh.Trimesh]]

    def scene(self) -> trimesh.Scene:
        sc = trimesh.Scene()
        counts: dict[str, int] = {}
        for label, mesh in self.parts:
            counts[label] = counts.get(label, 0) + 1
            sc.add_geometry(mesh, node_name=f"{label}_{counts[label]}", geom_name=f"{label}_{counts[label]}")
        return sc

    def merged(self) -> trimesh.Trimesh:
        return trimesh.util.concatenate([m for _, m in self.parts])


def build_engine(p: EngineParams, rng: np.random.Generator | None = None) -> GeneratedEngine:
    rng = rng or np.random.default_rng(0)
    _RES[0] = float(p.options.get("resolution", 1.0))
    B, S, L = p.bore, p.stroke, p.rod_length
    r = S / 2
    o = {"valve_cover": True, "oil_pan": True, "gasket": True, "rings": True, "bearings": True,
         "springs": True, "head_bolts": True, "flywheel": True, "timing": True, "plugs": True,
         "liners": True, "dohc": True, "head": True}
    o.update(p.options)
    if not o["head"]:
        for k in ("valve_cover", "springs", "plugs", "dohc", "head_bolts", "gasket"):
            o[k] = False
    j = lambda lo, hi: float(rng.uniform(lo, hi))  # jitter helper

    # Key dimensions ---------------------------------------------------------
    is_v = p.layout == "V"
    n_throws = p.n_cyl // 2 if is_v else p.n_cyl
    wide = bool(o.get("wide", False))
    rod_w = B * (j(0.18, 0.48) if wide else j(0.2, 0.25))
    web_t = B * j(0.15, 0.2)
    # The throw (crankpin + two webs) must leave room for a main journal within one pitch.
    room = p.pitch - 2 * web_t - 0.25 * B
    rod_w = max(0.12 * B, min(rod_w, (room - 1.5) / 2 if is_v else room - 1.0))
    pin_len = (2 * rod_w + 1.5) if is_v else rod_w + 1.0
    main_d = B * (j(0.58, 0.88) if wide else j(0.6, 0.7))
    crankpin_d = B * (j(0.48, 0.66) if wide else j(0.5, 0.58))
    piston_pin_d = B * (j(0.2, 0.42) if wide else j(0.22, 0.27))
    comp_h = B * (j(0.3, 0.55) if wide else j(0.32, 0.42))        # pin centre to crown
    piston_h = B * (j(0.55, 1.2) if wide else j(0.62, 0.82))
    piston_h = max(piston_h, comp_h + 0.7 * piston_pin_d)
    pin_bore_r = (0.0 if rng.random() < 0.3 else piston_pin_d * j(0.15, 0.33)) if wide else piston_pin_d * 0.3
    liner_t = B * j(0.05, 0.07)
    deck = r + L + comp_h + j(0.6, 1.2)
    liner_len = S + piston_h + B * 0.1
    gasket_t = j(0.9, 1.6)
    head_h = B * j(0.75, 0.95)
    head_z0 = deck + gasket_t
    head_top = head_z0 + head_h
    xs = (np.arange(n_throws) - (n_throws - 1) / 2) * p.pitch
    x_front, x_rear = xs[0] - p.pitch / 2, xs[-1] + p.pitch / 2

    # Crank throw phasing (even firing) and current rotation.
    if is_v:
        phases = [k * 360.0 / n_throws for k in range(n_throws)]
    elif p.n_cyl == 4:
        phases = [0, 180, 0, 180] if o.get("alt_phasing") else [0, 180, 180, 0]
    elif p.n_cyl == 6:
        phases = [0, 120, 240, 240, 120, 0]
    else:
        phases = [k * 720.0 / p.n_cyl % 360 for k in range(p.n_cyl)]
    throw_ang = [(p.crank_angle + ph) % 360 for ph in phases]

    banks = [(-p.bank_angle / 2, -0.55 * rod_w), (p.bank_angle / 2, 0.55 * rod_w)] if is_v else [(0.0, 0.0)]
    # cylinders: (x, bank_angle, throw_index)
    cylinders = [(xs[t] + dx, ang, t) for t in range(n_throws) for ang, dx in banks]

    parts: list[tuple[str, trimesh.Trimesh]] = []

    def add(label, man_or_mesh, T=None):
        mesh = to_trimesh(man_or_mesh) if isinstance(man_or_mesh, Man) else man_or_mesh
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return            # a boolean left nothing (rare random proportions): no such part
        if T is not None:
            mesh.apply_transform(T)
        parts.append((label, mesh))

    # Crankshaft ------------------------------------------------------------
    def pin_pos(theta):
        t = math.radians(theta)
        return (-r * math.sin(t), r * math.cos(t))

    crank = None
    main_x = [x_front + 0.0] + [(xs[k] + xs[k + 1]) / 2 for k in range(n_throws - 1)] + [x_rear]
    for k, x in enumerate(xs):
        py, pz = pin_pos(throw_ang[k])
        pin = cyl_x(crankpin_d / 2, pin_len + 0.4, 28).translate((x, py, pz))
        crank = pin if crank is None else crank + pin
        t = math.radians(throw_ang[k])
        web_sec = CS.batch_hull([CS.circle(main_d / 2 * 1.08, 32),
                                 CS.circle(crankpin_d / 2 * 1.2, 32).translate((py, pz))])
        cw = CS.circle(r * j(1.1, 1.35), 36).translate((r * 0.35 * math.sin(t), -r * 0.35 * math.cos(t)))
        web_sec = web_sec + cw
        for side in (-1, 1):
            wx = x + side * (pin_len / 2 + web_t / 2)
            crank = crank + extrude_yz(web_sec, web_t).translate((wx, 0, 0))
    for k in range(len(main_x)):
        if k == 0:
            a, b = x_front - 0.15 * B, xs[0] - pin_len / 2 - web_t
        elif k == len(main_x) - 1:
            a, b = xs[-1] + pin_len / 2 + web_t, x_rear + 0.15 * B
        else:
            a, b = xs[k - 1] + pin_len / 2 + web_t, xs[k] - pin_len / 2 - web_t
        crank = crank + cyl_x(main_d / 2, (b - a) + 0.4, 32).translate(((a + b) / 2, 0, 0))
    snout_len = B * j(0.35, 0.55)
    crank = crank + cyl_x(main_d * 0.36, snout_len, 24).translate((x_front - 0.15 * B - snout_len / 2 + 0.2, 0, 0))
    flange_t = B * 0.12
    flange_x = x_rear + 0.15 * B + flange_t / 2 - 0.2
    crank = crank + cyl_x(main_d * 0.85, flange_t, 36).translate((flange_x, 0, 0))
    add("crankshaft", crank)

    # Main bearing half-shells.
    bearing_t = max(1.6, 0.03 * B)
    if o["bearings"]:
        for k in range(len(main_x)):
            if k == 0:
                a, b = x_front - 0.15 * B, xs[0] - pin_len / 2 - web_t
            elif k == len(main_x) - 1:
                a, b = xs[-1] + pin_len / 2 + web_t, x_rear + 0.15 * B
            else:
                a, b = xs[k - 1] + pin_len / 2 + web_t, xs[k] - pin_len / 2 - web_t
            w = (b - a) * 0.75
            ring = tube_z(main_d / 2 + 0.06, main_d / 2 + bearing_t, w, 40).rotate((0, 90, 0))
            for s in (1, -1):
                half = ring ^ box(w * 2, main_d * 2, main_d * 2, (0, 0, s * (main_d + 0.4)))
                add("main_bearing", half.translate(((a + b) / 2, 0, 0)))

    # Per-cylinder parts ----------------------------------------------------
    Rp = B / 2 - 0.12
    ring_t_axial = max(1.0, 0.018 * B)
    ring_radial = 0.042 * B
    groove_z = [comp_h - 0.09 * B - k * (ring_t_axial + 0.045 * B) for k in range(3)]

    def make_piston():
        inner = Rp - 0.08 * B
        crown = 0.12 * B
        prof = [(0, comp_h), (Rp, comp_h), (Rp, comp_h - piston_h), (inner, comp_h - piston_h),
                (inner, comp_h - crown), (0, comp_h - crown)]
        pis = revolve(prof, 48)
        boss = cyl_x(piston_pin_d * 0.95, 2 * inner, 28)
        pis = pis + (boss ^ cyl_z(inner + 0.5, 4 * B, 48))
        # Slot for the rod's small end: it must stay below the crown, or it cuts the piston in two.
        z_lo, z_hi = -piston_pin_d * 1.6, min(piston_pin_d, comp_h - crown - 1.0)
        pis = pis - box(rod_w + 2.0, B, z_hi - z_lo, (0, 0, (z_lo + z_hi) / 2))
        pis = pis - cyl_x(piston_pin_d / 2 + 0.05, 2.5 * B, 28)
        for gz in groove_z:
            pis = pis - (tube_z(Rp - ring_radial - 0.25, Rp + 1, ring_t_axial + 0.15, 48)).translate((0, 0, gz))
        if rng.random() < 0.5:  # dished crown
            pis = pis - cyl_z(Rp * 0.6, 0.04 * B, 40).translate((0, 0, comp_h))
        return pis

    piston_man = make_piston()
    ring_man = (tube_z(Rp - ring_radial, B / 2 - 0.05, ring_t_axial, 48)
                - box(1.0, ring_radial * 4, ring_t_axial * 3, (0, -B / 2, 0)))
    pin_l = 2 * (Rp - (j(0.02, 0.08) if wide else 0.08) * B) - 1.0
    pin_man = cyl_x(piston_pin_d / 2, pin_l, 28)
    if pin_bore_r > 0.5:
        pin_man = pin_man - cyl_x(pin_bore_r, 3 * B, 20)

    def make_rod():
        rbe = crankpin_d / 2 + 0.1 * B
        rse = piston_pin_d / 2 + 0.07 * B
        sec = CS.circle(rbe, 40) + CS.circle(rse, 32).translate((0, L)) + CS.batch_hull([
            CS.square((0.26 * B, 1.0), True).translate((0, rbe * 0.6)),
            CS.square((0.16 * B, 1.0), True).translate((0, L - rse * 0.6))])
        sec = sec - CS.circle(crankpin_d / 2 + 0.15, 40) - CS.circle(piston_pin_d / 2 + 0.03, 32).translate((0, L))
        rod = extrude_yz(sec, rod_w)
        pocket_len = L - rbe - rse - 0.1 * B
        if pocket_len > 0.2 * B:
            for s in (-1, 1):
                pk = box(rod_w * 0.3, 0.1 * B, pocket_len, (s * rod_w / 2, 0, (rbe + L - rse) / 2))
                rod = rod - pk
        return rod

    rod_man = make_rod()
    cap_man = None
    if o.get("split_rod"):
        # Split at the big-end centre line with a 0.1 mm gap (keeps the shells apart in one STL).
        big = 4 * (crankpin_d + B)
        cap_man = rod_man ^ box(big, big, big, (0, 0, -big / 2 - 0.05))
        rod_man = rod_man - box(big, big, big, (0, 0, -big / 2 + 0.05))
    liner_man = tube_z(B / 2, B / 2 + liner_t, liner_len, 48).translate((0, 0, deck - liner_len / 2))

    for (cx, bang, t) in cylinders:
        bank_T = translate((cx, 0, 0)) @ rot_x(bang)
        alpha = math.radians(throw_ang[t] - bang)
        yc, zc = -r * math.sin(alpha), r * math.cos(alpha)
        s_pin = r * math.cos(alpha) + math.sqrt(L ** 2 - (r * math.sin(alpha)) ** 2)
        add("piston", piston_man, bank_T @ translate((0, 0, s_pin)))
        add("piston_pin", pin_man, bank_T @ translate((0, 0, s_pin)))
        if o["rings"]:
            for gz in groove_z:
                add("piston_ring", ring_man.rotate((0, 0, float(rng.uniform(0, 360)))),
                    bank_T @ translate((0, 0, s_pin + gz)))
        phi = math.degrees(math.atan2(yc, s_pin - zc))
        add("connecting_rod", rod_man, bank_T @ translate((0, yc, zc)) @ rot_x(phi))
        if cap_man is not None:
            add("connecting_rod", cap_man, bank_T @ translate((0, yc, zc)) @ rot_x(phi))
        if o["liners"]:
            add("cylinder_liner", liner_man, bank_T)

    # Engine block ------------------------------------------------------------
    bore_hole = B / 2 + (liner_t + 0.2 if o["liners"] else 0.0)
    block_len = (x_rear - x_front) + 0.12 * B
    wall = 0.12 * B
    skirt = -(r + j(0.35, 0.6) * B)
    cyl_bottom = deck - liner_len
    bolt_d = max(8.0, round(B * j(0.12, 0.14)))  # head bolts ~M10-M14
    bolt_xy = []  # head-bolt positions per bank, local (x, y)
    bxs = [xs[0] - p.pitch / 2 + 0.0] + [(xs[k] + xs[k + 1]) / 2 for k in range(n_throws - 1)] + [xs[-1] + p.pitch / 2]
    for bx in bxs:
        for sy in (-1, 1):
            bolt_xy.append((bx, sy * (B / 2 + liner_t + bolt_d * 1.2)))
    bank_w = B + 2 * liner_t + 4.5 * bolt_d
    cc_w = max(2 * (r + 0.25 * B) + 2 * wall, bank_w)

    blk = box(block_len, cc_w, (cyl_bottom - skirt) + 1.0, (0, 0, (cyl_bottom + skirt) / 2))
    for bang, dx in banks:
        T = rot_x(bang)
        h = deck - (cyl_bottom - 0.2 * B)
        bank_box = box(block_len, bank_w, h, (0, 0, deck - h / 2))
        blk = blk + bank_box.transform(T[:3, :])
    cavity = box(block_len - 2 * wall, cc_w - 2 * wall, (cyl_bottom - skirt) + 2.0, (0, 0, (cyl_bottom + skirt) / 2 - 1.0))
    blk = blk - cavity
    for k in range(len(main_x)):
        bx = main_x[k] if 0 < k < len(main_x) - 1 else (x_front + wall / 2 + 0.06 * B if k == 0 else x_rear - wall / 2 - 0.06 * B)
        bulk = box(0.08 * B, cc_w - 2 * wall + 1, (0.0 - skirt) + main_d * 0.6, (bx, 0, (skirt + main_d * 0.3) / 2))
        blk = blk + bulk
    blk = blk - cyl_x(main_d / 2 + bearing_t + 0.3, block_len * 2, 40)
    for (cx, bang, t) in cylinders:
        T = (translate((cx, 0, 0)) @ rot_x(bang))[:3, :]
        blk = blk - cyl_z(bore_hole, liner_len + 4, 48).translate((0, 0, deck - liner_len / 2 + 1)).transform(T)
        # crank throw clearance under each bore
        blk = blk - cyl_z(B / 2 + 0.05 * B, liner_len, 40).translate((0, 0, cyl_bottom + 1)).transform(T)
    for bang, dx in banks:
        T = rot_x(bang)[:3, :]
        for (bx, by) in bolt_xy:
            blk = blk - cyl_z(bolt_d / 2 + 0.4, 1.4 * B, 16).translate((bx, by, deck - 0.6 * B)).transform(T)
    add("engine_block", blk)

    # Head, gasket, valves, springs, cams, plugs, cover, head bolts (per bank) --
    vr_stem = max(2.5, 0.035 * B)
    for bang, dx in banks:
        TB = rot_x(bang)
        bank_cyls = [(cx, t) for (cx, ba, t) in cylinders if ba == bang]
        if o["gasket"]:
            g = box(block_len, bank_w, gasket_t, (0, 0, deck + gasket_t / 2))
            for cx, _ in bank_cyls:
                g = g - cyl_z(B / 2 + 1.0, 10, 48).translate((cx - dx * 0 - 0, 0, deck))
            for bx, by in bolt_xy:
                g = g - cyl_z(bolt_d / 2 + 0.6, 10, 16).translate((bx, by, deck))
            add("head_gasket", g.translate((0, 0, 0)), TB)
        if not o["head"]:
            continue
        head = box(block_len, bank_w, head_h, (0, 0, head_z0 + head_h / 2))
        valves = []  # (x, y, intake?)
        for cx, _ in bank_cyls:
            head = head - cyl_z(B / 2 * 0.95, 0.12 * B * 2, 48).translate((cx, 0, head_z0))
            if p.valves_per_cyl == 4:
                for vx in (-0.2 * B, 0.2 * B):
                    valves += [(cx + vx, 0.22 * B, True), (cx + vx, -0.22 * B, False)]
            else:
                valves += [(cx, 0.2 * B, True), (cx, -0.2 * B, False)]
            head = head - cyl_z(7.5, 3 * head_h, 24).translate((cx, 0, head_top))  # plug well
        for vx, vy, _ in valves:
            head = head - cyl_z(vr_stem + 0.6, 3 * head_h, 16).translate((vx, vy, head_top))
        for bx, by in bolt_xy:
            head = head - cyl_z(bolt_d / 2 + 0.4, 3 * head_h, 16).translate((bx, by, head_top))
        cam_ys = sorted({vy for _, vy, _ in valves}) if o["dohc"] else [0.0]
        spring_h = 0.72 * B  # free length; wire 0.045B, coil 0.28B, ~9 turns: realistic rate, no coil bind
        cam_base = 0.17 * B
        cam_z = head_top + spring_h + 0.06 * B + cam_base
        for cy in cam_ys:
            head = head - cyl_x(cam_base + 0.5, block_len * 2, 32).translate((0, cy, cam_z - 0.35 * B))
        add("cylinder_head", head, TB)

        valve_len = head_h + spring_h + 0.08 * B
        for vx, vy, intake in valves:
            Rh = (0.2 if intake else 0.17) * B * (0.82 if p.valves_per_cyl == 4 else 1.15)
            prof = [(0, 0), (Rh, 0), (Rh, 0.035 * B), (vr_stem * 2.2, 0.17 * B), (vr_stem, 0.27 * B),
                    (vr_stem, valve_len), (0, valve_len)]
            add("valve", revolve(prof, 32), TB @ translate((vx, vy, head_z0 + 0.12 * B)))
            if o["springs"]:
                sp = helix_spring(0.14 * B * j(0.92, 1.08), 0.0225 * B * j(0.9, 1.1), spring_h, float(rng.uniform(8.5, 10)))
                add("valve_spring", sp, TB @ translate((vx, vy, head_top + 0.3)))
        for cy in cam_ys:
            lobes_x = sorted({round(vx, 3) for vx, vy, _ in valves if (not o["dohc"]) or abs(vy - cy) < 1e-6})
            cam = cyl_x(cam_base * 0.75, (lobes_x[-1] - lobes_x[0]) + 0.6 * B, 28).translate(
                ((lobes_x[-1] + lobes_x[0]) / 2, 0, 0))
            lift = 0.11 * B
            for k, lx in enumerate(lobes_x):
                lobe = CS.batch_hull([CS.circle(cam_base, 32), CS.circle(cam_base * 0.5, 24).translate((0, lift + cam_base * 0.5))])
                lobe = lobe.rotate(float(rng.uniform(0, 360)))
                cam = cam + extrude_yz(lobe, 0.13 * B).translate((lx, 0, 0))
            for jx in np.linspace(lobes_x[0], lobes_x[-1], max(2, n_throws + 1)):
                cam = cam + cyl_x(cam_base * 0.9, 0.12 * B, 28).translate((jx + 0.15 * B, 0, 0))
            add("camshaft", cam, TB @ translate((0, cy, cam_z)))
            if o["timing"]:
                gr = gear_section(int(rng.integers(36, 50)), cam_base * 2.1, cam_base * 2.35)
                g = extrude_yz(gr - CS.circle(cam_base * 0.55, 24), 0.12 * B)
                add("timing_gear", g, TB @ translate((lobes_x[0] - 0.45 * B, cy, cam_z)))
        if o["plugs"]:
            for cx, _ in bank_cyls:
                prof = [(0, 0), (3.0, 0), (3.0, 2.0), (6.9, 2.0), (6.9, 19.0), (8.5, 19.0), (8.5, 21.0),
                        (5.5, 21.0), (5.5, 52.0), (3.0, 52.0), (3.0, 60.0), (0, 60.0)]
                plug = revolve(prof, 32) + cyl_z(10.2, 12.0, 6).translate((0, 0, 27.0))
                add("spark_plug", plug, TB @ translate((cx, 0, head_z0 - 1.0)))
        if o["head_bolts"]:
            hb_len = head_h + 0.45 * B
            bolt = cyl_z(bolt_d / 2, hb_len, 16).translate((0, 0, -hb_len / 2)) + cyl_z(bolt_d * 0.9, bolt_d * 0.7, 6).translate((0, 0, bolt_d * 0.35))
            for bx, by in bolt_xy:
                add("bolt", bolt, TB @ translate((bx, by, head_top + 0.2)))
        if o["valve_cover"]:
            cov_h = (cam_z + cam_base + lift + 0.08 * B) - head_top
            outer = box(block_len - 2, bank_w - 2, cov_h, (0, 0, head_top + cov_h / 2 + 0.3))
            inner = box(block_len - 8, bank_w - 8, cov_h, (0, 0, head_top + cov_h / 2 - 2.7))
            cov = outer - inner
            for cx, _ in bank_cyls:
                cov = cov - cyl_z(11.5, 3 * cov_h, 24).translate((cx, 0, head_top + cov_h))
            add("valve_cover", cov, TB)

    # Oil pan ---------------------------------------------------------------
    if o["oil_pan"]:
        depth = B * j(0.6, 1.0)
        pan_out = box(block_len - 4, cc_w - 4, depth, (0, 0, skirt - depth / 2 - 0.3))
        pan_in = box(block_len - 10, cc_w - 10, depth, (0, 0, skirt - depth / 2 + 2.7))
        pan = pan_out - pan_in
        flange = box(block_len, cc_w, 3.0, (0, 0, skirt - 1.8)) - box(block_len - 10, cc_w - 10, 10, (0, 0, skirt))
        add("oil_pan", pan + flange)

    # Flywheel + crank gear ------------------------------------------------------
    if o["flywheel"]:
        R = B * j(1.25, 1.6)
        fw_t = B * j(0.18, 0.26)
        disk = gear_section(int(rng.integers(100, 140)), R, R + 0.04 * B)
        fw = extrude_yz(disk - CS.circle(main_d * 0.22, 32), fw_t)
        hub_bolts = 6
        for k in range(hub_bolts):
            a = 2 * math.pi * k / hub_bolts
            fw = fw - cyl_x(max(4.0, 0.05 * B) + 0.7, fw_t * 3, 12).translate((0, main_d * 0.55 * math.cos(a), main_d * 0.55 * math.sin(a)))
        fw = fw - cyl_x(R * 0.75, fw_t * 0.5, 48).translate((fw_t * 0.3, 0, 0))  # recessed face
        fx = flange_x + flange_t / 2 + fw_t / 2 + 0.3
        add("flywheel", fw.translate((fx, 0, 0)))
        rb = max(4.0, 0.05 * B)  # ~M10 for a 100 mm bore
        fb = cyl_x(rb, fw_t + flange_t, 12) + cyl_x(rb * 1.7, rb * 1.2, 6).translate(((fw_t + flange_t) / 2 + rb * 0.6, 0, 0))
        for k in range(hub_bolts):
            a = 2 * math.pi * k / hub_bolts
            add("bolt", fb.translate((fx - flange_t / 2, main_d * 0.55 * math.cos(a), main_d * 0.55 * math.sin(a))))
    if o["timing"]:
        gr = gear_section(int(rng.integers(18, 25)), main_d * 0.62, main_d * 0.7)
        cg = extrude_yz(gr - CS.circle(main_d * 0.36 + 0.1, 28), 0.12 * B)
        add("timing_gear", cg.translate((x_front - 0.15 * B - snout_len * 0.6, 0, 0)))

    # Unrelated accessories (label "other") so the classifier learns to say no.
    if o.get("accessories", True):
        lo = np.min([m.bounds[0] for _, m in parts], axis=0)
        hi = np.max([m.bounds[1] for _, m in parts], axis=0)
        for _ in range(int(rng.integers(0, 5))):
            kind = rng.choice(["bracket", "pulley", "hose", "sensor", "housing"])
            if kind == "bracket":
                sec = CS.square((j(40, 120), j(4, 8))) + CS.square((j(4, 8), j(30, 90)))
                acc = extrude_yz(sec, j(20, 50)) - cyl_x(j(3, 6), 200, 16).translate((0, j(15, 30), 3))
            elif kind == "pulley":
                R = j(30, 80)
                w = j(15, 30)
                prof = [(5, -w / 2), (R, -w / 2), (R, -w / 4), (R * 0.85, 0), (R, w / 4), (R, w / 2), (5, w / 2)]
                acc = revolve(prof, 48)
            elif kind == "hose":
                acc = Man.revolve(CS.circle(j(8, 16), 16).translate((j(60, 140), 0)), 32, float(j(60, 150)))
                acc = acc - Man.revolve(CS.circle(j(4, 6), 16).translate((j(60, 140), 0)), 32, 40.0)
            elif kind == "sensor":
                acc = cyl_z(j(6, 10), j(30, 60), 24) + cyl_z(j(9, 14), 8, 6).translate((0, 0, 10))
            else:
                acc = box(j(80, 160), j(60, 110), j(60, 110)) + cyl_x(j(20, 35), j(170, 200), 32)
            pos = np.array([rng.uniform(lo[0], hi[0]), rng.choice([lo[1] - 60, hi[1] + 60]), rng.uniform(lo[2], hi[2])])
            add("other", acc, translate(pos) @ rot_x(float(rng.uniform(0, 360))))

    return GeneratedEngine(params=p, parts=parts)


def random_engine(seed: int) -> GeneratedEngine:
    rng = np.random.default_rng(seed)
    return build_engine(random_params(rng), rng)


def randomly_posed(engine: GeneratedEngine, seed: int) -> GeneratedEngine:
    """Same engine, rotated and moved arbitrarily (uploads have no fixed frame)."""
    rng = np.random.default_rng(seed)
    T = trimesh.transformations.random_rotation_matrix(rng.random(3))
    T[:3, 3] = rng.uniform(-500, 500, 3)
    parts = [(lab, m.copy().apply_transform(T)) for lab, m in engine.parts]
    return GeneratedEngine(params=engine.params, parts=parts)
