"""Engine balance and crankshaft torsional vibration, from the CAD geometry.

Balance: every crank throw's angle and axial position are read from the crankshaft itself,
every cylinder's direction from its piston. The shaking force and rocking couple of the
reciprocating masses are summed over one revolution and split into orders: 1st order
(once per revolution) and 2nd order (twice, from the rod's angularity, lambda = r / l).
Rotating unbalance (crankpins + big ends) is reported before counterweights.

Torsional vibration: a lumped (Holzer) model of the crank: one inertia per throw (crank share +
pistons and rods), journal stiffness between throws, a flywheel at one end. Its natural
frequencies, divided by the excitation orders of a four-stroke engine, give the critical speeds;
the "major" orders (multiples of half the cylinder count) excite the crank hardest.

Standard textbook models (Taylor, "The Internal-Combustion Engine in Theory and Practice";
Ker Wilson, "Practical Solution of Torsional Vibration Problems"). Estimates for comparing
designs; a torsional damper, crank web stiffness details and gas-torque amplitudes need a
dedicated torsional-vibration study for sign-off.
"""
from __future__ import annotations

import math

import numpy as np

STANDARD_PHASES = {2: [0, 180], 3: [0, 240, 120], 4: [0, 180, 180, 0], 5: [0, 144, 288, 72, 216],
                   6: [0, 120, 240, 240, 120, 0]}


def _basis(a):
    a = a / np.linalg.norm(a)
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(a, t)
    e1 /= np.linalg.norm(e1)
    return a, e1, np.cross(a, e1)


def _throws(d: dict) -> list[dict]:
    """Group the crankpin slices into throws: axial position (mm), angle (deg), radius (mm)."""
    a, e1, e2 = _basis(d["axis"])
    o = d["point"]
    pts = d["pin_points"]
    if len(pts) == 0:
        return []
    t = (pts - o) @ a
    order = np.argsort(t)
    t, pts = t[order], pts[order]
    gaps = np.diff(t)
    step = np.median(gaps) if len(gaps) else 1.0
    groups, cur = [], [0]
    for i in range(1, len(t)):
        if t[i] - t[i - 1] > 3 * step:
            groups.append(cur)
            cur = []
        cur.append(i)
    groups.append(cur)
    out = []
    for g in groups:
        off = pts[g].mean(0) - o
        off -= (off @ a) * a
        out.append({"x_mm": float(t[g].mean()), "angle_deg": float(math.degrees(math.atan2(off @ e2, off @ e1)) % 360),
                    "radius_mm": float(np.linalg.norm(off))})
    return out


def _orders(vec: np.ndarray, phi: np.ndarray, n: int) -> float:
    """Peak magnitude of the order-n part of a vector signal over one revolution."""
    c = np.exp(-1j * n * phi)
    X = 2 * (vec * c[:, None]).mean(0)
    rec = np.real(X[None, :] * np.exp(1j * n * phi)[:, None])
    return float(np.linalg.norm(rec, axis=1).max())


def shaking(throw_ang, throw_x, cyl_throw, cyl_beta, m_rec, m_rot, r, lam, omega):
    """1st/2nd-order reciprocating force (N) and couple (N m), rotating unbalance (N, N m)."""
    phi = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    x0 = np.mean([throw_x[k] for k in cyl_throw])
    F = np.zeros((len(phi), 2))
    M = np.zeros((len(phi), 2))
    for k, beta in zip(cyl_throw, cyl_beta):
        alpha = phi + math.radians(throw_ang[k]) - beta
        f = m_rec * r * omega ** 2 * (np.cos(alpha) + lam * np.cos(2 * alpha))
        u = np.array([math.cos(beta), math.sin(beta)])
        F += f[:, None] * u[None, :]
        lever = (throw_x[k] - x0) / 1000.0
        M += lever * f[:, None] * np.array([-u[1], u[0]])[None, :]     # couple about an axis normal to the crank
    rot = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in throw_ang])
    xs = (np.array(throw_x) - np.mean(throw_x)) / 1000.0
    rot_force = m_rot * r * omega ** 2 * np.linalg.norm(rot.sum(0))
    rot_couple = m_rot * r * omega ** 2 * np.linalg.norm((rot * xs[:, None]).sum(0))
    return {"primary_force_N": _orders(F, phi, 1), "secondary_force_N": _orders(F, phi, 2),
            "primary_couple_Nm": _orders(M, phi, 1), "secondary_couple_Nm": _orders(M, phi, 2),
            "rotating_force_N": float(rot_force), "rotating_couple_Nm": float(rot_couple)}


def analyse(asm) -> dict | None:
    from .. import materials
    from ..specs import crank_measurements
    cranks = [p for p in asm.parts if p.label == "crankshaft"]
    pistons = [p for p in asm.parts if p.label == "piston"]
    eng = asm.engine_model()
    if not cranks or not pistons or eng is None:
        return None
    crank = cranks[0]
    d: dict = {}
    crank_measurements(crank.mesh, crank.info, details=d)
    if "axis" not in d:
        return None
    throws = _throws(d)
    if not throws:
        return None
    a, e1, e2 = _basis(d["axis"])
    o = d["point"]
    throw_x = [t["x_mm"] for t in throws]
    throw_ang = [t["angle_deg"] for t in throws]
    cyl_throw, cyl_beta = [], []
    for p in pistons:
        v = p.info.center - o
        tx = float(v @ a)
        k = int(np.argmin([abs(tx - x) for x in throw_x]))
        u = v - (v @ a) * a
        cyl_throw.append(k)
        cyl_beta.append(math.atan2(u @ e2, u @ e1))
    r = eng.stroke_mm / 2000.0
    lam = eng.stroke_mm / 2 / eng.rod_mm
    f_big = eng.rod_big_end_frac
    m_rec = eng.m_piston_kg + (1 - f_big) * eng.m_rod_kg
    per_throw = np.bincount(cyl_throw, minlength=len(throws))
    m_rot_throw = f_big * eng.m_rod_kg * max(1, int(per_throw.max()))
    omega = eng.max_rpm * 2 * math.pi / 60
    sh = shaking(throw_ang, throw_x, cyl_throw, cyl_beta, m_rec, m_rot_throw, r, lam, omega)
    out = {"rpm": eng.max_rpm, "reciprocating_mass_kg": round(m_rec, 3), "rotating_mass_per_throw_kg": round(m_rot_throw, 3),
           "throws": [{**t, "cylinders": int(per_throw[i])} for i, t in enumerate(throws)],
           "phasing_deg": [round((t - throw_ang[0]) % 360) % 360 for t in throw_ang],
           **{k: round(v, 1) for k, v in sh.items()}}
    # Compare with the standard phasing for this inline cylinder count, if different.
    n = len(throws)
    if all(c == 1 for c in per_throw) and n in STANDARD_PHASES:
        std = [(throw_ang[0] + s) % 360 for s in STANDARD_PHASES[n]]
        rel = sorted(out["phasing_deg"])
        if rel != sorted(round(s) % 360 for s in STANDARD_PHASES[n]) or out["phasing_deg"] != [round(s) % 360 for s in STANDARD_PHASES[n]]:
            s2 = shaking(std, throw_x, cyl_throw, cyl_beta, m_rec, m_rot_throw, r, lam, omega)
            if s2["primary_couple_Nm"] < 0.5 * sh["primary_couple_Nm"] or s2["secondary_couple_Nm"] < 0.5 * sh["secondary_couple_Nm"]:
                out["standard_phasing"] = {"phasing_deg": STANDARD_PHASES[n],
                                           **{k: round(v, 1) for k, v in s2.items()}}
    out["torsional"] = torsional(asm, crank, throws, per_throw, m_rec, f_big * eng.m_rod_kg, r, eng, a)
    return out


def torsional(asm, crank, throws, per_throw, m_rec, m_rot, r, eng, axis) -> dict:
    from .. import materials
    mat = materials.get(crank.material)
    rho = mat.density * 1e-9                                     # kg/mm^3
    I = np.asarray(crank.mesh.moment_inertia) * rho * 1e-6       # kg m^2 about the centroid
    J_crank = float(axis @ I @ axis)
    n = len(throws)
    J = [J_crank / n + int(c) * (m_rec * r * r / 2 + m_rot * r * r) for c in per_throw]
    fly = [p for p in asm.parts if p.label == "flywheel"]
    if fly:
        Ifw = np.asarray(fly[0].mesh.moment_inertia) * materials.get(fly[0].material).density * 1e-15
        J_fw = float(axis @ Ifw @ axis)
        fw_note = "flywheel from the model"
    else:
        J_fw = 0.06 * eng.displacement_l + 0.02
        fw_note = f"no flywheel in the model: a typical {J_fw:.2f} kg m^2 is assumed"
    xs = [t["x_mm"] for t in throws]
    pitch = float(np.median(np.diff(xs))) if n > 1 else 100.0
    d = (eng.main_d_mm or 0.65 * eng.bore_mm) / 1000.0
    G = mat.E * 1e9 / (2 * (1 + mat.poisson))
    k_shaft = G * math.pi * d ** 4 / 32 / (1.6 * pitch / 1000.0)  # webs make a throw ~1.6x as flexible as a plain shaft
    k_fw = G * math.pi * d ** 4 / 32 / (0.8 * pitch / 1000.0)
    Js = J + [J_fw]
    m = len(Js)
    K = np.zeros((m, m))
    for i in range(m - 1):
        k = k_shaft if i < m - 2 else k_fw
        K[i, i] += k
        K[i + 1, i + 1] += k
        K[i, i + 1] -= k
        K[i + 1, i] -= k
    from scipy.linalg import eigh
    w2, _ = eigh(K, np.diag(Js))
    freqs = [float(math.sqrt(max(x, 0)) / (2 * math.pi)) for x in w2[1:3]]
    n_cyl = int(sum(per_throw))
    major = [k * n_cyl / 2 for k in range(1, 9)]
    crit = []
    lo, hi = 0.12 * eng.max_rpm, 1.1 * eng.max_rpm
    for mode, f in enumerate(freqs, 1):
        for q in [x / 2 for x in range(1, 25)]:
            rpm = 60 * f / q
            if lo <= rpm <= hi:
                crit.append({"mode": mode, "order": q, "rpm": round(rpm), "major": any(abs(q - mq) < 1e-6 for mq in major)})
    crit.sort(key=lambda c: (not c["major"], c["rpm"]))
    return {"natural_frequencies_hz": [round(f, 1) for f in freqs], "critical_speeds": crit[:12],
            "major_orders": major[:4], "flywheel": fw_note, "J_crank_kgm2": round(J_crank, 4),
            "stiffness_Nm_per_rad": round(k_shaft)}
