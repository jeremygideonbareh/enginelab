"""Robot joint torques and motor sizing from the CAD.

For every revolute joint of the link chain (links and joint axes as in the URDF export):
  * holding torque in the modelled pose, with gravity in its worst direction;
  * worst case: everything beyond the joint stretched straight out (upper bound from the link
    lengths), plus a payload at the tip;
  * acceleration torque at a chosen angular acceleration (stretched inertia);
and the motor that drives the joint (the motor/servo nearest the joint axis), with its rated
torque from a table of common robot actuators, so you see required vs available torque.

Statics + rigid-body inertia with exact masses and centres of mass from the B-rep. Gear ratios
inside servos are included in their rated torque; for a bare stepper/BLDC driving through an
external gearbox or belt, multiply its torque by the ratio (shown as the ratio needed).
"""
from __future__ import annotations

import math
import re

import numpy as np

G = 9.81
# (pattern on name/designation/parent, rated torque N m, what) — stall/holding torque, typical
ACTUATORS = [
    (r"xm540", 10.6, "Dynamixel XM540 (stall)"), (r"xm430", 4.1, "Dynamixel XM430 (stall)"),
    (r"xh430", 3.4, "Dynamixel XH430 (stall)"), (r"xl430|dc11", 1.5, "Dynamixel XL430 (stall)"),
    (r"xc430", 1.6, "Dynamixel XC430 (stall)"), (r"xl330.?m288|xl330|xc330|dc15", 0.52, "Dynamixel XL/XC330 (stall)"),
    (r"sts3215|st3215|scs215", 2.9, "Feetech STS3215 (stall, 12 V)"), (r"sts3032", 0.45, "Feetech STS3032 (stall)"),
    (r"mg996", 1.0, "MG996R hobby servo (stall)"), (r"mg90|sg90", 0.18, "SG90 / MG90 micro servo (stall)"),
    (r"nema[\s-]*34", 4.5, "NEMA 34 stepper (holding)"), (r"nema[\s-]*23", 1.2, "NEMA 23 stepper (holding)"),
    (r"nema[\s-]*17", 0.45, "NEMA 17 stepper (holding)"), (r"nema[\s-]*14", 0.2, "NEMA 14 stepper (holding)"),
    (r"nema[\s-]*11", 0.1, "NEMA 11 stepper (holding)"), (r"nema[\s-]*8\b", 0.03, "NEMA 8 stepper (holding)"),
]


def _actuator(p) -> tuple[float, str] | None:
    comp = p.component or {}
    text = " ".join([p.name or "", comp.get("designation") or "", comp.get("category") or "",
                     str((comp.get("size") or {}).get("bought_assembly", ""))] + list(p.path[-2:])).lower()
    for pat, tq, what in ACTUATORS:
        if re.search(pat, text):
            return tq, what
    return None


GEARBOX = re.compile(r"pgb|gear\s*box|gearbox|planetary|harmonic|strain\s*wave|cycloid|reducer|\d+\s*:\s*1", re.I)


def _up_axis(asm, links, mp) -> np.ndarray:
    """Which world axis is up: from the base link's centre of mass towards the rest of the robot,
    snapped to the nearest CAD axis (models are drawn axis-aligned)."""
    base = np.asarray(mp[0].get("com_mm", [0, 0, 0]), float)
    rest_m = sum(m["mass_kg"] for m in mp[1:]) or 1.0
    rest = sum(m["mass_kg"] * np.asarray(m.get("com_mm", base), float) for m in mp[1:]) / rest_m
    v = rest - base
    if np.linalg.norm(v) < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    k = int(np.argmax(np.abs(v)))
    up = np.zeros(3)
    up[k] = math.copysign(1.0, v[k])
    return up


def _order_chain(asm, links: list[dict]) -> tuple[list[dict], bool]:
    """Links in kinematic order from the base: start at the base link and repeatedly take the
    unvisited link that touches the current tip (bounding-box gap). Returns (links, clean) where
    clean=False means the chain had to jump (the CAD tree does not map neatly onto arm links)."""
    if len(links) < 3:
        return links, True
    boxes = []
    for l in links:
        b = np.array([asm.parts[i].mesh.bounds for i in l["ids"]])
        boxes.append((b[:, 0].min(0), b[:, 1].max(0)))
    gap = lambda a, b: float(np.linalg.norm(np.maximum(0, np.maximum(boxes[a][0] - boxes[b][1], boxes[b][0] - boxes[a][1]))))
    order, left, clean = [0], set(range(1, len(links))), True
    while left:
        tip = order[-1]
        nxt = min(left, key=lambda j: (gap(tip, j), j))
        if gap(tip, nxt) > 5.0:
            clean = False
        order.append(nxt)
        left.remove(nxt)
    return [links[i] for i in order], clean


def _line_dist(c, pivot, axis):
    d = np.asarray(c, float) - pivot
    return float(np.linalg.norm(d - (d @ axis) * axis))


def joint_torques(asm, payload_kg: float = 0.2, accel: float = 5.0) -> dict:
    from .robotics import default_links, suggest_chain
    links = default_links(asm)
    if len(links) < 2:
        return {"joints": [], "note": "Only one link was found (no sub-assemblies), so there are no joints to size."}
    links, clean = _order_chain(asm, links)
    mp = [asm.mass_properties(l["ids"]) for l in links]
    up = _up_axis(asm, links, mp)
    gvec = -up
    names = [l["name"] for l in links]
    structure_ok = clean and len(set(names)) == len(names) and len(links[0]["ids"]) < 0.6 * len(asm.parts)
    joints = []
    pivots = [None]
    chain = suggest_chain(asm, links)
    for k in range(1, len(links)):
        j = chain[k]
        pivots.append(np.asarray(j["pivot_mm"], float) if j["type"] == "revolute" else None)
        joints.append((k, j))
    # farthest point of the last link (tip) for the payload
    last = links[-1]["ids"]
    pts = np.vstack([asm.parts[i].mesh.vertices for i in last])
    out = []
    for k, j in joints:
        if j["type"] != "revolute":
            continue
        axis = np.asarray(j["axis"], float)
        axis /= np.linalg.norm(axis)
        p = pivots[k]
        down = list(range(k, len(links)))
        mass = sum(mp[i]["mass_kg"] for i in down)
        # modelled pose, worst gravity direction: |axis x first moment|
        M1 = sum(mp[i]["mass_kg"] * (np.asarray(mp[i].get("com_mm", p)) - p) for i in down) / 1000.0
        tip = pts[np.argmax([_line_dist(v, p, axis) for v in pts[:: max(1, len(pts) // 4000)]]) * max(1, len(pts) // 4000)]
        tip_arm = _line_dist(tip, p, axis) / 1000.0
        # gravity only loads a joint through its component perpendicular to the axis
        sin_g = float(np.linalg.norm(gvec - (gvec @ axis) * axis))
        hold_pose = G * abs(float(axis @ np.cross(M1, gvec))) + G * payload_kg * tip_arm * sin_g
        # stretched upper bound: each downstream COM as far out as the chain allows
        reach, stretched, J = 0.0, 0.0, 0.0
        for i in down:
            seg = 0.0 if i == k else (np.linalg.norm(pivots[i] - pivots[i - 1]) / 1000.0 if pivots[i] is not None
                                      and pivots[i - 1] is not None else 0.0)
            reach += seg
            ref = pivots[i] if pivots[i] is not None else p
            arm = reach + np.linalg.norm(np.asarray(mp[i].get("com_mm", ref)) - ref) / 1000.0
            stretched += mp[i]["mass_kg"] * arm
            I = np.asarray(mp[i].get("inertia_kgmm2", np.zeros((3, 3)))) * 1e-6
            J += float(axis @ I @ axis) + mp[i]["mass_kg"] * arm ** 2
        tip_reach = max(tip_arm, reach + (np.linalg.norm(tip - (pivots[-1] if pivots[-1] is not None else p)) / 1000.0))
        hold_worst = G * sin_g * (stretched + payload_kg * tip_reach)
        J += payload_kg * tip_reach ** 2
        acc = J * accel
        # the motor driving this joint: the actuator nearest the joint axis in the two links
        cand = []
        for i in links[k - 1]["ids"] + links[k]["ids"]:
            q = asm.parts[i]
            a = _actuator(q)
            if a and (q.label in ("motor", "unknown", "custom") or "servo" in (q.component or {}).get("category", "")
                      or "stepper" in (q.component or {}).get("category", "")):
                cand.append((_line_dist(q.info.center, p, axis), a, q))
        motor = None
        if cand:
            dist, (tq, what), q = min(cand, key=lambda c: c[0])
            need = hold_worst + acc
            geared = bool(GEARBOX.search(" ".join(q.path[-3:] + [q.name or ""])))
            verdict = ("OK" if need <= 0.5 * tq else "marginal" if need <= tq else "too weak")
            if geared and verdict != "OK":
                verdict = f"needs >= {need / (0.5 * tq):.0f}:1 reduction"
            motor = {"part": q.id, "name": q.name or "", "rated_Nm": tq, "type": what,
                     "distance_to_axis_mm": round(dist, 1), "through_gearbox": geared,
                     "load_ratio": round(need / tq, 2), "verdict": verdict,
                     "ratio_needed": round(need / (0.5 * tq), 1) if need > 0.5 * tq else None}
        out.append({"joint": f"{links[k - 1]['name']} → {links[k]['name']}", "axis": [round(x, 3) for x in axis],
                    "gravity_factor": round(sin_g, 3),
                    "moving_mass_kg": round(mass, 3), "hold_pose_Nm": round(hold_pose, 3),
                    "hold_worst_Nm": round(hold_worst, 3), "accel_Nm": round(acc, 3),
                    "required_Nm": round(hold_worst + acc, 3), "inertia_kgm2": round(J, 5), "motor": motor,
                    "reason": j.get("reason", "")})
    from . import materials
    custom = [p for p in asm.parts if p.label == "custom"]
    printed = sum(materials.get(p.material).category == "polymer" for p in custom)
    return {"payload_kg": payload_kg, "accel_rad_s2": accel, "joints": out, "up_axis": up.tolist(),
            "structure_ok": structure_ok, "custom_parts": len(custom), "custom_printed": printed,
            "note": ("Worst case = everything beyond the joint stretched straight out, gravity at its worst, payload at "
                     "the tip, plus the acceleration torque. 'OK' means the need is under half the actuator's stall/"
                     "holding torque (continuous use); a bare stepper/BLDC through a belt or gearbox needs the ratio shown.")}
