"""Robotics export: URDF with exact inertial properties.

Each link is a set of parts (by default one top-level sub-assembly per link). Links form a
chain in the given order. For every joint the axis and pivot are suggested from the
round parts that connect the two links (bearings first, then shafts, then any
axisymmetric part of the child near the parent); when nothing round connects them the
joint is "fixed". Units follow URDF: metres, kilograms, kg*m^2.
"""
from __future__ import annotations

import io
import zipfile
from xml.sax.saxutils import escape

import numpy as np
import trimesh


def default_links(asm) -> list[dict]:
    """Top-level sub-assemblies as links (loose top-level parts form the base)."""
    t = asm.tree
    if not t:
        return [{"name": "base_link", "ids": list(range(len(asm.parts)))}]
    node = t
    while len(node["children"]) == 1 and not node["parts"]:
        node = node["children"][0]                      # skip single wrapper levels
    links = []

    def ids(n):
        out = list(n["parts"])
        for c in n["children"]:
            out += ids(c)
        return out
    if node["parts"]:
        links.append({"name": "base_link", "ids": list(node["parts"])})
    for c in node["children"]:
        links.append({"name": c["name"], "ids": ids(c)})
    return [l for l in links if l["ids"]]


def _safe(name: str, used: set) -> str:
    s = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name).strip("_") or "link"
    base, k = s, 2
    while s in used:
        s = f"{base}_{k}"
        k += 1
    used.add(s)
    return s


def suggest_chain(asm, links: list[dict]) -> list:
    """Joints for a whole chain, worked out from the tip back to the base: the bearings a joint
    runs on are not offered to the joint below it (a link holds the bearings of the joint at its
    far end too, and they must not be mistaken for the joint at its near end)."""
    joints, used = [None] * len(links), set()
    for k in range(len(links) - 1, 0, -1):
        j = suggest_joint(asm, links[k - 1]["ids"], links[k]["ids"], exclude=used)
        joints[k] = j
        used |= set(j.get("from_parts", []))
    return joints


def suggest_joint(asm, parent_ids: list[int], child_ids: list[int], exclude: set | None = None) -> dict:
    """Axis + pivot from the round parts shared by / closest between two links."""
    pc = np.array([asm.parts[i].info.center for i in parent_ids])
    lo, hi = pc.min(0), pc.max(0)
    pad = 0.05 * float(np.linalg.norm(hi - lo) + 1e-9)
    cands = []
    for i in child_ids + parent_ids:
        if exclude and i in exclude:
            continue
        p = asm.parts[i]
        round_ = p.info.sym_score > 0.7 and p.info.features.get("cyl_frac", 0) > 0.25
        if not round_:
            continue
        cat = (p.component or {}).get("category", "") + " " + p.specs.get("kind", "")
        # Only parts that make rotation possible count: bearings, shafts, motor output shafts.
        if p.label == "bearing" or "bearing" in cat:
            rank = 0
        elif "shaft" in cat or p.label == "motor" or "gearbox" in cat:
            rank = 1
        else:
            continue
        c = np.asarray(p.info.sym_center)
        near = bool(np.all(c >= lo - pad) & np.all(c <= hi + pad))
        if i in child_ids and not near:
            continue
        cands.append((rank, -p.info.sym_diameter, i))
    if not cands:
        return {"type": "fixed", "reason": "no bearing, shaft or motor connects these links (rigid joint)"}
    # A joint runs on several coaxial parts (a bearing pair, the shaft through them); a gearbox
    # bearing sits on its own off-axis line. Cluster candidates by axis line and take the best
    # supported line (bearings count double, bigger parts more).
    lines = []
    for rank, negd, i in cands:
        p = asm.parts[i]
        a = np.array(p.info.sym_axis, dtype=float)
        a /= np.linalg.norm(a) + 1e-12
        c = np.asarray(p.info.sym_center, dtype=float)
        w = (2.0 if rank == 0 else 1.0) * p.info.sym_diameter     # big joint bearings dominate
        for ln in lines:
            if abs(float(a @ ln["axis"])) > 0.998:
                d = c - ln["point"]
                off = np.linalg.norm(d - (d @ ln["axis"]) * ln["axis"])
                if off < max(1.0, 0.05 * p.info.sym_diameter):
                    ln["members"].append(i)
                    ln["score"] += w
                    break
        else:
            lines.append({"axis": a, "point": c, "members": [i], "score": w})
    # The joint sits where the two links meet: favour lines passing near the interface.
    def box(ids):
        b = np.array([asm.parts[i].mesh.bounds for i in ids])
        return b[:, 0].min(0), b[:, 1].max(0)
    (plo, phi), (clo, chi) = box(parent_ids), box(child_ids)
    ilo, ihi = np.maximum(plo, clo), np.minimum(phi, chi)
    meet = (ilo + ihi) / 2 if np.all(ihi >= ilo) else (0.5 * (plo + phi) + 0.5 * (clo + chi)) / 2
    size = float(np.linalg.norm(chi - clo)) + 1e-9
    for ln in lines:
        d = meet - ln["point"]
        dist = float(np.linalg.norm(d - (d @ ln["axis"]) * ln["axis"]))
        ln["score"] *= float(np.exp(-dist / (0.25 * size)))
    best_line = max(lines, key=lambda l: l["score"])
    members = [asm.parts[i] for i in best_line["members"]]
    axis = best_line["axis"]
    pts = np.array([m.info.sym_center for m in members])
    pivot = pts.mean(0)
    names = ", ".join(sorted({(m.display_name if hasattr(m, "display_name") else None) or (m.name or str(m.id))
                              for m in members}))[:120]
    return {"type": "revolute", "axis": axis.tolist(), "pivot_mm": list(map(float, pivot)),
            "from_parts": [m.id for m in members],
            "reason": f"common axis of {len(members)} part(s): {names}"}


def link_inertial(asm, ids: list[int]) -> dict:
    mp = asm.mass_properties(ids)
    return mp


def build_urdf(asm, links: list[dict] | None = None, robot_name: str = "robot") -> tuple[str, dict, list]:
    links = links or default_links(asm)
    used: set = set()
    names = [_safe(l["name"], used) for l in links]
    xml = [f'<?xml version="1.0"?>', f'<robot name="{escape(_safe(robot_name, set()))}">']
    meshes, joints_out = {}, []
    frames = []                                         # world position (mm) of each link frame
    chain = suggest_chain(asm, links)
    for k, (l, nm) in enumerate(zip(links, names)):
        if k == 0:
            frames.append(np.zeros(3))
            joints_out.append(None)
        else:
            j = chain[k]
            pivot = np.asarray(j.get("pivot_mm", asm.mass_properties(l["ids"]).get("com_mm", [0, 0, 0])))
            frames.append(pivot)
            joints_out.append(j)
    for k, (l, nm) in enumerate(zip(links, names)):
        o = frames[k]
        mp = link_inertial(asm, l["ids"])
        m = max(mp["mass_kg"], 1e-6)
        com = (np.asarray(mp.get("com_mm", o)) - o) / 1000.0
        I = np.asarray(mp.get("inertia_kgmm2", np.eye(3) * 1e-3)) * 1e-6
        mesh = trimesh.util.concatenate([asm.parts[i].mesh for i in l["ids"]])
        mesh = mesh.copy()
        mesh.apply_translation(-o)
        mesh.apply_scale(0.001)
        fn = f"meshes/{nm}.stl"
        meshes[fn] = mesh.export(file_type="stl")
        xml += [f'  <link name="{nm}">',
                '    <inertial>',
                f'      <origin xyz="{com[0]:.6g} {com[1]:.6g} {com[2]:.6g}" rpy="0 0 0"/>',
                f'      <mass value="{m:.6g}"/>',
                f'      <inertia ixx="{I[0, 0]:.6g}" ixy="{I[0, 1]:.6g}" ixz="{I[0, 2]:.6g}" '
                f'iyy="{I[1, 1]:.6g}" iyz="{I[1, 2]:.6g}" izz="{I[2, 2]:.6g}"/>',
                '    </inertial>',
                f'    <visual><geometry><mesh filename="{fn}"/></geometry></visual>',
                f'    <collision><geometry><mesh filename="{fn}"/></geometry></collision>',
                '  </link>']
    for k in range(1, len(links)):
        j = joints_out[k]
        xyz = (frames[k] - frames[k - 1]) / 1000.0
        jn = f"{names[k - 1]}__{names[k]}"
        if j["type"] == "revolute":
            ax = j["axis"]
            xml += [f'  <joint name="{jn}" type="revolute">',
                    f'    <!-- suggested from the {escape(j["reason"])}; check the axis and set real limits -->',
                    f'    <parent link="{names[k - 1]}"/>', f'    <child link="{names[k]}"/>',
                    f'    <origin xyz="{xyz[0]:.6g} {xyz[1]:.6g} {xyz[2]:.6g}" rpy="0 0 0"/>',
                    f'    <axis xyz="{ax[0]:.6g} {ax[1]:.6g} {ax[2]:.6g}"/>',
                    '    <limit lower="-3.14159" upper="3.14159" effort="10" velocity="3"/>',
                    '  </joint>']
        else:
            xml += [f'  <joint name="{jn}" type="fixed">', f'    <!-- {escape(j["reason"])} -->',
                    f'    <parent link="{names[k - 1]}"/>', f'    <child link="{names[k]}"/>',
                    f'    <origin xyz="{xyz[0]:.6g} {xyz[1]:.6g} {xyz[2]:.6g}" rpy="0 0 0"/>',
                    '  </joint>']
    xml.append("</robot>")
    summary = [{"link": names[k], "parts": len(links[k]["ids"]),
                "mass_kg": round(link_inertial(asm, links[k]["ids"])["mass_kg"], 4),
                "joint_to_parent": joints_out[k]} for k in range(len(links))]
    return "\n".join(xml), meshes, summary


def urdf_zip(asm, links=None, robot_name="robot") -> tuple[bytes, list]:
    xml, meshes, summary = build_urdf(asm, links, robot_name)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{robot_name}.urdf", xml)
        for fn, data in meshes.items():
            z.writestr(fn, data)
        z.writestr("README.txt", "URDF exported by EngineLab. Masses/inertias are exact from the CAD B-rep using "
                                 "the materials chosen in the parts list. Revolute joints are suggestions from "
                                 "bearing/shaft axes: check axes, directions and limits before simulating.\n")
    return buf.getvalue(), summary
