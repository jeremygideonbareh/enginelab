"""Assembly breakdown helpers: fast per-definition features, domain detection,
general (non-engine) part labels, and the bill of materials.
"""
from __future__ import annotations

import copy
import re
from collections import OrderedDict

import numpy as np

from .features import ShapeInfo, context_features, shape_features

GENERAL_LABELS = ["fastener", "bearing", "linear_motion", "transmission", "motor", "structure",
                  "electronics", "spring", "seal", "fluid", "custom"]
FAMILY_TO_LABEL = {"fluid": "fluid", "fastener": "fastener", "bearing": "bearing", "linear motion": "linear_motion",
                   "transmission": "transmission", "motor": "motor", "structure": "structure",
                   "electronics": "electronics", "spring": "spring", "seal": "seal"}
DEFAULT_MATERIAL = {"fastener": "steel_bolt88", "bearing": "steel_52100", "linear_motion": "steel_52100",
                    "transmission": "steel_4140", "motor": "steel_1018", "structure": "al_6061",
                    "electronics": "pa66gf30", "spring": "steel_crsi", "seal": "pa66gf30", "fluid": "brass_c360",
                    "custom": "al_6061"}
PRINTED_HINT = re.compile(r"(\[a\]|_a\b|printed|\bpla\b|petg|\babs\b|\basa\b|cowling|duct)",
                          re.IGNORECASE)

_COPY_SUFFIX = re.compile(r"(\s*\(\d+\))+$|:\d+$|<\d+>$|\.\d{3}$|\s+v\d+$")   # ' v9' = Fusion 360 version


def display_name(name: str | None) -> str | None:
    """'M3x8 SHCS (1) (18)' -> 'M3x8 SHCS';  'Bracket:2' -> 'Bracket'."""
    if not name:
        return name
    prev = None
    while prev != name:
        prev = name
        name = _COPY_SUFFIX.sub("", name).strip()
    return name or prev


def _kabsch(a: np.ndarray, b: np.ndarray):
    """Rigid transform (R, t) with b ~ R a + t, and the RMS residual."""
    ca, cb = a.mean(0), b.mean(0)
    H = (a - ca).T @ (b - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = cb - R @ ca
    res = float(np.sqrt(np.mean(np.sum((a @ R.T + t - b) ** 2, axis=1))))
    return R, t, res


def _moved(info: ShapeInfo, R: np.ndarray, t: np.ndarray) -> ShapeInfo:
    out = copy.copy(info)
    out.center = R @ info.center + t
    out.sym_center = R @ info.sym_center + t
    out.axes = (R @ info.axes.T).T
    out.sym_axis = R @ info.sym_axis
    return out


def _features_job(items):
    return [shape_features(m, seed=i) for i, m in items]


def fast_shape_features(raws, progress=None, n_jobs: int = -1) -> list[ShapeInfo]:
    """Shape features once per part definition; instances get theirs by a rigid transform.
    Unique definitions are processed in parallel for large assemblies."""
    first: "OrderedDict[str, int]" = OrderedDict()
    keys = []
    for i, r in enumerate(raws):
        k = r.definition or f"__{i}"
        keys.append(k)
        first.setdefault(k, i)
    uniq = list(first.values())
    if len(uniq) > 150 and n_jobs != 1:
        from joblib import Parallel, delayed
        chunks = [uniq[j:j + 40] for j in range(0, len(uniq), 40)]
        res = Parallel(n_jobs=n_jobs)(delayed(_features_job)([(i, raws[i].mesh) for i in c]) for c in chunks)
        computed = {i: inf for c, r in zip(chunks, res) for i, inf in zip(c, r)}
    else:
        computed = {}
        for n, i in enumerate(uniq):
            computed[i] = shape_features(raws[i].mesh, seed=i)
            if progress and n % 50 == 0:
                progress(n, len(uniq))
    infos = []
    for i, r in enumerate(raws):
        j = first[keys[i]]
        base = computed[j]
        if i == j:
            infos.append(base)
            continue
        a, b = raws[j].mesh.vertices, r.mesh.vertices
        if len(a) == len(b):
            R, t, res = _kabsch(a, b)
            if res < 1e-3 * (np.linalg.norm(base.extents) + 1e-9):
                infos.append(_moved(base, R, t))
                continue
        infos.append(shape_features(r.mesh, seed=i))
    # Exact B-rep volume / area beat the tessellated ones.
    for inf, r in zip(infos, raws):
        if r.volume_exact:
            inf.volume = float(r.volume_exact)
            inf.watertight = True                      # a closed B-rep solid
            inf.features["watertight"] = 1.0
            inf.features["log_volume"] = float(np.log10(inf.volume + 1e-9))
        if r.area_exact:
            inf.area = float(r.area_exact)
        g = (r.brep or {}).get("genus")
        if g is not None:
            inf.genus = int(g)
            inf.features["genus"] = float(min(g, 30))
            inf.features["log_genus"] = float(np.log1p(g))
    return infos


def definition_groups(raws, infos) -> list[int] | None:
    """Exact instance groups from the CAD definitions (None if the file has none)."""
    if not raws or not all(r.definition for r in raws):
        return None
    ids: dict[str, int] = {}
    return [ids.setdefault(r.definition, len(ids)) for r in raws]


def is_engine(preds, names=None) -> bool:
    """Engine mode only when the core moving parts of an engine are clearly present:
    one crankshaft, and as many connecting rods as pistons (a robot's planetary gears or
    bearing rings can look like pistons on their own, but never in matching sets)."""
    # A part named "piston" / "connecting rod" counts whatever the shape model thinks.
    lab = [p.label for p in preds if p.confidence > 0.9 or getattr(p, "source", "") == "name"]
    n_p, n_r, n_c = lab.count("piston"), lab.count("connecting_rod"), lab.count("crankshaft")
    if names:
        text = " ".join(n.lower() for n in names if n)
        if any(k in text for k in ("piston", "crankshaft", "conrod", "connecting rod", "con rod")):
            return n_c <= 1 and n_p >= 1
    return n_c == 1 and n_p >= 1 and n_r in (n_p, 2 * n_p)      # rods, or rods + separate caps


def custom_kind(info: ShapeInfo) -> str:
    """Plain-language shape class for parts that are made for this design."""
    L1, L2, L3 = info.extents
    f = info.features
    if L3 < 0.08 * L2:
        return "plate / panel"
    round_ = info.sym_score > 0.75 and f.get("cyl_frac", 0) > 0.3   # square prisms also score high on symmetry
    if round_ and f.get("slender", 0) > 3:
        return "shaft / pin"
    if round_:
        return "turned part (bush, spacer, hub)"
    if f.get("wall_ratio", 1) < 0.18 and f.get("convexity", 1) < 0.35:
        return "thin-walled housing / cover"
    if L1 > 5 * L2:
        return "beam / profile"
    if f.get("convexity", 1) < 0.45:
        return "bracket / complex part"
    return "block / machined part"


def bought_occurrence(p):
    """(assembly name, occurrence id) if this part is a piece of a bought sub-assembly."""
    comp = p.component or {}
    name = (comp.get("size") or {}).get("bought_assembly")
    if not name or name not in p.path:
        return None
    k = len(p.path) - 1 - p.path[::-1].index(name)
    occ = p.path_ids[k] if len(p.path_ids) > k else tuple(p.path[:k + 1])
    return name, occ


def bom_key(p) -> tuple:
    """Identical parts share a parts-list line even when the CAD file gives every copy its
    own definition (Fusion 360 exports do): same name, same type, same volume, same size.
    Pieces of a bought sub-assembly (a rail carriage, a fan) form one line for that assembly."""
    comp = p.component or {}
    bo = bought_occurrence(p)
    if bo:
        return ("bought", bo[0].lower(), p.label)
    if comp.get("standard_part") and comp.get("designation") and comp.get("family") in (
            "fastener", "bearing", "linear motion"):
        # Standard parts with a full designation are interchangeable, whoever modelled them.
        return (comp["designation"].lower(), p.label)
    dims = tuple(round(float(x), 1) for x in sorted(p.info.extents))
    return ((display_name(p.name) or "").lower(), p.label, round(float(p.info.volume), 0), dims)


def build_bom(asm) -> list[dict]:
    from . import materials
    lines: "OrderedDict[tuple, dict]" = OrderedDict()
    for p in asm.parts:
        key = bom_key(p)
        comp = p.component or {}
        mass = p.specs.get("mass_kg")
        if key not in lines:
            lines[key] = {
                "item": len(lines) + 1, "name": display_name(p.name) or comp.get("designation") or p.label,
                "qty": 0, "label": p.label, "category": comp.get("category") or p.specs.get("kind") or p.label,
                "designation": comp.get("designation", ""), "standard_part": bool(comp.get("standard_part")),
                "material": materials.get(p.material).name, "unit_mass_kg": mass, "links": comp.get("links", {}),
                "path": " / ".join(p.path) if p.path else "", "part_ids": [],
                "size": comp.get("size", {}), "dims_mm": [round(float(x), 1) for x in p.info.extents],
            }
        ln = lines[key]
        bo = bought_occurrence(p)
        if bo:
            ln.setdefault("_occ", set()).add(bo[1])
            ln["qty"] = len(ln["_occ"])
            ln["name"] = bo[0]
            ln["bought_assembly"] = True
        else:
            ln["qty"] += 1
        ln["part_ids"].append(p.id)
    out = list(lines.values())
    for ln in out:
        ln.pop("_occ", None)
        if ln.get("bought_assembly"):
            ids = ln["part_ids"]
            ln["unit_mass_kg"] = round(sum(asm.parts[i].specs.get("mass_kg") or 0 for i in ids) / max(ln["qty"], 1), 5)
        ln["total_mass_kg"] = round((ln["unit_mass_kg"] or 0) * ln["qty"], 4)
    out.sort(key=lambda l: (not l["standard_part"], l["label"], l["name"] or ""))
    for i, ln in enumerate(out):
        ln["item"] = i + 1
    return out
