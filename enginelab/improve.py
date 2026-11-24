"""Design improvement: what-if simulations, part recommendations, and a continuous
self-improving optimiser.

A *design* is a set of changes to the engine as uploaded, one option per lever:
material, geometry (crown thickness, pin bore, shank section, crankpin diameter),
surface treatment (fillet rolling, nitriding, shot peening, hard anodising), coating
(skirt, ring face) and cooling (piston oil jets / galleries).

`evaluate()` runs the engine physics for a design: it recomputes the moving masses
(a lighter pin lowers the inertia load on the rod and crank), solves the engine cycle at
every duty point and evaluates every part's failure modes. It returns lives, safety
factors, mass and a relative cost.

`recommend()` tries every lever on one part, simulates each, and explains what helps.

`Optimizer` keeps running simulations in the background. A random-forest surrogate is
fitted to every simulation run so far and chooses the next designs (upper confidence
bound: predicted score + uncertainty), so the more it runs the better its predictions
and the better the designs it finds. Every simulation is stored in
data/optimizer/<engine>.json and reused when the same engine is opened again.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import materials
from .recommend import candidates
from .sim.life import DUTY_CYCLES, SERVICE_ITEMS, SimPart, aggregate_modes, solve_operating_points

STORE = os.path.join(os.path.dirname(__file__), "..", "data", "optimizer")
LIFE_CAP = 1e6


@dataclass
class Option:
    name: str
    explain: str = ""
    material: str | None = None
    spec: dict = field(default_factory=dict)        # key -> absolute value
    fatigue: dict = field(default_factory=dict)     # mode substring -> fatigue-strength factor
    wear: float = 1.0                               # wear-rate factor
    temp_delta: float = 0.0                         # deg C cooler
    dvol: float = 0.0                               # volume change per part, mm^3
    cost: float = 0.0                               # relative extra cost per part (process / coating)


@dataclass
class Lever:
    id: str
    label: str
    title: str
    options: list[Option]
    kind: str = "design"                            # material | design


NAMES = {"piston": "Piston", "piston_pin": "Piston pin", "connecting_rod": "Connecting rod",
         "crankshaft": "Crankshaft", "piston_ring": "Piston ring", "valve": "Valve", "valve_spring": "Valve spring",
         "camshaft": "Camshaft", "main_bearing": "Main bearing", "cylinder_liner": "Cylinder liner",
         "engine_block": "Engine block", "cylinder_head": "Cylinder head", "bolt": "Bolt", "timing_gear": "Timing gear",
         "flywheel": "Flywheel", "head_gasket": "Head gasket"}


def _non_caps(parts):
    return [p for p in parts if p.specs.get("rod_cap_of") is None]


def _base(asm, p) -> dict:
    """The part as uploaded (before any applied design): specs, volume, design ctx."""
    b = getattr(asm, "design_base", {}).get(p.id)
    if b:
        return b
    return {"specs": p.specs, "volume": p.info.volume,
            "ctx": {k: p.ctx.get(k) for k in ("design_fatigue", "design_wear", "temp_delta")}}


# ------------------------------------------------------------------ levers
def build_levers(asm) -> list[Lever]:
    e = asm.engine
    B = e.get("bore_mm") or e.get("piston_diameter_mm") or 86.0
    by = {}
    for p in asm.parts:
        by.setdefault(p.label, []).append(p)
    out: list[Lever] = []
    for label, ps in by.items():
        rep = _non_caps(ps)[0] if _non_caps(ps) else ps[0]
        if label not in NAMES:
            continue
        mats = []
        cur = rep.material
        rho0 = materials.get(cur).density
        for k in [cur] + [c for c in candidates(label) if c != cur]:
            m = materials.get(k)
            if k != cur and not 0.5 <= m.density / rho0 <= 2.0:
                continue        # e.g. a steel piston needs a different (thin-walled) design, not a material swap
            mats.append(Option(name=m.name if k != cur else f"{m.name} (current)", material=k,
                               explain=m.notes or m.category))
        out.append(Lever(f"{label}.material", label, "Material", mats, kind="material"))
        sp = _base(asm, rep)["specs"]
        if label == "piston":
            t = float(sp.get("crown_thickness_mm") or 0.1 * B)
            area = math.pi / 4 * (0.85 * B) ** 2
            out.append(Lever("piston.crown", label, "Crown thickness", [Option(f"{t:.1f} mm (as designed)")] + [
                Option(f"{t + d:.1f} mm ({d:+.0f} mm)", spec={"crown_thickness_mm": t + d}, dvol=area * d,
                       explain=("Crown bending stress falls with thickness squared (Roark plate); costs "
                                "reciprocating mass." if d > 0 else "Saves reciprocating mass if the crown has margin."))
                for d in (1.0, 2.0, -1.0) if t + d >= 0.06 * B]))
            out.append(Lever("piston.cooling", label, "Crown cooling", [
                Option("none (as designed)"),
                Option("oil-jet cooling", temp_delta=30.0, cost=1.0,
                       explain="An oil jet under the crown typically cools it 20-40 C, which slows thermal "
                               "fatigue and keeps the alloy's strength."),
                Option("cooling gallery", temp_delta=45.0, cost=3.0, dvol=0.03 * _base(asm, rep)["volume"],
                       explain="A cast-in oil gallery behind the ring belt cools the crown 40-60 C "
                               "(diesel / high-output practice); slightly heavier.")]))
            out.append(Lever("piston.skirt", label, "Skirt coating", [
                Option("none (as designed)"),
                Option("graphite skirt coating", wear=0.5, cost=0.4,
                       explain="Graphite-filled resin coatings roughly halve skirt scuffing/wear."),
                Option("MoS2 polymer skirt coating", wear=0.35, cost=0.7,
                       explain="MoS2/PTFE polymer coatings cut skirt wear and friction further.")]))
            out.append(Lever("piston.anodise", label, "Crown surface", [
                Option("as machined (as designed)"),
                Option("hard-anodised crown and top groove", fatigue={"thermal": 1.25}, cost=1.0,
                       explain="A hard-anodised layer resists crown thermal-fatigue cracking and "
                               "top-groove wear.")]))
        elif label == "piston_pin":
            do = float(sp.get("outer_diameter_mm") or 0.25 * B)
            di0 = float(sp.get("inner_diameter_mm") or 0.0)
            L = float(sp.get("pin_length_mm") or 0.8 * B)
            opts = [Option(f"{'solid' if di0 == 0 else f'{di0:.1f} mm bore'} (as designed)")]
            for f in (0.35, 0.45, 0.55, 0.62):
                di = round(f * do, 1)
                if di > di0 + 0.5:
                    opts.append(Option(f"{di:.1f} mm bore", spec={"inner_diameter_mm": di},
                                       dvol=-math.pi / 4 * (di ** 2 - di0 ** 2) * L,
                                       explain="A bored pin keeps most of its bending stiffness (it goes with "
                                               "do^4 - di^4) but sheds reciprocating mass, which unloads the rod, "
                                               "crankpin and bearings at speed."))
            if len(opts) > 1:
                out.append(Lever("piston_pin.bore", label, "Pin bore", opts))
        elif label == "connecting_rod":
            A = float(sp.get("min_shank_area_mm2") or 0.045 * B * B)
            Ls = 0.55 * float(e.get("rod_length_mm") or 1.65 * (e.get("stroke_mm") or B))
            out.append(Lever("connecting_rod.shank", label, "Shank section", [Option(f"{A:.0f} mm² (as designed)")] + [
                Option(f"{A * f:.0f} mm² ({(f - 1) * 100:+.0f}%)", spec={"min_shank_area_mm2": A * f},
                       dvol=A * (f - 1) * Ls,
                       explain=("A larger I-beam section lowers push-pull stress; adds rod mass." if f > 1 else
                                "A slimmer shank saves rod mass (and inertia load) where there is margin."))
                for f in (1.1, 1.2, 0.9, 0.85)]))
            out.append(Lever("connecting_rod.peen", label, "Big-end radii", [
                Option("as machined (as designed)"),
                Option("shot-peened big-end radii", fatigue={"eye": 1.2}, cost=0.5,
                       explain="Shot peening puts compressive stress in the eye radii: about +20% fatigue "
                               "strength.")]))
        elif label == "crankshaft":
            out.append(Lever("crankshaft.fillet", label, "Fillet treatment", [
                Option("plain ground fillets (as designed)"),
                Option("deep-rolled fillets", fatigue={"fillet": 1.5}, cost=1.0,
                       explain="Deep rolling the crankpin fillets adds 30-80% fatigue strength (compressive "
                               "residual stress); the standard fix for fillet cracks."),
                Option("gas nitrided", fatigue={"*": 1.35}, cost=1.5,
                       explain="Nitriding hardens the journals and adds 30-50% fatigue strength."),
                Option("deep-rolled + nitrided", fatigue={"fillet": 1.75}, cost=2.5,
                       explain="Both treatments: the high-output / motorsport choice.")]))
            d = float(sp.get("crankpin_diameter_mm") or 0.54 * B)
            n = max(1, int(e.get("cylinders", 1) or 1))
            out.append(Lever("crankshaft.pin_d", label, "Crankpin diameter", [Option(f"{d:.1f} mm (as designed)")] + [
                Option(f"{d + k:.1f} mm (+{k:.0f} mm)", spec={"crankpin_diameter_mm": d + k},
                       dvol=n * math.pi / 4 * ((d + k) ** 2 - d ** 2) * 0.45 * B,
                       explain="Bending stress at the fillet falls with diameter cubed; the rod big end "
                               "must grow to match.") for k in (2.0, 4.0)]))
        elif label == "main_bearing":
            w = float(sp.get("bearing_width_mm") or 0.3 * B)
            out.append(Lever("main_bearing.width", label, "Bearing width", [Option(f"{w:.1f} mm (as designed)")] + [
                Option(f"{w * f:.1f} mm (+{(f - 1) * 100:.0f}%)", spec={"bearing_width_mm": w * f},
                       explain="A wider shell spreads the load: lower specific pressure, thicker oil film, "
                               "less wear; the crank gets longer.") for f in (1.1, 1.2)]))
            out.append(Lever("main_bearing.overlay", label, "Shell overlay", [
                Option("plain lining (as designed)"),
                Option("polymer overlay coating", wear=0.5, cost=0.5,
                       explain="A polymer (PAI + solid lubricant) overlay roughly halves start-stop wear "
                               "(stop-start engines use it)."),
                Option("sputtered overlay", wear=0.35, cost=1.0,
                       explain="Sputtered AlSn overlay: the most wear-resistant shell layer (high-load diesels).")]))
        elif label == "cylinder_head":
            out.append(Lever("cylinder_head.bridge", label, "Valve-bridge cooling", [
                Option("as cast (as designed)"),
                Option("drilled valve-bridge cooling passages", temp_delta=25.0, cost=1.0,
                       explain="Coolant drillings through the exhaust valve bridge cool it 20-30 C: the "
                               "standard fix for bridge cracking.")]))
        elif label == "engine_block":
            out.append(Lever("engine_block.girdle", label, "Main bulkheads", [
                Option("separate main caps (as designed)"),
                Option("ladder-frame / main-cap girdle", fatigue={"bulkhead": 1.3}, cost=2.0, dvol=0.04 * _base(asm, rep)["volume"],
                       explain="Tying the main caps together in a ladder frame stiffens the bulkheads "
                               "and cuts their cyclic stress.")]))
        elif label == "cylinder_liner":
            out.append(Lever("cylinder_liner.coating", label, "Bore surface", [
                Option("as designed"),
                Option("plateau honing", wear=0.7, cost=0.3, explain="Plateau honing removes the peaks that "
                       "wear in, lowering bore and ring wear."),
                Option("plasma-sprayed iron coating", wear=0.4, cost=1.5, explain="Thermally sprayed bores "
                       "(e.g. PTWA) cut bore wear and friction.")]))
        elif label == "piston_ring":
            out.append(Lever("piston_ring.face", label, "Ring face", [
                Option("as designed (uncoated)"),
                Option("hard chrome plated", wear=0.5, cost=0.4, explain="Chrome-plated faces roughly halve "
                       "face wear."),
                Option("plasma-sprayed molybdenum", wear=0.4, cost=0.8, explain="Moly faces resist scuffing "
                       "and wear (top rings of high-output engines)."),
                Option("PVD CrN", wear=0.25, cost=1.5, explain="PVD chromium-nitride: about 4x less wear "
                       "than uncoated steel.")]))
    return out


# ------------------------------------------------------------------ evaluation
class Context:
    """Everything that stays fixed while designs change (built once per assembly)."""

    def __init__(self, asm, duty: str = "automotive"):
        self.asm = asm
        self.duty_key = duty if duty in DUTY_CYCLES else "automotive"
        self.duty = DUTY_CYCLES[self.duty_key]
        self.eng0 = asm.engine_model()
        if self.eng0 is None:
            raise ValueError("Design improvement needs an engine (pistons, rods and a crankshaft were not found). "
                             "Use the material recommendations and FEA for general machines.")
        asm._refresh_ctx(self.eng0)
        self.levers = build_levers(asm)
        self.by_id = {lv.id: lv for lv in self.levers}
        self.by_label: dict[str, list] = {}
        for p in asm.parts:
            self.by_label.setdefault(p.label, []).append(p)
        self.n_cyl = max(1, int(asm.engine.get("cylinders", 1) or 1))
        self.sig = hashlib.sha1(json.dumps(
            [sorted((k, v) for k, v in asm.engine.items() if isinstance(v, (int, float))),
             sorted((p.label, round(p.info.volume, 0), p.material) for p in asm.parts),
             [(lv.id, [o.name for o in lv.options]) for lv in self.levers]], default=str).encode()).hexdigest()[:16]

    def option(self, lever_id: str, value):
        lv = self.by_id[lever_id]
        if isinstance(value, str):          # any material key (material menu)
            m = materials.get(value)
            return Option(name=m.name, material=value)
        return lv.options[int(value)]

    def resolve(self, design: dict) -> dict[str, list[Option]]:
        per_label: dict[str, list[Option]] = {}
        for lid, v in (design or {}).items():
            if lid not in self.by_id or v in (0, None, "0"):
                continue
            per_label.setdefault(self.by_id[lid].label, []).append(self.option(lid, v))
        return per_label

    def material_of(self, p, opts) -> str:
        for o in opts:
            if o.material:
                return o.material
        return p.material

    def part_mass(self, p, opts) -> float:
        v0 = _base(self.asm, p)["volume"]
        vol = v0 + (0.0 if p.specs.get("rod_cap_of") is not None else sum(o.dvol for o in opts))
        return max(vol, 0.05 * v0) * 1e-9 * materials.get(self.material_of(p, opts)).density


def _evaluate_part(ctx: "Context", asm, label: str, rep, opts: list, eng, ops) -> dict | None:
    """Life, safety factors and weakest mode of one representative part under a design."""
    geom = asm.geom_for(rep)
    b = _base(asm, rep)
    spec = dict(b["specs"])
    pctx = dict(rep.ctx)
    ff: dict = dict(b["ctx"].get("design_fatigue") or {})
    wear = float(b["ctx"].get("design_wear") or 1.0)
    tdel = float(b["ctx"].get("temp_delta") or 0.0)
    for o in opts:
        spec.update(o.spec)
        for k, f in o.fatigue.items():
            ff[k] = ff.get(k, 1.0) * f
        wear *= o.wear
        tdel += o.temp_delta
    geom = dataclasses.replace(geom, specs=spec)
    pctx.update(design_fatigue=ff or None, design_wear=wear if wear != 1.0 else None, temp_delta=tdel)
    sp = SimPart(id=rep.id, label=label, material=ctx.material_of(rep, opts), geom=geom, ctx=pctx,
                 group=rep.group, name=rep.name or label)
    modes = aggregate_modes(sp, eng, ops, ctx.duty["starts_per_hour"])
    if not modes:
        return None
    life, sf_f, sf_s, dom, loc = LIFE_CAP, math.inf, math.inf, "", ""
    for m in modes:
        if m["kind"] == "fatigue" and m["rate"] > 0:
            h = 1.0 / m["rate"]
        elif m["kind"] in ("wear", "erosion") and m["rate"] > 0:
            h = m["limit"] / m["rate"]
        else:
            h = LIFE_CAP
        if h < life:
            life, dom, loc = h, m["name"], m["location"]
        if m["kind"] == "fatigue":
            sf_f = min(sf_f, m["safety_fatigue"])
        sf_s = min(sf_s, m["safety_static"], m["margin"])
    if not dom:
        dom = min(modes, key=lambda m: m["safety_fatigue"])["name"]
    sub = rep.ctx.get("subtype")
    return {"name": NAMES.get(label, label) + (f" ({sub})" if sub else ""), "life_h": min(life, LIFE_CAP),
            "sf_fatigue": None if not math.isfinite(sf_f) else round(float(sf_f), 3),
            "sf_static": None if not math.isfinite(sf_s) else round(float(sf_s), 3),
            "mode": dom, "location": loc}


def evaluate(ctx: Context, design: dict) -> dict:
    asm = ctx.asm
    per = ctx.resolve(design)
    masses = {}
    for label, ps in ctx.by_label.items():
        masses[label] = [ctx.part_mass(p, per.get(label, [])) for p in ps]
    n = ctx.n_cyl
    mean = lambda lab: float(np.mean(masses[lab])) if masses.get(lab) else None
    piston = mean("piston") or ctx.eng0.m_piston_kg
    pin = mean("piston_pin") or 0.0
    rings = sum(masses.get("piston_ring", [])) / n
    rods = ctx.by_label.get("connecting_rod", [])
    if rods:
        n_rod = max(1, len(_non_caps(rods)))
        m_rod = sum(masses["connecting_rod"]) / n_rod
    else:
        m_rod = ctx.eng0.m_rod_kg
    eng = dataclasses.replace(ctx.eng0, m_piston_kg=piston + pin + rings, m_rod_kg=m_rod)
    ops = solve_operating_points(eng, ctx.duty)
    parts_out = {}
    for label, ps in ctx.by_label.items():
        cand = _non_caps(ps) or ps
        # one representative per subtype (intake and exhaust valves run very differently)
        reps = list({p.ctx.get("subtype", ""): p for p in cand}.values())
        opts = per.get(label, [])
        worst = None
        for rep in reps:
            r = _evaluate_part(ctx, asm, label, rep, opts, eng, ops)
            if r and (worst is None or r["life_h"] < worst["life_h"] or
                      (r["life_h"] == worst["life_h"] and (r["sf_fatigue"] or 9) < (worst["sf_fatigue"] or 9))):
                worst = r
        if worst:
            worst["mass_kg"] = round(sum(masses[label]), 4)
            worst["count"] = len(ps)
            parts_out[label] = worst
    # Service items (spark plugs, gaskets...) are replaced on a schedule, as in the life simulation,
    # so they don't set the engine's life.
    structural = [k for k in parts_out if k not in SERVICE_ITEMS] or list(parts_out)
    sys_label = min(structural, key=lambda k: parts_out[k]["life_h"]) if parts_out else None
    sfs = [v for p in parts_out.values() for v in (p["sf_fatigue"], p["sf_static"]) if v is not None]
    cost = 0.0
    for label, ps in ctx.by_label.items():
        opts = per.get(label, [])
        for p, m in zip(ps, masses[label]):
            cost += m * materials.get(ctx.material_of(p, opts)).cost
        cost += 0.25 * len(ps) * sum(o.cost for o in opts)
    return {"parts": parts_out,
            "system_life_h": parts_out[sys_label]["life_h"] if sys_label else LIFE_CAP,
            "limiting_part": sys_label, "min_sf": min(sfs) if sfs else None,
            "total_mass_kg": round(sum(sum(v) for v in masses.values()), 3),
            "reciprocating_mass_kg": round(eng.m_piston_kg, 4), "rod_mass_kg": round(eng.m_rod_kg, 4),
            "cost_index": round(cost, 3)}


OBJECTIVES = {
    "balanced": "Longest life with a sensible safety margin, without adding much mass or cost",
    "life": "Longest life, whatever the mass or cost",
    "light": "Lightest engine that still meets the target life with safety factors >= 1.5",
}


def score(res: dict, base: dict, objective: str = "balanced", target: float = 10000.0) -> float:
    L = min(res["system_life_h"], LIFE_CAP)
    sf = res["min_sf"] if res["min_sf"] is not None else 5.0
    # Safety factors below 1.3 are close to forbidden (x20), and every change costs a little, so the
    # optimiser keeps only changes that clearly pay off (fewer changes = simpler to make and check).
    pen = sum(max(0.0, 1.3 - v) * 20.0 for p in res["parts"].values()
              for v in (p["sf_fatigue"], p["sf_static"]) if v is not None)
    pen += 0.004 * res.get("n_changes", 0)
    dm = (res["total_mass_kg"] - base["total_mass_kg"]) / max(base["total_mass_kg"], 1e-9)
    dc = (res["cost_index"] - base["cost_index"]) / max(base["cost_index"], 1e-9)
    if objective == "life":
        return math.log10(L) + 0.2 * min(sf, 3.0) - pen
    if objective == "light":
        short = max(0.0, math.log10(target) - math.log10(max(L, 1.0)))
        return -dm * 10 - 4 * short - 20 * max(0.0, 1.5 - sf) - 0.5 * dc - 0.004 * res.get("n_changes", 0)
    return math.log10(min(L, 3 * target)) + 0.3 * min(sf, 3.0) - 1.5 * dm - 0.6 * dc - pen


def describe(ctx: Context, design: dict) -> list[dict]:
    out = []
    for lid, v in (design or {}).items():
        if lid not in ctx.by_id or v in (0, None, "0"):
            continue
        lv = ctx.by_id[lid]
        o = ctx.option(lid, v)
        out.append({"lever": lid, "part": NAMES.get(lv.label, lv.label), "label": lv.label, "what": lv.title,
                    "change": o.name, "why": o.explain})
    return out


# ------------------------------------------------------------------ recommendations
def recommend(asm, part_id: int, duty: str = "automotive", target: float = 10000.0) -> dict:
    ctx = Context(asm, duty)
    p = asm.parts[part_id]
    label = p.label
    current = dict(getattr(asm, "design", {}) or {})
    base = evaluate(ctx, current)
    if label not in base["parts"]:
        return {"part": part_id, "label": label, "supported": False,
                "message": "This part type has no engine load model, so improvements can't be simulated. "
                           "Use the material menu and the FEA test instead.", "items": []}
    me = base["parts"][label]
    items = []
    for lv in [lv for lv in ctx.levers if lv.label == label]:
        for i, o in enumerate(lv.options):
            if i == 0 or current.get(lv.id) == i:
                continue
            d = {**current, lv.id: i}
            r = evaluate(ctx, d)
            q = r["parts"][label]
            life_ratio = q["life_h"] / max(me["life_h"], 1e-9)
            dsf = (q["sf_fatigue"] or q["sf_static"] or 0) - (me["sf_fatigue"] or me["sf_static"] or 0)
            dmass = r["total_mass_kg"] - base["total_mass_kg"]
            sys_ratio = r["system_life_h"] / max(base["system_life_h"], 1e-9)
            others, warnings, harm, gain = [], [], 0.0, 0.0
            for lab, qq in r["parts"].items():
                b = base["parts"].get(lab)
                if lab == label or not b:
                    continue
                if b["sf_fatigue"] and qq["sf_fatigue"] and abs(qq["sf_fatigue"] - b["sf_fatigue"]) >= 0.05:
                    txt = f"{qq['name']} safety {b['sf_fatigue']:.2f} -> {qq['sf_fatigue']:.2f}"
                    if qq["sf_fatigue"] < b["sf_fatigue"]:
                        # a change that weakens another part is a trade-off, not an improvement
                        drop = min(b["sf_fatigue"], 3.0) - min(qq["sf_fatigue"], 3.0)
                        harm += drop * (3.0 if qq["sf_fatigue"] < 1.5 else 0.7)
                        if qq["sf_fatigue"] < 1.5:
                            warnings.append(txt)
                            continue
                    else:
                        gain += 0.3 * (min(qq["sf_fatigue"], 3.0) - min(b["sf_fatigue"], 3.0))
                    others.append(txt)
            if life_ratio < 0.95:
                warnings.append(f"{me['name']} life x{life_ratio:.2f} ({q['mode']})")
                harm += -math.log10(max(life_ratio, 1e-6))
            if sys_ratio < 0.97 and life_ratio >= 0.95:
                warnings.append(f"engine life {base['system_life_h']:,.0f} -> {r['system_life_h']:,.0f} h")
                harm += -math.log10(max(sys_ratio, 1e-6))
            sf_b = me["sf_fatigue"] or me["sf_static"] or 9.9
            q_sf = q["sf_fatigue"] or q["sf_static"] or 9.9
            lighter = dmass < -0.002 * base["total_mass_kg"] and q_sf >= 1.5 and sys_ratio > 0.98 and not warnings
            # more safety only counts while the part is short of it
            better = life_ratio > 1.05 or (dsf > 0.03 and sf_b < 3.0) or sys_ratio > 1.02
            if not (better or lighter):
                continue
            benefit = (math.log10(max(life_ratio, 1e-6)) + 0.5 * min(dsf, 1.5) * (1.0 if sf_b < 3.0 else 0.2)
                       + math.log10(max(sys_ratio, 1e-6)) - 4.0 * max(0.0, dmass) / max(base["total_mass_kg"], 1e-9)
                       + 2.0 * max(0.0, -dmass) / max(base["total_mass_kg"], 1e-9) + (0.15 if lighter else 0.0)
                       + gain - harm
                       - 0.01 * max(0.0, 100 * (r["cost_index"] - base["cost_index"]) / max(base["cost_index"], 1e-9)))
            items.append({"lever": lv.id, "option": i, "what": lv.title, "change": o.name, "why": o.explain,
                          "kind": "trade-off" if warnings else ("lighter" if lighter else "stronger"),
                          "warnings": warnings,
                          "part_life_h": q["life_h"], "part_life_ratio": round(life_ratio, 3),
                          "part_sf_before": me["sf_fatigue"] or me["sf_static"], "part_sf_after": q["sf_fatigue"] or q["sf_static"],
                          "part_mode_after": q["mode"],
                          "system_life_before": base["system_life_h"], "system_life_after": r["system_life_h"],
                          "mass_change_kg": round(dmass, 4), "cost_change_pct": round(
                              100 * (r["cost_index"] - base["cost_index"]) / max(base["cost_index"], 1e-9), 1),
                          "also": others[:3], "benefit": benefit})
    items.sort(key=lambda x: (x["kind"] == "trade-off", -x["benefit"]))
    limiting = base["limiting_part"] == label
    status = (f"{me['name']} is the part that limits the engine's life: {me['mode']} at {me['location']}."
              if limiting else
              f"{me['name']}: weakest mode is {me['mode']}; the engine's life is limited by the "
              f"{base['parts'][base['limiting_part']]['name'].lower()}.")
    sf = me["sf_fatigue"] or me["sf_static"]
    if sf and sf > 4:
        status += f" It is far stronger than needed (safety factor {sf:.1f}): there is room to save mass."
    return {"part": part_id, "label": label, "supported": True, "status": status, "limiting": limiting,
            "current": me, "system_life_h": base["system_life_h"], "items": items[:8],
            "applied": describe(ctx, current)}


def whatif(asm, design: dict, duty: str = "automotive") -> dict:
    ctx = Context(asm, duty)
    current = dict(getattr(asm, "design", {}) or {})
    return {"before": evaluate(ctx, current), "after": evaluate(ctx, {**current, **design}),
            "changes": describe(ctx, {**current, **design})}


# ------------------------------------------------------------------ applying a design
def apply_design(asm, design: dict) -> None:
    """Make a design the assembly's current state (materials, specs, treatments, masses).
    Geometry and treatments are always applied relative to the part as uploaded."""
    ctx = Context(asm)
    if not hasattr(asm, "design_base"):
        asm.design_base = {p.id: {"specs": copy.deepcopy(p.specs), "volume": p.info.volume,
                                  "ctx": {k: p.ctx.get(k) for k in ("design_fatigue", "design_wear", "temp_delta")}}
                           for p in asm.parts}
    merged = {**(getattr(asm, "design", {}) or {}), **(design or {})}
    keep = {}
    for p in asm.parts:
        b = asm.design_base[p.id]
        p.specs = copy.deepcopy(b["specs"])
        p.info.volume = b["volume"]
        for k, v in b["ctx"].items():
            p.ctx[k] = v
    per = ctx.resolve(merged)
    for lid, v in merged.items():
        if lid in ctx.by_id and ctx.by_id[lid].kind != "material" and v not in (0, None):
            keep[lid] = v
    for label, opts in per.items():
        for p in ctx.by_label.get(label, []):
            cap = p.specs.get("rod_cap_of") is not None
            ff, wear, tdel = {}, 1.0, 0.0
            for o in opts:
                if o.material:
                    p.material = o.material
                    p.ctx["material_set_by_user"] = True
                if not cap:
                    p.specs.update(o.spec)
                    p.info.volume = max(p.info.volume + o.dvol, 0.05 * p.info.volume)
                for k, f in o.fatigue.items():
                    ff[k] = ff.get(k, 1.0) * f
                wear *= o.wear
                tdel += o.temp_delta
            p.ctx.update(design_fatigue=ff or None, design_wear=wear if wear != 1.0 else None, temp_delta=tdel)
    for p in asm.parts:
        p.specs["volume_cm3"] = round(p.info.volume / 1000, 3)
        p.specs["mass_kg"] = round(p.info.volume * 1e-9 * materials.get(p.material).density, 4)
    asm.design = keep
    asm.design_changes = describe(Context(asm), keep)
    asm._recompute_engine()


# ------------------------------------------------------------------ continuous optimiser
def _encode(ctx: Context, design: dict) -> np.ndarray:
    v = []
    for lv in ctx.levers:
        oh = np.zeros(len(lv.options))
        oh[int(design.get(lv.id, 0))] = 1.0
        v.extend(oh)
    return np.array(v)


class Optimizer:
    """Runs simulations continuously and learns which designs work (see module docstring)."""

    def __init__(self, asm, objective: str = "balanced", target: float = 10000.0, duty: str = "automotive",
                 budget: int = 2000, batch: int = 6):
        self.ctx = Context(asm, duty)
        self.objective = objective if objective in OBJECTIVES else "balanced"
        self.target, self.budget, self.batch = float(target), int(budget), int(batch)
        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None
        self.records: list[dict] = []       # {"design": {...}, "result": {...}}
        self.history: list[dict] = []
        self.errors: list[float] = []
        self.message = ""
        self.started = time.time()
        self.evals_this_run = 0
        self.loaded = 0
        self.base = evaluate(self.ctx, {})
        self.rng = np.random.default_rng(int(time.time()))
        self.path = os.path.join(STORE, f"{self.ctx.sig}.json")
        self._load()

    # -- persistence
    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            ids = {lv.id: len(lv.options) for lv in self.ctx.levers}
            for r in data.get("records", []):
                d = {k: int(v) for k, v in r["design"].items() if k in ids and int(v) < ids[k]}
                self.records.append({"design": d, "result": r["result"]})
            self.loaded = len(self.records)
        except (OSError, ValueError, KeyError):
            pass

    def _save(self):
        try:
            os.makedirs(STORE, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"engine": self.ctx.asm.engine, "records": self.records[-20000:]}, fh, default=float)
            os.replace(tmp, self.path)
        except OSError:
            pass

    # -- search
    def _key(self, d: dict) -> str:
        return json.dumps(sorted((k, int(v)) for k, v in d.items() if int(v)))

    def _score(self, res: dict) -> float:
        return score(res, self.base, self.objective, self.target)

    def _random_design(self) -> dict:
        d = {}
        for lv in self.ctx.levers:
            if self.rng.random() < 0.35:
                d[lv.id] = int(self.rng.integers(0, len(lv.options)))
        return d

    def _mutate(self, d: dict) -> dict:
        d = dict(d)
        for _ in range(int(self.rng.integers(1, 4))):
            lv = self.ctx.levers[int(self.rng.integers(0, len(self.ctx.levers)))]
            d[lv.id] = int(self.rng.integers(0, len(lv.options)))
        return d

    def _evaluate(self, d: dict) -> dict:
        res = evaluate(self.ctx, d)
        compact = {"system_life_h": res["system_life_h"], "min_sf": res["min_sf"],
                   "n_changes": sum(1 for v in d.values() if int(v)),
                   "total_mass_kg": res["total_mass_kg"], "cost_index": res["cost_index"],
                   "limiting_part": res["limiting_part"],
                   "parts": {k: {kk: v[kk] for kk in ("life_h", "sf_fatigue", "sf_static", "mode", "name", "mass_kg")}
                             for k, v in res["parts"].items()}}
        rec = {"design": {k: int(v) for k, v in d.items() if int(v)}, "result": compact}
        with self.lock:
            self.records.append(rec)
        return rec

    def _seed(self):
        seen = {self._key(r["design"]) for r in self.records}
        todo = [{}] + [{lv.id: i} for lv in self.ctx.levers for i in range(1, len(lv.options))]
        for d in todo:
            if not self.running:
                return
            if self._key(d) not in seen:
                self._evaluate(d)
                self.evals_this_run += 1
                seen.add(self._key(d))

    def _polish(self, passes: int = 2):
        """Coordinate descent from the best design: try every option of every lever, keep what
        helps. Benchmarked (tools/opt_bench.py): adding this to the surrogate search cut the gap to
        the best design by ~5x and the run-to-run spread by ~8x at the same number of simulations."""
        with self.lock:
            scores = {self._key(r["design"]): self._score(r["result"]) for r in self.records}
        best = max(self.records, key=lambda r: self._score(r["result"]))
        d, cur = dict(best["design"]), self._score(best["result"])
        for _ in range(passes):
            improved = False
            for j in self.rng.permutation(len(self.ctx.levers)):
                lv = self.ctx.levers[int(j)]
                for i in range(len(lv.options)):
                    if not self.running or self.evals_this_run >= self.budget:
                        return
                    if i == int(d.get(lv.id, 0)):
                        continue
                    t = {**d, lv.id: i}
                    k = self._key(t)
                    if k not in scores:
                        scores[k] = self._score(self._evaluate(t)["result"])
                        self.evals_this_run += 1
                    if scores[k] > cur + 1e-12:
                        d, cur, improved = t, scores[k], True
            if not improved:
                return

    def _loop(self):
        from sklearn.ensemble import ExtraTreesRegressor
        try:
            self.message = "Trying every single change once (the starting knowledge)"
            self._seed()
            kappa = 1.0
            it = 0
            polished = None
            while self.running and self.evals_this_run < self.budget:
                it += 1
                best_key = self._key(max(self.records, key=lambda r: self._score(r["result"]))["design"])
                final = self.evals_this_run >= 0.85 * self.budget
                if best_key != polished and (final or it % 6 == 0):
                    # polish only a new best: try every single change around it, keep what helps
                    self.message = ("Final polish around the best design" if final else
                                    "Polishing the best design lever by lever")
                    self._polish(passes=6 if final else 2)
                    polished = self._key(max(self.records, key=lambda r: self._score(r["result"]))["design"])
                    continue
                with self.lock:
                    recs = list(self.records)
                X = np.array([_encode(self.ctx, r["design"]) for r in recs])
                y = np.array([self._score(r["result"]) for r in recs])
                # Extra-trees surrogate: benchmarked best of RF / ET / GA / random search.
                model = ExtraTreesRegressor(n_estimators=120, min_samples_leaf=1, max_features=0.5,
                                            n_jobs=1, random_state=len(recs)).fit(X, y)
                order = np.argsort(-y)
                elite = [recs[i]["design"] for i in order[:8]]
                seen = {self._key(r["design"]) for r in recs}
                cands, keys = [], set()
                for _ in range(400):
                    d = self._mutate(elite[int(self.rng.integers(0, len(elite)))]) if self.rng.random() < 0.75 \
                        else self._random_design()
                    k = self._key(d)
                    if k not in seen and k not in keys:
                        keys.add(k)
                        cands.append(d)
                if not cands:
                    self.message = "Every reachable design near the best has been simulated"
                    break
                C = np.array([_encode(self.ctx, d) for d in cands])
                per_tree = np.stack([t.predict(C) for t in model.estimators_])
                mu, sd = per_tree.mean(0), per_tree.std(0)
                pick = np.argsort(-(mu + kappa * sd))[:self.batch]
                errs = []
                for i in pick:
                    if not self.running:
                        break
                    rec = self._evaluate(cands[i])
                    self.evals_this_run += 1
                    errs.append(abs(self._score(rec["result"]) - mu[i]))
                if errs:
                    self.errors.append(float(np.mean(errs)))
                kappa = max(0.15, kappa * 0.97)
                best = max(self.records, key=lambda r: self._score(r["result"]))
                self.history.append({"evaluations": len(self.records), "best_score": round(self._score(best["result"]), 4),
                                     "best_life_h": best["result"]["system_life_h"],
                                     "surrogate_error": round(float(np.mean(self.errors[-5:])), 4) if self.errors else None})
                self.message = (f"Learning: the surrogate model predicts each new design's score within "
                                f"{np.mean(self.errors[-5:]):.3f} on average" if self.errors else "Learning")
                if len(self.history) % 5 == 0:
                    self._save()
            if self.running:
                self.message = self.message or "Budget reached"
        except Exception as exc:     # never let the thread die silently
            self.message = f"Stopped: {exc.__class__.__name__}: {exc}"
        finally:
            self.running = False
            self._save()

    def start(self):
        if self.running:
            return
        self.running = True
        self.evals_this_run = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def status(self, top: int = 5) -> dict:
        with self.lock:
            recs = list(self.records)
        ranked = sorted(recs, key=lambda r: -self._score(r["result"]))
        base_rec = next((r for r in recs if not r["design"]), {"result": self.base})
        out_top, seen = [], set()
        for r in ranked:
            res = r["result"]
            sig = (round(res["system_life_h"], -1), round(res["total_mass_kg"], 2), round(res["min_sf"] or 0, 2))
            if sig in seen:            # same outcome, a lever that changes nothing: show one
                continue
            seen.add(sig)
            if len(out_top) >= top:
                break
            out_top.append({"design": r["design"], "score": round(self._score(r["result"]), 4),
                            "result": r["result"], "changes": describe(self.ctx, r["design"])})
        return {"running": self.running, "objective": self.objective, "objective_text": OBJECTIVES[self.objective],
                "target_hours": self.target, "duty": self.ctx.duty_key, "levers": len(self.ctx.levers),
                "design_space": int(np.prod([len(lv.options) for lv in self.ctx.levers], dtype=float)),
                "evaluations_total": len(recs), "evaluations_this_run": self.evals_this_run,
                "remembered_from_before": self.loaded, "message": self.message,
                "baseline": {"score": round(self._score(base_rec["result"]), 4), "result": base_rec["result"]},
                "top": out_top, "history": self.history[-400:],
                "elapsed_s": round(time.time() - self.started, 1)}
