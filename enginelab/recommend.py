"""Material recommendations, ranked by physics rather than a lookup table.

Every candidate material is run through the part's own load model over the
chosen duty cycle. Candidates are ranked on predicted life, mass (it matters
most for parts that move), cost and safety margin, and each one gets the
reasons behind its rank.
"""
from __future__ import annotations

import math

from .materials import MATERIALS, Material, get

CANDIDATES = {
    "piston": ["al_4032", "al_2618", "al_a356", "al_6061", "steel_4140"],
    "piston_pin": ["steel_8620c", "steel_52100", "steel_4140", "steel_4340", "ti_6al4v"],
    "connecting_rod": ["steel_38mnvs6", "steel_pf", "steel_4140", "steel_4340", "ti_6al4v", "al_7075"],
    "crankshaft": ["ci_ductile", "steel_38mnvs6", "steel_4140", "steel_4340", "steel_31crmov9n"],
    "camshaft": ["ci_chilled", "ci_ductile", "steel_8620c", "steel_1045", "steel_31crmov9n"],
    "valve": ["valve_21_4n", "inconel_751", "ti_6al4v", "ss_304", "steel_4140"],
    "valve_spring": ["steel_crsi", "ss_174ph", "steel_4140", "ti_6al4v"],
    "main_bearing": ["brg_alsn", "brg_cupb", "bronze_c954", "al_6061"],
    "piston_ring": ["steel_crsi", "ci_gray", "ci_ductile", "ss_174ph"],
    "cylinder_liner": ["ci_gray", "ci_cgi", "ci_ductile", "steel_1045"],
    "engine_block": ["ci_gray", "ci_cgi", "al_a319", "al_a356", "mg_az91"],
    "cylinder_head": ["al_a356", "al_a319", "ci_gray", "ci_cgi"],
    "head_gasket": ["ss_301fh", "ss_304"],          # MLS layers need spring stainless
    "bolt": ["steel_bolt109", "steel_bolt88", "steel_4140", "ss_174ph", "ti_6al4v"],
    "flywheel": ["ci_gray", "ci_ductile", "steel_1045", "steel_4140", "al_6061"],
    "timing_gear": ["steel_8620c", "steel_4140", "ci_ductile", "steel_1045", "pa66gf30"],
    "spark_plug": ["steel_1018"],
    "oil_pan": ["steel_1018", "al_a380", "pa66gf30", "mg_az91"],
    "valve_cover": ["al_a380", "pa66gf30", "mg_az91", "steel_1018"],
}
GENERIC = ["steel_1045", "steel_4140", "al_6061", "ss_304", "steel_1018", "pa66gf30"]

DEFAULT = {
    "piston": "al_4032", "piston_pin": "steel_8620c", "connecting_rod": "steel_38mnvs6",
    "crankshaft": "steel_38mnvs6", "camshaft": "ci_chilled", "valve": "valve_21_4n",
    "valve_spring": "steel_crsi", "main_bearing": "brg_alsn", "piston_ring": "steel_crsi",
    "cylinder_liner": "ci_gray", "engine_block": "al_a319", "cylinder_head": "al_a356",
    "head_gasket": "ss_301fh", "bolt": "steel_bolt109", "flywheel": "ci_gray",
    "timing_gear": "steel_8620c", "spark_plug": "steel_1018", "oil_pan": "steel_1018",
    "valve_cover": "al_a380",
}
MOVING = {"piston", "piston_pin", "connecting_rod", "valve", "valve_spring", "piston_ring",
          "crankshaft", "camshaft", "timing_gear"}

WHY = {
    "density": "light", "conductivity": "conducts heat away well", "cte": "low thermal expansion",
    "max_temp": "copes with the temperature",
}


def default_material(label: str) -> str:
    return DEFAULT.get(label, "steel_1045")


def candidates(label: str) -> list[str]:
    return CANDIDATES.get(label, GENERIC)


def rank_materials(label: str, evaluate_fn, volume_mm3: float, target_hours: float = 10000.0,
                   keys: list[str] | None = None) -> list[dict]:
    """`evaluate_fn(material) -> (life_hours, min_safety, notes)` runs the part's physics."""
    rows = []
    for key in keys or candidates(label):
        mat = get(key)
        life, safety, notes = evaluate_fn(mat)
        mass = volume_mm3 * 1e-9 * mat.density
        rows.append({"key": key, "name": mat.name, "life_hours": life, "min_safety": safety,
                     "mass_kg": mass, "cost_index": mass * mat.cost, "notes": notes, "material": mat})
    if not rows:
        return []
    life_cap = target_hours * 5
    lo_m = min(r["mass_kg"] for r in rows)
    lo_c = min(r["cost_index"] for r in rows)
    w_mass = 0.25 if label in MOVING else 0.08
    for r in rows:
        life = min(r["life_hours"], life_cap) if math.isfinite(r["life_hours"]) else life_cap
        s_life = math.log10(1 + life) / math.log10(1 + life_cap)
        s_mass = lo_m / r["mass_kg"] if r["mass_kg"] > 0 else 1
        s_cost = lo_c / r["cost_index"] if r["cost_index"] > 0 else 1
        s_safe = min(r["min_safety"], 3.0) / 3.0 if math.isfinite(r["min_safety"]) else 1.0
        score = 0.45 * s_life + w_mass * s_mass + 0.17 * s_cost + (0.38 - w_mass) * s_safe
        if r["life_hours"] < target_hours:
            score *= 0.6  # does not reach the target life
        if r["min_safety"] < 1.0:
            score *= 0.3  # yields or is overloaded somewhere
        r["score"] = round(score, 4)
    rows.sort(key=lambda r: -r["score"])
    best = rows[0]
    for r in rows:
        mat: Material = r.pop("material")
        reasons = []
        if r["life_hours"] >= target_hours:
            reasons.append(f"reaches the {target_hours:,.0f} h target")
        else:
            reasons.append(f"predicted life {r['life_hours']:,.0f} h is below the {target_hours:,.0f} h target")
        if r["min_safety"] < 1.0:
            reasons.append(f"overloaded (safety factor {r['min_safety']:.2f})")
        if r["mass_kg"] <= lo_m * 1.05:
            reasons.append("lightest option")
        if r["cost_index"] <= lo_c * 1.05:
            reasons.append("cheapest option")
        if r is not best and best["mass_kg"] > 0:
            dm = (r["mass_kg"] / best["mass_kg"] - 1) * 100
            if abs(dm) > 5:
                reasons.append(f"{abs(dm):.0f}% {'heavier' if dm > 0 else 'lighter'} than the top pick")
        if mat.notes:
            reasons.append(mat.notes)
        r["reasons"] = reasons
        r["life_hours"] = None if not math.isfinite(r["life_hours"]) else round(r["life_hours"], 0)
        r["min_safety"] = None if not math.isfinite(r["min_safety"]) else round(r["min_safety"], 2)
        r["mass_kg"] = round(r["mass_kg"], 4)
        r["cost_index"] = round(r["cost_index"], 3)
    return rows
