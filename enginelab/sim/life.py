"""Life simulation over thousands of operating hours.

1. The duty cycle (idle / city / highway / full load ... with time shares and
   starts per hour) is solved once per operating point with the cycle model.
2. Every part's failure modes are evaluated at every operating point and
   combined into hourly damage and wear rates.
3. A Monte Carlo time-stepping run (default 400 virtual engines) adds scatter
   in strength, loads, wear coefficients and Miner's critical damage, and
   couples degradation between parts:
     * bearing wear opens the clearance -> higher dynamic bearing load,
     * ring / bore wear lowers compression -> slightly lower gas loads,
     * gasket clamp relaxes with time -> sealing margin falls.
4. Results: failure probability by the horizon, B10/B50 lives, the dominant
   failure mode and location of every part, a system survival curve and a
   damage history for plotting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..materials import get as get_material
from .cycle import EngineModel, run_cycle
from .parts import Mode, PartGeom, gasket_relaxation, part_modes

DUTY_CYCLES = {
    "automotive": {"label": "Passenger car (mixed driving)", "starts_per_hour": 2.0, "modes": [
        {"name": "idle", "time": 0.20, "rpm": 0.12, "load": 0.08},
        {"name": "city", "time": 0.35, "rpm": 0.32, "load": 0.30},
        {"name": "highway", "time": 0.33, "rpm": 0.45, "load": 0.50},
        {"name": "hard acceleration", "time": 0.10, "rpm": 0.75, "load": 0.95},
        {"name": "redline", "time": 0.02, "rpm": 1.00, "load": 1.00}]},
    "generator": {"label": "Stationary generator", "starts_per_hour": 0.05, "modes": [
        {"name": "rated load", "time": 0.85, "rpm": 0.55, "load": 0.80},
        {"name": "light load", "time": 0.15, "rpm": 0.55, "load": 0.30}]},
    "marine": {"label": "Marine (propeller law)", "starts_per_hour": 0.2, "modes": [
        {"name": "idle", "time": 0.15, "rpm": 0.15, "load": 0.05},
        {"name": "cruise", "time": 0.70, "rpm": 0.70, "load": 0.70},
        {"name": "full power", "time": 0.15, "rpm": 0.95, "load": 0.95}]},
    "agricultural": {"label": "Tractor / off-road", "starts_per_hour": 0.3, "modes": [
        {"name": "idle", "time": 0.10, "rpm": 0.15, "load": 0.05},
        {"name": "transport", "time": 0.20, "rpm": 0.55, "load": 0.40},
        {"name": "field work", "time": 0.70, "rpm": 0.75, "load": 0.85}]},
    "racing": {"label": "Motorsport", "starts_per_hour": 0.5, "modes": [
        {"name": "pit / idle", "time": 0.10, "rpm": 0.20, "load": 0.10},
        {"name": "corner exit", "time": 0.30, "rpm": 0.65, "load": 0.70},
        {"name": "straights", "time": 0.40, "rpm": 0.88, "load": 1.00},
        {"name": "rev limiter", "time": 0.20, "rpm": 1.00, "load": 1.00}]},
    "durability_test": {"label": "Full-load durability test", "starts_per_hour": 0.0, "modes": [
        {"name": "rated power", "time": 1.00, "rpm": 0.90, "load": 1.00}]},
}


SERVICE_ITEMS = {"spark_plug"}  # wear out by design; reported as replacement intervals


@dataclass
class SimPart:
    id: int
    label: str
    material: str
    geom: PartGeom
    ctx: dict = field(default_factory=dict)
    group: int = 0
    name: str = ""


def solve_operating_points(eng: EngineModel, duty: dict):
    out = []
    for m in duty["modes"]:
        rpm = max(500.0, m["rpm"] * eng.max_rpm)
        out.append((m, run_cycle(eng, rpm, m["load"])))
    return out


def aggregate_modes(part: SimPart, eng: EngineModel, ops, starts_h: float) -> list[dict]:
    """Combine one part's modes over the duty cycle into hourly rates."""
    mat = get_material(part.material)
    table: dict[str, dict] = {}
    for m, cyc in ops:
        for md in part_modes(part.label, part.geom, mat, cyc, eng, starts_h, part.ctx):
            row = table.setdefault(md.name, {
                "name": md.name, "kind": md.kind, "location": md.location, "rate": 0.0, "limit": md.limit,
                "slope": md.slope, "safety_fatigue": math.inf, "safety_static": math.inf,
                "margin": math.inf, "temp_max": -273.0, "gas": md.gas_driven, "coupling": md.coupling,
                "note": md.note, "per_op": [], "worst_op": None, "_worst": -math.inf,
                "sigma_a": 0.0, "sigma_m": 0.0, "sigma_max": 0.0})
            contrib = m["time"] * md.rate
            row["rate"] += contrib
            row["per_op"].append({"op": m["name"], "rate": md.rate, "share": contrib})
            row["safety_fatigue"] = min(row["safety_fatigue"], md.safety_fatigue)
            row["safety_static"] = min(row["safety_static"], md.safety_static)
            row["margin"] = min(row["margin"], md.margin)
            row["temp_max"] = max(row["temp_max"], md.temp_c)
            severity = contrib if md.kind in ("fatigue", "wear", "erosion") else (
                1.0 / md.margin if md.margin > 0 else math.inf)
            if severity > row["_worst"]:
                row.update(_worst=severity, worst_op=m["name"], slope=md.slope, note=md.note,
                           sigma_a=md.sigma_a, sigma_m=md.sigma_m, sigma_max=md.sigma_max)
    return list(table.values())


def _std_normals(rng, S: int, D: int, sampler: str = "lhs", seed: int = 0) -> np.ndarray:
    """S x D standard normals. "mc": plain random. "lhs": Latin hypercube (every column is
    stratified into S equal-probability bins), which removes most of the sampling noise of
    quantiles such as B10. "sobol": scrambled Sobol quasi-random points."""
    from scipy.special import ndtri
    if sampler == "mc":
        return rng.standard_normal((S, D))
    if sampler == "sobol":
        from scipy.stats import qmc
        u = qmc.Sobol(d=D, scramble=True, seed=seed).random(S)
    else:
        u = (np.argsort(rng.random((S, D)), axis=0) + rng.random((S, D))) / S
    return ndtri(np.clip(u, 1e-12, 1 - 1e-12))


def simulate(parts: list[SimPart], eng: EngineModel, duty_key: str = "automotive", hours: float = 5000.0,
             samples: int = 400, steps: int = 240, seed: int = 0, duty: dict | None = None,
             sampler: str = "lhs") -> dict:
    duty = duty or DUTY_CYCLES[duty_key]
    starts_h = duty.get("starts_per_hour", 1.0)
    ops = solve_operating_points(eng, duty)
    rng = np.random.default_rng(seed)

    # Modes per distinct part (identical copies share their mode table).
    cache: dict = {}
    rows = []          # (instance index, mode dict)
    for pi, p in enumerate(parts):
        key = (p.label, p.group, p.material, p.ctx.get("subtype", ""))
        if key not in cache:
            cache[key] = aggregate_modes(p, eng, ops, starts_h)
        for md in cache[key]:
            rows.append((pi, md))

    n_parts, S = len(parts), samples
    t_fail_part = np.full((S, n_parts), np.inf)
    mode_of_fail = np.full((S, n_parts), -1)
    histories = {}
    t_grid = np.linspace(0, hours, steps + 1)
    dt = hours / steps

    if rows:
        R = len(rows)
        kind = np.array([r[1]["kind"] for r in rows])
        inst = np.array([r[0] for r in rows])
        rate0 = np.array([r[1]["rate"] for r in rows])
        limit = np.array([r[1]["limit"] if r[1]["kind"] in ("wear", "erosion") else 1.0 for r in rows])
        slope = np.array([min(r[1]["slope"], 25.0) for r in rows])
        gas = np.array([r[1]["gas"] for r in rows])
        bearing = np.array([r[1]["coupling"] == "bearing" for r in rows])
        margin0 = np.array([r[1]["margin"] for r in rows])
        static_sf = np.array([np.inf if r[1]["kind"] == "check" else r[1]["safety_static"] for r in rows])
        temp = np.array([r[1]["temp_max"] for r in rows])
        is_fat = kind == "fatigue"
        is_wear = (kind == "wear") | (kind == "erosion")
        is_seal = kind == "sealing"
        ring_bore = np.array([r[1]["name"] in ("face wear", "bore wear") for r in rows])

        # Scatter.
        Z = _std_normals(rng, S, 1 + n_parts + 3 * R, sampler, seed)
        load_sc = np.exp(0.05 * Z[:, :1])                                          # per virtual engine
        strength_sc = np.exp(0.07 * Z[:, 1:1 + n_parts])[:, inst]                   # per part instance
        o = 1 + n_parts
        miner = np.exp(0.30 * Z[:, o:o + R])
        wear_sc = np.exp(0.35 * Z[:, o + R:o + 2 * R])
        seal_sc = np.exp(0.08 * Z[:, o + 2 * R:o + 3 * R])

        fat_rate = np.where(is_fat, rate0, 0.0) * (load_sc / strength_sc) ** slope / miner
        wear_rate = np.where(is_wear, rate0, 0.0) * wear_sc
        damage = np.zeros((S, R))
        t_fail = np.full((S, R), np.inf)
        # Static failure (yield / buckling / slip) happens immediately. Local yielding up to ~10%
        # over the yield strength shakes down (residual stresses), so the threshold is 0.9.
        static_fail = (static_sf * strength_sc / load_sc) < 0.9
        t_fail[static_fail] = 0.0

        # Instance -> bearing wear row (for coupling).
        bearing_wear_row = {}
        for j in range(R):
            if bearing[j] and is_wear[j]:
                bearing_wear_row[inst[j]] = j
        couple_idx = np.array([bearing_wear_row.get(inst[j], -1) for j in range(R)])
        has_couple = bearing & (couple_idx >= 0)
        rb_cols = np.nonzero(ring_bore)[0]

        nominal_frac = np.zeros((steps + 1, R))
        nom_damage = np.zeros(R)
        for k in range(1, steps + 1):
            t0, t1 = t_grid[k - 1], t_grid[k]
            wear_frac = np.where(is_wear, damage, 0.0)
            gas_scale = 1.0 - 0.05 * (wear_frac[:, rb_cols].mean(1, keepdims=True) if len(rb_cols) else 0.0)
            f_b = np.ones((S, R))
            if has_couple.any():
                f_b[:, has_couple] = 1.0 + 0.6 * np.clip(wear_frac[:, couple_idx[has_couple]], 0, 2)
            mult = np.where(gas, gas_scale ** slope, 1.0) * f_b ** np.where(is_fat, slope, 1.0)
            inc = np.where(is_fat, fat_rate * mult, wear_rate * f_b / limit) * dt
            new = damage + inc
            crossed = (new >= 1.0) & (damage < 1.0) & ~np.isfinite(t_fail)
            frac = np.clip((1.0 - damage) / np.maximum(inc, 1e-300), 0, 1)
            t_fail = np.where(crossed, t0 + frac * dt, t_fail)
            damage = new
            # Sealing: margin falls as clamp relaxes.
            if is_seal.any():
                relax = gasket_relaxation(np.array([t1]), float(temp[is_seal].max()))[0]
                m_now = margin0[None, :] * seal_sc * (1 - relax) / np.maximum(gas_scale, 1e-6)
                lost = is_seal[None, :] & (m_now < 1.0) & ~np.isfinite(t_fail)
                t_fail = np.where(lost, t1, t_fail)
            # Nominal (no scatter) history for plotting.
            nom_damage += np.where(is_fat, rate0, np.where(is_wear, rate0 / limit, 0.0)) * dt
            nominal_frac[k] = nom_damage

        # Project lives past the horizon from the final rates.
        rem = ~np.isfinite(t_fail)
        final_rate = np.where(is_fat, fat_rate, wear_rate / limit)
        proj = hours + np.clip(1.0 - damage, 0, None) / np.maximum(final_rate, 1e-300)
        if is_seal.any():
            # margin(t) = m0 * (1 - a - k log10(1+t)) = 1  ->  t
            k_rel = 0.03 * (1 + max(float(temp[is_seal].max()) - 100, 0) / 100)
            m0s = margin0[None, :] * seal_sc
            need = (1 - 1 / np.maximum(m0s, 1e-9) - 0.08) / k_rel
            t_seal = np.where(m0s > 1, 10 ** np.clip(need, -3, 12) - 1, 0.0)
            proj = np.where(is_seal[None, :], np.maximum(t_seal, hours), proj)
        t_fail = np.where(rem, proj, t_fail)

        for pi in range(n_parts):
            cols = np.nonzero(inst == pi)[0]
            if len(cols):
                sub = t_fail[:, cols]
                t_fail_part[:, pi] = sub.min(1)
                mode_of_fail[:, pi] = cols[np.argmin(sub, 1)]
        for pi in range(n_parts):
            cols = np.nonzero(inst == pi)[0]
            if len(cols):
                histories[pi] = nominal_frac[:, cols].max(1)

    # ---------------------------------------------------------------- summarise
    def q(a, p):
        a = a[np.isfinite(a)]
        return float(np.percentile(a, p)) if len(a) else math.inf

    part_results = []
    for pi, p in enumerate(parts):
        tf = t_fail_part[:, pi]
        cols = [j for j, r in enumerate(rows) if r[0] == pi]
        modes = []
        for j in cols:
            md = rows[j][1]
            share = float(np.mean(mode_of_fail[:, pi] == j)) if len(cols) else 0.0
            nominal_h = (1.0 / md["rate"] if md["kind"] == "fatigue" and md["rate"] > 0 else
                         md["limit"] / md["rate"] if md["kind"] in ("wear", "erosion") and md["rate"] > 0 else None)
            modes.append({k: _clean(v) for k, v in md.items() if not k.startswith("_")} |
                         {"share_of_failures": round(share, 3), "nominal_hours": _clean(nominal_h)})
        modes.sort(key=lambda m: -m["share_of_failures"])
        has_model = bool(cols)
        dominant = modes[0] if modes else None
        part_results.append({
            "id": p.id, "label": p.label, "name": p.name, "material": p.material,
            "modelled": has_model,
            "prob_fail": round(float(np.mean(tf <= hours)), 4) if has_model else None,
            "b10_hours": _clean(q(tf, 10)) if has_model else None,
            "b50_hours": _clean(q(tf, 50)) if has_model else None,
            "dominant_mode": dominant["name"] if dominant else None,
            "location": dominant["location"] if dominant else None,
            "min_safety_fatigue": _clean(min([m["safety_fatigue"] or math.inf for m in modes] or [math.inf])),
            "min_safety_static": _clean(min([m["safety_static"] or math.inf for m in modes] or [math.inf])),
            "modes": modes,
            "history": [round(float(x), 5) for x in histories.get(pi, np.zeros(steps + 1))[::max(1, steps // 60)]],
        })
    service = np.array([p.label in SERVICE_ITEMS for p in parts]) if n_parts else np.zeros(0, bool)
    structural = ~service
    t_sys = t_fail_part[:, structural].min(1) if structural.any() else np.full(S, np.inf)
    surv_t = np.linspace(0, hours, 61)
    survival = [round(float(np.mean(t_sys > t)), 4) for t in surv_t]
    first = {}
    if n_parts:
        masked = np.where(structural[None, :], t_fail_part, np.inf)
        first_idx = np.argmin(masked, 1)
        for i in first_idx:
            first[parts[i].label] = first.get(parts[i].label, 0) + 1
    for r, p in zip(part_results, parts):
        r["service_item"] = p.label in SERVICE_ITEMS
        if r["service_item"] and r["b10_hours"]:
            r["service_interval_hours"] = round(r["b10_hours"] * 0.8, -1)
    ranking = sorted([r for r in part_results if r["modelled"] and not r["service_item"]],
                     key=lambda r: (r["b10_hours"] if r["b10_hours"] is not None else math.inf))
    return {
        "hours": hours, "samples": S, "duty": {"key": duty_key, **duty},
        "operating_points": [dict(name=m["name"], time=m["time"], **{k: round(v, 2) for k, v in c.stats.items()})
                             for m, c in ops],
        "parts": part_results,
        "system": {
            "survival_time": [round(float(t), 1) for t in surv_t],
            "survival": survival,
            "b10_hours": _clean(q(t_sys, 10)), "b50_hours": _clean(q(t_sys, 50)),
            "b10_ci90": _ci(t_sys, 10, seed), "b50_ci90": _ci(t_sys, 50, seed),
            "sampler": sampler,
            "prob_fail": round(float(np.mean(t_sys <= hours)), 4),
            "first_failure_share": {k: round(v / S, 3) for k, v in sorted(first.items(), key=lambda kv: -kv[1])},
        },
        "risk_ranking": [{"id": r["id"], "label": r["label"], "b10_hours": r["b10_hours"],
                          "mode": r["dominant_mode"], "location": r["location"]} for r in ranking[:15]],
        "history_time": [round(float(t), 1) for t in t_grid[::max(1, steps // 60)]],
    }


def _ci(t: np.ndarray, pct: float, seed: int = 0, n_boot: int = 300):
    """90% bootstrap confidence interval of a life percentile (sampling uncertainty only)."""
    t = t[np.isfinite(t)]
    if len(t) < 20:
        return None
    rng = np.random.default_rng(seed + 7)
    idx = rng.integers(0, len(t), (n_boot, len(t)))
    qs = np.percentile(t[idx], pct, axis=1)
    return [round(float(np.percentile(qs, 5)), 1), round(float(np.percentile(qs, 95)), 1)]


def _clean(v):
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if not math.isfinite(v):
            return None
        return round(v, 4)
    if isinstance(v, list):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v
