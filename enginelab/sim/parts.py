"""Load models: how each kind of part is stressed, worn or heated in service.

Every model turns one operating point (a solved engine cycle) plus the part's
measured geometry and material into a list of failure modes. Each mode is
either fatigue (stress amplitude/mean -> cycles to failure), wear (rate vs a
wear limit), sealing (margin), or erosion. Formulas are standard machine
design results (Shigley, Roark, Heywood, Hamrock); the docstring of each
model says which.
"""
from __future__ import annotations

import contextvars
import math
from dataclasses import dataclass, field

import numpy as np

from ..materials import Material
from .cycle import CycleResult, EngineModel
from .fatigue import evaluate, local_slope, sn_slope, NON_FERROUS

# Design modifiers for what-if studies (improve.py): {mode-name substring or "*": factor} raising
# fatigue strength (fillet rolling, nitriding, shot peening, hard anodising). Thread-safe.
_FATIGUE_FACTORS: contextvars.ContextVar = contextvars.ContextVar("fatigue_factors", default=None)

LUB_RING = 0.03       # fraction of ring/liner sliding in boundary contact (calibrated: ~10 um/1000 h)
LUB_SKIRT = 0.002
G_STEEL = 79.3e3      # shear modulus, MPa


@dataclass
class Mode:
    part: str
    name: str
    location: str
    kind: str                      # fatigue | wear | sealing | erosion | static | check (warning only)
    cycles_per_hour: float = 0.0
    sigma_a: float = 0.0
    sigma_m: float = 0.0
    sigma_max: float = 0.0
    cycles_to_failure: float = math.inf
    slope: float = 8.0             # S-N exponent (damage ~ stress^slope)
    safety_fatigue: float = math.inf
    safety_static: float = math.inf
    rate: float = 0.0              # wear/erosion: um per hour; fatigue: damage per hour
    limit: float = 1.0             # wear/erosion limit in um; fatigue: 1
    margin: float = math.inf       # sealing margin (>1 = sealed)
    temp_c: float = 20.0
    gas_driven: bool = False       # stress scales with cylinder pressure
    coupling: str = ""             # e.g. "bearing" (load grows as the bearing wears)
    note: str = ""

    def hours_to_failure(self) -> float:
        if self.kind == "fatigue":
            return 1.0 / self.rate if self.rate > 0 else math.inf
        if self.kind in ("wear", "erosion"):
            return self.limit / self.rate if self.rate > 0 else math.inf
        return math.inf

    def to_dict(self) -> dict:
        out = {k: (None if isinstance(v, float) and not math.isfinite(v) else
                   (round(v, 6) if isinstance(v, float) else v)) for k, v in self.__dict__.items()}
        h = self.hours_to_failure()
        out["hours_to_failure"] = None if not math.isfinite(h) else round(h, 1)
        return out


@dataclass
class PartGeom:
    """Geometry the load models need, measured where possible, otherwise typical ratios of the bore."""
    label: str
    specs: dict
    bore: float
    stroke: float
    rod: float
    extra: dict = field(default_factory=dict)

    def get(self, key: str, default: float) -> float:
        v = self.specs.get(key, self.extra.get(key))
        return float(v) if v not in (None, 0) else float(default)


def op_temperature(label: str, load: float, rpm_frac: float, subtype: str = "") -> float:
    """Typical metal temperatures (deg C) of engine parts vs load (Heywood, Ch. 12)."""
    l = float(np.clip(load, 0, 1.2))
    s = math.sqrt(max(rpm_frac, 0.05))
    table = {
        "piston": 140 + 150 * l * s, "piston_ring": 140 + 120 * l * s, "piston_pin": 120 + 50 * l,
        "cylinder_liner": 110 + 70 * l, "cylinder_head": 120 + 110 * l * s, "head_gasket": 110 + 70 * l,
        "valve": (300 + 450 * l * s) if subtype == "exhaust" else (200 + 150 * l * s),
        "valve_spring": 100 + 20 * l, "camshaft": 100 + 20 * l, "crankshaft": 100 + 25 * l,
        "connecting_rod": 105 + 30 * l, "main_bearing": 105 + 35 * l, "engine_block": 95 + 30 * l,
        "spark_plug": 450 + 400 * l, "oil_pan": 90 + 30 * l, "valve_cover": 90 + 20 * l,
        "flywheel": 50 + 20 * l, "timing_gear": 90 + 20 * l, "bolt": 100 + 30 * l,
    }
    return float(table.get(label, 80 + 20 * l))


def _fatigue(part, name, loc, mat: Material, sa, sm, cph, temp, smax=None, surface="machined",
             d_eff=None, load="bending", gas=False, coupling="", note="", reliability=0.5) -> Mode:
    ff = 1.0
    for key, f in (_FATIGUE_FACTORS.get() or {}).items():
        if key == "*" or key in name:
            ff *= float(f)
    if ff != 1.0:
        # A treatment that raises the endurance limit by ff is, in Goodman terms, the same as an
        # alternating stress lower by ff; static strength (smax) is unchanged.
        sa = sa / ff
        note = (note + " " if note else "") + f"surface treatment: fatigue strength x{ff:.2f}"
    res = evaluate(mat, sa, sm, smax=smax, surface=surface, d_eff_mm=d_eff, load=load,
                   temp_c=temp, reliability=reliability)
    slope = local_slope(res.sigma_ar, mat.uts * res.marin["kd"], res.Se, mat.category not in NON_FERROUS)
    over = max(0.0, temp - mat.max_temp)
    cycles = res.cycles * math.exp(-over / 30.0)  # running hotter than the material is rated for
    if over > 0:
        note = (note + " " if note else "") + f"runs {over:.0f} C above the material's service limit"
    rate = cph / cycles if cycles > 0 else math.inf
    return Mode(part=part, name=name, location=loc, kind="fatigue", cycles_per_hour=cph, sigma_a=sa,
                sigma_m=sm, sigma_max=(smax if smax is not None else abs(sa) + abs(sm)),
                cycles_to_failure=cycles, slope=slope, safety_fatigue=res.safety_fatigue,
                safety_static=res.safety_static, rate=rate, temp_c=temp, gas_driven=gas,
                coupling=coupling, note=note)


def _archard_rate_um_h(k: float, p_mpa: float, v_ms: float, hv: float, f_contact: float) -> float:
    """Archard: depth rate = K p v / H (H in Pa = 9.81e6 * HV)."""
    return k * (p_mpa * 1e6) * v_ms * 3600 / (hv * 9.81e6) * f_contact * 1e6


# ------------------------------------------------------------------ part models
def model_piston(g: PartGeom, mat, cyc: CycleResult, eng: EngineModel, temp, starts_h):
    """Crown: clamped circular plate under peak pressure (Roark, sigma = 3 p a^2 / 4 t^2).
    Crown thermal fatigue per start-stop. Skirt wear by side thrust (Archard)."""
    B = g.bore
    p = cyc.pressure_bar.max() / 10.0
    # Effective radius 0.34B: the pin bosses and skirt support the crown well inside the ring
    # land; calibrated to published piston FEA (40-80 MPa crown stress at 70-80 bar).
    a = 0.34 * B
    t = g.get("crown_thickness_mm", 0.1 * B)
    sig = 3 * p * a * a / (4 * t * t)
    cph = cyc.rpm * 30
    modes = [_fatigue("piston", "crown bending fatigue", "crown centre / ring-land fillet", mat,
                      sig / 2, sig / 2, cph, temp, smax=sig, d_eff=t, gas=True)]
    dT = 30 + 70 * cyc.load
    sth = mat.E * 1e3 * mat.cte * 1e-6 * dT / (2 * (1 - mat.poisson))
    modes.append(_fatigue("piston", "crown thermal fatigue", "crown surface (bowl rim)", mat,
                          sth / 2, 0.0, starts_h + 2.0, temp, smax=sth / 3,
                          note="one cycle per start/load change; secondary stress (3Sm rule for static)"))
    side = np.abs(cyc.side_force)
    p_skirt = float(side.mean()) / (0.5 * B * 0.6 * B)
    v = cyc.stats["mean_piston_speed_ms"]
    w = Mode(part="piston", name="skirt wear", location="thrust side of the skirt", kind="wear",
             rate=_archard_rate_um_h(mat.archard_k, p_skirt, v, mat.hardness_hv, LUB_SKIRT),
             limit=40.0, temp_c=temp, note="limit = coating / ovality allowance")
    return modes + [w]


def model_pin(g: PartGeom, mat, cyc, eng, temp, starts_h):
    """Hollow pin in bending between the bosses: M = F L / 8, Z = pi (do^4 - di^4) / 32 do."""
    B = g.bore
    do = g.get("outer_diameter_mm", 0.25 * B)
    di = g.specs.get("inner_diameter_mm", 0.55 * do) or 0.0
    span = 0.55 * B
    Fp = (cyc.pressure_bar - 1.0) * 0.1 * math.pi / 4 * B * B - (eng.m_piston_kg * 0.8) * cyc.piston_accel * 1.0
    z = math.pi * (do ** 4 - di ** 4) / (32 * do)
    sig = Fp * span / 8 / z
    return [_fatigue("piston_pin", "bending fatigue", "pin centre, inner surface", mat,
                     (sig.max() - sig.min()) / 2, (sig.max() + sig.min()) / 2, cyc.rpm * 30, temp,
                     smax=float(np.abs(sig).max()), surface="ground", d_eff=do, gas=True)]


def model_rod(g: PartGeom, mat, cyc, eng, temp, starts_h):
    """Shank: axial push-pull on the minimum section; big-end eye: curved-beam bending
    under inertia tension (K ~ 3); buckling: Johnson/Euler."""
    B = g.bore
    A = g.get("min_shank_area_mm2", 0.045 * B * B)
    thick = g.get("thickness_mm", 0.22 * B)
    Fc = float(cyc.rod_force.max())
    Ft = float(max(-cyc.rod_force.min(), 0.0))
    cph = cyc.rpm * 30
    sa = (Fc + Ft) / (2 * A)
    sm = (Ft - Fc) / (2 * A)
    modes = [_fatigue("connecting_rod", "shank push-pull fatigue", "I-beam shank, minimum section", mat,
                      1.2 * sa, 1.2 * sm, cph, temp, smax=1.2 * max(Fc, Ft) / A, surface="machined",
                      load="axial", gas=True, note="assumes a shot-peened forging (surface factor of a machined part)")]
    A_eye = 0.12 * B * thick
    s_eye = 3.0 * Ft / (2 * A_eye)
    modes.append(_fatigue("connecting_rod", "big-end eye bending (inertia)", "big-end cap / bolt bosses", mat,
                          s_eye / 2, s_eye / 2, cyc.rpm * 60, temp, smax=s_eye, surface="machined",
                          d_eff=0.12 * B))
    # Buckling about the weak axis (rod thickness), pinned-pinned, length = centre distance.
    k_r = 0.30 * thick
    Lr = g.rod
    E = mat.E * 1e3
    sy = mat.yield_strength * mat.strength_factor(temp)
    slender = Lr / k_r
    trans = math.sqrt(2 * math.pi ** 2 * E / sy)
    if slender < trans:
        scr = sy - (sy * slender / (2 * math.pi)) ** 2 / E
    else:
        scr = math.pi ** 2 * E / slender ** 2
    margin = scr * A / max(Fc, 1.0)
    modes.append(Mode(part="connecting_rod", name="buckling", location="whole shank", kind="static",
                      safety_static=margin, margin=margin, temp_c=temp,
                      note="Johnson column" if slender < trans else "Euler column"))
    return modes


def model_crank(g: PartGeom, mat, cyc, eng, temp, starts_h):
    """Crankpin fillet: bending (span between webs, Kt 2.0) plus journal torsion (Kt 1.5),
    combined with von Mises; one cycle per revolution."""
    B = g.bore
    d_pin = g.get("crankpin_diameter_mm", 0.54 * B)
    d_main = g.get("main_journal_diameter_mm", 0.65 * B)
    span = 0.6 * B
    r = g.stroke / 2 / 1000
    omega = cyc.rpm * 2 * math.pi / 60
    m_rot = eng.m_rod_kg * eng.rod_big_end_frac
    th = cyc.theta * math.pi / 180
    beta = np.arcsin((g.stroke / 2) / g.rod * np.sin(th))
    F_r = cyc.rod_force * np.cos(th + beta) - m_rot * r * omega ** 2
    M = F_r * span / 4
    sig = 2.0 * 32 * M / (math.pi * d_pin ** 3)
    tau_a = 1.5 * 16 * (cyc.journal_torque_range / 2 * 1000) / (math.pi * d_main ** 3)
    tau_m = 1.5 * 16 * (abs(cyc.journal_torque_mean) * 1000) / (math.pi * d_main ** 3)
    sa_b, sm_b = (sig.max() - sig.min()) / 2, (sig.max() + sig.min()) / 2
    sa = math.sqrt(sa_b ** 2 + 3 * tau_a ** 2)
    sm = math.sqrt(sm_b ** 2 + 3 * tau_m ** 2)
    return [_fatigue("crankshaft", "crankpin fillet fatigue (bending + torsion)",
                     "crankpin-to-web fillet, rear throw", mat, sa, sm, cyc.rpm * 60, temp,
                     smax=sa + sm, surface="ground", d_eff=d_pin, load="combined", gas=True,
                     note="assumes un-rolled fillets; fillet rolling or nitriding adds 30-80% fatigue strength")]


def _spring_forces(g: PartGeom, eng: EngineModel, cyc):
    B = g.bore
    d = g.get("wire_diameter_mm", max(2.0, 0.044 * B))
    D = g.get("mean_coil_diameter_mm", 0.22 * B)
    na = g.get("active_coils", 4.5)
    lift = g.get("valve_lift_mm", 0.11 * B)
    k = G_STEEL * d ** 4 / (8 * D ** 3 * na)            # N/mm
    f_min = k * 0.6 * lift                             # installed load
    f_max = f_min + k * lift                           # the spring itself only sees its deflection
    return d, D, na, k, f_min, f_max


def model_spring(g: PartGeom, mat, cyc, eng, temp, starts_h):
    """Torsional shear with the Bergstrasser factor; surge check (natural frequency
    vs the 13th cam harmonic)."""
    d, D, na, k, f_min, f_max = _spring_forces(g, eng, cyc)
    C = D / d
    Kb = (4 * C + 2) / (4 * C - 3)
    tau = lambda F: Kb * 8 * F * D / (math.pi * d ** 3)
    ta, tm = (tau(f_max) - tau(f_min)) / 2, (tau(f_max) + tau(f_min)) / 2
    # Natural frequency, fixed-fixed: fn = d / (2 pi na D^2) * sqrt(G / (2 rho))   (SI)
    fn = (d / 1000) / (2 * math.pi * na * (D / 1000) ** 2) * math.sqrt(G_STEEL * 1e6 / (2 * mat.density))
    cam_hz = cyc.rpm / 120
    surge = fn < 11 * cam_hz
    mult = 1.15 if surge else 1.0
    # Torsion: mean stress barely matters (Sines), so the shear amplitude is compared with the
    # shot-peened torsional endurance strength (Zimmerli: ~400 MPa for Cr-Si).
    from .fatigue import marin_factors, sn_cycles
    f = marin_factors(mat, surface="ground", d_eff_mm=d, temp_c=temp)
    s_sa = 0.577 * mat.endurance * f["ka"] * f["kb"] * f["kd"] * 1.3
    s_su = 0.67 * mat.uts * f["kd"]
    ta_eff = ta * mult
    ferrous = mat.category not in NON_FERROUS
    cycles = sn_cycles(ta_eff, s_su, s_sa, ferrous)
    slope = local_slope(ta_eff, s_su, s_sa, ferrous)
    t_max = tau(f_max) * mult
    t_allow = 0.45 * mat.uts * f["kd"]   # set (permanent sag) limit for spring steels
    note = (f"surge risk: natural frequency {fn:.0f} Hz < 11x cam frequency" if surge
            else f"natural frequency {fn:.0f} Hz")
    cph = cyc.rpm * 30
    set_margin = t_allow / t_max
    return [Mode(part="valve_spring", name="torsional fatigue", location="inner coil surface", kind="fatigue",
                 cycles_per_hour=cph, sigma_a=ta_eff, sigma_m=tm, sigma_max=t_max, cycles_to_failure=cycles,
                 slope=slope, safety_fatigue=s_sa / ta_eff, safety_static=math.inf,
                 rate=cph / cycles, temp_c=temp,
                 note=note + f"; shear {ta_eff:.0f} MPa amplitude vs {s_sa:.0f} MPa (peened)"),
            Mode(part="valve_spring", name="set (sag) check", location="whole coil", kind="check",
                 safety_static=set_margin, margin=set_margin, temp_c=temp,
                 note=f"max shear {t_max:.0f} MPa vs set limit {t_allow:.0f} MPa (0.45 Sut); "
                      "below 1 the spring slowly loses preload -> valve float")]


def model_valve(g: PartGeom, mat, cyc, eng, temp, starts_h, subtype="intake"):
    """Head: simply-supported disc under gas pressure (Roark 1.24 p r^2 / t^2) at its running
    temperature; seat recession wear from impact."""
    B = g.bore
    dh = g.get("head_diameter_mm", 0.36 * B)
    t = max(0.06 * dh, 1.5)
    p = cyc.pressure_bar.max() / 10
    sig = 1.24 * p * (dh / 2) ** 2 / (t * t) * 0.35  # tulip head: the stem fillet stiffens the disc
    modes = [_fatigue("valve", f"{subtype} valve head fatigue", "stem-to-head fillet", mat,
                      sig / 2, sig / 2, cyc.rpm * 30, temp, smax=sig, surface="ground", d_eff=t, gas=True)]
    v_seat = 0.25 + 0.5 * cyc.rpm / eng.max_rpm
    rate = _archard_rate_um_h(mat.archard_k, 25.0, v_seat * 0.002, mat.hardness_hv, 1.0) * (cyc.rpm * 30) / 3600
    modes.append(Mode(part="valve", name=f"{subtype} seat recession", location="valve face / seat",
                      kind="wear", rate=rate * (2.0 if subtype == "exhaust" else 1.0), limit=150.0,
                      temp_c=temp, note="micro-slip at seating; hotter exhaust valves wear faster"))
    return modes


def model_cam(g: PartGeom, mat, cyc, eng, temp, starts_h, spring_force=None):
    """Lobe: Hertz line contact with a steel flat follower; contact (pitting) fatigue.
    Contact force = spring force + valve-train inertia on the opening flank."""
    B = g.bore
    F = (spring_force if spring_force else 700.0 * (B / 86) ** 2) * (1.0 + 0.35 * (cyc.rpm / eng.max_rpm) ** 2)
    b = 0.13 * B
    R = 0.085 * B
    E_star = 1.0 / ((1 - mat.poisson ** 2) / (mat.E * 1e3) + (1 - 0.3 ** 2) / 210e3)
    pmax = math.sqrt(F * E_star / (math.pi * b * R))
    s_allow = 1.9 * mat.hardness_hv + 250
    cycles = 1e8 * (s_allow / pmax) ** 8.0
    over = max(0.0, temp - mat.max_temp)
    cycles *= math.exp(-over / 30.0)
    cph = cyc.rpm * 30
    return [Mode(part="camshaft", name="lobe contact fatigue (pitting)", location="lobe nose", kind="fatigue",
                 cycles_per_hour=cph, sigma_a=pmax / 2, sigma_m=pmax / 2, sigma_max=pmax,
                 cycles_to_failure=cycles, slope=8.0, safety_fatigue=s_allow / pmax,
                 safety_static=s_allow * 1.6 / pmax, rate=cph / cycles, temp_c=temp,
                 note=f"Hertz pressure {pmax:.0f} MPa vs allowable {s_allow:.0f} MPa")]


def model_bearing(g: PartGeom, mat, cyc, eng, temp, starts_h, v_layout=False):
    """Plain journal bearing: specific load vs lining fatigue strength; Ocvirk short-bearing
    film thickness; mixed-lubrication and start-up wear (Archard)."""
    B = g.bore
    d = g.get("main_journal_diameter_mm", 0.65 * B)
    w = g.get("bearing_width_mm", 0.3 * B)
    F = float(cyc.pin_force.max()) * (1.0 if v_layout else 0.6)
    p_spec = F / (d * w)
    p_allow = mat.bearing_pressure or 0.3 * mat.yield_strength
    cph = cyc.rpm * 60
    cycles = 2e8 * (p_allow / max(p_spec, 1e-6)) ** 5
    cycles *= math.exp(-max(0.0, temp - mat.max_temp) / 30.0)
    modes = [Mode(part="main_bearing", name="lining fatigue", location="loaded half-shell, lining",
                  kind="fatigue", cycles_per_hour=cph, sigma_a=p_spec / 2, sigma_m=p_spec / 2,
                  sigma_max=p_spec, cycles_to_failure=cycles, slope=5.0,
                  safety_fatigue=p_allow / p_spec, safety_static=2 * p_allow / p_spec,
                  rate=cph / cycles, temp_c=temp, coupling="bearing",
                  note=f"specific load {p_spec:.1f} MPa vs allowable {p_allow:.0f} MPa")]
    # Ocvirk short bearing: W = eta U L^3 / (4 c^2) * eps / (1-eps^2)^2 * sqrt(pi^2 (1-eps^2) + 16 eps^2)
    eta = 0.009 * math.exp(-0.025 * (temp - 100))
    R = d / 2000
    L = w / 1000
    c = 0.0005 * d / 1000
    U = cyc.rpm * 2 * math.pi / 60 * R
    def load(eps):
        return eta * U * L ** 3 / (4 * c * c) * eps / (1 - eps * eps) ** 2 * math.sqrt(
            math.pi ** 2 * (1 - eps * eps) + 16 * eps * eps)
    lo, hi = 1e-4, 0.9999
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if load(mid) < F else (lo, mid)
    h_min_um = c * (1 - 0.5 * (lo + hi)) * 1e6
    lam = h_min_um / 0.4
    # Asperity contact share: full below lambda 1, falling off fast, ~nil above lambda 3.
    f_contact = 1.0 if lam < 1 else (10 ** (-2 * (lam - 1)) if lam < 3 else 1e-4)
    v = U
    rate = _archard_rate_um_h(mat.archard_k, p_spec * 0.4, v, mat.hardness_hv, f_contact)
    start_wear = _archard_rate_um_h(mat.archard_k, 1.0, 0.5, mat.hardness_hv, 1.0) * 3.0 / 3600 * starts_h
    modes.append(Mode(part="main_bearing", name="wear (mixed lubrication + starts)", location="lining surface",
                      kind="wear", rate=rate + start_wear, limit=25.0, temp_c=temp, coupling="bearing",
                      note=f"min oil film {h_min_um:.2f} um (lambda {lam:.1f})"))
    if h_min_um < 0.3:
        modes[-1].note += "; film collapses at peak load: seizure risk"
    return modes


def model_ring(g: PartGeom, mat, cyc, eng, temp, starts_h):
    p_gas = float(np.mean(cyc.pressure_bar)) / 10
    p = 0.15 + 0.6 * p_gas
    v = cyc.stats["mean_piston_speed_ms"]
    return [Mode(part="piston_ring", name="face wear", location="running face (top ring)", kind="wear",
                 rate=_archard_rate_um_h(mat.archard_k, p, v, mat.hardness_hv, LUB_RING), limit=80.0,
                 temp_c=temp, gas_driven=True, note="limit: ~0.5 mm extra end gap -> blow-by")]


def model_bore(part: str, g: PartGeom, mat, cyc, eng, temp, starts_h):
    """Bore wear against the rings + hoop stress of the wall (p r / t)."""
    B = g.bore
    p_gas = float(np.mean(cyc.pressure_bar)) / 10
    p = 0.15 + 0.6 * p_gas
    v = cyc.stats["mean_piston_speed_ms"]
    t = g.get("wall_mm", 0.06 * B) if part == "cylinder_liner" else 0.08 * B
    hoop = cyc.pressure_bar.max() / 10 * (B / 2) / t
    return [
        Mode(part=part, name="bore wear", location="top-ring reversal zone", kind="wear",
             rate=_archard_rate_um_h(mat.archard_k, p, v, mat.hardness_hv, LUB_RING), limit=100.0,
             temp_c=temp, gas_driven=True, note="limit: 0.1 mm wear step / ovality"),
        _fatigue(part, "hoop-stress fatigue", "upper bore wall", mat, hoop / 2, hoop / 2, cyc.rpm * 30, temp,
                 smax=hoop, surface="machined", d_eff=t, gas=True),
    ]


def model_block(g: PartGeom, mat, cyc, eng, temp, starts_h, has_liners=True, v_layout=False):
    B = g.bore
    F = float(cyc.pin_force.max()) * (1.0 if v_layout else 0.6)
    A = (0.14 * B * 0.65 * B) if v_layout else (0.12 * B * 0.6 * B)   # V blocks: 4-bolt / cross-bolted mains
    s = 1.8 * F / A
    modes = [_fatigue("engine_block", "main bulkhead fatigue", "main-cap bolt boss in the bulkhead", mat,
                      s / 2, s / 2, cyc.rpm * 60, temp, smax=s, surface="as-cast", d_eff=0.12 * B, gas=True)]
    if not has_liners:
        modes += model_bore("engine_block", g, mat, cyc, eng, temp, starts_h)
    return modes


def model_head(g: PartGeom, mat, cyc, eng, temp, starts_h):
    """Valve bridge thermal fatigue (restrained expansion, E a dT / (1-v)) + fire deck bending."""
    B = g.bore
    dT = 30 + 70 * cyc.load
    sth = 0.5 * mat.E * 1e3 * mat.cte * 1e-6 * dT / (1 - mat.poisson)
    p = cyc.pressure_bar.max() / 10
    t = 0.12 * B
    sdeck = 0.3 * p * (0.5 * B / t) ** 2
    return [
        _fatigue("cylinder_head", "valve-bridge thermal fatigue", "bridge between exhaust valve seats", mat,
                 sth / 2, 0.0, starts_h + 1.0, temp, smax=sth / 3, surface="as-cast",
                 note="one thermal cycle per start / big load change; thermal stress is secondary "
                      "(self-limiting), so the static check uses the 3Sm rule"),
        _fatigue("cylinder_head", "fire-deck bending", "combustion chamber roof", mat, sdeck / 2, sdeck / 2,
                 cyc.rpm * 30, temp, smax=sdeck, surface="as-cast", d_eff=t, gas=True),
    ]


def bolt_preload(mat, d: float, torque_to_yield: bool = False) -> tuple[float, float]:
    """Preload: 75% of proof load (Shigley) or ~90% of yield for torque-to-yield head bolts."""
    At = 0.78 * math.pi / 4 * d * d
    if torque_to_yield:
        return 0.9 * mat.yield_strength * At, At
    proof = 0.85 * mat.yield_strength
    return 0.75 * proof * At, At


def _bolt_mode(mat, d, Fi, At, P, C, cph, temp, name, note):
    """Shigley ch. 8 preloaded bolt: fatigue factor from the preload line,
    n_f = Se (Sut - si) / (Sut sa + Se (sm - si)), with the fully corrected endurance
    limit of rolled threads (~0.155 Sut: 129 MPa for 8.8, 162 MPa for 10.9).
    Static check = joint separation n0 = Fi / (P (1 - C)); torque-to-yield bolts are meant to
    sit at yield, so yield itself is not a failure."""
    from .fatigue import sn_cycles
    kd = mat.strength_factor(temp)
    sut = mat.uts * kd
    se = 0.155 * mat.uts * kd
    si = Fi / At
    sa = C * P / (2 * At)
    sm = sa + si
    n_f = se * (sut - si) / (sut * sa + se * (sm - si)) if sa > 0 else math.inf
    s_ar = se / n_f if math.isfinite(n_f) and n_f > 0 else 0.0
    ferrous = mat.category not in NON_FERROUS
    cycles = sn_cycles(s_ar, sut, se, ferrous) if s_ar > 0 else math.inf
    slope = local_slope(s_ar, sut, se, ferrous)
    over = max(0.0, temp - mat.max_temp)
    cycles *= math.exp(-over / 30.0)
    n0 = Fi / (P * (1 - C)) if P > 0 else math.inf
    return Mode(part="bolt", name=name, location="first engaged thread", kind="fatigue", cycles_per_hour=cph,
                sigma_a=sa, sigma_m=sm, sigma_max=sm + sa, cycles_to_failure=cycles, slope=slope,
                safety_fatigue=n_f, safety_static=n0, rate=(cph / cycles) if cycles > 0 else math.inf,
                temp_c=temp, gas_driven=True, note=note)


def model_bolt(g: PartGeom, mat, cyc, eng, temp, starts_h, bolts_per_bore=4.0, role="head", n_joint=6):
    """Head bolts carry the combustion load through the joint; other bolts (flywheel, covers)
    are friction joints: the check there is slip under peak torque."""
    B = g.bore
    d = g.get("shank_diameter_mm", 0.1 * B)
    if role != "head":
        Fi, At = bolt_preload(mat, d)
        T_peak = abs(cyc.journal_torque_mean) + cyc.journal_torque_range / 2
        r_bc = 0.55 * g.get("main_journal_diameter_mm", 0.65 * B) / 1000
        slip = 0.15 * Fi * n_joint * r_bc / max(T_peak, 1e-6)
        m = _bolt_mode(mat, d, Fi, At, 0.05 * Fi, 0.25, cyc.rpm * 60, temp, "thread-root fatigue",
                       f"friction joint, preload {Fi / 1000:.1f} kN")
        m.safety_static = math.inf
        return [m, Mode(part="bolt", name="joint slip check", location="clamped faces", kind="check",
                        safety_static=slip, margin=slip, temp_c=temp,
                        note=f"slip margin {slip:.2f}; below 1 the joint relies on dowels "
                             "(expect fretting/loosening unless dowelled)")]
    Fi, At = bolt_preload(mat, d, torque_to_yield=True)
    F_gas = cyc.pressure_bar.max() / 10 * math.pi / 4 * (B + 6) ** 2
    P = F_gas / bolts_per_bore
    C = 0.25
    return [_bolt_mode(mat, d, Fi, At, P, C, cyc.rpm * 30, temp, "thread-root fatigue",
                       f"torque-to-yield preload {Fi / 1000:.1f} kN; external load {P / 1000:.1f} kN, C = {C}")]


def model_gasket(g: PartGeom, mat, cyc, eng, temp, starts_h, bolt_mat=None, bolt_d=None, bolts_per_bore=4.0):
    """Combustion seal: 40% of the clamp load sits on the fire ring; it relaxes with time
    (embedding + creep, log-time). Leaks once gas lift exceeds the remaining clamp."""
    B = g.bore
    d = bolt_d or 0.13 * B   # no head bolts in the model: assume typical ~M11 for an 86 mm bore
    from ..materials import get as _get
    Fi, _ = bolt_preload(bolt_mat or _get("steel_bolt109"), d, torque_to_yield=True)
    clamp = 0.4 * Fi * bolts_per_bore
    F_gas = cyc.pressure_bar.max() / 10 * math.pi / 4 * (B + 6) ** 2
    margin0 = clamp / F_gas
    return [Mode(part="head_gasket", name="combustion seal", location="fire ring around the bore",
                 kind="sealing", margin=margin0, temp_c=temp, gas_driven=True,
                 note=f"clamp {clamp / 1000:.0f} kN vs gas lift {F_gas / 1000:.0f} kN at peak pressure")]


def gasket_relaxation(hours: np.ndarray, temp: float) -> np.ndarray:
    """Fraction of gasket clamp lost: initial embedding + log-time creep (hotter = faster)."""
    k = 0.03 * (1 + max(temp - 100, 0) / 100)
    return 0.08 + k * np.log10(1 + hours)


def model_flywheel(g: PartGeom, mat, cyc, eng, temp, starts_h):
    """Rotating disc with a central bore: peak hoop stress at the bore = (3+v)/4 rho w^2 R^2."""
    R = g.get("tip_diameter_mm", g.get("diameter_mm", 2.8 * g.bore)) / 2000
    w_max = eng.max_rpm * 1.15 * 2 * math.pi / 60
    w_op = cyc.rpm * 2 * math.pi / 60
    s_op = (3 + mat.poisson) / 4 * mat.density * w_op ** 2 * R ** 2 / 1e6
    s_burst = (3 + mat.poisson) / 4 * mat.density * w_max ** 2 * R ** 2 / 1e6
    m = _fatigue("flywheel", "bore hoop-stress fatigue", "hub bore / bolt holes", mat, 1.5 * s_op / 2,
                 1.5 * s_op / 2, starts_h + 20.0, temp, smax=1.5 * s_burst, surface="machined",
                 note="cycles = starts + big speed changes; static check at 115% max rpm")
    return [m]


def model_gear(g: PartGeom, mat, cyc, eng, temp, starts_h, spring_force=None):
    """Lewis tooth bending + Hertz contact (pitting) for the timing drive."""
    B = g.bore
    m_mod = g.get("module_mm", 1.5)
    z = g.get("teeth", 40)
    b = g.get("face_width_mm", 0.12 * B)
    d = m_mod * z
    F_s = spring_force if spring_force else 700.0 * (B / 86) ** 2
    T = 4 * F_s * 0.04 * B / 1000                     # N-m, cam drive torque peak
    Ft = T * 1000 / (d / 2)
    V = math.pi * d / 1000 * cyc.rpm / 60
    Kv = (6.1 + V) / 6.1
    Y = 0.154 - 0.912 / max(z, 12) + 0.2
    sb = Kv * Ft / (b * m_mod * Y)
    sh = 191 * math.sqrt(max(Kv * Ft / (b * d) * 3, 1e-9))
    s_allow = 1.9 * mat.hardness_hv + 250
    cph = cyc.rpm * 60 * 0.5
    modes = [_fatigue("timing_gear", "tooth root bending (Lewis)", "tooth root fillet", mat, sb / 2, sb / 2,
                      cph, temp, smax=sb, surface="machined", d_eff=m_mod * 2.25)]
    cyc_p = 1e8 * (s_allow / sh) ** 8
    modes.append(Mode(part="timing_gear", name="flank pitting", location="tooth flank at pitch line",
                      kind="fatigue", cycles_per_hour=cph, sigma_a=sh / 2, sigma_m=sh / 2, sigma_max=sh,
                      cycles_to_failure=cyc_p, slope=8.0, safety_fatigue=s_allow / sh,
                      safety_static=1.6 * s_allow / sh, rate=cph / cyc_p, temp_c=temp))
    return modes


def model_plug(g: PartGeom, mat, cyc, eng, temp, starts_h):
    """Electrode erosion: ~6.7e-9 mm of gap per spark for nickel electrodes (≈30,000 km);
    precious-metal tips last 3-4x longer."""
    rate = 6.7e-9 * 1000 * cyc.rpm * 30 * (0.6 + 0.6 * cyc.load)
    return [Mode(part="spark_plug", name="electrode erosion", location="centre / ground electrode",
                 kind="erosion", rate=rate, limit=300.0, temp_c=temp,
                 note="nickel electrodes; iridium/platinum tips last 3-4x longer")]


def model_panel(part: str, g: PartGeom, mat, cyc, eng, temp, starts_h):
    """Thin panels (pans, covers): bending from engine vibration (a ~ 3-15 g)."""
    a = 9.81 * (3 + 12 * (cyc.rpm / eng.max_rpm) ** 2)
    t = max(g.get("wall_mm", 3.0), 1.0)
    span = 0.4 * g.get("width_mm", 300)
    s = 0.5 * mat.density * a * (span / 1000) ** 2 / (t / 1000) / 1e6
    return [_fatigue(part, "vibration fatigue", "panel centre / flange", mat, s, 0.0, cyc.rpm * 60, temp,
                     smax=s, surface="as-cast", d_eff=t)]


def part_modes(label: str, g: PartGeom, mat: Material, cyc: CycleResult, eng: EngineModel,
               starts_h: float, ctx: dict) -> list[Mode]:
    """Failure modes of one part at one operating point. Design modifiers in ctx:
    design_fatigue {mode substring | "*": factor}, design_wear (wear-rate factor, coatings),
    temp_delta (deg C cooler, e.g. piston oil jets)."""
    token = _FATIGUE_FACTORS.set(ctx.get("design_fatigue"))
    try:
        modes = _part_modes(label, g, mat, cyc, eng, starts_h, ctx)
    finally:
        _FATIGUE_FACTORS.reset(token)
    wf = ctx.get("design_wear")
    if wf:
        for m in modes:
            if m.kind in ("wear", "erosion"):
                m.rate *= float(wf)
    return modes


def _part_modes(label: str, g: PartGeom, mat: Material, cyc: CycleResult, eng: EngineModel,
                starts_h: float, ctx: dict) -> list[Mode]:
    rpm_frac = cyc.rpm / eng.max_rpm
    sub = ctx.get("subtype", "")
    temp = (ctx.get("temp_override") or op_temperature(label, cyc.load, rpm_frac, sub)) - float(ctx.get("temp_delta") or 0.0)
    v = ctx.get("v_layout", False)
    sf = ctx.get("spring_force")
    if label == "piston":
        return model_piston(g, mat, cyc, eng, temp, starts_h)
    if label == "piston_pin":
        return model_pin(g, mat, cyc, eng, temp, starts_h)
    if label == "connecting_rod":
        return model_rod(g, mat, cyc, eng, temp, starts_h)
    if label == "crankshaft":
        return model_crank(g, mat, cyc, eng, temp, starts_h)
    if label == "valve_spring":
        return model_spring(g, mat, cyc, eng, temp, starts_h)
    if label == "valve":
        return model_valve(g, mat, cyc, eng, temp, starts_h, subtype=sub or "intake")
    if label == "camshaft":
        return model_cam(g, mat, cyc, eng, temp, starts_h, spring_force=sf)
    if label == "main_bearing":
        return model_bearing(g, mat, cyc, eng, temp, starts_h, v_layout=v)
    if label == "piston_ring":
        return model_ring(g, mat, cyc, eng, temp, starts_h)
    if label == "cylinder_liner":
        return model_bore("cylinder_liner", g, mat, cyc, eng, temp, starts_h)
    if label == "engine_block":
        return model_block(g, mat, cyc, eng, temp, starts_h, has_liners=ctx.get("has_liners", True), v_layout=v)
    if label == "cylinder_head":
        return model_head(g, mat, cyc, eng, temp, starts_h)
    if label == "bolt":
        return model_bolt(g, mat, cyc, eng, temp, starts_h, bolts_per_bore=ctx.get("bolts_per_bore", 4.0),
                          role=ctx.get("bolt_role", "head"), n_joint=ctx.get("joint_bolts", 6))
    if label == "head_gasket":
        return model_gasket(g, mat, cyc, eng, temp, starts_h, bolt_mat=ctx.get("bolt_material"),
                            bolt_d=ctx.get("bolt_d"), bolts_per_bore=ctx.get("bolts_per_bore", 4.0))
    if label == "flywheel":
        return model_flywheel(g, mat, cyc, eng, temp, starts_h)
    if label == "timing_gear":
        return model_gear(g, mat, cyc, eng, temp, starts_h, spring_force=sf)
    if label == "spark_plug":
        return model_plug(g, mat, cyc, eng, temp, starts_h)
    if label in ("oil_pan", "valve_cover"):
        return model_panel(label, g, mat, cyc, eng, temp, starts_h)
    return []
