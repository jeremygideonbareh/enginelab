"""Engine cycle: thermodynamics + slider-crank dynamics.

Cylinder pressure comes from a single-zone first-law model with a Wiebe
heat-release curve, calibrated so full load reaches the target peak pressure.
The slider-crank gives piston motion; gas and inertia forces give rod force,
side thrust, crankpin load, crank torque and the torque carried by each main
journal (all cylinders phased for even firing).

Angles in degrees, 0 = firing TDC, cycle from -360 to +360.
SI units internally; results reported in N, N-m, bar, kW.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

DEG = math.pi / 180


@dataclass
class EngineModel:
    bore_mm: float = 86.0
    stroke_mm: float = 86.0
    rod_mm: float = 145.0
    n_cyl: int = 4
    compression_ratio: float = 10.5
    fuel: str = "gasoline"          # gasoline | diesel
    boost_bar: float = 0.0          # gauge boost at full load
    max_rpm: float = 6500.0
    peak_pressure_bar: float | None = None   # full-load peak; default from fuel/boost/CR
    m_piston_kg: float = 0.45       # piston + pin + rings
    m_rod_kg: float = 0.55
    rod_big_end_frac: float = 0.67  # share of rod mass treated as rotating
    crankpin_d_mm: float | None = None
    main_d_mm: float | None = None

    def full_load_peak_bar(self) -> float:
        if self.peak_pressure_bar:
            return self.peak_pressure_bar
        # Typical full-load peaks: NA petrol ~65-75 bar, turbo petrol ~100-120,
        # NA diesel ~120-140, turbo diesel ~160-220.
        if self.fuel == "diesel":
            return 130.0 * (self.compression_ratio / 17.0) ** 0.5 * (1.0 + 0.28 * self.boost_bar)
        return 68.0 * (self.compression_ratio / 10.5) ** 0.8 * (1.0 + 0.6 * self.boost_bar)

    @property
    def area_m2(self) -> float:
        return math.pi / 4 * (self.bore_mm / 1000) ** 2

    @property
    def displacement_l(self) -> float:
        return self.area_m2 * self.stroke_mm / 1000 * self.n_cyl * 1000


@dataclass
class CycleResult:
    rpm: float
    load: float
    theta: np.ndarray
    pressure_bar: np.ndarray
    piston_accel: np.ndarray       # m/s^2, + towards the crank
    rod_force: np.ndarray          # N, + = compression
    side_force: np.ndarray         # N
    pin_force: np.ndarray          # N, magnitude on the crankpin bearing
    torque_cyl: np.ndarray         # N-m, one cylinder
    journal_torque_range: float    # N-m, worst main journal torque range (max - min)
    journal_torque_mean: float
    stats: dict = field(default_factory=dict)


def wiebe(theta: np.ndarray, soc: float, duration: float, a: float = 5.0, m: float = 2.0) -> np.ndarray:
    x = np.clip((theta - soc) / duration, 0, None)
    return 1 - np.exp(-a * x ** (m + 1))


def run_cycle(eng: EngineModel, rpm: float, load: float, step_deg: float = 0.5) -> CycleResult:
    load = float(np.clip(load, 0.02, 1.2))
    r = eng.stroke_mm / 2000
    L = eng.rod_mm / 1000
    A = eng.area_m2
    lam = r / L
    omega = rpm * 2 * math.pi / 60
    theta = np.arange(-360.0, 360.0, step_deg)
    th = theta * DEG
    s = np.sin(th)
    x = r * (1 - np.cos(th)) + L - np.sqrt(L ** 2 - (r * s) ** 2)    # distance from TDC
    Vd = A * 2 * r
    Vc = Vd / (eng.compression_ratio - 1)
    V = Vc + A * x
    dV = np.gradient(V, th)

    diesel = eng.fuel == "diesel"
    dur = 60.0 if diesel else 50.0
    gamma, n_c = 1.30, 1.32
    i_bdc = np.searchsorted(theta, -180.0)
    i_evo = np.searchsorted(theta, 140.0)
    i_exh = np.searchsorted(theta, 180.0)
    dth = step_deg * DEG

    def intake(ld: float) -> float:
        """Intake manifold pressure: throttled at part load for spark ignition."""
        if diesel:
            return 1.0e5 * (1.0 + eng.boost_bar * ld)
        return 1.0e5 * (0.3 + 0.7 * min(ld, 1.0) + eng.boost_bar * max(ld - 0.5, 0) * 2)

    def heat(ld: float) -> float:
        """Fuel energy per cycle, limited by trapped air (volumetric efficiency 0.9,
        diesel lambda >= 1.45). 25% of it is lost to the walls (no separate heat-transfer model)."""
        m_air = 0.9 * intake(ld) * (Vc + Vd) / (287.0 * (330.0 if diesel else 320.0))
        if diesel:
            return ld * m_air / (14.5 * 1.45) * 42.5e6 * 0.75
        return m_air / 14.7 * 43.4e6 * 0.75

    def pressure(Q: float, soc: float, p_in: float, p_exh: float) -> np.ndarray:
        dxb = np.gradient(wiebe(theta, soc, dur), th)
        p = np.empty_like(theta)
        p[:i_bdc] = p_in
        p[i_bdc:] = p_in * (V[i_bdc] / V[i_bdc:]) ** n_c
        for i in range(np.searchsorted(theta, soc), i_evo):
            p[i + 1] = p[i] + ((gamma - 1) / V[i] * Q * dxb[i] - gamma * p[i] / V[i] * dV[i]) * dth
        blow = np.linspace(0, 1, i_exh - i_evo)   # blowdown, then exhaust stroke
        p[i_evo:i_exh] = p[i_evo] * (1 - blow) + p_exh * blow
        p[i_exh:] = p_exh
        return p

    def exhaust(ld: float) -> float:
        return 1.1e5 + 0.3e5 * eng.boost_bar * ld

    # Combustion timing is set so full load reaches the target peak pressure.
    key = (eng.bore_mm, eng.stroke_mm, eng.rod_mm, eng.compression_ratio, eng.fuel, eng.boost_bar,
           eng.full_load_peak_bar(), step_deg)
    if key not in _TIMING:
        target = eng.full_load_peak_bar() * 1e5
        soc_c = _calibrate_timing(
            lambda soc: pressure(heat(1.0), soc, intake(1.0), exhaust(1.0)).max(), target)
        q_scale = 1.0
        if pressure(heat(1.0), soc_c, intake(1.0), exhaust(1.0)).max() < 0.99 * target:
            # Even fully advanced timing cannot reach the requested peak with the trapped air:
            # the user asked for a harder test, so add fuel energy until it does.
            lo_q, hi_q = 1.0, 1.0
            while pressure(heat(1.0) * hi_q, soc_c, intake(1.0), exhaust(1.0)).max() < target and hi_q < 64:
                hi_q *= 2
            for _ in range(40):
                mid = 0.5 * (lo_q + hi_q)
                if pressure(heat(1.0) * mid, soc_c, intake(1.0), exhaust(1.0)).max() < target:
                    lo_q = mid
                else:
                    hi_q = mid
            q_scale = hi_q
        _TIMING[key] = (soc_c, q_scale)
    soc, q_scale = _TIMING[key]
    p_in = intake(load)
    p = pressure(heat(load) * q_scale, soc, p_in, exhaust(load))

    # Kinematics (exact, by differentiation of x on the fine grid).
    vel = np.gradient(x, th) * omega
    acc = np.gradient(np.gradient(x, th), th) * omega ** 2
    m_rec = eng.m_piston_kg + eng.m_rod_kg * (1 - eng.rod_big_end_frac)
    m_rot = eng.m_rod_kg * eng.rod_big_end_frac
    Fg = (p - 1.0e5) * A
    Fp = Fg - m_rec * acc                           # along the cylinder, + towards crank
    beta = np.arcsin(lam * s)
    F_rod = Fp / np.cos(beta)
    F_side = Fp * np.tan(beta)
    F_t = F_rod * np.sin(th + beta)                 # tangential at the crankpin
    F_r = F_rod * np.cos(th + beta)                 # radial (towards crank centre)
    torque = F_t * r
    # Crankpin bearing: rod force plus the centrifugal load of the rotating rod mass.
    F_cent = m_rot * r * omega ** 2
    pin = np.sqrt(F_t ** 2 + (F_r - F_cent) ** 2)

    # Multi-cylinder torque: even firing; the worst main journal carries the
    # accumulated torque of every cylinder in front of it.
    n = eng.n_cyl
    shift = int(round((720.0 / n) / step_deg)) if n > 0 else 0
    per_cyl = [np.roll(torque, k * shift) for k in range(n)]
    acc_t = np.cumsum(per_cyl, axis=0)
    ranges = acc_t.max(1) - acc_t.min(1)
    j = int(np.argmax(ranges))

    work = float(np.trapezoid(p, V)) if hasattr(np, "trapezoid") else float(np.trapz(p, V))
    imep = work / Vd
    power_kw = work * n * rpm / 120 / 1000
    stats = {
        "rpm": rpm, "load": load,
        "peak_pressure_bar": float(p.max() / 1e5),
        "peak_pressure_angle_deg": float(theta[np.argmax(p)]),
        "imep_bar": float(imep / 1e5),
        "indicated_power_kw": float(power_kw),
        "indicated_torque_nm": float(power_kw * 1000 / omega) if omega > 0 else 0.0,
        "mean_piston_speed_ms": float(2 * (2 * r) * rpm / 60),
        "max_piston_speed_ms": float(np.abs(vel).max()),
        "max_piston_accel_g": float(np.abs(acc).max() / 9.81),
        "max_rod_compression_kn": float(F_rod.max() / 1000),
        "max_rod_tension_kn": float(-F_rod.min() / 1000),
        "max_side_force_kn": float(np.abs(F_side).max() / 1000),
        "max_crankpin_load_kn": float(pin.max() / 1000),
        "journal_torque_range_nm": float(ranges[j]),
        "intake_pressure_bar": p_in / 1e5,
        "start_of_combustion_deg": soc,
    }
    return CycleResult(rpm=rpm, load=load, theta=theta, pressure_bar=p / 1e5, piston_accel=acc,
                       rod_force=F_rod, side_force=F_side, pin_force=pin, torque_cyl=torque,
                       journal_torque_range=float(ranges[j]), journal_torque_mean=float(acc_t[j].mean()),
                       stats=stats)


_TIMING: dict = {}


def _calibrate_timing(peak_fn, target: float, lo: float = -30.0, hi: float = 25.0) -> float:
    """Start of combustion (deg) giving the target peak pressure; earlier = higher peak."""
    if peak_fn(lo) <= target:
        return lo
    if peak_fn(hi) >= target:
        return hi
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if peak_fn(mid) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.05:
            break
    return 0.5 * (lo + hi)
