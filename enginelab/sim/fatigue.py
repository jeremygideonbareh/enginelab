"""Stress-life fatigue (Shigley / Marin method).

Se  = ka kb kc kd ke kf * Se'          (Marin factors)
S-N = straight line in log-log from f*Sut at 1e3 cycles to Se at 1e6
      (ferrous) or at 5e8 (non-ferrous, no true endurance limit).
Below the endurance limit, ferrous parts keep a shallow slope (k* = 45, about
5% strength loss per decade; Sonsino 2007, very-high-cycle regime), so damage
from scattered or variable loads is not ignored but steel keeps its
practically unlimited life under constant loads below the knee.
Mean stress: Goodman (tensile mean), none for compressive mean.
Damage: Palmgren-Miner.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..materials import Material

SURFACE = {  # Shigley Table 6-2: ka = a * Sut^b (Sut in MPa)
    "ground": (1.58, -0.085),
    "machined": (4.51, -0.265),
    "hot-rolled": (57.7, -0.718),
    "forged": (272.0, -0.995),
    "as-cast": (57.7, -0.718),
}
RELIABILITY = {0.5: 1.0, 0.9: 0.897, 0.95: 0.868, 0.99: 0.814, 0.999: 0.753, 0.9999: 0.702}
K_VHCF = 45.0   # slope beyond the knee for steels / irons (Sonsino)
NON_FERROUS = {"aluminium", "magnesium", "titanium", "polymer", "copper alloy", "bearing alloy"}


@dataclass
class FatigueResult:
    Se: float                 # corrected endurance limit, MPa
    sigma_ar: float           # equivalent fully-reversed amplitude, MPa
    cycles: float             # cycles to failure (inf if no damage)
    safety_fatigue: float     # Se / sigma_ar (infinite-life factor)
    safety_static: float      # yield / max stress
    marin: dict


def marin_factors(mat: Material, surface: str = "machined", d_eff_mm: float | None = None,
                  load: str = "bending", temp_c: float = 20.0, reliability: float = 0.99,
                  misc: float = 1.0) -> dict:
    a, b = SURFACE.get(surface, SURFACE["machined"])
    ka = min(1.0, a * mat.uts ** b)
    if mat.category in ("polymer", "ceramic", "bearing alloy"):
        ka = 1.0  # their endurance data already reflect the real surface
    if d_eff_mm is None or load == "axial":
        kb = 1.0
    elif d_eff_mm <= 51:
        kb = 1.24 * max(d_eff_mm, 2.79) ** -0.107
    else:
        kb = 1.51 * min(d_eff_mm, 254) ** -0.157
    kc = {"bending": 1.0, "axial": 0.85, "torsion": 0.59, "combined": 1.0}[load]
    kd = mat.strength_factor(temp_c)
    ke = RELIABILITY.get(reliability, 0.814)
    return {"ka": ka, "kb": kb, "kc": kc, "kd": kd, "ke": ke, "kf": misc}


def endurance_limit(mat: Material, **kw) -> tuple[float, dict]:
    f = marin_factors(mat, **kw)
    se = mat.endurance * f["ka"] * f["kb"] * f["kc"] * f["kd"] * f["ke"] * f["kf"]
    return se, f


def fatigue_fraction(sut: float) -> float:
    """Shigley Fig. 6-18: fraction of Sut reached at 1e3 cycles."""
    return float(np.interp(sut, [490, 1400], [0.9, 0.77]))


def sn_cycles(sigma_ar: float, sut: float, se: float, ferrous: bool) -> float:
    if sigma_ar <= 0:
        return math.inf
    s1 = fatigue_fraction(sut) * sut
    n_e = 1e6 if ferrous else 5e8
    if s1 <= se:
        s1 = se * 1.05
    k = math.log10(n_e / 1e3) / math.log10(s1 / se)   # S-N slope exponent: N ~ S^-k
    if sigma_ar >= s1:
        # Low-cycle region: continue the same line (capped at 1 cycle).
        return max(1.0, 1e3 * (s1 / sigma_ar) ** k)
    if sigma_ar >= se:
        return 1e3 * (s1 / sigma_ar) ** k
    if ferrous:
        return n_e * (se / sigma_ar) ** K_VHCF
    return n_e * (se / sigma_ar) ** k


def local_slope(sigma_ar: float, sut: float, se: float, ferrous: bool) -> float:
    """S-N exponent at this stress level (how strongly damage reacts to load scatter)."""
    if ferrous and sigma_ar < se:
        return K_VHCF
    return sn_slope(sut, se, ferrous)


def sn_slope(sut: float, se: float, ferrous: bool) -> float:
    s1 = max(fatigue_fraction(sut) * sut, se * 1.05)
    n_e = 1e6 if ferrous else 5e8
    return math.log10(n_e / 1e3) / math.log10(s1 / se)


def goodman(sa: float, sm: float, sut: float) -> float:
    """Equivalent fully reversed amplitude (Goodman); compressive mean gives no benefit."""
    if sm <= 0:
        return abs(sa)
    if sm >= sut:
        return math.inf
    return abs(sa) / (1 - sm / sut)


def evaluate(mat: Material, sa: float, sm: float, smax: float | None = None, **marin_kw) -> FatigueResult:
    se, f = endurance_limit(mat, **marin_kw)
    sut = mat.uts * f["kd"]
    sy = mat.yield_strength * f["kd"]
    sar = goodman(sa, sm, sut)
    ferrous = mat.category not in NON_FERROUS
    n = sn_cycles(sar, sut, se, ferrous) if math.isfinite(sar) else 1.0
    smax = abs(sm) + abs(sa) if smax is None else abs(smax)
    return FatigueResult(Se=se, sigma_ar=sar, cycles=n,
                         safety_fatigue=(se / sar) if sar > 0 else math.inf,
                         safety_static=(sy / smax) if smax > 0 else math.inf, marin=f)


def von_mises(sigma: float, tau: float) -> float:
    return math.sqrt(sigma ** 2 + 3 * tau ** 2)
