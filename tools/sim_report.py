"""Print every failure mode of a generated engine (with true labels) for sanity checking.

python tools/sim_report.py [seed] [duty] [hours]
"""
import math
import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from enginelab.generator import random_engine  # noqa: E402
from enginelab.io import RawPart  # noqa: E402
from enginelab import classify  # noqa: E402
from enginelab.classify import Prediction  # noqa: E402


def analyze_with_truth(seed: int):
    """Analyse a generated engine but take the true labels (isolates physics from the classifier)."""
    from enginelab import pipeline
    eng = random_engine(seed)
    raws = [RawPart(mesh=m, name=lab, source="cad-body") for lab, m in eng.parts]
    orig = pipeline.classify

    def truth(infos, ctx, names):
        return [Prediction(label=n, confidence=1.0, source="name", alternatives=[(n, 1.0)]) for n in names]
    pipeline.classify = truth
    try:
        asm = pipeline.analyze_parts(raws, f"engine_{seed}")
    finally:
        pipeline.classify = orig
    return eng, asm


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    duty = sys.argv[2] if len(sys.argv) > 2 else "automotive"
    hours = float(sys.argv[3]) if len(sys.argv) > 3 else 5000
    eng, asm = analyze_with_truth(seed)
    print("engine:", {k: asm.engine.get(k) for k in ("layout", "bore_mm", "stroke_mm", "rod_length_mm", "displacement_cc")})
    res = asm.run_simulation(duty, hours=hours, samples=300)
    print("model:", {k: round(v, 3) if isinstance(v, float) else v for k, v in res["engine_model"].items()})
    for op in res["operating_points"]:
        print(f"  {op['name']:18s} {op['rpm']:6.0f} rpm  load {op['load']:.2f}  pmax {op['peak_pressure_bar']:6.1f} bar"
              f"  {op['indicated_power_kw']:6.1f} kW  {op['indicated_torque_nm']:6.1f} Nm")
    seen = set()
    print(f"\n{'part':16s} {'mode':42s} {'kind':8s} {'sa':>7s} {'sm':>7s} {'SF_fat':>7s} {'SF_st':>7s} {'nominal h':>11s} {'Tmax':>5s}")
    for p in res["parts"]:
        key = (p["label"], p["material"])
        if key in seen:
            continue
        seen.add(key)
        for m in p["modes"]:
            nh = m["nominal_hours"]
            print(f"{p['label'][:16]:16s} {m['name'][:42]:42s} {m['kind'][:8]:8s} {m['sigma_a'] or 0:7.1f} {m['sigma_m'] or 0:7.1f} "
                  f"{(m['safety_fatigue'] or math.inf):7.2f} {(m['safety_static'] or math.inf):7.2f} "
                  f"{(nh if nh is not None else math.inf):11.0f} {m['temp_max']:5.0f}  {m.get('note') or ''}"[:220])
    s = res["system"]
    print("\nsystem B10", s["b10_hours"], "B50", s["b50_hours"], "P(fail by horizon)", s["prob_fail"])
    print("first failures:", s["first_failure_share"])
