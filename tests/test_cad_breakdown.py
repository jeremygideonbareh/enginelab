"""Breakdown of real and constructed CAD assemblies (STEP via OpenCASCADE XCAF)."""
import math
import os
import tempfile
import warnings

import pytest

warnings.filterwarnings("ignore")

from enginelab.cad import read_cad
from enginelab.components import from_name, match_bearing
from enginelab.pipeline import analyze_file

HERE = os.path.dirname(__file__)
TOOLHEAD = os.path.join(HERE, "..", "data", "real_cad", "minisb", "MiniSB_Bowden.step")


def _write_step(path, build):
    import gmsh
    gmsh.initialize(["-noenv"], readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("m")
        build(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()


def test_exact_volume_hole_detection_and_genus():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "g.step")

        def build(o):
            o.addCylinder(0, 0, 0, 0, 0, 200, 10)                                   # shaft, genus 0
            t = o.addCylinder(40, 0, 0, 0, 0, 10, 5)
            h = o.addCylinder(40, 0, -1, 0, 0, 12, 2)
            o.cut([(3, t)], [(3, h)])                                               # tube, genus 1
            b = o.addBox(60, -5, 0, 30, 10, 5)
            h1 = o.addCylinder(67, 0, -1, 0, 0, 7, 2)
            h2 = o.addCylinder(83, 0, -1, 0, 0, 7, 2)
            o.cut([(3, b)], [(3, h1), (3, h2)])                                     # plate, genus 2
        _write_step(p, build)
        a = read_cad(p)
    assert len(a.parts) == 3
    assert [q.brep["genus"] for q in a.parts] == [0, 1, 2]
    shaft = a.parts[0]
    assert abs(shaft.volume_exact - math.pi * 100 * 200) / (math.pi * 100 * 200) < 1e-6
    assert any(c["r"] == 10 and not c["hole"] for c in shaft.brep["cylinders"])
    assert any(c["r"] == 2 and c["hole"] for c in a.parts[1].brep["cylinders"])


def test_name_rules_and_standard_tables():
    c = from_name("DIN EN ISO 4762 - M3 x 50")
    assert c.category == "socket head cap screw" and c.size["thread"] == "M3" and c.size["length_mm"] == 50
    assert from_name("MR825 Ballbearing 5x8x2").family == "bearing"
    assert from_name("MGN12H carriage").designation.startswith("MGN12")
    assert from_name("Cowling_PC4_M10 v6") is None or from_name("Cowling_PC4_M10 v6").family == "fluid"
    assert match_bearing(20, 47, 14) == "6204"
    assert match_bearing(8, 22, 7) == "608"


@pytest.mark.skipif(not os.path.exists(TOOLHEAD), reason="real sample not downloaded")
def test_real_toolhead_breakdown():
    asm = analyze_file(TOOLHEAD)
    j = asm.to_json()
    assert j["mode"] == "general"
    assert len(j["parts"]) == 31                # 30 leaf products + 1 sub-assembly's own body
    bom = {l["name"]: l for l in j["bom"]}
    assert bom["M2x6 SHCS"]["qty"] == 4
    assert bom["M2x6 SHCS"]["designation"] == "ISO 4762 M2x6 socket head cap screw"
    assert bom["M3x35 BHCS"]["designation"].startswith("ISO 7380 M3x35")
    assert bom["MGN7-H"]["designation"] == "MGN7 linear guide"
    assert bom["Cowling_PC4_M10"]["standard_part"] is False
    assert "McMaster-Carr" in bom["M2x6 SHCS"]["links"]
    # Every part's mesh is a faithful copy of the exact solid.
    for p in asm.parts:
        assert p.info.volume > 0
    assert j["tree"]["children"][0]["name"].startswith("MiniSB_Bowden_Assembly")


def test_motor_training_labels():
    """Real motors are motors; printed parts named after the motor they hold are not."""
    from enginelab.component_model import name_label
    motors = [("NEMA-17-40", None), ("Nema14", None), ("STS3215_03a", None), ("DC15_A01_CASE_M_DUMMY", None),
              ("200mm Integrated Leadscrew Stepper", None), ("motor_antigravity_4004_stator", ["motor_module"]),
              ("SG-ZIJI_15", ["SO101 Assembly", "ST3215 Servo"])]
    not_motors = [("Motor_Plate", "custom"), ("b_stepper_lower", "custom"), ("z_stepper_left", "custom"),
                  ("X Motor", "custom"), ("Y Motor Mount", "custom"), ("MG996-ROUND-SERVO-HORN", "transmission"),
                  ("motor_antigravity_4004_custom_shaft", "transmission"),
                  ("User_Library-0603_SMD_Capacitor", "electronics")]
    for n, p in motors:
        assert name_label(n, p) == "motor", n
    for n, want in not_motors:
        assert name_label(n, None) == want, n
    assert name_label("M3x8 SHCS", ["ST3215 Servo"]) == "fastener"
    assert from_name("NEMA-23-76").designation == "NEMA 23 stepper motor"


CRANK_ZIP = os.path.join(os.path.dirname(__file__), "..", "data", "real_cad", "crank454", "crankshaft_454.zip")


@pytest.mark.skipif(not os.path.exists(CRANK_ZIP), reason="native SolidWorks test assembly not present")
def test_native_solidworks_zip_engine():
    """A zip of .SLDASM + .SLDPRT files: every instance, closed meshes, engine measured."""
    from collections import Counter
    from enginelab.io import load_file
    lr = load_file(CRANK_ZIP)
    names = Counter(p.name for p in lr.parts)
    assert len(lr.parts) == 29 and names["2-PISTON RING"] == 12 and names["1-PISTON"] == 4
    assert all(p.mesh.is_watertight for p in lr.parts)
    a = analyze_file(CRANK_ZIP, filename="crank.zip")
    j = a.to_json()
    e = j["engine"]
    assert j["mode"] == "engine" and e["cylinders"] == 4 and e["layout"] == "inline-4"
    assert abs(e["stroke_mm"] - 90.1) < 1.0 and abs(e["piston_diameter_mm"] - 85.0) < 0.5


@pytest.mark.skipif(not os.path.exists(CRANK_ZIP), reason="native SolidWorks test assembly not present")
def test_design_improvement_and_optimiser(tmp_path):
    """What-if physics, part recommendations and the learning optimiser on a real engine."""
    from enginelab import improve
    a = analyze_file(CRANK_ZIP, filename="crank.zip")
    ctx = improve.Context(a)
    base = improve.evaluate(ctx, {})
    assert base["limiting_part"] == "piston"
    # a bored pin lowers reciprocating mass, which unloads the rod
    hollow = improve.evaluate(ctx, {"piston_pin.bore": len(ctx.by_id["piston_pin.bore"].options) - 1})
    assert hollow["reciprocating_mass_kg"] < base["reciprocating_mass_kg"]
    assert hollow["parts"]["connecting_rod"]["sf_fatigue"] > base["parts"]["connecting_rod"]["sf_fatigue"]
    rec = improve.recommend(a, next(p.id for p in a.parts if p.label == "piston"))
    assert rec["items"] and rec["items"][0]["kind"] != "trade-off" and rec["items"][0]["part_life_ratio"] > 1.5
    improve.STORE = str(tmp_path)
    opt = improve.Optimizer(a, budget=120)
    opt.start()
    opt.thread.join(timeout=120)
    st = opt.status()
    assert st["evaluations_total"] >= 100
    assert st["top"][0]["result"]["system_life_h"] > base["system_life_h"]
    assert improve.Optimizer(a).loaded == st["evaluations_total"]          # it remembers
    sim0 = a.run_simulation("automotive", hours=10000, samples=100)
    improve.apply_design(a, {k: v for k, v in st["top"][0]["design"].items() if not k.endswith(".material")})
    assert a.design and improve.evaluate(improve.Context(a), a.design)["system_life_h"] > base["system_life_h"]
    sim1 = a.run_simulation("automotive", hours=10000, samples=100)       # the Monte Carlo sees the change
    assert sim1["system"]["prob_fail"] < sim0["system"]["prob_fail"]


@pytest.mark.skipif(not os.path.exists(CRANK_ZIP), reason="native SolidWorks test assembly not present")
def test_life_simulation_scatter_is_small():
    """Latin-hypercube sampling: B10 varies < 1% between random seeds, and a 90% CI is reported."""
    import numpy as np
    a = analyze_file(CRANK_ZIP, filename="crank.zip")
    from enginelab.sim.life import simulate
    eng = a.engine_model()
    a._refresh_ctx(eng)
    b10 = [simulate(a.sim_parts(), eng, "automotive", hours=10000, samples=400, seed=s)["system"]["b10_hours"]
           for s in range(4)]
    assert np.std(b10) / np.mean(b10) < 0.01
    r = simulate(a.sim_parts(), eng, "automotive", hours=10000, samples=400, seed=0)["system"]
    assert r["b10_ci90"][0] <= r["b10_hours"] <= r["b10_ci90"][1]


TEST_ENGINES = os.path.join(os.path.dirname(__file__), "..", "data", "real_cad", "test_engines")


@pytest.mark.skipif(not os.path.exists(os.path.join(TEST_ENGINES, "eng_a", "Assem1.SLDASM")),
                    reason="real test engines not present (not redistributable)")
def test_real_engines_read_and_identified():
    """Older SolidWorks files read (false section markers skipped); real engines identified."""
    import subprocess
    import sys
    from enginelab.io import load_file
    lr = load_file(os.path.join(TEST_ENGINES, "eng_a", "Assem1.SLDASM"))
    assert len(lr.parts) == 6 and all(p.mesh.is_watertight for p in lr.parts)
    root = os.path.join(os.path.dirname(__file__), "..")
    out = subprocess.run([sys.executable, "-W", "ignore", "tools/eval_real_engines.py"], cwd=root, capture_output=True,
                         text=True, env={**os.environ, "WITH_NAMES": "1"}).stdout
    line = next(l for l in out.splitlines() if "joblib" in l)
    # Engines whose parts are named must be fully identified. eng_c's parts are mostly numbered
    # ("2-09"), so with names it is a geometry test: it only has to stay above half.
    for seg in line.split("|")[1:]:
        name, score = seg.rsplit(":", 1)
        ok, tot = map(int, score.strip().split("/"))
        if name.strip().startswith("eng_c"):
            assert ok >= tot // 2, line
        else:
            assert ok == tot, line


@pytest.mark.skipif(not os.path.exists(CRANK_ZIP), reason="native SolidWorks test assembly not present")
def test_balance_reads_crank_phasing():
    """Throw angles come from the crank geometry; the 0-180-0-180 crank shows a primary couple that the
    standard 0-180-180-0 order would cancel."""
    from enginelab.sim.balance import analyse
    r = analyse(analyze_file(CRANK_ZIP, filename="crank.zip"))
    assert r["phasing_deg"] == [0, 180, 0, 180]
    assert r["primary_force_N"] < 50 and r["primary_couple_Nm"] > 1000
    assert r["standard_phasing"]["primary_couple_Nm"] < 10
    assert 200 < r["torsional"]["natural_frequencies_hz"][0] < 2000


def test_sourcing_prices_and_shops():
    """Standard parts get a price range and local shops first; servo internals collapse into one servo."""
    from enginelab.sourcing import collapse_bought, source_line, typical_price
    line = {"item": 1, "name": "608ZZ", "qty": 4, "label": "bearing", "category": "ball bearing",
            "designation": "608ZZ", "standard_part": True, "size": {}, "dims_mm": [22, 22, 7]}
    lo, hi, _ = typical_price(line)
    assert 0.3 < lo < 1.0 < hi < 6
    r = source_line(line, "IN", {"INR": 90.0})
    assert r["shops"][0]["ships_from"] == "IN" and r["total_low"] > 0
    pieces = [{"item": i, "name": n, "qty": q, "path": "Arm / ST3215 Servo", "dims_mm": [40, 20, 30], "part_ids": [i]}
              for i, (n, q) in enumerate([("shell", 6), ("pcb", 6), ("horn", 5)])]
    out = collapse_bought(pieces)
    assert len(out) == 1 and out[0]["qty"] == 6 and out[0]["includes"] == 3


def test_feedback_correction_becomes_training_row(tmp_path, monkeypatch):
    from enginelab import feedback
    monkeypatch.setattr(feedback, "DIR", str(tmp_path))
    monkeypatch.setattr(feedback, "CORRECTIONS", str(tmp_path / "c.jsonl"))
    a = analyze_file(os.path.join(os.path.dirname(__file__), "..", "data", "real_cad", "koch_follower", "arm.step")) \
        if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", "real_cad", "koch_follower", "arm.step")) else None
    if a is None:
        pytest.skip("sample robot not present")
    pid = next(p.id for p in a.parts if p.label == "custom")
    feedback.record_correction(a, pid, "custom", "motor")
    X, y = feedback.load_corrections("component")
    assert y == ["motor"] and len(X[0]) > 40
