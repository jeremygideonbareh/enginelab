"""End-to-end and unit tests.  Run: python -m pytest -q"""
import math
import os
import tempfile
import warnings
from collections import Counter

import numpy as np
import pytest
import trimesh

warnings.filterwarnings("ignore")

from enginelab import materials
from enginelab.features import shape_features
from enginelab.generator import LABELS, random_engine, randomly_posed
from enginelab.io import load_file
from enginelab.pipeline import analyze_file
from enginelab.segment import extract_parts, segment_fused
from enginelab.sim.cycle import EngineModel, run_cycle
from enginelab.sim.fatigue import evaluate, fatigue_fraction, goodman, sn_cycles
from enginelab.sim.fea import solve
from enginelab.specs import crank_measurements, part_specs

UNSEEN = [1001, 1002, 1003]  # training uses seeds 0..299


@pytest.fixture(scope="module")
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------- geometry
def test_generator_parts_are_closed_solids():
    for seed in range(5):
        for label, m in random_engine(seed).parts:
            assert label in LABELS
            assert m.is_watertight and m.volume > 0, label


@pytest.mark.parametrize("fmt", ["stl", "glb", "obj"])
def test_every_part_is_recovered_from_a_single_file(tmpdir, fmt):
    eng = randomly_posed(random_engine(1002), 5)
    path = os.path.join(tmpdir, f"e.{fmt}")
    if fmt == "stl":
        eng.merged().export(path)
    else:
        eng.scene().export(path)
    parts = extract_parts(load_file(path).parts)
    assert len(parts) == len(eng.parts)


def test_units_are_detected(tmpdir):
    eng = random_engine(1003)
    m = eng.merged().copy()
    m.apply_scale(0.001)  # metres
    path = os.path.join(tmpdir, "metres.stl")
    m.export(path)
    lr = load_file(path)
    assert lr.units == "m"
    ext = max(p.mesh.extents.max() for p in lr.parts)
    assert ext > 50  # back in millimetres


def test_step_files_keep_solid_bodies(tmpdir):
    import gmsh
    path = os.path.join(tmpdir, "asm.step")
    gmsh.initialize(["-noenv"], readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("a")
        gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, 200, 10)       # shaft
        gmsh.model.occ.addBox(-60, -60, 220, 120, 120, 40)        # plate
        gmsh.model.occ.addSphere(150, 0, 0, 25)                   # ball
        gmsh.model.occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()
    lr = load_file(path)
    assert len(lr.parts) == 3
    vols = sorted(round(abs(p.mesh.volume)) for p in lr.parts)
    expect = sorted([round(math.pi * 100 * 200), 120 * 120 * 40, round(4 / 3 * math.pi * 25 ** 3)])
    for v, e in zip(vols, expect):
        assert abs(v - e) / e < 0.03


def test_fused_mesh_is_cut_into_pieces():
    a = trimesh.creation.box((100, 100, 20))
    b = trimesh.creation.cylinder(15, 120)
    b.apply_translation((0, 0, 70))
    fused = trimesh.boolean.union([a, b], engine="manifold")
    pieces = segment_fused(fused, max_parts=6)
    assert 2 <= len(pieces) <= 6


# ---------------------------------------------------------------- identification + specs
@pytest.mark.parametrize("seed", UNSEEN)
def test_unseen_engines_identified_and_measured(tmpdir, seed):
    eng = randomly_posed(random_engine(seed), seed)
    path = os.path.join(tmpdir, f"u{seed}.stl")
    eng.merged().export(path)
    asm = analyze_file(path)
    truth = Counter(l for l, _ in eng.parts)
    found = Counter(p.label for p in asm.parts)
    wrong = sum(abs(truth[k] - found.get(k, 0)) for k in truth) / 2
    assert wrong / len(eng.parts) <= 0.02, (truth, found)
    e, p = asm.engine, eng.params
    assert abs(e["bore_mm"] - p.bore) / p.bore < 0.01
    assert abs(e["stroke_mm"] - p.stroke) / p.stroke < 0.01
    assert abs(e["rod_length_mm"] - p.rod_length) / p.rod_length < 0.01
    assert e["cylinders"] == p.n_cyl


def test_spring_and_gear_measurements():
    from enginelab.generator import gear_section, extrude_yz, to_trimesh, helix_spring, CS
    spring = helix_spring(12.0, 2.0, 60, 8)
    s = part_specs("valve_spring", spring, shape_features(spring))
    assert abs(s["wire_diameter_mm"] - 4.0) < 0.3
    assert abs(s["mean_coil_diameter_mm"] - 24.0) < 1.0
    gear = to_trimesh(extrude_yz(gear_section(37, 40, 44) - CS.circle(10, 24), 12))
    g = part_specs("timing_gear", gear, shape_features(gear))
    assert g["teeth"] == 37


# ---------------------------------------------------------------- physics
def test_cycle_matches_typical_engines():
    na = run_cycle(EngineModel(bore_mm=86, stroke_mm=86, rod_mm=143, n_cyl=4), 6000, 1.0).stats
    assert 64 < na["peak_pressure_bar"] < 72
    assert 90 < na["indicated_power_kw"] < 150          # 2.0 L NA petrol
    turbo = run_cycle(EngineModel(bore_mm=110, stroke_mm=135, rod_mm=215, n_cyl=6, compression_ratio=17,
                                  fuel="diesel", boost_bar=2.0, max_rpm=2200, m_piston_kg=2.5, m_rod_kg=2.8),
                      1800, 1.0).stats
    assert 180 < turbo["peak_pressure_bar"] < 230
    assert 1200 < turbo["indicated_torque_nm"] < 2000   # 7.7 L turbo diesel


def test_fatigue_basics():
    steel = materials.get("steel_4140")
    assert goodman(100, 0, 1000) == 100
    assert goodman(100, 500, 1000) == pytest.approx(200)
    se = evaluate(steel, 1, 0).Se
    f = fatigue_fraction(steel.uts)
    assert sn_cycles(f * steel.uts, steel.uts, se, True) == pytest.approx(1e3, rel=0.05)
    assert sn_cycles(se * 1.0001, steel.uts, se, True) == pytest.approx(1e6, rel=0.01)


def test_fea_matches_beam_theory():
    steel = materials.get("steel_4140")
    bar = trimesh.creation.box((20, 20, 200))
    L, I = 200, 20 ** 4 / 12
    r = solve(bar, steel, [0, 0, 1], "bending", 1000.0)
    beam = 1000 * ((L - 0.04 * L) - 0.14 * L) * 10 / I
    assert abs(r["peak_von_mises_mpa"] - beam) / beam < 0.06
    r = solve(bar, steel, [0, 0, 1], "axial", 40000.0)
    assert abs(r["peak_von_mises_mpa"] - 100) / 100 < 0.06


def test_life_simulation_is_sane(tmpdir):
    eng = random_engine(1)
    path = os.path.join(tmpdir, "sim.stl")
    eng.merged().export(path)
    asm = analyze_file(path)
    car = asm.run_simulation("automotive", hours=5000, samples=200)
    race = asm.run_simulation("racing", hours=5000, samples=200)
    assert car["system"]["b10_hours"] > 2000
    assert race["system"]["b10_hours"] < car["system"]["b10_hours"]
    plugs = [p for p in car["parts"] if p["label"] == "spark_plug"]
    assert plugs and plugs[0]["service_item"]


def test_material_ranking_prefers_piston_alloys(tmpdir):
    eng = random_engine(1)
    path = os.path.join(tmpdir, "rec.stl")
    eng.merged().export(path)
    asm = analyze_file(path)
    pid = next(p.id for p in asm.parts if p.label == "piston")
    ranking = asm.recommend(pid)
    assert ranking[0]["key"] in ("al_4032", "al_2618")
