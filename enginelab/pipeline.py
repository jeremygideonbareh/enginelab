"""End-to-end analysis: file -> parts -> identification -> specs -> engine model.

`analyze_file` returns an `Assembly` that keeps the meshes (for the viewer,
FEA and re-runs) and can produce JSON for the API.
"""
from __future__ import annotations

import math
import re
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
import trimesh

from . import materials
from .classify import Prediction, analyse_shapes, classify
from .features import ShapeInfo, copy_groups
from .io import RawPart, load_file
from .recommend import default_material, rank_materials, candidates
from .segment import extract_parts
from .breakdown import display_name
from .sim.cycle import EngineModel
from .sim.life import DUTY_CYCLES, SimPart, aggregate_modes, simulate, solve_operating_points
from .sim.parts import PartGeom, _spring_forces
from .specs import engine_specs, part_specs


@dataclass
class Part:
    id: int
    mesh: trimesh.Trimesh
    info: ShapeInfo
    name: str | None
    source: str
    pred: Prediction
    group: int
    label: str
    material: str
    specs: dict = field(default_factory=dict)
    ctx: dict = field(default_factory=dict)
    path: list = field(default_factory=list)
    definition: str | None = None
    color: tuple | None = None
    component: dict | None = None      # standard-part recognition + sourcing links
    brep: dict = field(default_factory=dict)
    path_ids: list = field(default_factory=list)
    com_mm: list | None = None         # centre of mass (world, mm)
    inertia_unit: list | None = None   # inertia about the COM per unit density (mm^5)

    def mass_props(self) -> dict:
        """Mass (kg), centre of mass (mm) and inertia about the COM (kg*mm^2) with the
        current material."""
        rho = materials.get(self.material).density * 1e-9      # kg/mm^3
        out = {"mass_kg": self.info.volume * rho}
        if self.com_mm is not None:
            out["com_mm"] = list(map(float, self.com_mm))
        if self.inertia_unit is not None:
            out["inertia_kgmm2"] = (np.asarray(self.inertia_unit) * rho).tolist()
        return out


@dataclass
class Assembly:
    id: str
    filename: str
    units: str
    parts: list[Part]
    engine: dict
    timings: dict
    warnings: list[str] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    last_sim: dict | None = None
    mode: str = "engine"               # engine | general
    tree: dict | None = None           # assembly tree (CAD files)
    info: dict = field(default_factory=dict)

    # ------------------------------------------------------------ engine model
    def engine_model(self, **overrides) -> EngineModel | None:
        e = self.engine
        if not e.get("cylinders"):
            return None
        bore = e.get("bore_mm") or e.get("piston_diameter_mm")
        stroke = e.get("stroke_mm") or bore
        rod = e.get("rod_length_mm") or 1.65 * stroke
        fuel = overrides.get("fuel", self.settings.get("fuel", "gasoline"))
        piston_speed_limit = 13.0 if fuel == "diesel" else 20.0
        max_rpm = overrides.get("max_rpm") or self.settings.get("max_rpm") or \
            round(min(9000.0, piston_speed_limit * 60 / (2 * stroke / 1000)), -2)
        m_pist, m_rod = self.moving_masses()
        return EngineModel(
            bore_mm=bore, stroke_mm=stroke, rod_mm=rod, n_cyl=int(e["cylinders"]),
            compression_ratio=overrides.get("compression_ratio", self.settings.get("compression_ratio",
                                                                                   17.0 if fuel == "diesel" else 10.5)),
            fuel=fuel, boost_bar=overrides.get("boost_bar", self.settings.get("boost_bar", 0.0)),
            max_rpm=float(max_rpm),
            peak_pressure_bar=overrides.get("peak_pressure_bar", self.settings.get("peak_pressure_bar")),
            m_piston_kg=m_pist, m_rod_kg=m_rod,
            crankpin_d_mm=e.get("crankpin_diameter_mm"), main_d_mm=e.get("main_journal_diameter_mm"))

    def moving_masses(self) -> tuple[float, float]:
        def mean_mass(label):
            ps = [p for p in self.parts if p.label == label]
            if not ps:
                return None
            return float(np.mean([p.info.volume * 1e-9 * materials.get(p.material).density for p in ps]))
        n = max(1, self.engine.get("cylinders", 1))
        rings = [p for p in self.parts if p.label == "piston_ring"]
        ring_mass = sum(p.info.volume * 1e-9 * materials.get(p.material).density for p in rings) / n
        piston = mean_mass("piston")
        pin = mean_mass("piston_pin") or 0.0
        bore = self.engine.get("bore_mm") or 86
        if piston is None:
            piston = 0.45 * (bore / 86) ** 3
        rods = [p for p in self.parts if p.label == "connecting_rod"]
        if rods and any(p.specs.get("rod_cap_of") is not None for p in rods):
            # rod + its cap per cylinder (caps are separate parts)
            rod = sum(p.info.volume * 1e-9 * materials.get(p.material).density for p in rods) / max(
                1, len(rods) - sum(p.specs.get("rod_cap_of") is not None for p in rods))
        else:
            rod = mean_mass("connecting_rod") or 0.55 * (bore / 86) ** 3
        return piston + pin + ring_mass, rod

    def geom_for(self, p: Part) -> PartGeom:
        e = self.engine
        bore = e.get("bore_mm") or e.get("piston_diameter_mm") or 86.0
        stroke = e.get("stroke_mm") or bore
        rod = e.get("rod_length_mm") or 1.65 * stroke
        extra = {k: v for k, v in e.items() if isinstance(v, (int, float))}
        return PartGeom(label=p.label, specs=p.specs, bore=bore, stroke=stroke, rod=rod, extra=extra)

    def sim_parts(self) -> list[SimPart]:
        # a rod cap is simulated as part of its rod
        return [SimPart(id=p.id, label=p.label, material=p.material, geom=self.geom_for(p), ctx=p.ctx,
                        group=p.group, name=p.name or f"{p.label} #{p.id}") for p in self.parts
                if p.specs.get("rod_cap_of") is None]

    # ------------------------------------------------------------ actions
    def run_simulation(self, duty: str = "automotive", hours: float = 5000, samples: int = 1000,
                       custom_duty: dict | None = None, **overrides) -> dict:
        eng = self.engine_model(**overrides)
        if eng is None:
            raise ValueError("No pistons were identified, so no engine cycle can be built. "
                             "Use the component test bench for individual parts.")
        self._refresh_ctx(eng)
        t = time.time()
        res = simulate(self.sim_parts(), eng, duty, hours=hours, samples=samples,
                       duty=custom_duty or DUTY_CYCLES[duty])
        res["engine_model"] = {k: v for k, v in eng.__dict__.items()}
        res["engine_model"]["displacement_l"] = round(eng.displacement_l, 3)
        res["elapsed_s"] = round(time.time() - t, 2)
        self.last_sim = res
        return res

    def recommend(self, part_id: int, duty: str = "automotive", target_hours: float = 10000) -> list[dict]:
        p = self.parts[part_id]
        eng = self.engine_model()
        geom = self.geom_for(p)
        if eng is None:
            # No engine: rank on static properties at room temperature only.
            def ev(mat):
                return math.inf, mat.yield_strength / 100.0, []
            return rank_materials(p.label, ev, p.info.volume, target_hours)
        self._refresh_ctx(eng)
        duty_d = DUTY_CYCLES[duty]
        ops = solve_operating_points(eng, duty_d)

        def ev(mat):
            sp = SimPart(id=p.id, label=p.label, material=mat.key, geom=geom, ctx=p.ctx, group=p.group)
            modes = aggregate_modes(sp, eng, ops, duty_d["starts_per_hour"])
            life, safety, notes = math.inf, math.inf, []
            for m in modes:
                if m["kind"] == "fatigue" and m["rate"] > 0:
                    life = min(life, 1 / m["rate"])
                elif m["kind"] in ("wear", "erosion") and m["rate"] > 0:
                    life = min(life, m["limit"] / m["rate"])
                safety = min(safety, m["safety_static"], m["margin"])
                if m["kind"] == "fatigue":
                    safety = min(safety, m["safety_fatigue"] * 1.0 if m["safety_fatigue"] else math.inf)
                if m.get("note"):
                    notes.append(m["note"])
            return life, safety, notes
        return rank_materials(p.label, ev, p.info.volume, target_hours)

    def test_part(self, part_id: int, rpm: float | None = None, load: float = 1.0,
                  peak_pressure_bar: float | None = None, temp_c: float | None = None,
                  material: str | None = None, sweep: bool = True) -> dict:
        """Component test bench: run one part's load models at chosen conditions, plus a
        pressure sweep showing where it stops being safe."""
        from .sim.cycle import run_cycle
        from .sim.parts import part_modes
        p = self.parts[part_id]
        mat = materials.get(material or p.material)
        eng = self.engine_model() or EngineModel(bore_mm=86, stroke_mm=86, rod_mm=143, n_cyl=4)
        if peak_pressure_bar:
            eng.peak_pressure_bar = float(peak_pressure_bar)
        rpm = float(rpm or 0.75 * eng.max_rpm)
        self._refresh_ctx(eng)
        ctx = dict(p.ctx)
        if temp_c is not None:
            ctx["temp_override"] = float(temp_c)
        geom = self.geom_for(p)
        cyc = run_cycle(eng, rpm, load)
        modes = part_modes(p.label, geom, mat, cyc, eng, 1.0, ctx)
        out = {"part": p.id, "label": p.label, "material": mat.key, "material_name": mat.name,
               "conditions": {"rpm": rpm, "load": load, "peak_pressure_bar": round(cyc.stats["peak_pressure_bar"], 1),
                              "temp_c": temp_c, "max_rpm": eng.max_rpm},
               "cycle": {k: round(v, 3) for k, v in cyc.stats.items()},
               "modes": [m.to_dict() for m in modes],
               "pressure_trace": {"theta": cyc.theta[::4].round(1).tolist(),
                                  "bar": cyc.pressure_bar[::4].round(2).tolist()}}
        if not modes:
            out["note"] = ("No built-in load model for this part type; use the FEA tab to test it "
                           "under a force, torque or pressure.")
        if sweep and modes:
            base = eng.full_load_peak_bar()
            pts = []
            for f in np.linspace(0.25, 2.5, 19):
                eng.peak_pressure_bar = base * f
                c2 = run_cycle(eng, rpm, 1.0)
                ms = part_modes(p.label, geom, mat, c2, eng, 1.0, ctx)
                sf_f = min([m.safety_fatigue for m in ms if m.kind == "fatigue"] or [math.inf])
                sf_s = min([m.safety_static for m in ms if m.kind in ("fatigue", "static")] or [math.inf])
                life = min([m.hours_to_failure() for m in ms] or [math.inf])
                pts.append({"peak_bar": round(c2.stats["peak_pressure_bar"], 1),
                            "safety_fatigue": None if not math.isfinite(sf_f) else round(sf_f, 3),
                            "safety_static": None if not math.isfinite(sf_s) else round(sf_s, 3),
                            "life_hours_at_this_point": None if not math.isfinite(life) else round(life, 1)})
            out["pressure_sweep"] = pts
            lim = next((q["peak_bar"] for q in pts if q["safety_fatigue"] is not None and q["safety_fatigue"] < 1), None)
            out["infinite_life_limit_bar"] = lim
            lim_s = next((q["peak_bar"] for q in pts if q["safety_static"] is not None and q["safety_static"] < 1), None)
            out["static_limit_bar"] = lim_s
        return out

    def fea(self, part_id: int, case: str = "axial", magnitude: float = 10000.0, axis: str = "auto",
            material: str | None = None, temp_c: float = 20.0) -> dict:
        from .sim.fea import solve
        p = self.parts[part_id]
        mat = materials.get(material or p.material)
        axes = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0]),
                "long": p.info.axes[0], "short": p.info.axes[2], "sym": p.info.sym_axis}
        ax = axes.get(axis, p.info.sym_axis if p.info.sym_score > 0.6 else p.info.axes[0])
        res = solve(p.mesh, mat, ax, case, magnitude, temp_c=temp_c)
        res.update(part=p.id, material=mat.key, material_name=mat.name, axis=[round(float(x), 4) for x in ax])
        return res

    def mass_properties(self, ids: list[int]) -> dict:
        """Combined mass, centre of mass and inertia tensor (about the combined COM) of any
        set of parts: a robot link, a sub-assembly, the whole machine."""
        props = [self.parts[i].mass_props() for i in ids]
        m = sum(p["mass_kg"] for p in props)
        out = {"parts": len(ids), "mass_kg": round(m, 5)}
        if m > 0 and all("com_mm" in p for p in props):
            com = sum(np.asarray(p["com_mm"]) * p["mass_kg"] for p in props) / m
            out["com_mm"] = [round(float(x), 3) for x in com]
            if all("inertia_kgmm2" in p for p in props):
                I = np.zeros((3, 3))
                for p in props:
                    d = np.asarray(p["com_mm"]) - com
                    I += np.asarray(p["inertia_kgmm2"]) + p["mass_kg"] * (d @ d * np.eye(3) - np.outer(d, d))
                out["inertia_kgmm2"] = [[round(float(x), 4) for x in row] for row in I]
                w, v = np.linalg.eigh(I)
                out["principal_moments_kgmm2"] = [round(float(x), 4) for x in w]
                out["principal_axes"] = [[round(float(x), 4) for x in v[:, k]] for k in range(3)]
        return out

    def set_label(self, part_id: int, label: str):
        p = self.parts[part_id]
        p.label = label
        p.pred = Prediction(label=label, confidence=1.0, source="user", alternatives=p.pred.alternatives)
        p.material = default_material(label)
        p.specs = part_specs(label, p.mesh, p.info, materials.get(p.material).density)
        self._recompute_engine()

    def set_material(self, part_id: int, key: str, all_copies: bool = True):
        materials.get(key)
        target = self.parts[part_id]
        for p in self.parts:
            if p is target or (all_copies and p.group == target.group and p.label == target.label):
                p.material = key
                p.ctx["material_set_by_user"] = True
                p.specs["mass_kg"] = round(p.info.volume * 1e-9 * materials.get(key).density, 4)

    # ------------------------------------------------------------ internals
    def _recompute_engine(self):
        pair_rod_caps(self.parts)
        labels = [p.label for p in self.parts]
        self.engine = engine_specs(labels, [p.specs for p in self.parts], [p.info for p in self.parts])
        _assign_context(self)

    def _refresh_ctx(self, eng: EngineModel):
        springs = [p for p in self.parts if p.label == "valve_spring"]
        f_max = None
        if springs:
            from .sim.cycle import run_cycle
            cyc = run_cycle(eng, eng.max_rpm, 1.0)
            f_max = _spring_forces(self.geom_for(springs[0]), eng, cyc)[5]
        for p in self.parts:
            if p.label in ("camshaft", "timing_gear"):
                p.ctx["spring_force"] = f_max

    def to_json(self) -> dict:
        counts: dict[str, int] = {}
        for p in self.parts:
            counts[p.label] = counts.get(p.label, 0) + 1
        from .breakdown import bom_key, build_bom
        qty: dict = {}
        for p in self.parts:
            k = bom_key(p)
            qty[k] = qty.get(k, 0) + 1
        return {
            "id": self.id, "filename": self.filename, "units": self.units, "timings": self.timings,
            "warnings": self.warnings, "engine": self.engine, "counts": counts, "settings": self.settings,
            "mode": self.mode, "tree": self.tree, "bom": build_bom(self), "info": self.info,
            "design_changes": getattr(self, "design_changes", []),
            "parts": [{
                "id": p.id, "name": p.name, "label": p.label, "confidence": round(p.pred.confidence, 3),
                "source": p.pred.source, "alternatives": [[a, round(c, 3)] for a, c in p.pred.alternatives],
                "group": p.group, "material": p.material,
                "material_name": materials.get(p.material).name, "specs": p.specs,
                "center": [round(float(x), 2) for x in p.info.center],
                "bounds": [[round(float(x), 2) for x in p.mesh.bounds[0]], [round(float(x), 2) for x in p.mesh.bounds[1]]],
                "subtype": p.ctx.get("subtype"),
                "path": p.path, "definition": p.definition, "color": p.color, "component": p.component,
                "quantity": qty.get(bom_key(p), 1),
                "com_mm": [round(float(x), 3) for x in p.com_mm] if p.com_mm is not None else None,
                "display_name": display_name(p.name),
            } for p in self.parts],
        }

    def export_glb(self) -> bytes:
        sc = trimesh.Scene()
        for p in self.parts:
            sc.add_geometry(p.mesh, node_name=f"part_{p.id}", geom_name=f"part_{p.id}")
        return sc.export(file_type="glb")


def pair_rod_caps(parts: list) -> None:
    """Connecting rods modelled as rod + bolted cap: find each cap's rod, measure the pair
    together (eye centres, big-end bore), and mark the cap so it is not simulated as a
    second rod. Caps are named "...cap", or are the smaller of two touching rod parts."""
    from .specs import rod_cap_measurements
    rods = [p for p in parts if p.label == "connecting_rod"]
    for p in rods:
        p.specs.pop("rod_cap_of", None)
    if len(rods) < 2:
        return
    vmax = max(p.info.volume for p in rods)
    is_cap = lambda p: bool(re.search(r"\bcap\b|_cap|cap$", (p.name or "").lower())) or p.info.volume < 0.75 * vmax
    caps = [p for p in rods if is_cap(p)]
    bodies = [p for p in rods if p not in caps]
    if not caps or not bodies:
        return
    gap = lambda a, b: float(np.linalg.norm(np.maximum(0, np.maximum(a.mesh.bounds[0] - b.mesh.bounds[1],
                                                                       b.mesh.bounds[0] - a.mesh.bounds[1]))))
    def in_plane(cap, rod):
        # A cap sits in its own rod's thickness slab; in a V engine the neighbouring rod on the
        # same crankpin is offset along the pin by about one rod thickness.
        n = np.asarray(rod.info.axes[2], float)
        return abs(float((cap.info.center - rod.info.center) @ n)) < 0.3 * max(rod.info.extents[2], 1e-6)

    taken = set()
    for cap in caps:
        cand = sorted((b for b in bodies if b.id not in taken and in_plane(cap, b)),
                      key=lambda b: (gap(cap, b), float(np.linalg.norm(cap.mesh.centroid - b.mesh.centroid))))
        if not cand or gap(cap, cand[0]) > 2.0:
            continue
        rod = cand[0]
        taken.add(rod.id)
        cap.specs["rod_cap_of"] = rod.id
        cap.specs["role"] = "connecting rod cap (bolted to the rod; simulated with it)"
        try:
            m = rod_cap_measurements(rod.mesh, rod.info, cap.mesh)
        except Exception:
            m = {}
        if m:
            rod.specs.update(m)
            rod.specs["cap_part"] = cap.id


def _assign_context(asm: Assembly):
    parts = asm.parts
    e = asm.engine
    v_layout = str(e.get("layout", "")).startswith("V")
    has_liners = any(p.label == "cylinder_liner" for p in parts)
    n_cyl = max(1, int(e.get("cylinders", 1) or 1))
    # Without liners the block itself is the running surface: assume cast iron, not bare aluminium.
    if not has_liners:
        for p in parts:
            if p.label == "engine_block" and materials.get(p.material).category == "aluminium"                     and not p.ctx.get("material_set_by_user"):
                p.material = "ci_gray"
                p.specs["mass_kg"] = round(p.info.volume * 1e-9 * materials.get("ci_gray").density, 4)
                msg = ("No cylinder liners were found, so the block is assumed to be cast iron "
                       "(a bare aluminium bore would need a Nikasil/plasma coating).")
                if msg not in asm.warnings:
                    asm.warnings.append(msg)
    # Intake valves are the bigger ones.
    valves = [p for p in parts if p.label == "valve"]
    if valves:
        d = np.array([p.specs.get("head_diameter_mm", p.info.sym_diameter) for p in valves])
        split = d.max() / max(d.min(), 1e-6) > 1.05
        thr = (d.max() + d.min()) / 2
        for p, di in zip(valves, d):
            p.ctx["subtype"] = ("intake" if di >= thr else "exhaust") if split else "exhaust"
        if not split:
            asm.warnings.append("All valves are the same size, so every valve is treated as an exhaust valve "
                                "(the hotter, conservative case).")
    # Head bolts run parallel to the cylinders.
    pistons = [p for p in parts if p.label == "piston"]
    bolts = [p for p in parts if p.label == "bolt"]
    head_bolts = bolts
    if pistons and bolts:
        cyl_axes = np.array([p.info.sym_axis for p in pistons])
        head_bolts = [b for b in bolts if np.max(np.abs(cyl_axes @ b.info.sym_axis)) > 0.95]
    head_ids = {id(b) for b in head_bolts}
    for b in bolts:
        b.ctx["bolt_role"] = "head" if id(b) in head_ids else "other"
        b.ctx["joint_bolts"] = max(1, len(bolts) - len(head_bolts))
    nb = len(head_bolts)
    bolts_per_bore = 4.0 if nb >= 2 * (n_cyl + 1) else max(2.0, nb / n_cyl) if nb else 4.0
    bolt_d = None
    bolt_mat = None
    if head_bolts:
        ds = [b.specs.get("shank_diameter_mm") for b in head_bolts if b.specs.get("shank_diameter_mm")]
        bolt_d = float(np.median(ds)) if ds else None
        bolt_mat = materials.get(head_bolts[0].material)
    # Bearing width from volume: half-shell volume = pi/8 (Do^2 - Di^2) w.
    d_main = e.get("main_journal_diameter_mm")
    for p in parts:
        p.ctx.update(v_layout=v_layout, has_liners=has_liners, bolts_per_bore=bolts_per_bore,
                     bolt_d=bolt_d, bolt_material=bolt_mat)
        if p.label == "main_bearing" and d_main:
            do = float(p.info.extents[0])
            di = min(d_main, do * 0.98)
            ring = math.pi / 8 * (do * do - di * di)
            if ring > 0:
                p.specs["bearing_width_mm"] = round(p.info.volume / ring, 2)
                p.specs["inner_diameter_mm"] = round(di, 2)


def analyze_file(path: str, units: str = "auto", filename: str | None = None, settings: dict | None = None,
                 fused: str = "auto", progress=None) -> Assembly:
    say = progress or (lambda *a: None)
    t0 = time.time()
    say("Reading the file and its assembly structure")
    lr = load_file(path, units=units, progress=lambda n: say(f"Tessellating parts: {n} done"))
    t1 = time.time()
    say(f"Splitting into parts ({len(lr.parts)} bodies)")
    raws = extract_parts(lr.parts, fused=fused)
    t2 = time.time()
    return analyze_parts(raws, filename or path, lr.units, settings,
                         timings={"load_s": t1 - t0, "segment_s": t2 - t1}, info=lr.info, tree=lr.tree,
                         progress=say)


def analyze_parts(raws: list[RawPart], filename: str, units: str = "mm", settings: dict | None = None,
                  timings: dict | None = None, info: dict | None = None, tree: dict | None = None,
                  progress=None) -> Assembly:
    from .breakdown import (DEFAULT_MATERIAL, FAMILY_TO_LABEL, PRINTED_HINT, custom_kind, definition_groups,
                            fast_shape_features, is_engine)
    from .components import recognise
    from .features import context_features
    from .specs import gear_teeth
    say = progress or (lambda *a: None)
    timings = dict(timings or {})
    t = time.time()
    say(f"Measuring the shape of {len(raws)} parts")
    infos = fast_shape_features(raws, progress=lambda n, m: say(f"Measuring shapes: {n}/{m}"))
    ctx = context_features(infos)
    timings["features_s"] = time.time() - t
    t = time.time()
    say("Identifying parts")
    names = [display_name(r.name) for r in raws]
    preds = classify(infos, ctx, names)
    groups = definition_groups(raws, infos) or copy_groups(infos)
    engine_mode = is_engine(preds, names)
    timings["classify_s"] = time.time() - t
    t = time.time()
    say("Recognising standard components and measuring specifications")
    learned = None
    if True:   # general assemblies use it to label parts; engines use it as a second opinion (below)
        try:
            from .component_model import predict as cm_predict, row as cm_row
            learned = cm_predict([cm_row(inf, c, r.brep, r.n_solids) for r, inf, c in zip(raws, infos, ctx)],
                                 [r.path for r in raws], alone=len(raws) == 1)
        except Exception:
            learned = None
    parts = []
    comp_cache: dict = {}
    SECOND_OPINION = {"fastener": "bolt", "bearing": "main_bearing", "spring": "valve_spring"}
    for i, (r, inf, pr) in enumerate(zip(raws, infos, preds)):
        key = r.definition or f"g{groups[i]}"
        if key not in comp_cache:
            pre = {}
            if inf.features.get("teeth", 0) > 0.4 and inf.sym_score > 0.6:
                try:
                    pre = gear_teeth(r.mesh, inf)
                except Exception:
                    pre = {}
            c = recognise(names[i], inf, r.brep, pre, path=r.path)
            comp_cache[key] = (c.to_dict() if c else None, pre)
        comp, pre = comp_cache[key]
        if engine_mode:
            label = pr.label
            pred = pr
            # Second opinion from the component model (trained on ~9,000 real parts): it knows real
            # screws, bearings and springs far better than the engine model trained on generated engines.
            # Real engines, held out: 72% -> 75% (tools/eval_real_engines.py + scratch ensemble test).
            if learned and pr.source != "name":
                ml_label, ml_conf = learned[0][i], learned[1][i]
                eng = SECOND_OPINION.get(ml_label)
                if eng and ml_conf >= 0.8 and eng != pr.label:
                    label = eng
                    pred = Prediction(label=eng, confidence=ml_conf, source="learned model",
                                      alternatives=[(eng, ml_conf)] + [a for a in pr.alternatives if a[0] != eng][:2])
        else:
            fam = comp["family"] if comp else None
            label = FAMILY_TO_LABEL.get(fam, "custom")
            conf = comp["confidence"] if comp else 0.5
            source = comp["source"] if comp else "geometry"
            if learned and (comp is None or comp.get("source") == "geometry"):
                ml_label, ml_conf = learned[0][i], learned[1][i]
                # Names are the strongest evidence; the learned model decides for unnamed parts.
                if comp is None and ml_conf >= 0.6:
                    label, conf, source = ml_label, ml_conf, "learned model"
                elif comp is not None and ml_label == label:
                    conf = max(conf, ml_conf)
            alts = [(label, conf)]
            if pr.confidence > 0.9 and pr.label not in ("other", "unknown"):
                alts.append((f"engine part: {pr.label}", pr.confidence))   # what the engine model thinks
            pred = Prediction(label=label, confidence=conf, source=source, alternatives=alts)
        mat = default_material(label) if engine_mode else DEFAULT_MATERIAL.get(label, "al_6061")
        if not engine_mode and label == "custom" and names[i] and PRINTED_HINT.search(names[i]):
            mat = "asa"   # looks 3D printed (Voron-style naming, version suffix, ducts/cowlings)
        if not engine_mode and comp and comp.get("category") == "heat-set threaded insert":
            mat = "brass_c360"
        sp = part_specs(label, r.mesh, inf, materials.get(mat).density)
        sp.update({k: v for k, v in pre.items() if k not in sp})
        if r.volume_exact:
            sp["volume_cm3"] = round(r.volume_exact / 1000, 3)
            sp["mass_kg"] = round(r.volume_exact * 1e-9 * materials.get(mat).density, 4)
            sp["volume_source"] = "exact B-rep"
        if r.n_solids > 1:
            sp["bodies"] = r.n_solids
        if (not comp or not comp.get("standard_part")) and not (engine_mode and label not in ("other", "unknown")):
            sp["kind"] = custom_kind(inf)     # plain shape class only for parts with no engine role
        com, I = r.com, r.inertia_unit
        if com is None:
            # Mesh uploads: mass properties from the (closed) mesh at unit density.
            try:
                if r.mesh.is_watertight or inf.features.get("watertight", 0) > 0:
                    mp = r.mesh.mass_properties
                    com = [float(x) for x in mp["center_mass"]]
                    I = (np.asarray(mp["inertia"]) * (inf.volume / max(abs(mp["volume"]), 1e-12))).tolist()
            except Exception:
                com, I = None, None
        parts.append(Part(id=i, mesh=r.mesh, info=inf, name=r.name, source=r.source, pred=pred, group=groups[i],
                          label=label, material=mat, specs=sp, path=list(r.path), definition=r.definition,
                          color=r.color, component=comp, brep=r.brep, com_mm=com, inertia_unit=I,
                          path_ids=list(r.path_ids)))
    if engine_mode:
        pair_rod_caps(parts)
    eng = engine_specs([p.label for p in parts], [p.specs for p in parts], infos) if engine_mode else {}
    timings["specs_s"] = time.time() - t
    asm = Assembly(id=uuid.uuid4().hex[:12], filename=filename, units=units, parts=parts, engine=eng,
                   timings={k: round(v, 3) for k, v in timings.items()}, settings=settings or {},
                   mode="engine" if engine_mode else "general", tree=tree, info=dict(info or {}))
    if info and info.get("detected_units") and units != info.get("detected_units"):
        asm.warnings.append(f"Units set to {units} (file looked like {info['detected_units']}).")
    if any(p.source == "segment" for p in parts):
        asm.warnings.append("The model was one fused mesh, so parts were cut along concave creases. "
                            "Upload a STEP file or an assembly with separate bodies for exact parts.")
    unknown = sum(p.label == "unknown" for p in parts)
    if unknown:
        asm.warnings.append(f"{unknown} part(s) could not be identified with confidence; you can label them.")
    _assign_context(asm)
    return asm
