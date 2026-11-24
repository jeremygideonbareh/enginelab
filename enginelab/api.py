"""HTTP API + static web UI.  Run: python run.py  (then open http://127.0.0.1:8000)"""
from __future__ import annotations

import os
import tempfile
import threading
from collections import OrderedDict

import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__, materials
from .generator import LABELS, EngineParams, build_engine, random_engine
from .io import SUPPORTED_EXT
from .pipeline import Assembly, analyze_file
from .sim.life import DUTY_CYCLES

WEB = os.path.join(os.path.dirname(__file__), "..", "web")
BETA = os.environ.get("ENGINELAB_BETA") == "1"          # hosted public beta: tighter limits
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", 100 if BETA else 200))
MAX_ASSEMBLIES = 8 if BETA else 12
MAX_JOBS = 2 if BETA else 4                               # analyses running at the same time
MAX_OPTIMIZERS = 2 if BETA else 6
MAX_OPT_BUDGET = 3000 if BETA else 50000
_job_slots = threading.Semaphore(MAX_JOBS)

app = FastAPI(title="EngineLab", version=__version__)


@app.on_event("startup")
def _restore_feedback():
    try:
        from . import feedback
        feedback.restore()
    except Exception:
        pass
_store: "OrderedDict[str, Assembly]" = OrderedDict()
_lock = threading.Lock()
_jobs: dict[str, dict] = {}


def _start_job(fn, label: str) -> dict:
    """Run a long analysis in a background thread; the browser polls /api/jobs/{id}."""
    import time
    import uuid
    jid = uuid.uuid4().hex[:10]
    job = {"id": jid, "status": "running", "message": label, "started": time.time(), "result": None, "error": None}
    _jobs[jid] = job

    def say(msg):
        job["message"] = str(msg)

    def run():
        if not _job_slots.acquire(blocking=False):
            say("Waiting for a free slot (other analyses are running)")
            _job_slots.acquire()
        try:
            asm = fn(say)
            _keep(asm)
            job["result"] = asm.id
            job["status"] = "done"
        except Exception as exc:  # report the reason to the user
            job["error"] = str(exc) or exc.__class__.__name__
            job["status"] = "error"
        finally:
            _job_slots.release()
    threading.Thread(target=run, daemon=True).start()
    return {"job": jid}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    import time
    job = _jobs.get(jid)
    if not job:
        raise HTTPException(404, "Unknown job")
    return {k: v for k, v in job.items() if k != "started"} | {"elapsed_s": round(time.time() - job["started"], 1)}


def _keep(asm: Assembly) -> Assembly:
    with _lock:
        _store[asm.id] = asm
        while len(_store) > MAX_ASSEMBLIES:
            _store.popitem(last=False)
    return asm


def _get(aid: str) -> Assembly:
    asm = _store.get(aid)
    if asm is None:
        raise HTTPException(404, "Assembly not found (the server keeps the most recent uploads only).")
    return asm


def _part(asm: Assembly, pid: int):
    if not 0 <= pid < len(asm.parts):
        raise HTTPException(404, "Part not found")
    return asm.parts[pid]


@app.exception_handler(ValueError)
async def _value_error(_, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# ------------------------------------------------------------------ reference data
@app.get("/api/info")
def info():
    from .classify import load_model
    try:
        metrics = load_model()["metrics"]
    except FileNotFoundError:
        metrics = None
    from .breakdown import GENERAL_LABELS
    return {"version": __version__, "beta": BETA, "max_upload_mb": MAX_UPLOAD_MB,
            "formats": sorted(SUPPORTED_EXT), "labels": LABELS + ["unknown"],
            "general_labels": GENERAL_LABELS,
            "classifier": metrics,
            "duty_cycles": {k: {"label": v["label"], "starts_per_hour": v["starts_per_hour"], "modes": v["modes"]}
                            for k, v in DUTY_CYCLES.items()}}


@app.get("/api/materials")
def list_materials():
    return materials.all_materials()


# ------------------------------------------------------------------ analysis
@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), units: str = Form("auto"), fuel: str = Form("gasoline"),
                  fused: str = Form("auto")):
    name = os.path.basename(file.filename or "upload")
    ext = os.path.splitext(name)[1].lower()
    if ext not in SUPPORTED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use one of {sorted(SUPPORTED_EXT)}")
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB")
    mb = len(data) / 1e6

    def work(say):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model" + ext)
            with open(path, "wb") as fh:
                fh.write(data)
            if ext in (".step", ".stp", ".iges", ".igs"):
                say(f"Reading CAD geometry and assembly structure ({mb:.0f} MB; about 1-2 s per MB)")
            elif ext in (".zip", ".sldasm", ".sldprt"):
                say("Unpacking SolidWorks / archive files")
            return analyze_file(path, units=units, filename=name, settings={"fuel": fuel}, fused=fused,
                                progress=say)
    return _start_job(work, f"Uploaded {name} ({mb:.1f} MB)")


SAMPLES = {
    "open6x": ("Open6X 6-axis robot arm (19 MB, 692 parts)", "data/real_cad/open6x/Open6X-v2-ROBOT-ARM.step"),
    "odri_biped": ("ODRI biped robot, brushless actuators (27 MB, 345 parts)", "data/real_cad/odri_biped/biped_6dof_v1.STEP"),
    "so101": ("SO-101 LeRobot arm, servo driven (20 MB, 259 parts)", "data/real_cad/so101/SO101_Assembly.step"),
    "koch": ("Koch low-cost robot arm (2.5 MB, 115 parts)", "data/real_cad/koch_follower/arm.step"),
    "crank454": ("Inline-4 crank train, native SolidWorks zip (2.6 MB, 29 parts)", "data/real_cad/crank454/crankshaft_454.zip"),
    "switchwire": ("Voron Switchwire printer (90 MB, 712 parts)", "data/real_cad/switchwire/Switchwire_Assembly_v1_STEP.step"),
    "voron0": ("Voron 0.2 printer (179 MB, 1,683 parts)", "data/real_cad/voron0/V0.2R1_Master_Assembly.step"),
    "voron_toolhead": ("Voron 0 MiniSB toolhead (13 MB, 31 parts)", "data/real_cad/minisb/MiniSB_Bowden.step"),
    "voron24": ("Voron 2.4 3D printer, full assembly (241 MB, ~1,300 parts)", "data/real_cad/voron24/Voron_2.4r2_Assembly.step"),
    "trident": ("Voron Trident 3D printer, full assembly (185 MB)", "data/real_cad/trident/trident_assembly.step"),
}
ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")


@app.get("/api/samples")
def samples():
    return [{"key": k, "title": t, "size_mb": round(os.path.getsize(os.path.join(ROOT_DIR, p)) / 1e6, 1)}
            for k, (t, p) in SAMPLES.items() if os.path.exists(os.path.join(ROOT_DIR, p))]


@app.post("/api/samples/{key}")
def run_sample(key: str):
    if key not in SAMPLES:
        raise HTTPException(404, "Unknown sample")
    title, rel = SAMPLES[key]
    path = os.path.join(ROOT_DIR, rel)
    if not os.path.exists(path):
        raise HTTPException(404, "Sample file not downloaded")

    def work(say):
        return analyze_file(path, filename=os.path.basename(path), settings={"fuel": "gasoline"}, progress=say)
    return _start_job(work, f"Analysing sample: {title}")


@app.post("/api/demo")
def demo(seed: int = Body(1, embed=True), layout: str | None = Body(None, embed=True),
         cylinders: int | None = Body(None, embed=True), fmt: str = Body("stl", embed=True)):
    """Generate an engine, export it as a plain mesh file (no part names) and analyse it
    exactly like an upload."""
    rng = np.random.default_rng(seed)
    if layout or cylinders:
        from .generator import random_params
        p = random_params(rng)
        if layout:
            p.layout = layout
            p.bank_angle = 90.0 if layout == "V" else 0.0
        if cylinders:
            p.n_cyl = int(cylinders)
        if p.layout == "V" and p.n_cyl % 2:
            p.n_cyl += 1
        p.pitch = p.bore * (1.36 if p.layout == "V" else 1.2)
        eng = build_engine(p, rng)
    else:
        eng = random_engine(seed)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"demo_engine_{seed}.{fmt}")
        if fmt == "glb":
            # GLB keeps separate meshes but we strip names so identification is from geometry.
            import trimesh
            sc = trimesh.Scene()
            for i, (_, m) in enumerate(eng.parts):
                sc.add_geometry(m, node_name=f"node_{i}", geom_name=f"mesh_{i}")
            sc.export(path)
        else:
            eng.merged().export(path)
        asm = analyze_file(path, filename=f"demo_engine_{seed}.{fmt}", settings={"fuel": "gasoline"})
    truth = [lab for lab, _ in eng.parts]
    out = _keep(asm).to_json()
    out["demo_truth"] = {
        "layout": eng.params.layout, "cylinders": eng.params.n_cyl, "bore_mm": round(eng.params.bore, 2),
        "stroke_mm": round(eng.params.stroke, 2), "rod_length_mm": round(eng.params.rod_length, 2),
        "displacement_cc": round(eng.params.displacement_cc, 1), "part_count": len(truth),
        "label_counts": {k: truth.count(k) for k in sorted(set(truth))},
    }
    return out


@app.get("/api/assembly/{aid}")
def get_assembly(aid: str):
    return _get(aid).to_json()


@app.get("/api/assembly/{aid}/massprops")
def mass_props(aid: str, ids: str = ""):
    """Combined mass properties of a set of parts (comma-separated ids; empty = whole model)."""
    asm = _get(aid)
    sel = [int(x) for x in ids.split(",") if x.strip().isdigit()] if ids else list(range(len(asm.parts)))
    sel = [i for i in sel if 0 <= i < len(asm.parts)]
    if not sel:
        raise HTTPException(400, "No valid part ids")
    return asm.mass_properties(sel)


@app.get("/api/assembly/{aid}/robot/links")
def robot_links(aid: str):
    """Default robot links (top-level sub-assemblies) and suggested joints, for review."""
    from .robotics import build_urdf, default_links
    asm = _get(aid)
    links = default_links(asm)
    _, _, summary = build_urdf(asm, links, "preview")
    return {"links": [{"name": l["name"], "ids": l["ids"]} for l in links], "summary": summary}


@app.get("/api/assembly/{aid}/robot/torques")
def robot_torques(aid: str, payload_kg: float = 0.2, accel: float = 5.0):
    """Joint torques (holding, worst case, acceleration) against the motors found at each joint."""
    from .robot_sizing import joint_torques
    return joint_torques(_get(aid), float(np.clip(payload_kg, 0, 1000)), float(np.clip(accel, 0, 1000)))


@app.post("/api/assembly/{aid}/robot/urdf")
def robot_urdf(aid: str, body: dict = Body(default={})):
    """URDF + STL meshes as a zip. Body: {"links": [{"name", "ids"}...], "name": "robot"} (optional)."""
    from .robotics import urdf_zip
    asm = _get(aid)
    links = body.get("links") or None
    if links:
        for l in links:
            l["ids"] = [int(i) for i in l.get("ids", []) if 0 <= int(i) < len(asm.parts)]
        links = [l for l in links if l["ids"]]
    name = body.get("name") or os.path.splitext(asm.filename)[0]
    data, _ = urdf_zip(asm, links, name)
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{name}_urdf.zip"'})


@app.get("/api/assembly/{aid}/bom.csv")
def bom_csv(aid: str):
    import csv
    import io
    asm = _get(aid)
    rows = asm.to_json()["bom"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["item", "qty", "name", "category", "designation", "standard part", "material", "unit mass kg",
                "total mass kg", "size L x W x H mm", "assembly path", "search link 1", "search link 2"])
    for r in rows:
        links = list(r["links"].values())
        w.writerow([r["item"], r["qty"], r["name"], r["category"], r["designation"], "yes" if r["standard_part"] else "no",
                    r["material"], r["unit_mass_kg"], r["total_mass_kg"], " x ".join(map(str, r["dims_mm"])),
                    r["path"], links[0] if links else "", links[1] if len(links) > 1 else ""])
    name = os.path.splitext(asm.filename)[0] + "_BOM.csv"
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/assembly/{aid}/model.glb")
def model_glb(aid: str):
    return Response(_get(aid).export_glb(), media_type="model/gltf-binary")


@app.post("/api/assembly/{aid}/settings")
def set_settings(aid: str, settings: dict = Body(...)):
    asm = _get(aid)
    allowed = {"fuel", "compression_ratio", "boost_bar", "max_rpm", "peak_pressure_bar"}
    for k, v in settings.items():
        if k in allowed:
            asm.settings[k] = v if v not in ("", None) else None
    asm.settings = {k: v for k, v in asm.settings.items() if v is not None}
    eng = asm.engine_model()
    return {"settings": asm.settings, "engine_model": eng.__dict__ if eng else None}


@app.post("/api/assembly/{aid}/part/{pid}/label")
def relabel(aid: str, pid: int, label: str = Body(..., embed=True)):
    asm = _get(aid)
    _part(asm, pid)
    from .breakdown import GENERAL_LABELS
    if label not in LABELS + ["unknown"] + GENERAL_LABELS:
        raise HTTPException(400, f"Unknown label '{label}'")
    old = asm.parts[pid].label
    asm.set_label(pid, label)
    try:      # the correction becomes a training example (shape numbers only)
        from . import feedback
        feedback.record_correction(asm, pid, old, label)
    except Exception:
        pass
    return asm.to_json()


@app.get("/api/assembly/{aid}/balance")
def balance(aid: str):
    """Engine balance (shaking forces / couples) and crank torsional critical speeds."""
    from .sim.balance import analyse
    asm = _get(aid)
    r = analyse(asm)
    if r is None:
        raise HTTPException(400, "Balance analysis needs an engine with a crankshaft and pistons.")
    return r


@app.get("/api/regions")
def regions():
    from .sourcing import REGIONS
    return [{"code": k, "name": v["name"], "currency": v["currency"]} for k, v in REGIONS.items()]


@app.get("/api/assembly/{aid}/sourcing")
def sourcing(aid: str, region: str = "IN"):
    """Price estimates, shops and a shopping plan for the parts list, for one country."""
    from .sourcing import source_assembly
    return source_assembly(_get(aid), region.upper())


@app.get("/api/live_price")
def live_price(q: str, qty: int = 1):
    from .sourcing import live_price as lp, live_providers
    if not live_providers():
        raise HTTPException(400, "No distributor API key is configured on this server (e.g. MOUSER_API_KEY).")
    return {"query": q, "results": lp(q[:80], max(1, min(qty, 100000)))}


@app.post("/api/feedback")
def send_feedback(body: dict = Body(...)):
    """Free-text problem report from the app (no personal data is asked for)."""
    from . import feedback
    try:
        feedback.record_report(str(body.get("text", "")), body.get("mode"), body.get("label"), body.get("page"))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"ok": True}


@app.get("/api/feedback/summary")
def feedback_summary():
    from . import feedback
    return feedback.summary()


@app.post("/api/assembly/{aid}/part/{pid}/material")
def set_material(aid: str, pid: int, key: str = Body(..., embed=True), all_copies: bool = Body(True, embed=True)):
    asm = _get(aid)
    _part(asm, pid)
    try:
        asm.set_material(pid, key, all_copies)
    except KeyError as exc:
        raise HTTPException(400, str(exc))
    return asm.to_json()


@app.post("/api/assembly/{aid}/materials/bulk")
def bulk_material(aid: str, label: str = Body(..., embed=True), key: str = Body(..., embed=True)):
    """Set the material of every part of one type, e.g. all custom parts are PETG prints."""
    asm = _get(aid)
    try:
        materials.get(key)
    except KeyError as exc:
        raise HTTPException(400, str(exc))
    n = 0
    for p in asm.parts:
        if p.label == label:
            p.material = key
            p.ctx["material_set_by_user"] = True
            p.specs["mass_kg"] = round(p.info.volume * 1e-9 * materials.get(key).density, 4)
            n += 1
    out = asm.to_json()
    out["changed"] = n
    return out


@app.get("/api/assembly/{aid}/part/{pid}/recommend")
def recommend(aid: str, pid: int, duty: str = "automotive", target_hours: float = 10000):
    asm = _get(aid)
    _part(asm, pid)
    if duty not in DUTY_CYCLES:
        raise HTTPException(400, "Unknown duty cycle")
    return {"part": pid, "duty": duty, "target_hours": target_hours,
            "ranking": asm.recommend(pid, duty, target_hours)}


@app.post("/api/assembly/{aid}/part/{pid}/test")
def test_part(aid: str, pid: int, body: dict = Body(...)):
    asm = _get(aid)
    _part(asm, pid)
    return asm.test_part(pid, rpm=body.get("rpm"), load=float(body.get("load", 1.0)),
                         peak_pressure_bar=body.get("peak_pressure_bar"), temp_c=body.get("temp_c"),
                         material=body.get("material"), sweep=bool(body.get("sweep", True)))


@app.post("/api/assembly/{aid}/part/{pid}/fea")
def fea(aid: str, pid: int, body: dict = Body(...)):
    asm = _get(aid)
    _part(asm, pid)
    case = body.get("case", "axial")
    if case not in ("axial", "bending", "torsion", "pressure"):
        raise HTTPException(400, "case must be axial, bending, torsion or pressure")
    return asm.fea(pid, case=case, magnitude=float(body.get("magnitude", 10000)), axis=body.get("axis", "auto"),
                   material=body.get("material"), temp_c=float(body.get("temp_c", 20)))


@app.post("/api/assembly/{aid}/simulate")
def simulate(aid: str, body: dict = Body(...)):
    asm = _get(aid)
    duty = body.get("duty", "automotive")
    custom = body.get("custom_duty")
    if duty not in DUTY_CYCLES and not custom:
        raise HTTPException(400, "Unknown duty cycle")
    hours = float(np.clip(float(body.get("hours", 5000)), 10, 200000))
    samples = int(np.clip(int(body.get("samples", 1000)), 50, 4000))
    overrides = {k: body[k] for k in ("fuel", "compression_ratio", "boost_bar", "max_rpm", "peak_pressure_bar")
                 if body.get(k) not in (None, "")}
    for k in ("compression_ratio", "boost_bar", "max_rpm", "peak_pressure_bar"):
        if k in overrides:
            overrides[k] = float(overrides[k])
    return asm.run_simulation(duty if duty in DUTY_CYCLES else "automotive", hours=hours, samples=samples,
                              custom_duty=custom, **overrides)


# ------------------------------------------------------------------ design improvement
_optimizers: dict = {}


def _improve_call(fn):
    try:
        return fn()
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/assembly/{aid}/part/{pid}/improve")
def improve_part(aid: str, pid: int, duty: str = "automotive", target_hours: float = 10000):
    from . import improve
    asm = _get(aid)
    _part(asm, pid)
    return _improve_call(lambda: improve.recommend(asm, pid, duty, target_hours))


@app.post("/api/assembly/{aid}/whatif")
def whatif(aid: str, body: dict = Body(...)):
    """Preview a change without applying it: {"design": {"piston.material": "al_2618", ...}}."""
    from . import improve
    asm = _get(aid)
    return _improve_call(lambda: improve.whatif(asm, body.get("design") or {}, body.get("duty", "automotive")))


@app.post("/api/assembly/{aid}/design")
def apply_design(aid: str, body: dict = Body(...)):
    """Apply changes (merged with those already applied); {"reset": true} returns to the upload."""
    from . import improve
    asm = _get(aid)
    if body.get("reset"):
        asm.design = {}
        design = {}
    else:
        design = body.get("design") or {}
    _improve_call(lambda: improve.apply_design(asm, design))
    asm.last_sim = None
    return asm.to_json()


@app.post("/api/assembly/{aid}/optimize/start")
def optimize_start(aid: str, body: dict = Body(...)):
    from . import improve
    asm = _get(aid)
    old = _optimizers.get(aid)
    if old:
        old.stop()
    running = [k for k, o in _optimizers.items() if o.running and k != aid]
    while len(running) >= MAX_OPTIMIZERS:          # the oldest run yields to the new one
        _optimizers[running.pop(0)].stop()
    opt = _improve_call(lambda: improve.Optimizer(
        asm, objective=body.get("objective", "balanced"), target=float(body.get("target_hours", 10000)),
        duty=body.get("duty", "automotive"), budget=int(np.clip(int(body.get("budget", 2000)), 20, MAX_OPT_BUDGET))))
    _optimizers[aid] = opt
    opt.start()
    return opt.status()


@app.post("/api/assembly/{aid}/optimize/stop")
def optimize_stop(aid: str):
    opt = _optimizers.get(aid)
    if not opt:
        raise HTTPException(404, "No optimiser has run on this assembly")
    opt.stop()
    return opt.status()


@app.get("/api/assembly/{aid}/optimize")
def optimize_status(aid: str):
    opt = _optimizers.get(aid)
    if not opt:
        return {"running": False, "evaluations_total": 0, "history": [], "top": []}
    return opt.status()


@app.post("/api/assembly/{aid}/optimize/apply")
def optimize_apply(aid: str, body: dict = Body(...)):
    """Apply one of the optimiser's best designs (rank 0 = best) to the assembly."""
    from . import improve
    asm = _get(aid)
    opt = _optimizers.get(aid)
    if not opt:
        raise HTTPException(404, "No optimiser has run on this assembly")
    top = opt.status(top=10)["top"]
    k = int(body.get("rank", 0))
    if not 0 <= k < len(top):
        raise HTTPException(400, "No design at that rank")
    design = {}
    for lid, v in top[k]["design"].items():
        lv = opt.ctx.by_id[lid]
        design[lid] = lv.options[int(v)].material if lv.kind == "material" else int(v)   # materials by key
    asm.design = {}
    _improve_call(lambda: improve.apply_design(asm, design))
    asm.last_sim = None
    return asm.to_json()


# ------------------------------------------------------------------ web UI
@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))


app.mount("/static", StaticFiles(directory=WEB), name="static")
