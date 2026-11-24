"""Load 3D models into a list of raw parts.

CAD formats (STEP/IGES/BREP) go through gmsh's OpenCASCADE kernel: every solid
body becomes one part and keeps its name, which is the most accurate split.
Mesh formats (STL/OBJ/PLY/GLB/GLTF/OFF/3MF) go through trimesh: named nodes are
kept, and every node is split further into its closed shells later.

All geometry is returned in millimetres.
"""
from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field

import numpy as np
import trimesh

CAD_EXT = {".step", ".stp", ".iges", ".igs", ".brep"}
MESH_EXT = {".stl", ".obj", ".ply", ".glb", ".gltf", ".off", ".3mf"}
NATIVE_EXT = {".sldprt", ".sldasm"}          # SolidWorks, read without SolidWorks (solidworks.py)
ARCHIVE_EXT = {".zip"}                        # an assembly with its part files
SUPPORTED_EXT = CAD_EXT | MESH_EXT | NATIVE_EXT | ARCHIVE_EXT

UNIT_SCALE = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4, "ft": 304.8}

# Names exporters give to unnamed geometry; these carry no meaning.
_GENERIC_NAME = re.compile(
    r"^(geometry|mesh|object|node|body|solid|part|shape|group|default|untitled|"
    r"unnamed|model|scene|component|polysurface|brep|volume)?[\s_\-.]*\d*(\.\d+)?$",
    re.IGNORECASE,
)

_gmsh_lock = threading.Lock()


@dataclass
class RawPart:
    mesh: trimesh.Trimesh
    name: str | None = None
    source: str = "shell"  # cad-body | named-node | shell | segment
    path: list = field(default_factory=list)     # sub-assembly path (CAD files)
    definition: str | None = None                # shared by every instance of one part definition
    color: tuple | None = None
    brep: dict = field(default_factory=dict)     # exact surface statistics (CAD files)
    volume_exact: float | None = None
    area_exact: float | None = None
    n_solids: int = 1
    com: list | None = None
    inertia_unit: list | None = None
    path_ids: list = field(default_factory=list)


@dataclass
class LoadResult:
    parts: list[RawPart]
    units: str
    scale_to_mm: float
    info: dict = field(default_factory=dict)
    tree: dict | None = None                     # assembly tree from CAD files


def clean_name(name: str | None) -> str | None:
    if not name:
        return None
    name = os.path.basename(str(name).replace("\\", "/").strip())
    name = re.sub(r"\.(stl|obj|step|stp|ply|glb)$", "", name, flags=re.I)
    if not name or _GENERIC_NAME.match(name):
        return None
    return name


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Weld vertices and drop degenerate/duplicate faces left by exporters."""
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def detect_units(extent_max: float) -> str:
    """Guess the file's units from the size of the whole model.

    Machines people upload are roughly 20 mm to 20 m. A model whose largest
    extent is under 10 units is almost certainly in metres.
    """
    if extent_max <= 0:
        return "mm"
    if extent_max < 10:
        return "m"
    return "mm"


def load_file(path: str, units: str = "auto", progress=None) -> LoadResult:
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXT)}")
    tree = None
    if ext in ARCHIVE_EXT:
        return _load_zip(path, units, progress)
    if ext in NATIVE_EXT:
        from .solidworks import read_solidworks
        parts, info, tree = read_solidworks(path, progress)
        detected = "mm"
    elif ext in CAD_EXT:
        try:
            parts, info, tree = _load_cad_xcaf(path, progress)
        except ImportError:
            parts, info = _load_cad(path)
        detected = "mm"  # OpenCASCADE converts CAD to millimetres
    else:
        parts, info = _load_mesh(path)
        bounds = np.vstack([p.mesh.bounds for p in parts]) if parts else np.zeros((2, 3))
        extent = float((bounds.max(0) - bounds.min(0)).max())
        detected = detect_units(extent)
    if not parts:
        raise ValueError("The file contains no surface geometry.")
    use = detected if units == "auto" else units
    if use not in UNIT_SCALE:
        raise ValueError(f"Unknown units '{units}'.")
    scale = UNIT_SCALE[use]
    if scale != 1.0:
        for p in parts:
            p.mesh.apply_scale(scale)
    info["detected_units"] = detected
    return LoadResult(parts=parts, units=use, scale_to_mm=scale, info=info, tree=tree)


def _load_zip(path: str, units: str = "auto", progress=None) -> LoadResult:
    """A .zip upload: a SolidWorks assembly with its parts, or one CAD / mesh file."""
    import shutil
    import tempfile
    import zipfile
    tmp = tempfile.mkdtemp(prefix="enginelab_zip_")
    try:
        with zipfile.ZipFile(path) as z:
            total = 0
            for item in z.infolist():
                if item.is_dir():
                    continue
                total += item.file_size
                if total > 4 * 1024 ** 3:
                    raise ValueError("The archive unpacks to more than 4 GB.")
                target = os.path.realpath(os.path.join(tmp, item.filename))
                if not target.startswith(os.path.realpath(tmp) + os.sep):
                    continue                     # never write outside the temp folder
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(item) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        found = []
        for root, _, files in os.walk(tmp):
            found += [os.path.join(root, f) for f in files]
        if any(os.path.splitext(f)[1].lower() in NATIVE_EXT for f in found):
            from .solidworks import read_parts_folder, read_solidworks, top_file_in_folder
            main = top_file_in_folder(tmp)
            parts, info, tree = read_parts_folder(main) if os.path.isdir(main) else read_solidworks(main, progress)
            info["archive_main_file"] = os.path.relpath(main, tmp) if main != tmp else "(all parts)"
            if not parts:
                raise ValueError("The file contains no surface geometry.")
            info["detected_units"] = "mm"
            return LoadResult(parts=parts, units="mm", scale_to_mm=1.0, info=info, tree=tree)
        usable = [f for f in found if os.path.splitext(f)[1].lower() in CAD_EXT | MESH_EXT]
        if not usable:
            raise ValueError("The archive contains no supported CAD or mesh file.")
        return load_file(max(usable, key=os.path.getsize), units, progress)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _load_cad_xcaf(path: str, progress=None):
    """Exact assembly structure via OpenCASCADE XCAF (names, tree, instances, colours)."""
    import hashlib
    import pickle
    from .cad import read_cad
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
    from .cad import CACHE_VERSION
    cache = os.path.join(cache_dir, f"cad_v{CACHE_VERSION}_{h.hexdigest()}.pkl")
    asm = None
    if os.path.exists(cache):
        try:
            with open(cache, "rb") as fh:
                asm = pickle.load(fh)   # our own cache file, written below
        except Exception:
            asm = None
    if asm is None:
        asm = read_cad(path, progress=progress)
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache, "wb") as fh:
                pickle.dump(asm, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
    parts = [RawPart(mesh=p.mesh, name=clean_name(p.name) or p.name, source="cad-body", path=p.path,
                     definition=p.definition, color=p.color, brep=p.brep, volume_exact=p.volume_exact,
                     area_exact=p.area_exact, n_solids=p.n_solids,
                     com=getattr(p, "com", None), inertia_unit=getattr(p, "inertia_unit", None),
                     path_ids=list(getattr(p, "path_ids", []))) for p in asm.parts]
    return parts, {"format": "cad", **asm.info}, asm.tree


def _load_mesh(path: str) -> tuple[list[RawPart], dict]:
    ext = os.path.splitext(path)[1].lower()
    kwargs = {"split_object": True, "group_material": False} if ext == ".obj" else {}
    scene = trimesh.load(path, force="scene", process=True, **kwargs)
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    parts: list[RawPart] = []
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        geom = scene.geometry.get(geom_name)
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue
        mesh = geom.copy()
        mesh.apply_transform(transform)
        clean_mesh(mesh)
        name = clean_name(node) or clean_name(geom_name)
        if name and name.lower() == stem:
            name = None  # the file name, not a part name
        parts.append(RawPart(mesh=mesh, name=name, source="named-node" if name else "shell"))
    return parts, {"format": ext[1:], "nodes": len(parts)}


def _load_cad(path: str) -> tuple[list[RawPart], dict]:
    try:
        import gmsh
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("STEP/IGES import needs gmsh: pip install gmsh") from exc

    with _gmsh_lock:
        gmsh.initialize(["-noenv"], readConfigFiles=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("Geometry.OCCImportLabels", 1)
            gmsh.model.add("import")
            gmsh.model.occ.importShapes(path)
            gmsh.model.occ.synchronize()
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
            diag = float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])) or 1.0
            gmsh.option.setNumber("Mesh.MeshSizeMax", diag / 60)
            gmsh.option.setNumber("Mesh.MeshSizeMin", diag / 4000)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 40)  # elements per 2*pi of curvature
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.model.mesh.generate(2)

            node_tags, coords, _ = gmsh.model.mesh.getNodes()
            coords = np.asarray(coords).reshape(-1, 3)
            index = np.zeros(int(node_tags.max()) + 1, dtype=np.int64)
            index[node_tags.astype(np.int64)] = np.arange(len(node_tags))

            def surface_faces(stag: int) -> np.ndarray:
                types, _, ntags = gmsh.model.mesh.getElements(2, stag)
                out = []
                for t, nt in zip(types, ntags):
                    if t == 2:  # 3-node triangle
                        out.append(index[np.asarray(nt, dtype=np.int64)].reshape(-1, 3))
                return np.vstack(out) if out else np.zeros((0, 3), dtype=np.int64)

            parts: list[RawPart] = []
            volumes = gmsh.model.getEntities(3)
            for _, vtag in volumes:
                faces = [surface_faces(abs(s)) for _, s in
                         gmsh.model.getBoundary([(3, vtag)], oriented=False, recursive=False)]
                faces = np.vstack(faces) if faces else np.zeros((0, 3), dtype=np.int64)
                if len(faces) == 0:
                    continue
                mesh = trimesh.Trimesh(coords, faces, process=True)
                mesh.remove_unreferenced_vertices()
                trimesh.repair.fix_normals(mesh)
                raw = gmsh.model.getEntityName(3, vtag)
                name = clean_name(raw.split("/")[-1] if raw else None)
                parts.append(RawPart(mesh=mesh, name=name, source="cad-body"))
            if not volumes:  # surface-only CAD: hand all faces on to shell splitting
                faces = [surface_faces(s) for _, s in gmsh.model.getEntities(2)]
                faces = np.vstack([f for f in faces if len(f)]) if faces else np.zeros((0, 3))
                if len(faces):
                    mesh = trimesh.Trimesh(coords, faces, process=True)
                    parts.append(RawPart(mesh=mesh, name=None, source="shell"))
            return parts, {"format": "cad", "solids": len(volumes)}
        finally:
            gmsh.finalize()
