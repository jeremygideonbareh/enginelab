"""Exact CAD assembly reader (STEP / IGES) built on OpenCASCADE XCAF (OCP).

What it keeps from the file, so a big assembly breaks down exactly as it was designed:
  * the product structure: sub-assemblies, part names, instance names, nesting path,
  * every instance with its placement (a screw used 200 times = 200 parts, 1 definition),
  * colours,
  * exact surface types of every face (plane, cylinder, cone, sphere, torus, B-spline)
    and the radius / axis of cylindrical faces, telling holes from shafts.

Tessellation is done once per part definition, relative to that part's own size
(small screws get fine meshes, big plates coarse ones), then reused for every instance.
"""
from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np
import trimesh

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.Bnd import Bnd_Box
from OCP.GeomAbs import (GeomAbs_BSplineSurface, GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane,
                         GeomAbs_Sphere, GeomAbs_Torus)
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.Quantity import Quantity_Color
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.IGESCAFControl import IGESCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_Tool
from OCP.collections import Sequence_TDF_Label as TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID, TopAbs_SHELL
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.XCAFDoc import XCAFDoc_ColorSurf, XCAFDoc_ColorGen, XCAFDoc_DocumentTool

SURF_NAMES = {GeomAbs_Plane: "plane", GeomAbs_Cylinder: "cylinder", GeomAbs_Cone: "cone",
              GeomAbs_Sphere: "sphere", GeomAbs_Torus: "torus", GeomAbs_BSplineSurface: "bspline"}


# Default names exporters give unnamed shapes. Pure numbers ("300", "250") are real product
# names (length variants), so they are NOT generic.
CACHE_VERSION = 3   # bump whenever read_cad output changes, so cached reads are redone

GENERIC_NAME = re.compile(r"^((solid|compound|shell|face|body|brep)\s*\d*|open cascade step translator.*|)$",
                          re.IGNORECASE)


@dataclass
class CadPart:
    """One placed instance of a part definition."""
    mesh: trimesh.Trimesh
    name: str                     # part (definition) name
    instance_name: str            # name of this occurrence in its parent
    path: list[str]               # sub-assembly names from the root down to the parent
    definition: str               # key shared by every instance of the same part definition
    color: tuple | None = None
    n_solids: int = 1
    brep: dict = field(default_factory=dict)   # exact face statistics of the definition
    volume_exact: float | None = None          # mm^3, from the B-rep (not the mesh)
    area_exact: float | None = None
    path_ids: list = field(default_factory=list)   # one id per assembly occurrence on the path
    com: list | None = None                    # centre of mass, world frame, mm
    inertia_unit: list | None = None           # 3x3 inertia about the COM per unit density (mm^5), world axes


@dataclass
class CadAssembly:
    parts: list[CadPart]
    tree: dict                    # nested {"name", "children": [...], "parts": [index...]}
    info: dict


def _name(label: TDF_Label) -> str:
    attr = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
        return attr.Get().ToExtString()
    return ""


def _entry(label: TDF_Label) -> str:
    """Stable id of a label in the document, e.g. '0:1:1:7' (same for every instance)."""
    from OCP.TCollection import TCollection_AsciiString
    s = TCollection_AsciiString()
    TDF_Tool.Entry_s(label, s)
    return s.ToCString()


def _color(ctool, label) -> tuple | None:
    c = Quantity_Color()
    for kind in (XCAFDoc_ColorSurf, XCAFDoc_ColorGen):
        try:
            ok = ctool.GetColor_s(label, kind, c)
        except Exception:
            ok = False
        if ok:
            return (round(c.Red(), 3), round(c.Green(), 3), round(c.Blue(), 3))
    return None


def _brep_stats(shape) -> dict:
    """Surface-type areas and cylinder list (radius, area, axis, hole?) of a part definition."""
    areas: dict[str, float] = {}
    cyl = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    n = 0
    while exp.More():
        face = TopoDS.Face(exp.Current())
        n += 1
        try:
            surf = BRepAdaptor_Surface(face, True)
            t = surf.GetType()
            props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face, props, False, True)   # from the triangulation
            a = props.Mass()
            key = SURF_NAMES.get(t, "other")
            areas[key] = areas.get(key, 0.0) + a
            if t == GeomAbs_Cylinder:
                c = surf.Cylinder()
                ax = c.Axis()
                d = ax.Direction()
                p = ax.Location()
                # A face whose normal points at the axis is a hole (concave), else a shaft.
                u = 0.5 * (surf.FirstUParameter() + surf.LastUParameter())
                v = 0.5 * (surf.FirstVParameter() + surf.LastVParameter())
                pnt = surf.Value(u, v)
                radial = np.array([pnt.X() - p.X(), pnt.Y() - p.Y(), pnt.Z() - p.Z()])
                dd = np.array([d.X(), d.Y(), d.Z()])
                radial -= (radial @ dd) * dd
                outward = c.Direct()  # right-handed frame -> normal points outward
                reversed_ = face.Orientation() == TopAbs_REVERSED
                hole = (outward == reversed_)
                cyl.append({"r": round(c.Radius(), 4), "area": round(a, 3), "hole": bool(hole),
                            "axis": [round(d.X(), 4), round(d.Y(), 4), round(d.Z(), 4)],
                            "origin": [round(p.X(), 3), round(p.Y(), 3), round(p.Z(), 3)]})
        except Exception:
            pass
        exp.Next()
    total = sum(areas.values()) or 1.0
    return {"faces": n, "area_by_type": {k: round(v / total, 4) for k, v in areas.items()},
            "cylinders": cyl[:200], "_area": total}


def _tessellate(shape, size_hint: float) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a shape in its own frame (deflection scales with the part size) and pull
    the triangles out through OpenCASCADE's binary STL writer (C++, no Python loops)."""
    import io
    from OCP.StlAPI import StlAPI_Writer
    lin = max(size_hint * 0.004, 0.01)
    BRepMesh_IncrementalMesh(shape, lin, False, 0.35, True)
    buf = io.BytesIO()
    w = StlAPI_Writer()
    w.ASCIIMode = False
    if not w.Write(shape, buf):
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    data = buf.getvalue()
    if len(data) < 84:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    n = int(np.frombuffer(data, dtype="<u4", count=1, offset=80)[0])
    rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    tri = np.frombuffer(data, dtype=rec, count=n, offset=84)["v"].astype(np.float64)
    extra = _fallback_faces(shape, lin)
    if len(extra):
        tri = np.concatenate([tri, extra])
    verts = tri.reshape(-1, 3)
    faces = np.arange(len(verts), dtype=np.int64).reshape(-1, 3)
    return verts, faces


def _fallback_faces(shape, lin: float) -> np.ndarray:
    """Triangles for faces the mesher skipped (seen on 45-degree countersink cones in real
    files): sample the face's own surface on a UV grid and keep the cells inside its boundary."""
    from OCP.BRepClass import BRepClass_FaceClassifier
    from OCP.TopAbs import TopAbs_IN, TopAbs_ON
    from OCP.gp import gp_Pnt2d
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face(exp.Current())
        exp.Next()
        t = BRep_Tool.Triangulation_s(face, TopLoc_Location())
        if t is not None and t.NbTriangles() > 0:
            continue
        try:
            s = BRepAdaptor_Surface(face, True)
            u0, u1 = s.FirstUParameter(), s.LastUParameter()
            v0, v1 = s.FirstVParameter(), s.LastVParameter()
            if not all(map(math.isfinite, (u0, u1, v0, v1))):
                continue
            nu, nv = 48, max(4, min(48, int(abs(v1 - v0) / max(lin, 1e-6)) + 2))
            us, vs = np.linspace(u0, u1, nu + 1), np.linspace(v0, v1, nv + 1)
            P = np.empty((nu + 1, nv + 1, 3))
            for i, u in enumerate(us):
                for j, v in enumerate(vs):
                    q = s.Value(u, v)
                    P[i, j] = (q.X(), q.Y(), q.Z())
            clf = BRepClass_FaceClassifier()
            rev = face.Orientation() == TopAbs_REVERSED
            for i in range(nu):
                for j in range(nv):
                    clf.Perform(face, gp_Pnt2d(0.5 * (us[i] + us[i + 1]), 0.5 * (vs[j] + vs[j + 1])), 1e-7)
                    if clf.State() not in (TopAbs_IN, TopAbs_ON):
                        continue
                    a, b, c, d = P[i, j], P[i + 1, j], P[i + 1, j + 1], P[i, j + 1]
                    t1, t2 = ((a, c, b), (a, d, c)) if rev else ((a, b, c), (a, c, d))
                    out.extend([t1, t2])
        except Exception:
            continue
    return np.asarray(out, dtype=np.float64).reshape(-1, 3, 3) if out else np.zeros((0, 3, 3))


def _trsf_matrix(loc: TopLoc_Location) -> np.ndarray:
    t = loc.Transformation()
    m = np.eye(4)
    for r in range(3):
        for c in range(4):
            m[r, c] = t.Value(r + 1, c + 1)
    return m


def _brep_genus(shape, n_solids: int) -> int:
    """Through-holes from B-rep topology: chi = V - E + 2F - L summed over closed shells,
    genus = (2 * shells - chi) / 2.  (Seam edges count once, which is what makes this exact.)"""
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_VERTEX, TopAbs_WIRE
    from OCP.TopExp import TopExp
    from OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as TopTools_IndexedMapOfShape
    counts = []
    for kind in (TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE):
        m = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(shape, kind, m)
        counts.append(m.Extent())
    V, E, F, L = counts
    chi = V - E + 2 * F - L
    return max(0, int(round((2 * max(n_solids, 1) - chi) / 2)))


def _count_solids(shape) -> int:
    n = 0
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        n += 1
        exp.Next()
    return n


def read_cad(path: str, max_parts: int = 20000, progress=None) -> CadAssembly:
    t0 = time.time()
    ext = os.path.splitext(path)[1].lower()
    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    if ext in (".iges", ".igs"):
        reader = IGESCAFControl_Reader()
    else:
        reader = STEPCAFControl_Reader()
        reader.SetNameMode(True)
        reader.SetColorMode(True)
        reader.SetLayerMode(True)
    if reader.ReadFile(path) != IFSelect_RetDone:
        raise ValueError("The CAD file could not be read (is it a valid STEP/IGES file?).")
    reader.Transfer(doc)
    t_read = time.time() - t0
    stool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    ctool = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())

    defs: dict[str, dict] = {}       # definition key -> cached tessellation + stats
    parts: list[CadPart] = []

    def definition(label: TDF_Label, shape) -> dict:
        key = _entry(label)
        if key in defs:
            return defs[key]
        box = Bnd_Box()
        BRepBndLib.Add_s(shape, box)
        if box.IsVoid():
            defs[key] = {"key": key, "empty": True}
            return defs[key]
        pmin, pmax = box.CornerMin(), box.CornerMax()
        xmin, ymin, zmin = pmin.X(), pmin.Y(), pmin.Z()
        xmax, ymax, zmax = pmax.X(), pmax.Y(), pmax.Z()
        size = math.dist((xmin, ymin, zmin), (xmax, ymax, zmax))
        v, f = _tessellate(shape, size)
        vp = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, vp)          # exact, from the B-rep
        c = vp.CentreOfMass()
        M = vp.MatrixOfInertia()                          # about the centre of mass, density 1
        inertia = [[M.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)]
        com = [c.X(), c.Y(), c.Z()]
        stats = _brep_stats(shape)
        try:
            stats["genus"] = _brep_genus(shape, _count_solids(shape))
        except Exception:
            pass
        defs[key] = {"key": key, "empty": len(f) == 0, "v": v, "f": f, "name": _name(label),
                     "brep": stats, "volume": abs(vp.Mass()), "area": stats.pop("_area"),
                     "com": com, "inertia": inertia,
                     "n_solids": max(1, _count_solids(shape)), "color": _color(ctool, label)}
        return defs[key]

    occ_counter = [0]

    def walk(label: TDF_Label, loc: np.ndarray, path: list[str], node: dict, inst_name: str, inst_color,
             path_ids: list | None = None):
        path_ids = path_ids or []
        if len(parts) >= max_parts:
            return
        ref = TDF_Label()
        if stool.IsReference_s(label) and stool.GetReferredShape_s(label, ref):
            sub_loc = loc @ _trsf_matrix(stool.GetLocation_s(label))
            walk(ref, sub_loc, path, node, _name(label) or inst_name, _color(ctool, label) or inst_color, path_ids)
            return
        if stool.IsAssembly_s(label):
            name = _name(label) or inst_name or "assembly"
            child = {"name": name, "children": [], "parts": []}
            node["children"].append(child)
            comps = TDF_LabelSequence()
            stool.GetComponents_s(label, comps, False)
            occ_counter[0] += 1
            for i in range(1, comps.Length() + 1):
                walk(comps.Value(i), loc, path + [name], child, "", inst_color, path_ids + [occ_counter[0]])
            # Bodies with default names ("SOLID", "COMPOUND"...) are the bodies of one
            # multi-body component, not separate parts: fold them into that component.
            generic = [k for k in child["parts"] if parts[k] is not None and GENERIC_NAME.match(parts[k].name)]
            if generic and GENERIC_NAME.match(name):
                # Unnamed solids under an unnamed compound are separate parts: number them.
                for n, k in enumerate(generic, 1):
                    parts[k].name = parts[k].instance_name = f"Part {len(parts) - len(generic) + n}"
                generic = []
            if generic:
                merged = _merge(parts, generic, name, _entry(label), path)
                merged.path_ids = list(path_ids)
                for k in generic:
                    parts[k] = None
                child["parts"] = [k for k in child["parts"] if parts[k] is not None]
                if not child["parts"] and not child["children"]:
                    node["children"].remove(child)     # the whole "assembly" was one part
                    parts.append(merged)
                    node["parts"].append(len(parts) - 1)
                else:
                    parts.append(merged)
                    child["parts"].append(len(parts) - 1)
            return
        shape = stool.GetShape_s(label)
        if shape.IsNull():
            return
        d = definition(label, shape)
        if d.get("empty"):
            return
        mesh = trimesh.Trimesh(d["v"].copy(), d["f"].copy(), process=False)
        mesh.apply_transform(loc)
        R = loc[:3, :3]
        com_w = (loc @ np.r_[d["com"], 1.0])[:3].tolist()
        sign = 1.0 if d["inertia"][0][0] >= 0 else -1.0     # reversed shapes give negative mass props
        I_w = (R @ (sign * np.asarray(d["inertia"])) @ R.T).tolist()
        mesh.merge_vertices()
        idx = len(parts)
        parts.append(CadPart(mesh=mesh, name=d["name"] or inst_name or f"part_{idx}",
                             instance_name=inst_name or d["name"], path=path, definition=d["key"],
                             color=inst_color or d["color"], n_solids=d["n_solids"], brep=d["brep"],
                             volume_exact=d["volume"], area_exact=d["area"], com=com_w, inertia_unit=I_w,
                             path_ids=list(path_ids)))
        node["parts"].append(idx)
        if progress and idx % 100 == 0:
            progress(idx)

    roots = TDF_LabelSequence()
    stool.GetFreeShapes(roots)
    tree = {"name": os.path.basename(path), "children": [], "parts": []}
    for i in range(1, roots.Length() + 1):
        walk(roots.Value(i), np.eye(4), [], tree, "", None)
    # Compact the part list (merged bodies left gaps) and renumber the tree.
    remap, kept = {}, []
    for i, p in enumerate(parts):
        if p is not None:
            remap[i] = len(kept)
            kept.append(p)

    def renumber(n):
        n["parts"] = [remap[k] for k in n["parts"] if k in remap]
        for c in n["children"]:
            renumber(c)
    renumber(tree)
    info = {"format": ext[1:], "read_s": round(t_read, 2), "total_s": round(time.time() - t0, 2),
            "definitions": len({p.definition for p in kept}), "instances": len(kept),
            "truncated": len(kept) >= max_parts}
    return CadAssembly(parts=kept, tree=tree, info=info)


def _merge(parts, idx, name, key, path) -> CadPart:
    ps = [parts[k] for k in idx]
    mesh = trimesh.util.concatenate([p.mesh for p in ps]) if len(ps) > 1 else ps[0].mesh
    cyl = [c for p in ps for c in p.brep.get("cylinders", [])][:200]
    areas: dict = {}
    tot = sum(p.area_exact or 0 for p in ps) or 1.0
    for p in ps:
        for k, v in p.brep.get("area_by_type", {}).items():
            areas[k] = areas.get(k, 0.0) + v * (p.area_exact or 0) / tot
    vols = [p.volume_exact or 0.0 for p in ps]
    V = sum(vols) or 1.0
    com = (sum(np.asarray(p.com) * v for p, v in zip(ps, vols) if p.com is not None) / V).tolist() \
        if all(p.com is not None for p in ps) else None
    I = None
    if com is not None and all(p.inertia_unit is not None for p in ps):
        I = np.zeros((3, 3))
        for p, v in zip(ps, vols):
            d = np.asarray(p.com) - np.asarray(com)
            I += np.asarray(p.inertia_unit) + v * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        I = I.tolist()
    return CadPart(mesh=mesh, name=name, instance_name=name, path=path, definition=key,
                   com=com, inertia_unit=I,
                   color=ps[0].color, n_solids=sum(p.n_solids for p in ps),
                   brep={"faces": sum(p.brep.get("faces", 0) for p in ps),
                         "area_by_type": {k: round(v, 4) for k, v in areas.items()}, "cylinders": cyl},
                   volume_exact=sum(p.volume_exact or 0 for p in ps), area_exact=tot)
