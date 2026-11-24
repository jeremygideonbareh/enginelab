"""Native SolidWorks files (.SLDPRT, .SLDASM, or a .zip of them) without SolidWorks.

A SolidWorks 2015+ file is a container of raw-deflate sections. Each section header is
    14 00 06 00 08 00 | u32 | u32 | u32 compressed size | u32 size | u32 name length | name
with the name's bytes nibble-swapped ("Contents/DisplayLists", "swXmlContents/COMPINSTANCETREE").

* Geometry: the exact B-rep is a Parasolid stream (no free reader exists), but every part
  also stores the triangulated display model SolidWorks draws on screen
  ("Contents/DisplayLists"). Per B-rep face it holds triangle strips:
      k, len_1..len_k | 12 100 2 n | n x float32 xyz (metres) | 12 100 2 n | n x float32 normals
  Stitched together these give a closed mesh of the part.
* Assembly structure: "swXmlContents/COMPINSTANCETREE" is XML listing every component
  instance with its file, name, suppression state and 4x4 transform (row-vector convention:
  p' = p @ R + t, metres).

Limits, stated plainly: the mesh is SolidWorks' display tessellation (its chordal tolerance,
not the exact surfaces), so volumes are mesh volumes; exact surface statistics are not
available. For exact geometry export STEP from SolidWorks (or import into Onshape / Fusion).
"""
from __future__ import annotations

import os
import re
import struct
import xml.etree.ElementTree as ET
import zlib

import numpy as np
import trimesh

SW_EXT = {".sldprt", ".sldasm"}
_MARK = b"\x14\x00\x06\x00\x08\x00"
_TAG = struct.pack("<iii", 12, 100, 2)
_NS = "{http://www.solidworks.com/sw2003/schema}"


class SolidWorksError(ValueError):
    pass


def _swap(b: bytes) -> bytes:
    return bytes(((x << 4) & 0xF0) | (x >> 4) for x in b)


_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def _ole_sections(data: bytes):
    """Pre-2015 SolidWorks files are OLE compound documents; streams ending in "__ZLB" are
    zlib-compressed. Names are returned without the suffix, like the newer container."""
    import io
    import olefile
    ole = olefile.OleFileIO(io.BytesIO(data))
    try:
        for parts in ole.listdir():
            name = "/".join(parts)
            raw = ole.openstream(name).read()
            if name.endswith("__ZLB"):
                name = name[:-5]
                raw = _unchunk(raw)
            yield name, raw
    finally:
        ole.close()


_CHUNK_SIG = bytes.fromhex("231dd571da8148a2a85898b21b89ef99")


def _unchunk(b: bytes) -> bytes:
    """SolidWorks chunked compression: a 16-byte signature, then repeated
    (u32 uncompressed size, u32 compressed size, deflate data) chunks."""
    if b[:16] != _CHUNK_SIG:
        for w in (15, -15):
            try:
                return zlib.decompress(b, w)
            except zlib.error:
                pass
        return b
    out, i = [], 16
    while i + 8 <= len(b):
        _, cs = struct.unpack_from("<II", b, i)
        if cs == 0 or i + 8 + cs > len(b):
            break
        blob = b[i + 8:i + 8 + cs]
        for w in (-15, 15):
            try:
                out.append(zlib.decompress(blob, w))
                break
            except zlib.error:
                continue
        else:
            break
        i += 8 + cs
    return b"".join(out)



def sections(data: bytes):
    """Yield (name, payload) for every section of a SolidWorks file."""
    if data[:8] == _OLE_MAGIC:
        yield from _ole_sections(data)
        return
    i = 0
    while True:
        i = data.find(_MARK, i)
        if i < 0 or i + 26 > len(data):
            return
        _, _, csize, usize, nlen = struct.unpack_from("<IIIII", data, i + 6)
        if not (0 < nlen < 512 and csize <= len(data) - i):
            i += 1
            continue
        name = _swap(data[i + 26:i + 26 + nlen]).decode("latin-1")
        start = i + 26 + nlen
        raw = data[start:start + csize]
        try:
            payload = zlib.decompress(raw, -15)
        except zlib.error:
            payload = raw if csize == usize else None
        # The marker bytes also occur by chance inside compressed data. Only a readable name and a
        # payload that decompresses to the stated size make a real section; otherwise step on by
        # one byte (jumping by a bogus size would skip real sections).
        if payload is None or not name.isprintable() or (usize and len(payload) != usize):
            i += 1
            continue
        yield name, payload
        i = start + csize


def _faces(d: bytes):
    """(vertices, normals, strip lengths) of every tessellated face in a display list."""
    i = 0
    while True:
        i = d.find(_TAG, i)
        if i < 0:
            return
        n = struct.unpack_from("<i", d, i + 12)[0]
        vstart = i + 16
        vend = vstart + 12 * n
        if not (3 <= n < 5_000_000 and vend + 16 <= len(d) and d[vend:vend + 12] == _TAG
                and struct.unpack_from("<i", d, vend + 12)[0] == n):
            i += 4
            continue
        strips = None
        for k in range(1, min(n, 20000)):
            p = i - 4 * (k + 1)
            if p < 0:
                break
            vals = struct.unpack_from(f"<{k + 1}i", d, p)
            if vals[0] == k and min(vals[1:]) >= 3 and sum(vals[1:]) == n:
                strips = vals[1:]
                break
        V = np.frombuffer(d, dtype="<f4", count=3 * n, offset=vstart).reshape(-1, 3).astype(float)
        N = np.frombuffer(d, dtype="<f4", count=3 * n, offset=vend + 16).reshape(-1, 3).astype(float)
        if strips is not None and np.isfinite(V).all():
            yield V, N, strips
        i = vend + 16 + 12 * n


def _strip_triangles(V, N, strips) -> np.ndarray:
    tris, base = [], 0
    for L in strips:
        j = np.arange(L - 2)
        a, b, c = base + j, base + j + 1, base + j + 2
        odd = j % 2 == 1
        a2, b2 = np.where(odd, b, a), np.where(odd, a, b)
        tris.append(np.stack([a2, b2, c], 1))
        base += L
    T = np.vstack(tris)
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    flip = (fn * N[T].mean(1)).sum(1) < 0          # follow the stored vertex normals
    T[flip] = T[flip][:, [0, 2, 1]]
    return T


def part_mesh(path: str) -> trimesh.Trimesh:
    """Display mesh of one .SLDPRT in millimetres."""
    with open(path, "rb") as fh:
        data = fh.read()
    Vs, Ts, off = [], [], 0
    for name, payload in sections(data):
        if name != "Contents/DisplayLists":
            continue
        for V, N, strips in _faces(payload):
            T = _strip_triangles(V, N, strips)
            Vs.append(V)
            Ts.append(T + off)
            off += len(V)
    if not Vs:
        raise SolidWorksError(f"{os.path.basename(path)}: no stored display geometry found "
                              "(save it with a shaded view in SolidWorks, or export STEP).")
    m = trimesh.Trimesh(np.vstack(Vs) * 1000.0, np.vstack(Ts), process=False)
    m.merge_vertices(digits_vertex=4)
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(m, multibody=True)
    return m


def _instance_tree(path: str):
    with open(path, "rb") as fh:
        data = fh.read()
    xml = next((p for n, p in sections(data) if n == "swXmlContents/COMPINSTANCETREE"), None)
    if xml is None:
        raise SolidWorksError(f"{os.path.basename(path)}: no component tree stored in this assembly.")
    t = xml.decode("utf-8", "replace")
    return ET.fromstring(t[t.find("<?xml"):].encode("utf-8"))


def _find_file(folder: str, sw_path: str) -> str | None:
    base = re.split(r"[\\/]", sw_path)[-1].lower()
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower() == base:
                return os.path.join(root, f)
    return None


def read_solidworks(path: str, progress=None):
    """-> (parts, info, tree) in the same shape as the STEP reader (RawPart list, info, tree)."""
    from .io import RawPart
    ext = os.path.splitext(path)[1].lower()
    meshes: dict[str, trimesh.Trimesh] = {}
    parts: list = []
    missing: set[str] = set()
    folder = os.path.dirname(os.path.abspath(path))

    def mesh_of(fn):
        if fn not in meshes:
            meshes[fn] = part_mesh(fn)
        return meshes[fn]

    def add_part(fn, name, M, path_names, path_ids, node):
        m = mesh_of(fn).copy()
        T = np.eye(4)
        T[:3, :3] = M[:3, :3].T          # row-vector -> column-vector convention
        T[:3, 3] = M[3, :3] * 1000.0
        m.apply_transform(T)
        n_bodies = len(m.split(only_watertight=False)) if len(m.faces) < 200_000 else 1
        parts.append(RawPart(mesh=m, name=name, source="cad-body", path=list(path_names),
                             definition=os.path.basename(fn).lower(), n_solids=max(1, n_bodies),
                             path_ids=list(path_ids)))
        node["parts"].append(len(parts) - 1)
        if progress and len(parts) % 20 == 0:
            progress(len(parts))

    counter = [0]

    def walk_asm(asm_path, M_parent, path_names, path_ids, node, depth=0):
        if depth > 12:
            return
        root = _instance_tree(asm_path)
        files = {f.get("id"): (f.get("swPath"), f.get("swDocType")) for f in root.iter(_NS + "swFile")}
        models = {m.get("id"): m for m in root.iter(_NS + "swModel")}
        this = os.path.splitext(os.path.basename(asm_path))[0].lower()
        top = next((m for m in models.values()
                    if files.get(m.get("swFileRef"), ("", ""))[1] == "ASSEMBLY"
                    and os.path.splitext(re.split(r"[\\/]", files[m.get("swFileRef")][0])[-1])[0].lower() == this),
                   None) or next((m for m in models.values() if m.get("swBoundingBox")), None)
        if top is None:
            raise SolidWorksError(f"{os.path.basename(asm_path)}: could not find the assembly in its component tree.")
        for r in top.findall(_NS + "swReference"):
            if r.get("swSuppressed") == "YES":
                continue
            model = models.get(r.get("swModelRef"))
            if model is None:
                continue
            sw_path, doc = files.get(model.get("swFileRef"), ("", ""))
            M = np.array([float(x) for x in r.get("swTransform", "").split()] or np.eye(4).ravel()).reshape(4, 4)
            Mw = M @ M_parent                 # row-vector transforms compose left to right
            name = r.get("swComponentName") or r.get("swName") or os.path.basename(sw_path)
            fn = _find_file(folder, sw_path)
            if fn is None:
                missing.add(re.split(r"[\\/]", sw_path)[-1])
                continue
            if doc == "ASSEMBLY":
                counter[0] += 1
                child = {"name": name, "children": [], "parts": []}
                node["children"].append(child)
                walk_asm(fn, Mw, path_names + [name], path_ids + [counter[0]], child, depth + 1)
            else:
                add_part(fn, name, Mw, path_names, path_ids, node)

    tree = {"name": os.path.basename(path), "children": [], "parts": []}
    if ext == ".sldasm":
        name = os.path.splitext(os.path.basename(path))[0]
        node = {"name": name, "children": [], "parts": []}
        tree["children"].append(node)
        counter[0] += 1
        walk_asm(path, np.eye(4), [name], [counter[0]], node)
    elif ext == ".sldprt":
        add_part(path, os.path.splitext(os.path.basename(path))[0], np.eye(4), [], [], tree)
    else:
        raise SolidWorksError(f"Not a SolidWorks file: {path}")
    if missing and not parts:
        raise SolidWorksError("The assembly references part files that were not uploaded: "
                              + ", ".join(sorted(missing)) + ". Upload a .zip with the assembly and all its parts.")
    info = {"format": "solidworks", "definitions": len(meshes), "instances": len(parts),
            "geometry": "SolidWorks display tessellation (export STEP for exact surfaces)",
            "missing_files": sorted(missing)}
    return parts, info, tree


def top_file_in_folder(folder: str) -> str:
    """The main file of an unpacked zip: the assembly no other assembly uses, else the only part."""
    asms, prts = [], []
    for root, _, files in os.walk(folder):
        for f in files:
            e = os.path.splitext(f)[1].lower()
            (asms if e == ".sldasm" else prts if e == ".sldprt" else []).append(os.path.join(root, f))
    if asms:
        used = set()
        for a in asms:
            try:
                root = _instance_tree(a)
                used |= {re.split(r"[\\/]", f.get("swPath", ""))[-1].lower() for f in root.iter(_NS + "swFile")
                         if f.get("swDocType") == "ASSEMBLY"
                         and re.split(r"[\\/]", f.get("swPath", ""))[-1].lower() != os.path.basename(a).lower()}
            except Exception:
                continue
        tops = [a for a in asms if os.path.basename(a).lower() not in used] or asms
        return max(tops, key=os.path.getsize)
    if len(prts) == 1:
        return prts[0]
    if prts:
        return folder              # several loose parts: load them all
    raise SolidWorksError("The archive contains no SolidWorks .SLDASM or .SLDPRT files.")


def read_parts_folder(folder: str):
    """Several loose .SLDPRT files (no assembly): every part once, at its own origin."""
    from .io import RawPart
    parts, tree = [], {"name": os.path.basename(folder), "children": [], "parts": []}
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(".sldprt"):
                m = part_mesh(os.path.join(root, f))
                parts.append(RawPart(mesh=m, name=os.path.splitext(f)[0], source="cad-body",
                                     definition=f.lower()))
                tree["parts"].append(len(parts) - 1)
    return parts, {"format": "solidworks", "instances": len(parts), "definitions": len(parts)}, tree
