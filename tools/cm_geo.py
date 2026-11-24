"""Geometry of named parts matching a regex.  python tools/cm_geo.py <regex> [assembly-glob]"""
import glob, re, sys, warnings
sys.path.insert(0, "."); warnings.filterwarnings("ignore")
import numpy as np
from enginelab.breakdown import display_name
from enginelab.components import from_name
from enginelab.io import load_file
from enginelab.segment import extract_parts
pat = re.compile(sys.argv[1], re.I)
seen = set()
for f in sorted(glob.glob(f"data/real_cad/{sys.argv[2] if len(sys.argv) > 2 else '*'}/*.step")):
    for r in extract_parts(load_file(f).parts):
        n = display_name(r.name)
        if not pat.search(n or ""):
            continue
        ext = np.sort(r.mesh.extents)[::-1]
        key = (f, n, tuple(np.round(ext)))
        if key in seen:
            continue
        seen.add(key)
        c = from_name(n)
        a = (r.brep or {}).get("area_by_type", {})
        top = ", ".join(f"{k} {v:.2f}" for k, v in sorted(a.items(), key=lambda x: -x[1])[:3])
        print(f"{f.split('/')[-2][:10]:10s} {n[:30]:30s} ext {ext.round(1)} vol {(r.volume_exact or 0)/1e3:6.1f}cm3 "
              f"sol {r.n_solids} faces {(r.brep or {}).get('faces')} | {top} | {c.family + '/' + c.category if c else None}",
              flush=True)
