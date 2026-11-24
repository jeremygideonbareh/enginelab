"""Engine part classifier on REAL engine CAD from geometry alone (names hidden), against
hand-written truth (data/real_cad/test_engines/truth.json) plus the crank454 assembly.
A folder means loose parts: each is classified on its own, as a single-part upload.
python tools/eval_real_engines.py [model.joblib ...]"""
import glob
import json
import os
import sys
import warnings
from collections import Counter

import joblib

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import enginelab.classify as C  # noqa: E402
from enginelab.breakdown import display_name, fast_shape_features  # noqa: E402
from enginelab.features import context_features  # noqa: E402
from enginelab.io import RawPart, load_file  # noqa: E402
from enginelab.segment import extract_parts  # noqa: E402
from enginelab.solidworks import part_mesh  # noqa: E402

BASE = "data/real_cad/test_engines"
truth = json.load(open(os.path.join(BASE, "truth.json")))
cases = []          # (case name, list of (raws list, truth labels)) ; each raws list classified together
for key, t in truth.items():
    p = os.path.join(BASE, key)
    if os.path.isdir(p):
        groups = []
        for f in sorted(glob.glob(os.path.join(p, "*.SLDPRT"))):
            n = os.path.splitext(os.path.basename(f))[0]
            if n in t:
                groups.append(([RawPart(mesh=part_mesh(f), name=n)], [t[n]]))
        cases.append((key, groups))
    else:
        raws = extract_parts(load_file(p).parts)
        lab = [t.get(display_name(r.name)) for r in raws]
        cases.append((key, [(raws, lab)]))
crank = "data/real_cad/crank454/crankshaft_454.zip"
if os.path.exists(crank):
    raws = extract_parts(load_file(crank).parts)
    cases.append(("crank454", [(raws, [C.label_from_name(display_name(r.name)) for r in raws])]))
prepared = []
for name, groups in cases:
    for raws, lab in groups:
        infos = fast_shape_features(raws)
        prepared.append((name, infos, context_features(infos), lab, [display_name(r.name) for r in raws]))
for m in (sys.argv[1:] or [C.MODEL_PATH]):
    C._cached = joblib.load(m)
    tot_ok = tot = 0
    per = Counter()
    wrong = Counter()
    for name, infos, ctx, lab, nm in prepared:
        preds = C.classify(infos, ctx, nm if os.environ.get("WITH_NAMES") else [None] * len(infos))
        for p, t in zip(preds, lab):
            if t:
                tot += 1
                per[(name, "n")] += 1
                if p.label == t:
                    tot_ok += 1
                    per[(name, "ok")] += 1
                else:
                    wrong[(name, t, p.label)] += 1
    print(f"{os.path.basename(m)}: {tot_ok}/{tot} = {100 * tot_ok / tot:.0f}% | " +
          " | ".join(f"{n}: {per[(n, 'ok')]}/{per[(n, 'n')]}" for n in dict.fromkeys(c[0] for c in prepared)))
    for (n, t, q), k in wrong.most_common(12):
        print(f"      {n:22s} {t:15s} -> {q} x{k}")
