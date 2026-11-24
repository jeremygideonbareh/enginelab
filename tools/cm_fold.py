"""Train on every real assembly except one and show what goes wrong on that one, by name.
python tools/cm_fold.py <held-out step file> [model kind]"""
import glob
import sys
import warnings
from collections import Counter

import numpy as np

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from enginelab.breakdown import display_name  # noqa: E402
from enginelab.component_model import (BREP_KEYS, assembly_rows, fit_text_model, make_model, name_label,  # noqa: E402
                                       oof_text_features, text_features)
from enginelab.io import load_file  # noqa: E402
from enginelab.segment import extract_parts  # noqa: E402

import os
norm = lambda f: os.path.normcase(os.path.abspath(f))
hold = sys.argv[1]
kind = sys.argv[2] if len(sys.argv) > 2 else "rf"
files = [f for f in sorted(set(glob.glob("data/real_cad/*/*.step") + glob.glob("data/real_cad/*/*.STEP")),
                           key=norm) if norm(f) != norm(hold)]
files = list({norm(f): f for f in files}.values())   # Windows globs are case-insensitive: drop duplicates
assert norm(hold) not in {norm(f) for f in files}
data = {f: assembly_rows(f) for f in files + [hold]}
classes = sorted({l for _, y, _ in data.values() for l in y})
feat = sorted({k for X, _, _ in data.values() for r in X for k in r})
base = lambda rows: np.array([[r.get(k, np.nan) for k in feat] for r in rows], dtype=float)
texts = [t for f in files for t in data[f][2]]
labels = [l for f in files for l in data[f][1]]
groups = [f for f in files for _ in data[f][1]]
Xtr = np.hstack([np.vstack([base(data[f][0]) for f in files]), oof_text_features(texts, labels, groups, classes)])
clf = make_model(kind).fit(Xtr, np.array(labels))
Xh, yh, Th = data[hold]
Xte = np.hstack([base(Xh), text_features(fit_text_model(texts, labels), Th, classes)])
p = clf.predict(Xte)
raws = extract_parts(load_file(hold).parts)
names = [display_name(r.name) for r in raws if name_label(r.name, r.path) is not None]
print(f"accuracy {np.mean(p == np.array(yh)):.3f} on {len(yh)} parts")
c = Counter((t, q, n) for t, q, n in zip(yh, p, names) if t != q)
for (t, q, n), k in c.most_common(30):
    print(f"  {t:13s} -> {q:13s} {k:3d} x {n[:55]}")
