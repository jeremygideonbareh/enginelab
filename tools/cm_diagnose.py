"""Where does the component model go wrong? Leave-one-assembly-out predictions with the
part names, so label noise and real mistakes can be told apart.
python tools/cm_diagnose.py
"""
import glob
import sys
import warnings
from collections import Counter

import numpy as np

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from enginelab.component_model import BREP_KEYS, assembly_rows  # noqa: E402
from enginelab.std_generator import dataset_rows  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

FILES = sorted(glob.glob("data/real_cad/*/*.step"))


def names_of(path):
    from enginelab.breakdown import display_name
    from enginelab.component_model import name_label
    from enginelab.io import load_file
    from enginelab.segment import extract_parts
    raws = extract_parts(load_file(path).parts)
    return [(display_name(r.name), " / ".join(r.path[-2:])) for r in raws if name_label(r.name, r.path) is not None]


data = {f: assembly_rows(f) for f in FILES}
names = {f: names_of(f) for f in FILES}
syn_X, syn_y = [], []
for k in range(30):
    X, y = dataset_rows(200, seed=1000 + k)
    syn_X += X
    syn_y += y
keys = sorted({k for X, _ in data.values() for r in X for k in r} | {k for r in syn_X for k in r})
M = lambda rows: np.array([[r.get(k, np.nan) for k in keys] for r in rows], dtype=float)
conf = Counter()
examples = []
for hold in FILES:
    Xs, ys = [M(syn_X)], [np.array(syn_y)]
    for f in FILES:
        if f != hold:
            Xs.append(M(data[f][0]))
            ys.append(np.array(data[f][1]))
    clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.06, class_weight="balanced", random_state=0)
    clf.fit(np.vstack(Xs), np.concatenate(ys))
    X, y = data[hold]
    p = clf.predict(M(X))
    for (nm, path), t, q in zip(names[hold], y, p):
        if t != q:
            conf[(t, q)] += 1
            examples.append((t, q, nm, path, hold.split("/")[-1][:12]))
print("most common errors (true -> predicted):")
for (t, q), n in conf.most_common(15):
    print(f"  {t:14s} -> {q:14s} {n}")
    for e in [e for e in examples if e[0] == t and e[1] == q][:6]:
        print(f"        {e[2][:40]:40s} | {e[3][:40]:40s} | {e[4]}")
