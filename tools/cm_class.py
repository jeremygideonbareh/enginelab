"""Leave-one-assembly-out view of ONE class: every real part with that true label or that
prediction, by name.   python tools/cm_class.py motor [kind] [--synthetic=N]"""
import glob, os, sys, warnings
from collections import Counter
import numpy as np
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
from enginelab.breakdown import display_name  # noqa: E402
from enginelab.component_model import (BREP_KEYS, assembly_rows, fit_text_model, make_model, name_label,  # noqa: E402
                                       oof_text_features, text_features)
from enginelab.io import load_file  # noqa: E402
from enginelab.segment import extract_parts  # noqa: E402

target = sys.argv[1]
kind = next((a for a in sys.argv[2:] if not a.startswith("--")), "rf")
nsyn = int(next((a.split("=")[1] for a in sys.argv[2:] if a.startswith("--synthetic=")), "0"))
norm = lambda f: os.path.normcase(os.path.abspath(f))
files = list({norm(f): f for f in sorted(glob.glob("data/real_cad/*/*.step") + glob.glob("data/real_cad/*/*.STEP"))}.values())
data = {f: assembly_rows(f) for f in files}
names = {}
for f in files:
    raws = extract_parts(load_file(f).parts)
    names[f] = [(display_name(r.name), " / ".join(display_name(p) for p in r.path[-3:-1]))
                for r in raws if name_label(r.name, r.path) is not None]
syn_X, syn_y = [], []
if nsyn:
    from joblib import Parallel, delayed
    from enginelab.std_generator import dataset_rows
    for X, y in Parallel(n_jobs=-1)(delayed(dataset_rows)(200, seed=1000 + k) for k in range(nsyn // 200)):
        syn_X += X; syn_y += y
classes = sorted({l for _, y, _ in data.values() for l in y} | set(syn_y))
feat = sorted({k for X, _, _ in data.values() for r in X for k in r} | {k for r in syn_X for k in r})
base = lambda rows: np.array([[r.get(k, np.nan) for k in feat] for r in rows], dtype=float)
rows = []
for hold in files:
    tr = [f for f in files if f != hold]
    texts = [t for f in tr for t in data[f][2]]; labels = [l for f in tr for l in data[f][1]]
    groups = [f for f in tr for _ in data[f][1]]
    Xtr = np.hstack([np.vstack([base(data[f][0]) for f in tr]), oof_text_features(texts, labels, groups, classes)])
    ytr = np.array(labels)
    if syn_X:
        Xtr = np.vstack([Xtr, np.hstack([base(syn_X), np.full((len(syn_X), len(classes)), np.nan)])])
        ytr = np.concatenate([ytr, syn_y])
    clf = make_model(kind).fit(Xtr, ytr)
    Xh, yh, Th = data[hold]
    Xte = np.hstack([base(Xh), text_features(fit_text_model(texts, labels), Th, classes)])
    P = clf.predict_proba(Xte); cl = list(clf.classes_)
    for (n, par), t, pr in zip(names[hold], yh, P):
        q = cl[int(pr.argmax())]
        if t == target or q == target:
            rows.append((os.path.basename(os.path.dirname(hold)), t, q, pr[cl.index(target)], n, par))
tp = sum(r[1] == target and r[2] == target for r in rows)
print(f"{target}: true {sum(r[1]==target for r in rows)}  predicted {sum(r[2]==target for r in rows)}  correct {tp}")
c = Counter((a, t, q, n, par) for a, t, q, _, n, par in rows)
for (a, t, q, n, par), k in sorted(c.items(), key=lambda x: (x[0][1] != target, x[0][0])):
    print(f"{'OK ' if t == q else 'XX '}{a[:12]:12s} {t:12s}->{q:12s} {k:3d}x {n[:38]:38s} | {par[:40]}")
