"""Leave-one-assembly-out comparison of component-classifier training variants, reporting
overall accuracy, macro F1 and the spread of accuracy across held-out assemblies.
python tools/cm_variants.py [variant ...]"""
import glob
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

from enginelab.component_model import (BREP_KEYS, CONTEXT_KEYS, assembly_rows, fit_text_model,  # noqa: E402
                                       graft_context, oof_text_features, text_features)

norm = lambda f: os.path.normcase(os.path.abspath(f))
FILES = list({norm(f): f for f in sorted(glob.glob("data/real_cad/*/*.step") + glob.glob("data/real_cad/*/*.STEP"))}.values())
data = {f: assembly_rows(f) for f in FILES}
from joblib import Parallel, delayed  # noqa: E402
from enginelab.std_generator import dataset_rows  # noqa: E402
syn_X, syn_y = [], []
for X, y in Parallel(n_jobs=-1)(delayed(dataset_rows)(200, seed=1000 + k) for k in range(80)):
    syn_X += X
    syn_y += y
classes = sorted({l for _, y, _ in data.values() for l in y} | set(syn_y))
feat = sorted({k for X, _, _ in data.values() for r in X for k in r} | {k for r in syn_X for k in r})
names = feat + [f"ctx_{c}" for c in classes]
blank_idx = [names.index(k) for k in BREP_KEYS if k in names] + list(range(len(feat), len(names)))
asm_idx = [names.index(k) for k in CONTEXT_KEYS if k in names]
base = lambda rows: np.array([[r.get(k, np.nan) for k in feat] for r in rows], dtype=float)
S0 = np.hstack([base(syn_X), np.full((len(syn_X), len(classes)), np.nan)])


def model(kind):
    imp = SimpleImputer(strategy="constant", fill_value=-1.0)
    if kind == "et":
        return make_pipeline(imp, ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt",
                                                       class_weight="balanced", n_jobs=-1, random_state=0))
    if kind == "rf_tuned":
        return make_pipeline(imp, RandomForestClassifier(n_estimators=600, min_samples_leaf=1, max_features=0.25,
                                                         class_weight="balanced_subsample", n_jobs=-1, random_state=0))
    return make_pipeline(imp, RandomForestClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt",
                                                     class_weight="balanced_subsample", n_jobs=-1, random_state=0))


def weights(tr, scheme):
    """Per-row training weights for the real rows of the training assemblies."""
    ws = []
    for f in tr:
        X, y, _ = data[f]
        n = len(y)
        w = np.ones(n)
        if scheme in ("dedup", "both"):
            keyrows = [tuple(np.round([r.get(k, 0) or 0 for k in ("log_L2", "log_L3", "flatness", "elongation")], 3)) + (l,)
                       for r, l in zip(X, y)]
            cnt = {}
            for k in keyrows:
                cnt[k] = cnt.get(k, 0) + 1
            w = np.array([1.0 / np.sqrt(cnt[k]) for k in keyrows])
        if scheme in ("asm", "both"):
            w = w * (1.0 / w.sum()) * 600.0             # every assembly carries the same total weight
        ws.append(w)
    return np.concatenate(ws)


def run(variant):
    kind = {"et": "et", "rf_tuned": "rf_tuned"}.get(variant, "rf")
    scheme = {"asm_weight": "asm", "dedup": "dedup", "both": "both"}.get(variant, None)
    Y, P, per = [], [], {}
    for hold in FILES:
        tr = [f for f in FILES if f != hold]
        texts = [t for f in tr for t in data[f][2]]
        labels = [l for f in tr for l in data[f][1]]
        groups = [f for f in tr for _ in data[f][1]]
        full = np.hstack([np.vstack([base(data[f][0]) for f in tr]), oof_text_features(texts, labels, groups, classes)])
        bl = full.copy()
        bl[:, blank_idx] = np.nan
        S = graft_context(S0, syn_y, full, labels, asm_idx)
        wr = weights(tr, scheme) if scheme else np.ones(len(labels))
        ws = np.full(len(syn_y), float(np.median(wr)))
        clf = model(kind).fit(np.vstack([full, bl, S]), np.concatenate([labels, labels, syn_y]),
                              **{f"{model(kind).steps[-1][0]}__sample_weight": np.concatenate([wr, wr, ws])})
        Xh, yh, Th = data[hold]
        Xte = np.hstack([base(Xh), text_features(fit_text_model(texts, labels), Th, classes)])
        p = clf.predict(Xte)
        per[os.path.basename(os.path.dirname(hold))] = accuracy_score(yh, p)
        Y += list(yh)
        P += list(p)
    acc = np.array(list(per.values()))
    rep = f1_score(Y, P, labels=["motor"], average=None)[0]
    print(f"{variant:12s} acc {accuracy_score(Y, P):.4f}  macroF1 {f1_score(Y, P, average='macro'):.4f}  "
          f"per-assembly mean {acc.mean():.3f} sd {acc.std():.3f} worst {acc.min():.3f}  motorF1 {rep:.3f}  "
          + " ".join(f"{k[:6]}={v:.2f}" for k, v in per.items()), flush=True)


for v in (sys.argv[1:] or ["baseline", "asm_weight", "dedup", "both", "et", "rf_tuned"]):
    t = time.time()
    run(v)
    print(f"   ({time.time() - t:.0f}s)", flush=True)
