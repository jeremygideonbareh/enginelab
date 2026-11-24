"""Does adding labelled REAL engine parts to the generated training data help on real engines?
Leave-one-real-engine-out: train on generated engines + the other real engines (weighted),
test on the held-out real engine from geometry alone.   python tools/engine_real_mix.py [weights...]"""
import glob
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

import enginelab.classify as C  # noqa: E402
from enginelab.breakdown import display_name, fast_shape_features  # noqa: E402
from enginelab.features import context_features  # noqa: E402
from enginelab.io import RawPart, load_file  # noqa: E402
from enginelab.segment import extract_parts  # noqa: E402
from enginelab.solidworks import part_mesh  # noqa: E402


def real_engines():
    """{engine: (feature rows, labels)} with hand truth (test_engines/truth.json) or name labels."""
    base = "data/real_cad/test_engines"
    truth = json.load(open(os.path.join(base, "truth.json")))
    out = {}
    for key, t in truth.items():
        p = os.path.join(base, key)
        rows, labs = [], []
        if os.path.isdir(p):           # loose parts: each one alone
            for f in sorted(glob.glob(os.path.join(p, "*.SLDPRT"))):
                n = os.path.splitext(os.path.basename(f))[0]
                if n in t:
                    infos = fast_shape_features([RawPart(mesh=part_mesh(f), name=n)])
                    rows.append({**infos[0].features, **context_features(infos)[0]})
                    labs.append(t[n])
        else:
            raws = extract_parts(load_file(p).parts)
            infos = fast_shape_features(raws)
            ctx = context_features(infos)
            for r, i, c in zip(raws, infos, ctx):
                lab = t.get(display_name(r.name))
                if lab:
                    rows.append({**i.features, **c})
                    labs.append(lab)
        out[key.split("/")[0]] = (rows, labs)
    crank = "data/real_cad/crank454/crankshaft_454.zip"
    if os.path.exists(crank):
        raws = extract_parts(load_file(crank).parts)
        infos = fast_shape_features(raws)
        ctx = context_features(infos)
        out["crank454"] = ([{**i.features, **c} for i, c in zip(infos, ctx)],
                           [C.label_from_name(display_name(r.name)) for r in raws])
    return out


if __name__ == "__main__":
    weights = [float(w) for w in sys.argv[1:]] or [0.0, 20.0, 100.0]
    X, y, groups, names = C.build_dataset(1200)
    real = real_engines()
    M = {k: (np.array([[r.get(n, np.nan) for n in names] for r in v[0]], float), np.array(v[1])) for k, v in real.items()}
    print({k: len(v[1]) for k, v in M.items()})
    for w in weights:
        tot = ok = 0
        per = {}
        for hold in M:
            others = [k for k in M if k != hold]
            Xr = np.vstack([X] + [M[k][0] for k in others]) if others else X
            yr = np.concatenate([y] + [M[k][1] for k in others]) if others else y
            sw = np.concatenate([np.ones(len(y))] + [np.full(len(M[k][1]), w) for k in others])
            clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=31,
                                                 l2_regularization=1e-3, class_weight="balanced", random_state=0)
            clf.fit(Xr, yr, sample_weight=sw if w > 0 else None)
            p = clf.predict(M[hold][0])
            per[hold] = f"{int((p == M[hold][1]).sum())}/{len(p)}"
            ok += int((p == M[hold][1]).sum())
            tot += len(p)
        print(f"weight {w:6.1f}: {ok}/{tot} = {100 * ok / tot:.0f}% | " + " ".join(f"{k} {v}" for k, v in per.items()), flush=True)
