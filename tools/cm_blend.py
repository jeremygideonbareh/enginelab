"""Leave-one-assembly-out test of blending the full prediction with a mesh-only prediction
(B-rep + parent features blanked), so synthetic examples count for CAD inputs too.
python tools/cm_blend.py [kind] [--synthetic=N]"""
import glob, os, sys, warnings
import numpy as np
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, f1_score, classification_report  # noqa: E402
from enginelab.component_model import (BREP_KEYS, assembly_rows, fit_text_model, make_model,  # noqa: E402
                                       oof_text_features, text_features)

kind = next((a for a in sys.argv[1:] if not a.startswith("--")), "rf")
nsyn = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--synthetic=")), "16000"))
norm = lambda f: os.path.normcase(os.path.abspath(f))
files = list({norm(f): f for f in sorted(glob.glob("data/real_cad/*/*.step") + glob.glob("data/real_cad/*/*.STEP"))}.values())
data = {f: assembly_rows(f) for f in files}
from joblib import Parallel, delayed
from enginelab.std_generator import dataset_rows
syn_X, syn_y = [], []
for X, y in Parallel(n_jobs=-1)(delayed(dataset_rows)(200, seed=1000 + k) for k in range(nsyn // 200)):
    syn_X += X; syn_y += y
classes = sorted({l for _, y, _ in data.values() for l in y} | set(syn_y))
feat = sorted({k for X, _, _ in data.values() for r in X for k in r} | {k for r in syn_X for k in r})
names = feat + [f"ctx_{c}" for c in classes]
blank_idx = [names.index(k) for k in BREP_KEYS if k in names] + list(range(len(feat), len(names)))
base = lambda rows: np.array([[r.get(k, np.nan) for k in feat] for r in rows], dtype=float)
S0 = np.hstack([base(syn_X), np.full((len(syn_X), len(classes)), np.nan)])
CTX = ["rel_L1", "rel_vol", "vol_rank", "copies", "copy_line", "axis_vs_line", "long_vs_line", "copy_spacing",
       "neighbors", "inside_bigger", "encloses", "sym_vs_main", "long_vs_main", "coaxial"]
ctx_idx = [names.index(k) for k in CTX if k in names]
transplant = "--transplant" in sys.argv


def graft(S0, Xreal, yreal, rng):
    """Synthetic parts get the assembly context of a real training part with the same label."""
    S = S0.copy()
    yreal = np.asarray(yreal)
    for lab in set(syn_y):
        rows = np.nonzero(np.asarray(syn_y) == lab)[0]
        pool = np.nonzero(yreal == lab)[0]
        if len(pool) == 0:
            pool = np.arange(len(yreal))
        S[np.ix_(rows, ctx_idx)] = Xreal[np.ix_(rng.choice(pool, len(rows)), ctx_idx)]
    return S
Y, P = [], {w: [] for w in (0.0, 0.25, 0.4, 0.5, 0.6, 1.0)}
for hold in files:
    tr = [f for f in files if f != hold]
    texts = [t for f in tr for t in data[f][2]]; labels = [l for f in tr for l in data[f][1]]
    groups = [f for f in tr for _ in data[f][1]]
    full = np.hstack([np.vstack([base(data[f][0]) for f in tr]), oof_text_features(texts, labels, groups, classes)])
    bl = full.copy(); bl[:, blank_idx] = np.nan
    S = graft(S0, full, labels, np.random.default_rng(0)) if transplant else S0
    clf = make_model(kind).fit(np.vstack([full, bl, S]), np.concatenate([labels, labels, syn_y]))
    Xh, yh, Th = data[hold]
    Xte = np.hstack([base(Xh), text_features(fit_text_model(texts, labels), Th, classes)])
    Xm = Xte.copy(); Xm[:, blank_idx] = np.nan
    pf, pm = clf.predict_proba(Xte), clf.predict_proba(Xm)
    Y += list(yh)
    for w in P:
        P[w] += list(clf.classes_[((1 - w) * pf + w * pm).argmax(1)])
    print(os.path.basename(hold), " ".join(f"w{w}:{accuracy_score(yh, P[w][-len(yh):]):.3f}" for w in P), flush=True)
for w in P:
    rep = classification_report(Y, P[w], output_dict=True, zero_division=0)
    print(f"w={w}: acc {accuracy_score(Y, P[w]):.4f} macroF1 {f1_score(Y, P[w], average='macro'):.4f} "
          f"motor {rep['motor']['f1-score']:.3f} (rec {rep['motor']['recall']:.2f}) elec {rep['electronics']['f1-score']:.3f} "
          f"trans {rep['transmission']['f1-score']:.3f} bearing {rep['bearing']['f1-score']:.3f}")
