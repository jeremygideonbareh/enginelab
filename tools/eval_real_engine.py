"""Engine part classifier on REAL engine CAD, from geometry alone (part names hidden).
Truth comes from the part names. python tools/eval_real_engine.py <cad file> [model.joblib ...]"""
import sys
import warnings
from collections import Counter

import joblib

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import enginelab.classify as C  # noqa: E402
from enginelab.breakdown import display_name, fast_shape_features  # noqa: E402
from enginelab.features import context_features  # noqa: E402
from enginelab.io import load_file  # noqa: E402
from enginelab.segment import extract_parts  # noqa: E402

path = sys.argv[1]
models = sys.argv[2:] or [C.MODEL_PATH]
raws = extract_parts(load_file(path).parts)
names = [display_name(r.name) for r in raws]
truth = [C.label_from_name(n) for n in names]
infos = fast_shape_features(raws)
ctx = context_features(infos)
for m in models:
    C._cached = joblib.load(m)
    preds = C.classify(infos, ctx, [None] * len(raws))
    ok = [p.label == t for p, t in zip(preds, truth) if t]
    print(f"{m}: {sum(ok)}/{len(ok)} parts right ({100 * sum(ok) / max(len(ok), 1):.0f}%) from geometry alone")
    wrong = Counter((t, p.label) for p, t in zip(preds, truth) if t and p.label != t)
    for (t, q), k in wrong.most_common():
        print(f"    {k:3d} x {t:16s} -> {q}")
