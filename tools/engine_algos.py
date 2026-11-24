"""Which algorithm trained on generated engines generalises best to REAL engines (geometry only)?
python tools/engine_algos.py"""
import sys, time, warnings
import numpy as np
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
sys.argv = sys.argv[:1]
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
import enginelab.classify as C
from tools.engine_real_mix import real_engines
X, y, groups, names = C.build_dataset(1200)
real = real_engines()
M = {k: (np.array([[r.get(n, np.nan) for n in names] for r in v[0]], float), np.array(v[1])) for k, v in real.items()}
Xi = np.nan_to_num(X, nan=-1.0)
algos = {
    "HGB (current)": lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1e-3, class_weight="balanced", random_state=0),
    "HGB regularised": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=1.0, class_weight="balanced", random_state=0),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=0),
    "ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=0),
}
for name, mk in algos.items():
    t = time.time()
    clf = mk().fit(X if "HGB" in name else Xi, y)
    res, ok, tot = [], 0, 0
    for k, (Xr, yr) in M.items():
        p = clf.predict(Xr if "HGB" in name else np.nan_to_num(Xr, nan=-1.0))
        ok += int((p == yr).sum()); tot += len(yr); res.append(f"{k} {int((p == yr).sum())}/{len(yr)}")
    print(f"{name:18s} real engines {ok}/{tot} = {100 * ok / tot:.0f}% | {' '.join(res)} | {time.time() - t:.0f}s", flush=True)
