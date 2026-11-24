"""Learned component classifier for robotics / mechanical assemblies.

Training data are REAL CAD assemblies: every part whose name identifies it
("M3x8 SHCS", "683-ZZ", "MGN9H carriage", "NEMA17", "Bracket_Left"...) is a labelled
example, plus synthetic standard parts. The model recognises the same families WITHOUT
the part's own name, so unnamed parts (STL uploads, "Body1", "SOLID") are identified too.

Features
  * shape descriptors + assembly context (copies, coaxial parts, size rank...)
  * exact B-rep surface statistics (surface-type shares, hole/shaft cylinders, genus...)
  * parent context: a text model over the names of the sub-assemblies the part sits in
    ("Fans", "Bowden_Hardware", "NEMA17-40-PGB"), which real CAD keeps even when the part
    itself is unnamed. It is fitted inside every training fold only (no leakage).
Every real example is also added with B-rep and parent features blanked, so one model
serves CAD files, plain meshes and parts with no parents.

Evaluation is leave-one-assembly-out: train on all assemblies but one, test on the one
it has never seen. Several algorithms are compared that way and the best is kept.

    python -m enginelab.component_model [--synthetic=12000] [files...]
"""
from __future__ import annotations

import glob
import math
import os
import re
import sys
import time

import joblib
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "models", "component_classifier.joblib")
# Model for files that contain ONE part (trained with "uploaded alone" rows): 56% -> 90% on lone parts
# held out by assembly, while the assembly model above stays best inside assemblies (91.0% vs 90.1%).
ALONE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "models", "component_classifier_alone.joblib")
ROWS_VERSION = 7
SURF = ["plane", "cylinder", "cone", "sphere", "torus", "bspline", "other"]
BREP_KEYS = [f"brep_{s}" for s in SURF] + ["brep_holes", "brep_shafts", "brep_max_shaft_rel", "brep_min_hole_rel",
                                            "brep_faces_log", "brep_genus", "brep_solids", "brep_radii",
                                            "brep_cyl_concentration", "brep_hole_frac"]


# Assembly-context features (features.context_features). A synthetic part has no real
# assembly around it, so it borrows the context of a real training part with the same label;
# otherwise "relative volume 0, no neighbours" alone tells synthetic from real and the forest
# ignores synthetic examples for real CAD.
CONTEXT_KEYS = ["rel_L1", "rel_vol", "vol_rank", "copies", "copy_line", "axis_vs_line", "long_vs_line",
                "copy_spacing", "neighbors", "inside_bigger", "encloses", "sym_vs_main", "long_vs_main", "coaxial"]


def alone_context(M: np.ndarray, names: list[str]) -> np.ndarray:
    """The assembly context every part gets when it is uploaded ON ITS OWN (features.context_features
    of a one-part list), computed from the shape columns already in M. Without these rows the only
    lone parts in training were library electronics, so the model called every lone part electronics."""
    M = M.copy()
    col = {n: i for i, n in enumerate(names)}
    L = [10 ** M[:, col[k]] if k in col else np.ones(len(M)) for k in ("log_L1", "log_L2", "log_L3")]
    vals = {k: 0.0 for k in CONTEXT_KEYS}
    vals["long_vs_main"] = 1.0
    for k, v in vals.items():
        if k in col:
            M[:, col[k]] = v
    if "rel_L1" in col:
        M[:, col["rel_L1"]] = L[0] / np.sqrt(L[0] ** 2 + L[1] ** 2 + L[2] ** 2)
    if "sym_vs_main" in col and "sym_is_long" in col:
        M[:, col["sym_vs_main"]] = M[:, col["sym_is_long"]]
    return M


def graft_context(S: np.ndarray, syn_y, X_real: np.ndarray, y_real, idx: list[int], seed: int = 0) -> np.ndarray:
    S = S.copy()
    rng = np.random.default_rng(seed)
    syn_y, y_real = np.asarray(syn_y), np.asarray(y_real)
    if len(y_real) == 0 or not idx:
        return S
    for lab in np.unique(syn_y):
        rows = np.nonzero(syn_y == lab)[0]
        pool = np.nonzero(y_real == lab)[0]
        if len(pool) == 0:
            pool = np.arange(len(y_real))
        S[np.ix_(rows, idx)] = X_real[np.ix_(rng.choice(pool, len(rows)), idx)]
    return S


def assembly_weights(X: list[dict], y: list[str], total: float = 600.0) -> np.ndarray:
    """Training weight of each labelled part of ONE assembly. Identical copies (200 M3 screws)
    count as sqrt(n) parts, not n, and every assembly carries the same total weight, so a
    1,400-part printer does not drown out a 100-part robot arm. Measured leave-one-assembly-out
    (tools/cm_variants.py): per-assembly accuracy mean 89.7% -> 91.0%, worst 74% -> 79%."""
    keys = [tuple(np.round([r.get(k, 0) or 0 for k in ("log_L2", "log_L3", "flatness", "elongation")], 3)) + (l,)
            for r, l in zip(X, y)]
    cnt: dict = {}
    for k in keys:
        cnt[k] = cnt.get(k, 0) + 1
    w = np.array([1.0 / math.sqrt(cnt[k]) for k in keys])
    return w * (total / w.sum()) if len(w) else w


def _fit(model, X, y, w):
    """fit() with sample weights through a Pipeline (named last step) or a plain estimator."""
    if hasattr(model, "steps"):
        return model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
    return model.fit(X, y, sample_weight=w)


def brep_features(brep: dict | None, info, n_solids: int = 1) -> dict:
    if not brep or "area_by_type" not in brep:
        return {k: np.nan for k in BREP_KEYS}
    a = brep.get("area_by_type", {})
    cyl = brep.get("cylinders", [])
    holes = [c for c in cyl if c["hole"]]
    shafts = [c for c in cyl if not c["hole"]]
    L1 = float(max(info.extents[0], 1e-6))
    areas = np.array([c["area"] for c in cyl]) if cyl else np.zeros(0)
    radii = {round(c["r"], 2) for c in cyl}
    return {**{f"brep_{s}": float(a.get(s, 0.0)) for s in SURF},
            "brep_holes": float(np.log1p(len(holes))), "brep_shafts": float(np.log1p(len(shafts))),
            "brep_max_shaft_rel": (2 * max(c["r"] for c in shafts) / L1) if shafts else 0.0,
            "brep_min_hole_rel": (2 * min(c["r"] for c in holes) / L1) if holes else 0.0,
            "brep_faces_log": float(np.log1p(brep.get("faces", 0))),
            "brep_genus": float(min(brep.get("genus", 0) or 0, 50)), "brep_solids": float(min(n_solids, 20)),
            "brep_radii": float(np.log1p(len(radii))),
            "brep_cyl_concentration": float(areas.max() / areas.sum()) if areas.sum() > 0 else 0.0,
            "brep_hole_frac": float(sum(c["area"] for c in holes) / areas.sum()) if areas.sum() > 0 else 0.0}


def row(info, ctx: dict, brep, n_solids=1) -> dict:
    L = np.maximum(info.extents, 1e-6)
    return {**info.features, **ctx, **brep_features(brep, info, n_solids),
            "log_L2": float(np.log10(L[1])), "log_L3": float(np.log10(L[2])),
            "flatness": float(L[2] / L[1]), "elongation": float(L[1] / L[0])}


def parent_text(path: list | None) -> str:
    """Names of the sub-assemblies a part sits in (not its own name), normalised."""
    if not path:
        return ""
    from .breakdown import display_name
    words = " ".join(display_name(p) or "" for p in path[-3:])
    return re.sub(r"[_\-./()]+", " ", words.lower()).strip()


BOUGHT_MOTOR = re.compile(r"servo|nema[\s_-]*\d|\bsts?\d{4}|dynamixel|\b(xl|xc|xm|xh|xw)[\s-]?\d{3}|mg99\d|sg90|"
                          r"gear\s*motor|bldc|outrunner", re.I)


def _bought_parent(path: list | None) -> str | None:
    """'motor' when the part sits inside a bought motor / servo sub-assembly (its own pieces
    are then pieces of that motor), else None. Mounts and adapters named after a motor don't count."""
    from .breakdown import display_name
    from .components import CUSTOM_HINT
    for p in (path or [])[-2:]:
        p = display_name(p) or ""
        if BOUGHT_MOTOR.search(p) and not CUSTOM_HINT.search(p) and \
                not re.search(r"assembly|assy|arm|robot|pgb|gear|planet|drive|lead\s*screw|module", p, re.I):
            return "motor"
    return None


def name_label(name: str | None, path: list | None) -> str | None:
    """Label from the CAD name: a component family, 'custom' for a named made-for-this
    design part, None when the name says nothing (exporter defaults)."""
    from .breakdown import FAMILY_TO_LABEL, display_name
    from .components import CUSTOM_HINT, ELECTRONIC_PART, ROBOTIS_HOUSING, from_name
    from .cad import GENERIC_NAME
    n = display_name(name)
    if not n or GENERIC_NAME.match(n) or n.lower().startswith("part "):
        return None
    c = from_name(n)
    if path and ELECTRONIC_PART.search(display_name(path[-1]) or ""):
        return None            # a pin / body of an IC package: the rules name it from its parent
    if c is None and not CUSTOM_HINT.search(n) and _bought_parent(path):
        return "motor"         # an unnamed-kind piece of a bought servo / motor (shell, case) is that motor
    strong = c is not None and (c.family == "fastener" and bool(c.designation) or bool(ROBOTIS_HOUSING.search(n))
                                or c.category == "pulley / idler" and re.search("roller|idler|pulley", n, re.I))
    if CUSTOM_HINT.search(n) and not strong:
        return "custom"
    if c is None:
        return "custom"
    if c.family == "structure" and c.category in ("plate / panel", "bracket / mount"):
        return "custom"
    return FAMILY_TO_LABEL.get(c.family, "custom")


def assembly_rows(path: str):
    """(rows, labels, parent texts) of one assembly; cached by file hash + version."""
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    cache = os.path.join(os.path.dirname(__file__), "..", "data", "cache",
                         f"cmrows_v{ROWS_VERSION}_{h.hexdigest()}.joblib")
    if os.path.exists(cache):
        return joblib.load(cache)
    out = _assembly_rows(path)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    joblib.dump(out, cache)
    return out


WHOLE_FILE_LABEL = {"3d_models": "electronics"}     # folder -> every file is one part of this family


def _assembly_rows(path: str):
    from .breakdown import fast_shape_features
    from .features import context_features
    from .io import load_file
    from .segment import extract_parts
    lr = load_file(path)
    raws = extract_parts(lr.parts)
    whole = WHOLE_FILE_LABEL.get(os.path.basename(os.path.dirname(os.path.abspath(path))))
    if whole:
        # A file that is ONE bought component (a connector or board model from a parts library):
        # its pieces carry generic names, so the whole file is one example of that family.
        import trimesh
        from .io import RawPart
        m = trimesh.util.concatenate([r.mesh for r in raws])
        raws = [RawPart(mesh=m, name=os.path.splitext(os.path.basename(path))[0], n_solids=len(raws))]
    infos = fast_shape_features(raws, n_jobs=1)
    ctx = context_features(infos)
    X, y, T = [], [], []
    if whole:
        return [row(infos[0], ctx[0], None, raws[0].n_solids)], [whole], [""]
    for r, inf, c in zip(raws, infos, ctx):
        lab = name_label(r.name, r.path)
        if lab is None:
            continue
        X.append(row(inf, c, r.brep, r.n_solids))
        y.append(lab)
        T.append(parent_text(r.path))
    return X, y, T


# ---------------------------------------------------------------- parent-context text model
def fit_text_model(texts, labels):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    keep = [i for i, t in enumerate(texts) if t]
    if len(set(labels[i] for i in keep)) < 2:
        return None
    m = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
                      LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced"))
    m.fit([texts[i] for i in keep], [labels[i] for i in keep])
    return m


def text_features(model, texts, classes) -> np.ndarray:
    out = np.full((len(texts), len(classes)), np.nan)
    if model is None:
        return out
    idx = [i for i, t in enumerate(texts) if t]
    if idx:
        P = model.predict_proba([texts[i] for i in idx])
        mc = list(model.classes_)
        for j, c in enumerate(classes):
            if c in mc:
                out[idx, j] = P[:, mc.index(c)]
    return out


def oof_text_features(texts, labels, groups, classes) -> np.ndarray:
    """Out-of-fold parent-context features for training rows (grouped by assembly)."""
    out = np.full((len(texts), len(classes)), np.nan)
    ug = sorted(set(groups))
    if len(ug) < 2:
        return out
    for g in ug:
        tr = [i for i in range(len(texts)) if groups[i] != g]
        te = [i for i in range(len(texts)) if groups[i] == g]
        m = fit_text_model([texts[i] for i in tr], [labels[i] for i in tr])
        out[te] = text_features(m, [texts[i] for i in te], classes)
    return out


# ---------------------------------------------------------------- learners
def make_model(kind: str):
    from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
                                  VotingClassifier)
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    hgb = lambda: HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, max_leaf_nodes=48,
                                                 min_samples_leaf=10, l2_regularization=0.5,
                                                 class_weight="balanced", random_state=0)
    et = lambda: make_pipeline(SimpleImputer(strategy="constant", fill_value=-1.0),
                               ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt",
                                                    class_weight="balanced", n_jobs=-1, random_state=0))
    rf = lambda: make_pipeline(SimpleImputer(strategy="constant", fill_value=-1.0),
                               RandomForestClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt",
                                                      class_weight="balanced_subsample", n_jobs=-1, random_state=0))
    if kind == "hgb":
        return hgb()
    if kind == "et":
        return et()
    if kind == "rf":
        return rf()
    if kind == "ensemble":
        return VotingClassifier([("hgb", hgb()), ("et", et()), ("rf", rf())], voting="soft")
    if kind == "rf_hgb":   # forest is best on printers, boosting on robot arms: blend both
        return VotingClassifier([("hgb", hgb()), ("rf", rf())], voting="soft")
    raise ValueError(kind)


def train(files: list[str], path: str = MODEL_PATH, report: bool = True, synthetic: int = 0,
          kinds=("hgb", "rf", "rf_hgb")) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    data = {}
    for f in files:
        t = time.time()
        X, y, T = assembly_rows(f)
        data[os.path.basename(f)] = (X, y, T)
        if report:
            print(f"{os.path.basename(f)}: {len(y)} labelled parts ({time.time() - t:.0f}s)", flush=True)
    # library files (one component each) form one group, held out together
    lib = [f for f in files if os.path.basename(os.path.dirname(os.path.abspath(f))) in WHOLE_FILE_LABEL]
    if lib:
        merged = ([], [], [])
        for f in lib:
            X, y, T = data.pop(os.path.basename(f))
            merged[0].extend(X)
            merged[1].extend(y)
            merged[2].extend(T)
        data["component_library"] = merged
    syn_X, syn_y = [], []
    if synthetic:
        from joblib import Parallel, delayed
        from .std_generator import dataset_rows
        t = time.time()
        chunks = Parallel(n_jobs=-1)(delayed(dataset_rows)(200, seed=1000 + k) for k in range(max(1, synthetic // 200)))
        for X, y in chunks:
            syn_X += X
            syn_y += y
        if report:
            print(f"synthetic standard parts: {len(syn_y)} ({time.time() - t:.0f}s)", flush=True)
    # Corrections people made in the app are an extra training "assembly" (shape numbers only,
    # no parent names); they are never held out for evaluation.
    from .feedback import load_corrections
    fb_X, fb_y = load_corrections("component")
    if fb_y and report:
        print(f"user corrections: {len(fb_y)}", flush=True)
    classes = sorted({l for _, y, _ in data.values() for l in y} | set(syn_y) | set(fb_y))
    feat = sorted({k for X, _, _ in data.values() for r in X for k in r} | {k for r in syn_X for k in r})
    ctx_names = [f"ctx_{c}" for c in classes]
    names = feat + ctx_names
    brep_idx = [names.index(k) for k in BREP_KEYS if k in names]
    ctx_idx = [names.index(k) for k in ctx_names]
    asm_idx = [names.index(k) for k in CONTEXT_KEYS if k in names]

    def base(rows):
        return np.array([[r.get(k, np.nan) for k in feat] for r in rows], dtype=float)

    def assemble(keys, test_key=None):
        """Training matrix for `keys` (+ synthetic), and the test matrix for test_key."""
        texts, labels, groups, wts = [], [], [], []
        for k in keys:
            Xk, y, T = data[k]
            texts += T
            labels += y
            groups += [k] * len(y)
            wts.append(assembly_weights(Xk, y))
        oof = oof_text_features(texts, labels, groups, classes)
        Xr = np.vstack([base(data[k][0]) for k in keys]) if keys else np.zeros((0, len(feat)))
        yr = np.array(labels)
        full = np.hstack([Xr, oof])
        blank = full.copy()                      # same part as a plain mesh with no parents
        blank[:, brep_idx] = np.nan
        blank[:, ctx_idx] = np.nan
        alone = alone_context(full, names)        # the same part uploaded on its own (no parents)
        use_alone = os.environ.get("CM_ALONE_ROWS", "1") != "0"
        alone[:, ctx_idx] = np.nan
        wr = np.concatenate(wts) if wts else np.zeros(0)
        parts, ys, ws = ([full, blank, alone], [yr, yr, yr], [wr, wr, 0.5 * wr]) if use_alone else ([full, blank], [yr, yr], [wr, wr])
        if fb_y:
            F = np.hstack([base(fb_X), np.full((len(fb_X), len(classes)), np.nan)])
            parts.append(F)
            ys.append(np.array(fb_y))
            ws.append(assembly_weights(fb_X, fb_y) * 2.0)       # people checked these: count double
        if syn_X:
            S = np.hstack([base(syn_X), np.full((len(syn_X), len(classes)), np.nan)])
            parts.append(graft_context(S, syn_y, full, yr, asm_idx))
            ys.append(np.array(syn_y))
            ws.append(np.full(len(syn_y), float(np.median(wr)) if len(wr) else 1.0))
            if use_alone:
                parts.append(alone_context(S, names))   # a lone screw / bearing / motor file
                ys.append(np.array(syn_y))
                ws.append(np.full(len(syn_y), 0.5 * float(np.median(wr)) if len(wr) else 0.5))
        Xtr, ytr = np.vstack(parts), np.concatenate(ys)
        assemble.weights = np.concatenate(ws)
        Xte = None
        if test_key:
            tm = fit_text_model(texts, labels)
            Xt, _, Tt = data[test_key]
            Xte = np.hstack([base(Xt), text_features(tm, Tt, classes)])
        return Xtr, ytr, Xte

    keys = list(data)
    results = {k: {} for k in kinds}
    preds = {k: {"y": [], "p": [], "pm": [], "pa": [], "conf": [], "confm": []} for k in kinds}
    if len(keys) > 1:
        for hold in keys:
            Xtr, ytr, Xte = assemble([k for k in keys if k != hold], hold)
            y = data[hold][1]
            Xm = Xte.copy()
            Xm[:, brep_idx] = np.nan
            Xm[:, ctx_idx] = np.nan
            Xa = alone_context(Xte, names)
            Xa[:, ctx_idx] = np.nan
            for kind in kinds:
                clf = _fit(make_model(kind), Xtr, ytr, assemble.weights)
                P, Pm = clf.predict_proba(Xte), clf.predict_proba(Xm)
                cl = np.asarray(clf.classes_)
                p, pm = cl[P.argmax(1)], cl[Pm.argmax(1)]
                preds[kind]["pa"] += list(cl[clf.predict_proba(Xa).argmax(1)])
                preds[kind]["conf"] += list(P.max(1))
                preds[kind]["confm"] += list(Pm.max(1))
                results[kind][hold] = {"parts": len(y), "acc_cad": round(float(accuracy_score(y, p)), 4),
                                       "acc_mesh_only": round(float(accuracy_score(y, pm)), 4)}
                preds[kind]["y"] += list(y)
                preds[kind]["p"] += list(p)
                preds[kind]["pm"] += list(pm)
            if report:
                print(f"  held out {hold}: " + " | ".join(f"{k} {results[k][hold]['acc_cad']:.3f}" for k in kinds),
                      flush=True)
    summary = {}
    for kind in kinds:
        y, p, pm = preds[kind]["y"], preds[kind]["p"], preds[kind]["pm"]
        if y:
            summary[kind] = {"acc_cad": round(float(accuracy_score(y, p)), 4),
                             "acc_mesh_only": round(float(accuracy_score(y, pm)), 4),
                             "acc_alone": round(float(accuracy_score(y, preds[kind]["pa"])), 4),
                             "macro_f1": round(float(f1_score(y, p, average="macro")), 4)}
    best = max(summary, key=lambda k: (summary[k]["acc_cad"] + summary[k]["macro_f1"])) if summary else kinds[0]
    if report and summary:
        for k, v in summary.items():
            print(f"{k:9s} {v}")
        print(f"best: {best}")
        print(classification_report(preds[best]["y"], preds[best]["p"], digits=3, zero_division=0))
    calibrator, calib = None, None
    if preds[best]["y"]:
        # Confidence calibration on held-out assemblies: raw forest "probabilities" are not
        # probabilities. Isotonic regression maps them to the measured chance of being right.
        from sklearn.isotonic import IsotonicRegression
        conf = np.array(preds[best]["conf"] + preds[best]["confm"])
        ok = np.concatenate([np.array(preds[best]["y"]) == np.array(preds[best]["p"]),
                             np.array(preds[best]["y"]) == np.array(preds[best]["pm"])]).astype(float)
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(conf, ok)
        cal = calibrator.predict(conf)
        bins = np.linspace(0, 1, 11)
        ece = lambda c: float(sum(abs(ok[(c >= a) & (c < b)].mean() - c[(c >= a) & (c < b)].mean()) *
                                  ((c >= a) & (c < b)).mean() for a, b in zip(bins[:-1], bins[1:])
                                  if ((c >= a) & (c < b)).any()))
        calib = {"ece_raw": round(ece(conf), 4), "ece_calibrated": round(ece(cal), 4),
                 "share_below_0.6": round(float((cal < 0.6).mean()), 4),
                 "accuracy_when_confident": round(float(ok[cal >= 0.6].mean()), 4) if (cal >= 0.6).any() else None,
                 "accuracy_when_unsure": round(float(ok[cal < 0.6].mean()), 4) if (cal < 0.6).any() else None}
        if report:
            print("confidence calibration:", calib, flush=True)
    Xtr, ytr, _ = assemble(keys)
    final = _fit(make_model(best), Xtr, ytr, assemble.weights)
    text_model = fit_text_model([t for k in keys for t in data[k][2]], [l for k in keys for l in data[k][1]])
    metrics = {"algorithms": summary, "best": best, "per_assembly": results[best] if results else {},
               "assemblies": keys, "examples": int(sum(len(v[1]) for v in data.values())),
               "synthetic_examples": len(syn_y), "calibration": calib, "user_corrections": len(fb_y)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": final, "features": names, "base_features": feat, "classes": classes,
                 "text_model": text_model, "metrics": metrics, "calibrator": calibrator}, path)
    return metrics


_cached = None


_cached_alone = None


def predict(rows: list[dict], paths: list | None = None, alone: bool = False):
    """(labels, confidences) for parts; None when no model has been trained.
    alone=True: the file holds one part (no assembly around it), so the lone-part model is used."""
    global _cached, _cached_alone
    if alone and os.path.exists(ALONE_MODEL_PATH):
        if _cached_alone is None:
            _cached_alone = joblib.load(ALONE_MODEL_PATH)
        b = _cached_alone
    else:
        if _cached is None:
            if not os.path.exists(MODEL_PATH):
                return None
            _cached = joblib.load(MODEL_PATH)
        b = _cached
    clf = b["model"]
    if "base_features" not in b:   # older bundle without parent context
        M = np.array([[r.get(k, np.nan) for k in b["features"]] for r in rows], dtype=float)
    else:
        base = np.array([[r.get(k, np.nan) for k in b["base_features"]] for r in rows], dtype=float)
        texts = [parent_text(p) for p in (paths or [None] * len(rows))]
        M = np.hstack([base, text_features(b.get("text_model"), texts, b["classes"])])
    P = clf.predict_proba(M)
    cls = list(clf.classes_)
    idx = P.argmax(1)
    conf = P.max(1)
    if b.get("calibrator") is not None:       # measured chance of being right (held-out assemblies)
        conf = b["calibrator"].predict(conf)
    return [cls[i] for i in idx], [float(c) for c in conf]


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    args = sys.argv[1:]
    n_syn = 0
    if args and args[0].startswith("--synthetic="):
        n_syn = int(args.pop(0).split("=")[1])
    kinds = ("rf", "hgb", "rf_hgb")
    if args and args[0].startswith("--kinds="):
        kinds = tuple(args.pop(0).split("=")[1].split(","))
    files = args or sorted(glob.glob("data/real_cad/*/*.step"))
    train(files, synthetic=n_syn, kinds=kinds)
