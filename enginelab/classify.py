"""Identify what each part is.

Ensemble, strongest signal first:
  1. Part names from the CAD file ("Piston_02", "con rod", "Kurbelwelle"...).
  2. A gradient-boosted classifier on shape + assembly-context features,
     trained on procedurally generated engines.
  3. Assembly reasoning: identical copies share one label, and parts that
     exist once per engine (block, crankshaft, flywheel...) are not repeated.
Parts whose best probability is low are reported as "unknown" with the
top guesses, instead of being forced into a class.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import joblib
import numpy as np

from .features import ShapeInfo, context_features, copy_groups, shape_features
from .generator import LABELS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "models", "part_classifier.joblib")
UNKNOWN_THRESHOLD = 0.40

# Keyword patterns per label; more specific patterns are listed first.
NAME_PATTERNS: list[tuple[str, str]] = [
    ("piston_ring", r"(piston|comp(ression)?|oil|scraper)[\s_\-]*ring|\bring\b|kolbenring"),
    ("piston_pin", r"(piston|gudgeon|wrist)[\s_\-]*pin|kolbenbolzen"),
    ("connecting_rod", r"con(necting)?[\s_\-]*rod|conrod|\brod\b|pleuel|biel"),
    ("piston", r"piston|kolben"),
    ("crankshaft", r"crank[\s_\-]*sh[ai]ft|crankshaft|\bcrank\b|kurbelwelle|vilebrequin"),   # "CRANKSHIFT" is a common misspelling
    ("camshaft", r"cam[\s_\-]*shaft|camshaft|nockenwelle"),
    ("valve_spring", r"valve[\s_\-]*spring|\bspring\b|ventilfeder"),
    ("valve_cover", r"valve[\s_\-]*cover|rocker[\s_\-]*cover|cam[\s_\-]*cover|zylinderkopfhaube"),
    ("valve", r"valve|ventil|intake[\s_\-]*v|exhaust[\s_\-]*v"),
    ("cylinder_liner", r"liner|sleeve|laufbuchse"),
    ("head_gasket", r"gasket|dichtung"),
    ("cylinder_head", r"(cylinder|cyl)[\s_\-]*head|\bhead\b|zylinderkopf"),
    ("engine_block", r"block|crank[\s_\-]*case|(lower|upper)[\s_\-]*case|\bfins?\b|cylinder[\s_\-]*barrel|motorblock|carter"),
    ("main_bearing", r"bearing|shell|lager"),
    ("flywheel", r"fly[\s_\-]*wheel|schwungrad|volant"),
    ("timing_gear", r"gear|sprocket|pinion|zahnrad"),
    ("spark_plug", r"spark|plug|kerze|glow"),
    ("oil_pan", r"oil[\s_\-]*pan|sump|oelwanne|ölwanne"),
    ("bolt", r"bolt|screw|stud|schraube|nut\b|washer"),
]

# Parts that normally appear once per engine.
SINGLETONS = {"engine_block", "crankshaft", "flywheel", "oil_pan"}


def label_from_name(name: str | None) -> str | None:
    if not name:
        return None
    n = name.lower().replace("_", " ")
    for label, pat in NAME_PATTERNS:
        if re.search(pat, n):
            return label
    return None


@dataclass
class Prediction:
    label: str
    confidence: float
    source: str                   # name | model | assembly
    alternatives: list[tuple[str, float]]


def analyse_shapes(meshes, seed: int = 0) -> tuple[list[ShapeInfo], list[dict]]:
    infos = [shape_features(m, seed=seed + i) for i, m in enumerate(meshes)]
    ctx = context_features(infos)
    return infos, ctx


# ---------------------------------------------------------------- training data
def _engine_rows(seed: int):
    """One generated engine -> training rows for the full assembly, for every part type
    uploaded on its own, and for random partial assemblies (people upload sub-assemblies)."""
    from .generator import random_engine, randomly_posed
    rng = np.random.default_rng(seed + 77)
    eng = randomly_posed(random_engine(seed), seed + 10_000)
    labels = [lab for lab, _ in eng.parts]
    infos, ctx = analyse_shapes([m for _, m in eng.parts], seed=seed)
    rows_i, rows_c, rows_l = list(infos), list(ctx), list(labels)
    first = {}
    for i, lab in enumerate(labels):
        first.setdefault(lab, i)
    for i in first.values():                      # single parts
        rows_i.append(infos[i])
        rows_c.append(context_features([infos[i]])[0])
        rows_l.append(labels[i])
    for _ in range(2):                            # partial assemblies
        k = max(2, int(len(labels) * rng.uniform(0.1, 0.6)))
        idx = rng.choice(len(labels), size=k, replace=False)
        sub_ctx = context_features([infos[i] for i in idx])
        for j, i in enumerate(idx):
            rows_i.append(infos[i])
            rows_c.append(sub_ctx[j])
            rows_l.append(labels[i])
    return rows_i, rows_c, rows_l


def _engine_rows_safe(seed: int):
    try:
        return _engine_rows(seed)
    except Exception as exc:          # a rare degenerate random engine must not stop training
        return ("failed", seed, f"{exc.__class__.__name__}: {exc}")


def build_dataset(n_engines: int, start_seed: int = 0, n_jobs: int = -1):
    """Generated-engine training rows; cached on disk per generator version (rebuilding takes
    ~20 minutes for 1,200 engines)."""
    import hashlib
    here = os.path.dirname(__file__)
    ver = hashlib.sha1(b"".join(open(os.path.join(here, f), "rb").read()
                                for f in ("generator.py", "features.py", "classify.py"))).hexdigest()[:10]
    cache = os.path.join(here, "..", "data", "cache", f"engine_ds_{n_engines}_{start_seed}_{ver}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        return z["X"].astype(float), z["y"].astype(str), z["groups"], [str(n) for n in z["names"]]
    X, y, groups, names = _build_dataset(n_engines, start_seed, n_jobs)
    try:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, X=X.astype(np.float32), y=y, groups=groups, names=np.array(names))
    except OSError:
        pass
    return X, y, groups, names


def _build_dataset(n_engines: int, start_seed: int = 0, n_jobs: int = -1):
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_engine_rows_safe)(s) for s in range(start_seed, start_seed + n_engines))
    failed = [r for r in results if r[0] == "failed"]
    if failed:
        print(f"skipped {len(failed)} degenerate generated engines: " +
              "; ".join(f"seed {s}: {m}" for _, s, m in failed[:5]), flush=True)
    results = [r for r in results if r[0] != "failed"]
    names = None
    X, y, groups = [], [], []
    for gi, (infos, ctx, labels) in enumerate(results):
        for info, c, lab in zip(infos, ctx, labels):
            merged = {**info.features, **c}
            if names is None:
                names = list(merged)
            X.append([merged[k] for k in names])
            y.append(lab)
            groups.append(gi)
    return np.asarray(X), np.asarray(y), np.asarray(groups), names


def train(n_engines: int = 240, path: str = MODEL_PATH, n_jobs: int = -1, report: bool = True):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import GroupShuffleSplit

    X, y, groups, names = build_dataset(n_engines, n_jobs=n_jobs)
    from .feedback import load_corrections
    fb_X, fb_y = load_corrections("engine")
    if fb_y:
        # corrections made in the app: added (x5) to training, never to the hold-out test
        F = np.array([[r.get(k, np.nan) for k in names] for r in fb_X], dtype=float)
        X = np.vstack([X] + [F] * 5)
        y = np.concatenate([y] + [np.array(fb_y)] * 5)
        groups = np.concatenate([groups, np.full(5 * len(fb_y), -1)])
        if report:
            print(f"user corrections: {len(fb_y)}")
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    tr, te = next(split.split(X, y, groups))
    te = te[groups[te] != -1]
    tr = np.union1d(tr, np.nonzero(groups == -1)[0])
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=31,
                                         l2_regularization=1e-3, class_weight="balanced", random_state=0)
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    metrics = {"holdout_accuracy": float(accuracy_score(y[te], pred)), "n_train": int(len(tr)),
               "n_test": int(len(te)), "n_engines": n_engines}
    if report:
        print(classification_report(y[te], pred, digits=3, zero_division=0))
        print(metrics)
    clf.fit(X, y)  # final model uses every engine
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": clf, "features": names, "metrics": metrics}, path)
    return metrics


_cached = None


def load_model(path: str = MODEL_PATH):
    global _cached
    if _cached is None:
        if not os.path.exists(path):
            raise FileNotFoundError("No trained part classifier. Run: python -m enginelab.train")
        _cached = joblib.load(path)
    return _cached


# ---------------------------------------------------------------- inference
def classify(infos: list[ShapeInfo], ctx: list[dict], names: list[str | None]) -> list[Prediction]:
    bundle = load_model()
    clf, feat_names = bundle["model"], bundle["features"]
    X = np.asarray([[{**i.features, **c}[k] for k in feat_names] for i, c in zip(infos, ctx)])
    proba = clf.predict_proba(X)
    classes = list(clf.classes_)

    # Name hints: blend strongly towards the named class.
    from_name = [label_from_name(n) for n in names]
    for i, lab in enumerate(from_name):
        if lab in classes:
            proba[i] = 0.25 * proba[i]
            proba[i, classes.index(lab)] += 0.75

    # Identical copies share a label: average their probabilities.
    groups = np.asarray(copy_groups(infos))
    for g in np.unique(groups):
        idx = np.nonzero(groups == g)[0]
        if len(idx) > 1:
            proba[idx] = proba[idx].mean(0)

    # Singleton parts: keep the most likely candidate, demote the others.
    source = ["name" if lab else "model" for lab in from_name]
    for lab in SINGLETONS & set(classes):
        k = classes.index(lab)
        top = np.nonzero(proba.argmax(1) == k)[0]
        if len(top) > 1:
            keep = top[np.argmax(proba[top, k])]
            for i in top:
                if i != keep and from_name[i] != lab:
                    proba[i, k] *= 0.2
                    proba[i] /= proba[i].sum()
                    source[i] = "assembly"

    out = []
    for i in range(len(infos)):
        order = np.argsort(-proba[i])
        best = classes[order[0]]
        conf = float(proba[i, order[0]])
        alts = [(classes[k], float(proba[i, k])) for k in order[:3]]
        if conf < UNKNOWN_THRESHOLD and from_name[i] is None:
            best = "unknown"
        out.append(Prediction(label=best, confidence=conf, source=source[i], alternatives=alts))
    return out
