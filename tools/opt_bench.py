"""Benchmark design-search algorithms on one engine: best score reached for a fixed number of
simulations, over several seeds (mean, spread, gap to the best design ever found).
python tools/opt_bench.py <assembly pickle> [budget] [seeds]"""
import json
import pickle
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor  # noqa: E402

from enginelab import improve  # noqa: E402

asm = pickle.load(open(sys.argv[1], "rb"))
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 400
SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
ctx = improve.Context(asm)
base = improve.evaluate(ctx, {})
LEV = ctx.levers
CACHE: dict = {}


def key(d):
    return tuple(int(d.get(lv.id, 0)) for lv in LEV)


def f(d):
    k = key(d)
    if k not in CACHE:
        CACHE[k] = improve.score(improve.evaluate(ctx, d), base)
    return CACHE[k]


class Run:
    """Counts distinct simulations per run (repeats are free, as they would be with a cache)."""

    def __init__(self, budget):
        self.budget, self.seen, self.best, self.best_d, self.curve = budget, set(), -1e9, {}, []

    def __call__(self, d):
        k = key(d)
        if k not in self.seen:
            if len(self.seen) >= self.budget:
                raise StopIteration
            self.seen.add(k)
            v = f(d)
            if v > self.best:
                self.best, self.best_d = v, dict(d)
            self.curve.append(self.best)
            return v
        return f(d)


def enc(d):
    v = []
    for lv in LEV:
        oh = np.zeros(len(lv.options))
        oh[int(d.get(lv.id, 0))] = 1
        v.extend(oh)
    return v


def rand_design(rng, p=0.35):
    return {lv.id: int(rng.integers(0, len(lv.options))) for lv in LEV if rng.random() < p}


def mutate(rng, d):
    d = dict(d)
    for _ in range(int(rng.integers(1, 4))):
        lv = LEV[int(rng.integers(0, len(LEV)))]
        d[lv.id] = int(rng.integers(0, len(lv.options)))
    return d


def seed_singles(run):
    run({})
    for lv in LEV:
        for i in range(1, len(lv.options)):
            run({lv.id: i})


def coord_descent(run, d, rng=None, passes=6):
    d = dict(d)
    cur = run(d)
    for _ in range(passes):
        improved = False
        order = list(LEV) if rng is None else [LEV[i] for i in rng.permutation(len(LEV))]
        for lv in order:
            best_i, best_v = int(d.get(lv.id, 0)), cur
            for i in range(len(lv.options)):
                if i == int(d.get(lv.id, 0)):
                    continue
                v = run({**d, lv.id: i})
                if v > best_v + 1e-12:
                    best_i, best_v = i, v
            if best_i != int(d.get(lv.id, 0)):
                d[lv.id] = best_i
                cur = best_v
                improved = True
        if not improved:
            break
    return d, cur


def alg_random(run, rng):
    run({})
    while True:
        run(rand_design(rng))


def alg_surrogate(run, rng, model="rf", polish=False, kappa0=1.0):
    seed_singles(run)
    kappa = kappa0
    it = 0
    while True:
        ds = [dict(zip([lv.id for lv in LEV], k)) for k in run.seen]
        X = np.array([enc(d) for d in ds])
        y = np.array([f(d) for d in ds])
        M = (RandomForestRegressor if model == "rf" else ExtraTreesRegressor)(
            n_estimators=120, max_features=0.5, n_jobs=1, random_state=int(rng.integers(1e9))).fit(X, y)
        elite = [ds[i] for i in np.argsort(-y)[:8]]
        cands, ks = [], set()
        for _ in range(400):
            d = mutate(rng, elite[int(rng.integers(0, len(elite)))]) if rng.random() < 0.75 else rand_design(rng)
            k = key(d)
            if k not in run.seen and k not in ks:
                ks.add(k)
                cands.append(d)
        if not cands:
            return
        C = np.array([enc(d) for d in cands])
        P = np.stack([t.predict(C) for t in M.estimators_])
        for i in np.argsort(-(P.mean(0) + kappa * P.std(0)))[:6]:
            run(cands[i])
        kappa = max(0.15, kappa * 0.97)
        it += 1
        if polish and it % 8 == 0:
            coord_descent(run, run.best_d, rng, passes=2)


def alg_coord(run, rng):
    seed_singles(run)
    # start from the best single changes combined greedily, then iterated local search
    d, v = coord_descent(run, run.best_d, rng)
    while True:
        start = d if rng.random() < 0.7 else run.best_d
        d2, v2 = coord_descent(run, mutate(rng, mutate(rng, start)), rng)
        if v2 >= v:
            d, v = d2, v2


def alg_ga(run, rng, pop=24):
    seed_singles(run)
    ds = [dict(zip([lv.id for lv in LEV], k)) for k in run.seen]
    P = sorted(ds, key=f, reverse=True)[:pop]
    while True:
        kids = []
        for _ in range(pop):
            a, b = P[int(rng.integers(0, len(P)))], P[int(rng.integers(0, len(P)))]
            c = {lv.id: (a.get(lv.id, 0) if rng.random() < 0.5 else b.get(lv.id, 0)) for lv in LEV}
            if rng.random() < 0.6:
                c = mutate(rng, c)
            run(c)
            kids.append(c)
        P = sorted(P + kids, key=f, reverse=True)[:pop]


ALGS = {"random": alg_random, "surrogate_rf (current)": lambda r, g: alg_surrogate(r, g, "rf"),
        "surrogate_et": lambda r, g: alg_surrogate(r, g, "et"),
        "surrogate_rf + polish": lambda r, g: alg_surrogate(r, g, "rf", polish=True),
        "surrogate_et + polish": lambda r, g: alg_surrogate(r, g, "et", polish=True),
        "coordinate descent + restarts": alg_coord, "genetic": alg_ga}
import os as _os
if _os.environ.get("ALGS"):
    ALGS = {k: v for k, v in ALGS.items() if any(s in k for s in _os.environ["ALGS"].split(","))}
res = {}
for name, alg in ALGS.items():
    bests, t = [], time.time()
    for s in range(SEEDS):
        run = Run(BUDGET)
        try:
            alg(run, np.random.default_rng(100 + s))
        except StopIteration:
            pass
        bests.append(run.best)
    res[name] = bests
    print(f"{name:32s} best {np.mean(bests):.4f} +- {np.std(bests):.4f}  (min {min(bests):.4f})  {time.time() - t:.0f}s",
          flush=True)
top = max(CACHE.values())
print(f"best score ever found: {top:.4f} (from {len(CACHE)} distinct designs)")
for name, b in res.items():
    print(f"  gap to best  {name:32s} mean {top - np.mean(b):.4f}  worst {top - min(b):.4f}")
