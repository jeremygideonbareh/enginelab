"""Honest accuracy evaluation of part identification.

  python tools/evaluate.py

1. Unseen engines (seeds never used in training), exported as one unnamed STL.
2. Out-of-range engines: bores far outside the training range (40-50 mm, 125-150 mm).
3. Single parts uploaded on their own (no assembly context).
Per-part accuracy is measured by matching every found part to the true part with the
nearest centre and volume.
"""
import os
import sys
import tempfile
import warnings
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from enginelab.generator import build_engine, random_engine, random_params, randomly_posed  # noqa: E402
from enginelab.pipeline import analyze_file, analyze_parts  # noqa: E402
from enginelab.io import RawPart  # noqa: E402


def per_part(eng, asm):
    tc = np.array([m.bounds.mean(0) for _, m in eng.parts])
    tv = np.array([abs(m.volume) for _, m in eng.parts])
    ok, total = 0, 0
    confusion = Counter()
    for p in asm.parts:
        c = p.mesh.bounds.mean(0)
        d = np.linalg.norm(tc - c, axis=1) + 5 * np.abs(np.log((tv + 1) / (p.info.volume + 1)))
        truth = eng.parts[int(np.argmin(d))][0]
        total += 1
        if truth == p.label:
            ok += 1
        else:
            confusion[(truth, p.label)] += 1
    return ok, total, confusion


def run(name, engines):
    ok = tot = 0
    conf = Counter()
    per_label = defaultdict(lambda: [0, 0])
    with tempfile.TemporaryDirectory() as d:
        for i, eng in enumerate(engines):
            path = os.path.join(d, f"e{i}.stl")
            eng.merged().export(path)
            asm = analyze_file(path)
            o, t, c = per_part(eng, asm)
            ok += o
            tot += t
            conf.update(c)
            for lab, _ in eng.parts:
                per_label[lab][1] += 1
            for (truth, _), n in c.items():
                per_label[truth][0] += n
    print(f"\n{name}: {ok}/{tot} parts correct = {100 * ok / tot:.2f}%")
    if conf:
        print("  most common confusions (true -> predicted):", conf.most_common(6))
    return ok / tot


if __name__ == "__main__":
    unseen = [randomly_posed(random_engine(s), s) for s in range(2000, 2012)]
    run("Unseen engines (12, STL, random pose)", unseen)

    ood = []
    for s, lo, hi in [(3000, 40, 50), (3001, 42, 50), (3002, 125, 150), (3003, 130, 150)]:
        rng = np.random.default_rng(s)
        p = random_params(rng)
        p.bore = float(rng.uniform(lo, hi))
        p.stroke = p.bore * float(rng.uniform(0.8, 1.2))
        p.rod_length = p.stroke * 1.7
        p.pitch = p.bore * (1.2 if p.layout == "inline" else 1.36)
        ood.append(randomly_posed(build_engine(p, rng), s))
    run("Out-of-range bores (40-50 mm and 125-150 mm)", ood)

    # Single parts: every distinct part type uploaded alone.
    ok = tot = 0
    wrong = Counter()
    for s in (2100, 2101, 2102):
        eng = random_engine(s)
        seen = set()
        for lab, m in eng.parts:
            if lab in seen:
                continue
            seen.add(lab)
            # The engine classifier itself, on a part with no assembly around it.
            from enginelab.classify import analyse_shapes, classify
            infos, ctx = analyse_shapes([m])
            got = classify(infos, ctx, [None])[0].label
            tot += 1
            ok += got == lab
            if got != lab:
                wrong[(lab, got)] += 1
    print(f"\nSingle parts without assembly context: {ok}/{tot} = {100 * ok / tot:.1f}%")
    if wrong:
        print("  confusions:", wrong.most_common(8))
