"""Full analysis of a big CAD file with a written report:
python tools/analyze_big.py file.step report.json
"""
import json
import sys
import time
import warnings
from collections import Counter

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from enginelab.pipeline import analyze_file  # noqa: E402

t = time.time()
asm = analyze_file(sys.argv[1], progress=lambda m: print(f"[{time.time() - t:7.1f}s] {m}", flush=True))
j = asm.to_json()
bom = j["bom"]


def depth(n):
    return 1 + max([depth(c) for c in n["children"]] or [0])


report = {
    "file": sys.argv[1], "seconds": round(time.time() - t, 1), "timings": j["timings"], "info": j["info"],
    "mode": j["mode"], "parts": len(j["parts"]), "bom_lines": len(bom),
    "tree_depth": depth(j["tree"]) if j["tree"] else None,
    "labels": Counter(p["label"] for p in j["parts"]),
    "standard_lines": sum(1 for l in bom if l["standard_part"]),
    "with_designation": sum(1 for l in bom if l["designation"]),
    "top_lines": [(l["qty"], l["name"], l["category"], l["designation"]) for l in sorted(bom, key=lambda l: -l["qty"])[:40]],
    "custom_sample": [(l["qty"], l["name"], l["category"]) for l in bom if l["label"] == "custom"][:40],
}
json.dump(report, open(sys.argv[2], "w"), indent=1, default=str)
print(json.dumps({k: report[k] for k in ("seconds", "parts", "bom_lines", "tree_depth", "labels",
                                         "standard_lines", "with_designation")}, default=str))
