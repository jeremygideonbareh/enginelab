"""Independent ground truth from the raw STEP text: count leaf part instances by walking
PRODUCT_DEFINITION / NEXT_ASSEMBLY_USAGE_OCCURRENCE.   python tools/step_structure.py file.step"""
import re, sys
from collections import defaultdict
ent = {}
pat = re.compile(r"#(\d+)\s*=\s*([A-Z_0-9]+)\s*\((.*)\)\s*;", re.S)
buf = ""
with open(sys.argv[1], encoding="latin-1") as fh:
    for line in fh:
        buf += line
        if line.rstrip().endswith(";"):
            m = pat.match(buf.strip())
            if m and m.group(2) in ("PRODUCT_DEFINITION", "NEXT_ASSEMBLY_USAGE_OCCURRENCE", "PRODUCT_DEFINITION_FORMATION",
                                    "PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE", "PRODUCT"):
                ent[int(m.group(1))] = (m.group(2), m.group(3))
            buf = ""
refs = lambda s: [int(x) for x in re.findall(r"#(\d+)", s)]
children = defaultdict(list); is_child = set()
for k, (t, a) in ent.items():
    if t == "NEXT_ASSEMBLY_USAGE_OCCURRENCE":
        r = refs(a); parent, child = r[0], r[1]
        children[parent].append(child); is_child.add(child)
pds = [k for k, (t, a) in ent.items() if t == "PRODUCT_DEFINITION"]
roots = [p for p in pds if p not in is_child]
memo = {}
def leaves(p):
    if p in memo: return memo[p]
    n = 1 if not children[p] else sum(leaves(c) for c in children[p])
    memo[p] = n; return n
print("product definitions", len(pds), "roots", len(roots), "assemblies", len(children),
      "leaf instances", sum(leaves(r) for r in roots))
