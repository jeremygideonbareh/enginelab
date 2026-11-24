"""Per-product-name comparison: STEP NAUO tree leaf counts vs parts found by the reader.
python tools/compare_structure.py file.step cache.pkl"""
import pickle, re, sys
sys.path.insert(0, ".")
from collections import Counter, defaultdict
pat = re.compile(r"#(\d+)\s*=\s*([A-Z_0-9]+)\s*\((.*)\)\s*;", re.S)
ent, buf = {}, ""
KEEP = {"PRODUCT_DEFINITION", "NEXT_ASSEMBLY_USAGE_OCCURRENCE", "PRODUCT_DEFINITION_FORMATION",
        "PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE", "PRODUCT"}
with open(sys.argv[1], encoding="latin-1") as fh:
    for line in fh:
        buf += line
        if line.rstrip().endswith(";"):
            m = pat.match(buf.strip())
            if m and m.group(2) in KEEP:
                ent[int(m.group(1))] = (m.group(2), m.group(3))
            buf = ""
refs = lambda s: [int(x) for x in re.findall(r"#(\d+)", s)]
strs = lambda s: re.findall(r"'((?:[^']|'')*)'", s)
children = defaultdict(list); is_child = set()
for k, (t, a) in ent.items():
    if t == "NEXT_ASSEMBLY_USAGE_OCCURRENCE":
        r = refs(a); children[r[0]].append(r[1]); is_child.add(r[1])
def pd_name(pd):
    r = refs(ent[pd][1]); pdf = ent.get(r[0]); prod = ent.get(refs(pdf[1])[0]) if pdf else None
    return strs(prod[1])[1] if prod and len(strs(prod[1])) > 1 else "?"
pds = [k for k, (t, a) in ent.items() if t == "PRODUCT_DEFINITION"]
roots = [p for p in pds if p not in is_child]
truth = Counter()
def walk(p, mult=1):
    if not children[p]:
        truth[pd_name(p)] += 1
    for c in children[p]:
        walk(c)
for r in roots:
    walk(r)
a = pickle.load(open(sys.argv[2], "rb"))
found = Counter(p.name for p in a.parts)
print("truth leaf instances", sum(truth.values()), "found parts", sum(found.values()))
diff = [(n, truth.get(n, 0), found.get(n, 0)) for n in set(truth) | set(found) if truth.get(n, 0) != found.get(n, 0)]
diff.sort(key=lambda x: -abs(x[1] - x[2]))
print("names with different counts:", len(diff))
for n, t, f in diff[:30]:
    print(f"  {n[:50]:50s} truth {t:4d} found {f:4d}")
