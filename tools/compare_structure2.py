"""Independent count from the raw STEP text: every product instance (leaf or not) whose
product owns solid geometry.   python tools/compare_structure2.py file.step cache.pkl"""
import io, pickle, re, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from collections import Counter, defaultdict
sys.path.insert(0, ".")
pat = re.compile(r"#(\d+)\s*=\s*([A-Z_0-9]+)\s*\((.*)\)\s*;", re.S)
ent, buf = {}, ""
SOLID_REPS = {"ADVANCED_BREP_SHAPE_REPRESENTATION", "MANIFOLD_SURFACE_SHAPE_REPRESENTATION",
              "FACETED_BREP_SHAPE_REPRESENTATION", "GEOMETRICALLY_BOUNDED_SURFACE_SHAPE_REPRESENTATION"}
KEEP = {"PRODUCT_DEFINITION", "NEXT_ASSEMBLY_USAGE_OCCURRENCE", "PRODUCT_DEFINITION_FORMATION",
        "PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE", "PRODUCT", "PRODUCT_DEFINITION_SHAPE",
        "SHAPE_DEFINITION_REPRESENTATION", "SHAPE_REPRESENTATION_RELATIONSHIP", "SHAPE_REPRESENTATION"} | SOLID_REPS
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
pds_of_shape = {}           # PRODUCT_DEFINITION_SHAPE -> product definition
rep_of_pd = defaultdict(list)
srr = defaultdict(list)     # shape rep -> related reps
for k, (t, a) in ent.items():
    r = refs(a)
    if t == "NEXT_ASSEMBLY_USAGE_OCCURRENCE":
        children[r[0]].append(r[1]); is_child.add(r[1])
    elif t == "PRODUCT_DEFINITION_SHAPE" and r:
        pds_of_shape[k] = r[-1]
for k, (t, a) in ent.items():
    r = refs(a)
    if t == "SHAPE_DEFINITION_REPRESENTATION" and len(r) >= 2 and r[0] in pds_of_shape:
        rep_of_pd[pds_of_shape[r[0]]].append(r[1])
    elif t == "SHAPE_REPRESENTATION_RELATIONSHIP" and len(r) >= 2:
        srr[r[0]].append(r[1]); srr[r[1]].append(r[0])
def has_solid(pd):
    for rep in rep_of_pd.get(pd, []):
        if ent.get(rep, ("",))[0] in SOLID_REPS:
            return True
        for other in srr.get(rep, []):
            if ent.get(other, ("",))[0] in SOLID_REPS:
                return True
    return False
def step_decode(s):
    """STEP string escapes: backslash-X2-backslash hex backslash-X0-backslash (UTF-16), X hh, '' -> '."""
    s = re.sub(r"\\X2\\([0-9A-Fa-f]+)\\X0\\",
               lambda m: bytes.fromhex(m.group(1)).decode("utf-16-be", "replace"), s)
    s = re.sub(r"\\X\\([0-9A-Fa-f]{2})", lambda m: bytes.fromhex(m.group(1)).decode("latin-1"), s)
    return " ".join(s.replace("''", "'").split())
def pd_name(pd):
    r = refs(ent[pd][1]); pdf = ent.get(r[0]); prod = ent.get(refs(pdf[1])[0]) if pdf else None
    return step_decode(strs(prod[1])[1]) if prod and len(strs(prod[1])) > 1 else "?"
pds = [k for k, (t, a) in ent.items() if t == "PRODUCT_DEFINITION"]
roots = [p for p in pds if p not in is_child]
truth = Counter()
def walk(p):
    if has_solid(p):
        truth[pd_name(p)] += 1
    for c in children[p]:
        walk(c)
for r in roots:
    walk(r)
a = pickle.load(open(sys.argv[2], "rb"))
found = Counter(" ".join(p.name.split()) for p in a.parts)
print("product instances owning solids:", sum(truth.values()), "| parts found:", sum(found.values()))
diff = [(n, truth.get(n, 0), found.get(n, 0)) for n in set(truth) | set(found) if truth.get(n, 0) != found.get(n, 0)]
diff.sort(key=lambda x: -abs(x[1] - x[2]))
print("names with different counts:", len(diff))
for n, t, f in diff[:25]:
    print(f"  {n[:50]:50s} truth {t:4d} found {f:4d}")
