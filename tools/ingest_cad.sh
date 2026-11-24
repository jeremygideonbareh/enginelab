#!/bin/sh
# Read + verify each new real assembly: full analysis (fills the CAD cache), then the
# independent structure check against its raw STEP file.
for f in "$@"; do
  d=$(dirname "$f")
  python tools/analyze_big.py "$f" "$d/report.json" > "$d/log.txt" 2>&1
  H=$(python -c "
import hashlib,sys; h=hashlib.sha1(); f=open(sys.argv[1],'rb')
for c in iter(lambda: f.read(1<<22), b''): h.update(c)
print(h.hexdigest())" "$f")
  python tools/compare_structure2.py "$f" "data/cache/cad_v3_$H.pkl" > "$d/compare.txt" 2>&1
done
