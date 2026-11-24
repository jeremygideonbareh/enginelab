"""Build the GitHub release assets (compressed models + permissively licensed sample CAD) into release/
and write their checksums to tools/release_assets.json.   python tools/make_release_assets.py <tag>"""
import hashlib
import json
import os
import sys
import zipfile

import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG = sys.argv[1] if len(sys.argv) > 1 else "v0.1.0"
REPO = "jeremygideonbareh/enginelab"
OUT = os.path.join(ROOT, "release")
MODELS = ["part_classifier.joblib", "component_classifier.joblib", "component_classifier_alone.joblib"]
SAMPLES = ["data/real_cad/koch_follower/arm.step", "data/real_cad/so101/SO101_Assembly.step",
           "data/real_cad/odri_biped/biped_6dof_v1.STEP"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def entry(p):
    return {"name": os.path.basename(p), "size": os.path.getsize(p), "sha256": sha256(p)}


os.makedirs(OUT, exist_ok=True)
models = []
for m in MODELS:
    dst = os.path.join(OUT, m)
    joblib.dump(joblib.load(os.path.join(ROOT, "data", "models", m)), dst, compress=3)
    models.append(entry(dst))
    print(m, f"{os.path.getsize(dst) / 1e6:.1f} MB", flush=True)
z = os.path.join(OUT, f"enginelab-samples-{TAG}.zip")
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
    for s in SAMPLES:
        zf.write(os.path.join(ROOT, s), s)
    zf.write(os.path.join(ROOT, "deploy", "NOTICE.md"), "data/real_cad/SAMPLES_NOTICE.md")
print("samples", f"{os.path.getsize(z) / 1e6:.1f} MB")
manifest = {"tag": TAG, "base_url": f"https://github.com/{REPO}/releases/download/{TAG}",
            "models": models, "samples": entry(z)}
json.dump(manifest, open(os.path.join(ROOT, "tools", "release_assets.json"), "w"), indent=1)
print("wrote tools/release_assets.json")
