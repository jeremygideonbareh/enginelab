"""Assemble the folder that is published as the hosted beta (Hugging Face Space / any Docker host).
Only code, the current trained models and permissively licensed samples go in; nothing private.

    python tools/build_space.py [out_dir]      (default: deploy/space)
"""
import hashlib
import os
import shutil
import sys

import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "deploy", "space"))
SAMPLES = ["data/real_cad/koch_follower/arm.step", "data/real_cad/so101/SO101_Assembly.step",
           "data/real_cad/odri_biped/biped_6dof_v1.STEP"]
MODELS = ["data/models/component_classifier.joblib", "data/models/component_classifier_alone.joblib",
          "data/models/part_classifier.joblib"]


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(os.path.join(ROOT, "enginelab"), os.path.join(OUT, "enginelab"), ignore=ign)
    shutil.copytree(os.path.join(ROOT, "web"), os.path.join(OUT, "web"), ignore=ign)
    shutil.copy(os.path.join(ROOT, "run.py"), OUT)
    for f, dst in [("Dockerfile", "Dockerfile"), ("requirements.txt", "requirements.txt"),
                   ("SPACE_README.md", "README.md"), ("NOTICE.md", "NOTICE.md")]:
        shutil.copy(os.path.join(ROOT, "deploy", f), os.path.join(OUT, dst))
    for m in MODELS:
        dst = os.path.join(OUT, m)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        joblib.dump(joblib.load(os.path.join(ROOT, m)), dst, compress=3)      # smaller download
    from enginelab.cad import CACHE_VERSION
    os.makedirs(os.path.join(OUT, "data", "cache"), exist_ok=True)
    for s in SAMPLES:
        src = os.path.join(ROOT, s)
        dst = os.path.join(OUT, s)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        cache = os.path.join(ROOT, "data", "cache", f"cad_v{CACHE_VERSION}_{sha1(src)}.pkl")
        if os.path.exists(cache):          # first open of a sample is instant
            shutil.copy(cache, os.path.join(OUT, "data", "cache"))
    if os.path.isdir(os.path.join(ROOT, "data", "samples")):
        shutil.copytree(os.path.join(ROOT, "data", "samples"), os.path.join(OUT, "data", "samples"))
    os.makedirs(os.path.join(OUT, "data", "optimizer"), exist_ok=True)
    with open(os.path.join(OUT, ".dockerignore"), "w") as fh:
        fh.write("__pycache__\n*.pyc\n")
    total = 0
    for dp, _, fs in os.walk(OUT):
        for f in fs:
            total += os.path.getsize(os.path.join(dp, f))
    print(f"built {OUT}: {total / 1e6:.0f} MB")


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    main()
