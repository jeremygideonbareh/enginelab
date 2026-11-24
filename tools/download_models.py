"""Download the trained models (and optionally the sample robot CAD) from the GitHub release.

    python tools/download_models.py            # models only (~45 MB)
    python tools/download_models.py --samples  # also the three open-source sample robots (~50 MB)

Files are checked against the SHA-256 sums in tools/release_assets.json before they are kept.
To train the models yourself instead, see README.md ("Training the models").
"""
import hashlib
import json
import os
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = json.load(open(os.path.join(ROOT, "tools", "release_assets.json"), encoding="utf-8"))
BASE = MANIFEST["base_url"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(asset, dest):
    if os.path.exists(dest) and sha256(dest) == asset["sha256"]:
        print(f"ok (already here)  {os.path.relpath(dest, ROOT)}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"{BASE}/{asset['name']}"
    tmp = dest + ".part"
    print(f"downloading {asset['name']} ({asset['size'] / 1e6:.1f} MB) ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "enginelab-downloader"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    if sha256(tmp) != asset["sha256"]:
        os.remove(tmp)
        sys.exit(f"checksum mismatch for {asset['name']} - download again or report an issue")
    os.replace(tmp, dest)
    print(f"ok  {os.path.relpath(dest, ROOT)}")


def main():
    for a in MANIFEST["models"]:
        fetch(a, os.path.join(ROOT, "data", "models", a["name"]))
    if "--samples" in sys.argv:
        s = MANIFEST["samples"]
        z = os.path.join(ROOT, "data", "cache", s["name"])
        fetch(s, z)
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                if m.startswith("/") or ".." in m.split("/"):
                    sys.exit(f"unsafe path in archive: {m}")
            zf.extractall(ROOT)
        print("samples unpacked into data/real_cad/ (see data/real_cad/SAMPLES_NOTICE.md for their licences)")


if __name__ == "__main__":
    main()
