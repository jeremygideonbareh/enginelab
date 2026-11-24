"""User feedback that makes the AI better: label corrections become training examples, and
free-text problem reports are kept for review.

Privacy: a correction stores only the part's numeric shape features (size ratios, surface
statistics, assembly-context numbers) and the two labels. No geometry, no part or file names,
no sub-assembly names, nothing about the user.

Storage: data/feedback/*.jsonl. On a host with an ephemeral disk (Hugging Face Spaces), set
HF_TOKEN and FEEDBACK_REPO (a private dataset repo, e.g. "you/enginelab-feedback") and the files
are uploaded there every few minutes, then downloaded again at start-up.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time

import numpy as np

DIR = os.path.join(os.path.dirname(__file__), "..", "data", "feedback")
CORRECTIONS = os.path.join(DIR, "corrections.jsonl")
REPORTS = os.path.join(DIR, "reports.jsonl")
_lock = threading.Lock()
_last_sync = [0.0]


def _clean(v):
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return v if math.isfinite(v) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def _append(path: str, rec: dict) -> None:
    os.makedirs(DIR, exist_ok=True)
    with _lock, open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=_clean) + "\n")
    maybe_sync()


def part_features(asm, part_id: int) -> dict:
    """The numbers the classifiers see for one part (engine and component models)."""
    from .component_model import row as cm_row
    from .features import context_features
    ctx = context_features([p.info for p in asm.parts])
    p = asm.parts[part_id]
    eng = {**p.info.features, **ctx[part_id]}
    comp = cm_row(p.info, ctx[part_id], p.brep or None, int(p.specs.get("bodies", 1) or 1))
    return {"engine": {k: _clean(v) for k, v in eng.items()}, "component": {k: _clean(v) for k, v in comp.items()}}


def record_correction(asm, part_id: int, old_label: str, new_label: str) -> None:
    if old_label == new_label:
        return
    feats = part_features(asm, part_id)
    _append(CORRECTIONS, {"time": round(time.time()), "mode": asm.mode, "old": old_label, "label": new_label,
                          "engine_features": feats["engine"], "component_features": feats["component"],
                          "copies": sum(1 for q in asm.parts if q.group == asm.parts[part_id].group)})


def record_report(text: str, mode: str | None = None, label: str | None = None, page: str | None = None) -> None:
    text = (text or "").strip()[:2000]
    if not text:
        raise ValueError("Empty report")
    _append(REPORTS, {"time": round(time.time()), "mode": mode, "label": label, "page": page, "text": text})


def load_corrections(kind: str) -> tuple[list[dict], list[str]]:
    """(feature rows, labels) of saved corrections for the "engine" or "component" model.
    The latest correction per identical feature row wins (people change their minds)."""
    if not os.path.exists(CORRECTIONS):
        return [], []
    from .breakdown import GENERAL_LABELS
    latest: dict = {}
    with open(CORRECTIONS, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("label") in (None, "unknown"):
                continue
            general = r["label"] in GENERAL_LABELS
            if (kind == "component") != general:
                continue
            f = r["component_features" if kind == "component" else "engine_features"]
            key = json.dumps(sorted((k, round(v, 4) if isinstance(v, float) else v) for k, v in f.items()))
            latest[key] = ({k: (np.nan if v is None else v) for k, v in f.items()}, r["label"])
    X = [v[0] for v in latest.values()]
    y = [v[1] for v in latest.values()]
    return X, y


def summary() -> dict:
    out = {"corrections": 0, "reports": 0, "by_label": {}}
    for path, key in ((CORRECTIONS, "corrections"), (REPORTS, "reports")):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    out[key] += 1
                    if key == "corrections":
                        try:
                            lab = json.loads(line)["label"]
                            out["by_label"][lab] = out["by_label"].get(lab, 0) + 1
                        except ValueError:
                            pass
    out["synced_to"] = os.environ.get("FEEDBACK_REPO") or None
    return out


# ------------------------------------------------------------------ optional persistent sync
def _hub():
    repo, token = os.environ.get("FEEDBACK_REPO"), os.environ.get("HF_TOKEN")
    if not (repo and token):
        return None
    try:
        from huggingface_hub import HfApi
        return HfApi(token=token), repo
    except ImportError:
        return None


def maybe_sync(min_interval: float = 300.0) -> None:
    hub = _hub()
    if not hub or time.time() - _last_sync[0] < min_interval:
        return
    _last_sync[0] = time.time()

    def run():
        api, repo = hub
        for path in (CORRECTIONS, REPORTS):
            if os.path.exists(path):
                try:
                    api.upload_file(path_or_fileobj=path, path_in_repo=os.path.basename(path), repo_id=repo,
                                    repo_type="dataset", commit_message="feedback sync")
                except Exception:
                    pass
    threading.Thread(target=run, daemon=True).start()


def restore() -> None:
    """At start-up on an ephemeral host: fetch the feedback files saved in the dataset repo."""
    hub = _hub()
    if not hub:
        return
    api, repo = hub
    os.makedirs(DIR, exist_ok=True)
    for name in ("corrections.jsonl", "reports.jsonl"):
        target = os.path.join(DIR, name)
        if os.path.exists(target):
            continue
        try:
            from huggingface_hub import hf_hub_download
            src = hf_hub_download(repo_id=repo, filename=name, repo_type="dataset", token=os.environ["HF_TOKEN"])
            with open(src, "rb") as a, open(target, "wb") as b:
                b.write(a.read())
        except Exception:
            pass
