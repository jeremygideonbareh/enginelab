"""Quick matplotlib renders of labelled assemblies (for checking geometry)."""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PALETTE = plt.get_cmap("tab20").colors


def render(parts, path, hide=(), elev=20, azim=-60, title=""):
    labels = sorted({l for l, _ in parts})
    color = {l: PALETTE[i % 20] for i, l in enumerate(labels)}
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    allv = []
    light = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    for label, mesh in parts:
        if label in hide:
            continue
        tris = mesh.triangles
        shade = 0.45 + 0.55 * np.clip(mesh.face_normals @ light, 0, 1)
        cols = np.clip(np.array(color[label])[None, :] * shade[:, None], 0, 1)
        pc = Poly3DCollection(tris, facecolors=cols, edgecolors="none")
        ax.add_collection3d(pc)
        allv.append(mesh.vertices)
    v = np.vstack(allv)
    c, h = v.mean(0), (v.max(0) - v.min(0)).max() / 2
    ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h); ax.set_zlim(c[2] - h, c[2] + h)
    ax.view_init(elev, azim)
    ax.set_axis_off()
    handles = [plt.Line2D([0], [0], marker="s", ls="", color=color[l], label=l) for l in labels if l not in hide]
    ax.legend(handles=handles, loc="upper left", fontsize=7)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    sys.path.insert(0, ".")
    from enginelab.generator import random_engine

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    out = sys.argv[2] if len(sys.argv) > 2 else f"engine_{seed}.png"
    e = random_engine(seed)
    hide = ("engine_block", "valve_cover", "oil_pan", "cylinder_head", "head_gasket")
    render(e.parts, out, hide=hide, title=f"{e.params.layout}{e.params.n_cyl} bore {e.params.bore:.0f} stroke {e.params.stroke:.0f} (cutaway)")
    render(e.parts, out.replace(".png", "_full.png"), title="full assembly")
