"""Step 1: In which CT layers does ink show up?

For Frag1, compare CT intensity at ink pixels vs clear-papyrus pixels, layer by layer.
Ink/non-ink comes from the existing 2D hand-traced labels (inklabels.png), shrunk and
grown so stroke edges are excluded. Reports raw and locally background-subtracted
contrast per layer (Cohen's d and ROC AUC).

Outputs (in outputs/step1/): layer_profile.csv, layer_profile.png

Usage:  python step1_ink_depth_profile.py [data_dir]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.metrics import roc_auc_score

Image.MAX_IMAGE_PIXELS = None
SEED = 0
N_SAMPLE = 200_000
EDGE_ERODE = 3     # px removed from inside strokes
BG_GAP = 20        # px between strokes and "clear papyrus" pixels
LOCAL_WIN = 101    # px window for local background mean


def load_mask(path):
    return np.array(Image.open(path)) > 0


def sample(mask, n, rng):
    idx = np.flatnonzero(mask)
    return rng.choice(idx, size=min(n, idx.size), replace=False)


def cohens_d(a, b):
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / s if s > 0 else 0.0


def main():
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("D:/DBBun/data/vesuvius/Frag1")
    out = Path(__file__).parent / "outputs" / "step1"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    frag = load_mask(data / "mask.png")
    ink = load_mask(data / "inklabels.png")
    ink_core = ndimage.binary_erosion(ink, iterations=EDGE_ERODE) & frag
    clear = frag & ~ndimage.binary_dilation(ink, iterations=BG_GAP)
    clear = ndimage.binary_erosion(clear, iterations=BG_GAP)  # also keep away from fragment edge
    ink_idx, bg_idx = sample(ink_core, N_SAMPLE, rng), sample(clear, N_SAMPLE, rng)
    print(f"ink pixels: {ink_core.sum():,} (sampled {ink_idx.size:,}); "
          f"clear papyrus: {clear.sum():,} (sampled {bg_idx.size:,})")

    layers = sorted((data / "surface_volume").glob("*.tif"))
    rows = []
    for i, p in enumerate(layers):
        v = np.array(Image.open(p), dtype=np.float32)
        local = v - ndimage.uniform_filter(v, LOCAL_WIN)
        ri, rb = v.ravel()[ink_idx], v.ravel()[bg_idx]
        li, lb = local.ravel()[ink_idx], local.ravel()[bg_idx]
        y = np.r_[np.ones(ri.size), np.zeros(rb.size)]
        rows.append(dict(
            layer=i, ink_mean=ri.mean(), bg_mean=rb.mean(),
            d_raw=cohens_d(ri, rb), auc_raw=roc_auc_score(y, np.r_[ri, rb]),
            d_local=cohens_d(li, lb), auc_local=roc_auc_score(y, np.r_[li, lb]),
        ))
        r = rows[-1]
        print(f"layer {i:02d}  ink {r['ink_mean']:8.0f}  bg {r['bg_mean']:8.0f}  "
              f"d_raw {r['d_raw']:+.3f}  auc_raw {r['auc_raw']:.3f}  "
              f"d_local {r['d_local']:+.3f}  auc_local {r['auc_local']:.3f}")

    keys = list(rows[0])
    with open(out / "layer_profile.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.5g}" for k in keys) + "\n")

    L = np.array([r["layer"] for r in rows])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(L, [r["ink_mean"] for r in rows], label="ink (hand labels, core)")
    ax[0].plot(L, [r["bg_mean"] for r in rows], label="clear papyrus")
    ax[0].set(title="Mean CT intensity by layer", xlabel="surface-volume layer", ylabel="intensity")
    ax[0].legend()
    ax[1].plot(L, [r["d_raw"] for r in rows], label="raw")
    ax[1].plot(L, [r["d_local"] for r in rows], label=f"local bg-subtracted ({LOCAL_WIN}px)")
    ax[1].axhline(0, color="gray", lw=0.8)
    ax[1].set(title="Ink vs papyrus effect size (Cohen's d)", xlabel="layer")
    ax[1].legend()
    ax[2].plot(L, [r["auc_raw"] for r in rows], label="raw")
    ax[2].plot(L, [r["auc_local"] for r in rows], label="local")
    ax[2].axhline(0.5, color="gray", lw=0.8)
    ax[2].set(title="Single-voxel separability (ROC AUC)", xlabel="layer")
    ax[2].legend()
    fig.suptitle("Frag1 (PHerc. Paris 2 Fr. 47): where does ink appear in depth?")
    fig.tight_layout()
    fig.savefig(out / "layer_profile.png", dpi=130)
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
