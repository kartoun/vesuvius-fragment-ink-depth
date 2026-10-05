"""Step 2: Re-measure the ink signal relative to each pixel's own papyrus surface.

The fragment surface is not flat, so a fixed layer index sits at different depths in
different places. For sampled ink and clear-papyrus columns, find the surface as the
steepest intensity drop (papyrus -> air) and compare profiles aligned on that point.
Also checks whether ink and clear areas sit at different surface heights (a confound).

Outputs (in outputs/step2/): aligned_profile.csv, aligned_profile.png, surface_height.txt

Usage:  python step2_surface_aligned_profile.py [data_dir]
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

from step1_ink_depth_profile import (BG_GAP, EDGE_ERODE, N_SAMPLE, SEED,
                                     cohens_d, load_mask, sample)

SEARCH = (20, 50)   # layers in which to look for the papyrus->air drop
OFFSETS = np.arange(-15, 16)  # layers relative to surface


def main():
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("D:/DBBun/data/vesuvius/Frag1")
    out = Path(__file__).parent / "outputs" / "step2"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    frag = load_mask(data / "mask.png")
    ink = load_mask(data / "inklabels.png")
    ink_core = ndimage.binary_erosion(ink, iterations=EDGE_ERODE) & frag
    clear = frag & ~ndimage.binary_dilation(ink, iterations=BG_GAP)
    clear = ndimage.binary_erosion(clear, iterations=BG_GAP)
    idx = np.r_[sample(ink_core, N_SAMPLE, rng), sample(clear, N_SAMPLE, rng)]
    is_ink = np.r_[np.ones(N_SAMPLE, bool), np.zeros(idx.size - N_SAMPLE, bool)]

    layers = sorted((data / "surface_volume").glob("*.tif"))
    cols = np.empty((idx.size, len(layers)), np.float32)
    for i, p in enumerate(layers):
        cols[:, i] = np.array(Image.open(p), dtype=np.float32).ravel()[idx]
        print(f"\rread layer {i + 1}/{len(layers)}", end="")
    print()

    smooth = ndimage.gaussian_filter1d(cols, 1.5, axis=1)
    grad = np.diff(smooth, axis=1)
    lo, hi = SEARCH
    surf = lo + np.argmin(grad[:, lo:hi], axis=1)  # steepest drop

    hi_ink, hi_bg = surf[is_ink], surf[~is_ink]
    msg = (f"surface layer, ink:   mean {hi_ink.mean():.2f}  median {np.median(hi_ink):.0f}  sd {hi_ink.std():.2f}\n"
           f"surface layer, clear: mean {hi_bg.mean():.2f}  median {np.median(hi_bg):.0f}  sd {hi_bg.std():.2f}\n"
           f"Cohen's d (ink - clear): {cohens_d(hi_ink.astype(float), hi_bg.astype(float)):+.3f}\n")
    print(msg)
    (out / "surface_height.txt").write_text(msg)

    # Aligned columns; normalise each column by its own papyrus peak and air level
    rows_ok = (surf + OFFSETS.min() >= 0) & (surf + OFFSETS.max() < cols.shape[1])
    take = surf[rows_ok, None] + OFFSETS[None, :]
    aligned = np.take_along_axis(cols[rows_ok], take, axis=1)
    peak = aligned[:, OFFSETS < 0].max(axis=1, keepdims=True)
    air = aligned[:, OFFSETS > 8].mean(axis=1, keepdims=True)
    norm = (aligned - air) / np.maximum(peak - air, 1)
    y = is_ink[rows_ok]

    res = []
    for j, o in enumerate(OFFSETS):
        a, b = aligned[y, j], aligned[~y, j]
        na, nb = norm[y, j], norm[~y, j]
        lab = np.r_[np.ones(a.size), np.zeros(b.size)]
        res.append((o, a.mean(), b.mean(), cohens_d(a, b), roc_auc_score(lab, np.r_[a, b]),
                    cohens_d(na, nb), roc_auc_score(lab, np.r_[na, nb])))
    with open(out / "aligned_profile.csv", "w") as f:
        f.write("offset,ink_mean,bg_mean,d_raw,auc_raw,d_norm,auc_norm\n")
        for r in res:
            f.write(",".join(f"{v:.5g}" for v in r) + "\n")
    for r in res:
        print(f"offset {r[0]:+3d}  d_raw {r[3]:+.3f} auc_raw {r[4]:.3f}  d_norm {r[5]:+.3f} auc_norm {r[6]:.3f}")

    R = np.array(res)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].hist([hi_ink, hi_bg], bins=np.arange(lo, hi + 1) - 0.5, density=True,
               label=["ink", "clear papyrus"])
    ax[0].set(title="Detected surface layer", xlabel="layer of steepest drop")
    ax[0].legend()
    ax[1].plot(R[:, 0], R[:, 1], label="ink")
    ax[1].plot(R[:, 0], R[:, 2], label="clear papyrus")
    ax[1].axvline(0, color="gray", lw=0.8)
    ax[1].set(title="Surface-aligned mean intensity", xlabel="layers from surface (+ = toward air)")
    ax[1].legend()
    ax[2].plot(R[:, 0], R[:, 3], label="raw")
    ax[2].plot(R[:, 0], R[:, 5], label="per-column normalised")
    ax[2].axhline(0, color="gray", lw=0.8)
    ax[2].axvline(0, color="gray", lw=0.8)
    ax[2].set(title="Ink vs papyrus effect size (Cohen's d)", xlabel="layers from surface")
    ax[2].legend()
    fig.suptitle("Frag1: ink signal aligned to each pixel's papyrus surface")
    fig.tight_layout()
    fig.savefig(out / "aligned_profile.png", dpi=130)
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
