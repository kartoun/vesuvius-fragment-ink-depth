"""Does the v4 measured band follow the surface better where there is more relief?

Answers khj1222's question on villa #192 (2026-10-06). At the 64 px cell level, compares
the v4 band centre with the mean exposed-surface layer of the cell's ink pixels, overall,
by row band, and by local relief (sd of cell surface in a 5 x 5 cell window), separately
for rows the occluder model was trained on and rows it never saw.

Usage:  python relief_check.py [outputs/khj_recipes[_tag]]
"""
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from scipy.stats import spearmanr

from common import LABELS_OUT, OUT, load_png_mask
from khj_recipes import CELL

HOLDOUT = (3298, 4432)   # step-4 held-out rows; the step-4 occluder trained outside +-96 rows
BUFFER = 96
V3_CENTRE = 32.61


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "khj_recipes"
    C = np.load(folder / "inkdepth.npz")["center"]
    frag = load_png_mask("mask.png")
    ink = load_png_mask("inklabels.png") & frag
    surf = tifffile.imread(LABELS_OUT / "surface.tif").astype(np.float32)
    rel = tifffile.imread(LABELS_OUT / "surface_reliable.tif") > 0
    core = ndimage.binary_erosion(ink, iterations=3) & rel
    H, W = frag.shape
    gh, gw = C.shape
    cid = (np.arange(H)[:, None] // CELL) * gw + (np.arange(W)[None] // CELL)
    n = np.bincount(cid[core], minlength=gh * gw).reshape(gh, gw)
    s = np.bincount(cid[core], weights=surf[core], minlength=gh * gw).reshape(gh, gw)
    ok = n >= 64
    smean = np.where(ok, s / np.maximum(n, 1), np.nan)
    rows_c = np.arange(gh) * CELL + CELL / 2
    held = ((rows_c >= HOLDOUT[0]) & (rows_c < HOLDOUT[1]))[:, None]
    train = ((rows_c < HOLDOUT[0] - BUFFER) | (rows_c >= HOLDOUT[1] + BUFFER))[:, None]
    local = ndimage.generic_filter(np.nan_to_num(smean, nan=np.nanmedian(smean)), np.std, size=5)

    def mad(o):
        return np.median(np.abs(o - np.median(o)))

    out = [f"occluder depths from: {folder}"]

    def stats(mask, label):
        if mask.sum() < 10:
            return
        a, b = C[mask], smean[mask]
        out.append(f"{label:36s} cells {mask.sum():4d}  relief sd {np.std(b):4.2f}  corr {np.corrcoef(a, b)[0, 1]:+.2f} "
                   f"(rho {spearmanr(a, b).statistic:+.2f})  slope {np.polyfit(b, a, 1)[0]:+.2f}  "
                   f"MAD v4 {mad(a - b):.2f} v3 {mad(V3_CENTRE - b):.2f}")

    stats(ok, "whole fragment")
    stats(ok & held, "step-4 held-out rows")
    stats(ok & train, "step-4 training rows")
    for e in range(0, H, 1134):
        m = ((rows_c >= e) & (rows_c < e + 1134))[:, None]
        stats(ok & m, f"rows {e:5d}-{min(e + 1134, H):5d}")
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 99)]:
        m = ok & (local >= lo) & (local < hi)
        stats(m & train, f"training rows, local relief {lo}-{hi}")
        stats(m & held, f"held-out rows, local relief {lo}-{hi}")
    txt = "\n".join(out)
    print(txt)
    (folder / "relief_check.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
