"""Step 3: Build surface-following 3D ink labels for Frag1 (plus the flat baseline).

1. Surface map: for every pixel, the layer where intensity drops fastest from papyrus
   to air (smoothed along depth), searched in layers SEARCH. Pixels where the drop sits
   on the search boundary are unreliable. The map is smoothed with a block median.
2. Two label volumes (Z, H, W), values 0 = not ink, 1 = ink, 2 = ignore:
   - flat:  2D hand-traced ink outline copied through every layer (current practice).
   - shell: ink only within SHELL layers of the local surface; other layers of an ink
            column are "not ink".
   Both: outside the fragment, a thin band along stroke edges, and ink columns with an
   unreliable surface are ignored, so the only difference between them is depth.

Outputs:
  D:/DBBun/data/vesuvius/Frag1_3d_labels/{surface.tif, labels_flat.zarr, labels_shell.zarr}
  outputs/step3/{surface_map.png, cross_sections.png, summary.txt}

Usage:  python step3_build_labels.py
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import zarr
from scipy import ndimage

from common import DATA, IGNORE, INK, LABELS_OUT, NOT_INK, OUT, load_png_mask, load_stack

SEARCH = (10, 56)      # layers searched for the papyrus->air drop
SHELL = (-4, 5)        # ink layers relative to surface, inclusive (from step 2)
EDGE_BAND = 3          # px ignored on each side of stroke outlines
BLOCK = 8              # px block size for surface smoothing
OUTLIER = 4            # layers; raw vs smoothed surface disagreement -> unreliable
AGREE_WIN = 15         # px neighbourhood for reliability vote
AGREE_FRAC = 0.5       # fraction of neighbourhood that must agree
ROWS = 256             # rows per processing chunk


def surface_map(stack, frag):
    Z, H, W = stack.shape
    surf = np.zeros((H, W), np.float32)
    lo, hi = SEARCH
    for r in range(0, H, ROWS):
        c = stack[:, r:r + ROWS].astype(np.float32)
        c = ndimage.gaussian_filter1d(c, 1.5, axis=0)
        g = np.diff(c, axis=0)[lo:hi]
        surf[r:r + ROWS] = lo + np.argmin(g, axis=0)
        print(f"\rsurface rows {min(r + ROWS, H)}/{H}", end="", flush=True)
    print()
    lo_hit = (surf <= lo) | (surf >= hi - 1)
    valid = frag & ~lo_hit

    # block median over valid pixels, nearest-fill gaps, upsample
    hb, wb = -(-H // BLOCK), -(-W // BLOCK)
    pad = np.full((hb * BLOCK, wb * BLOCK), np.nan, np.float32)
    pad[:H, :W] = np.where(valid, surf, np.nan)
    blocks = pad.reshape(hb, BLOCK, wb, BLOCK).transpose(0, 2, 1, 3).reshape(hb, wb, -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks are filled below
        med = np.nanmedian(blocks, axis=2)
    have = ~np.isnan(med)
    _, (iy, ix) = ndimage.distance_transform_edt(~have, return_indices=True)
    med = med[iy, ix]
    med = ndimage.median_filter(med, 5)
    smooth = ndimage.zoom(med, BLOCK, order=1)[:H, :W]
    # reliable where most raw detections in the neighbourhood agree with the smoothed surface
    agree = valid & (np.abs(surf - smooth) <= OUTLIER)
    reliable = frag & (ndimage.uniform_filter(agree.astype(np.float32), AGREE_WIN) >= AGREE_FRAC)
    return surf, np.rint(smooth).astype(np.int16), reliable


def build(stack_shape, frag, ink, surf, reliable):
    Z, H, W = stack_shape
    edge = ndimage.binary_dilation(ink, iterations=EDGE_BAND) & ~ndimage.binary_erosion(ink, iterations=EDGE_BAND)
    ignore2d = ~frag | edge

    flat = np.where(ink, INK, NOT_INK).astype(np.uint8)
    flat[ignore2d] = IGNORE
    flat[ink & ~reliable] = IGNORE  # same supervised columns in both label sets

    LABELS_OUT.mkdir(parents=True, exist_ok=True)
    kw = dict(shape=(Z, H, W), chunks=(Z, 256, 256), dtype="uint8", fill_value=IGNORE, overwrite=True)
    zf = zarr.create_array(store=str(LABELS_OUT / "labels_flat.zarr"), **kw)
    zs = zarr.create_array(store=str(LABELS_OUT / "labels_shell.zarr"), **kw)
    z = np.arange(Z)[:, None, None]
    counts = np.zeros((2, 3), np.int64)
    for r in range(0, H, ROWS):
        sl = slice(r, r + ROWS)
        f2, s2 = flat[sl], surf[sl]
        vol_flat = np.broadcast_to(f2, (Z, *f2.shape)).copy()
        rel = z - s2[None]
        in_shell = (rel >= SHELL[0]) & (rel <= SHELL[1])
        vol_shell = np.where(in_shell & (f2 == INK)[None], INK, NOT_INK).astype(np.uint8)
        vol_shell[:, f2 == IGNORE] = IGNORE
        zf[:, sl] = vol_flat
        zs[:, sl] = vol_shell
        for k, v in enumerate((vol_flat, vol_shell)):
            counts[k] += np.bincount(v.ravel(), minlength=3)[:3]
        print(f"\rlabel rows {min(r + ROWS, H)}/{H}", end="", flush=True)
    print()
    return zf, zs, counts


def figures(stack, surf, reliable, frag, zf, zs, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 7))
    s = np.where(frag, surf, np.nan)
    im = ax[0].imshow(s[::4, ::4], cmap="viridis")
    fig.colorbar(im, ax=ax[0], shrink=0.7, label="surface layer")
    ax[0].set_title("Smoothed surface layer")
    ax[1].imshow(np.where(frag, reliable, np.nan)[::4, ::4], cmap="gray")
    ax[1].set_title(f"Reliable surface (white): {reliable[frag].mean():.1%} of fragment")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out / "surface_map.png", dpi=110)
    plt.close(fig)

    # cross-sections through the rows with the most ink
    ink_rows = np.argsort((zf[32] == INK).sum(axis=1))[-1]
    rows = [ink_rows, ink_rows - 600, ink_rows + 600]
    fig, ax = plt.subplots(len(rows), 3, figsize=(16, 2.6 * len(rows)))
    for i, r in enumerate(rows):
        r = int(np.clip(r, 0, stack.shape[1] - 1))
        cols = np.flatnonzero(zf[32, r] == INK)
        c0 = int(np.clip((cols.mean() if cols.size else stack.shape[2] / 2) - 400, 0, stack.shape[2] - 800))
        cs = slice(c0, c0 + 800)
        ct = stack[:, r, cs].astype(np.float32)
        for j, (title, lab) in enumerate((("CT", None), ("flat labels", zf[:, r, cs]), ("surface-shell labels", zs[:, r, cs]))):
            ax[i, j].imshow(ct, cmap="gray", aspect="auto")
            if lab is not None:
                ov = np.zeros((*lab.shape, 4))
                ov[lab == INK] = (1, 0.2, 0.2, 0.55)
                ov[lab == IGNORE] = (0.2, 0.5, 1, 0.35)
                ax[i, j].imshow(ov, aspect="auto")
            ax[i, j].plot(np.arange(800), surf[r, cs], color="yellow", lw=0.6)
            ax[i, j].set_title(f"{title} - row {r}, cols {c0}-{c0 + 800}", fontsize=9)
            ax[i, j].set_ylabel("layer")
    fig.suptitle("Frag1 cross-sections: red = ink, blue = ignore, yellow = detected surface")
    fig.tight_layout()
    fig.savefig(out / "cross_sections.png", dpi=110)
    plt.close(fig)


def main():
    out = OUT / "step3"
    out.mkdir(parents=True, exist_ok=True)
    frag = load_png_mask("mask.png")
    ink = load_png_mask("inklabels.png") & frag
    stack = load_stack()

    raw, surf, reliable = surface_map(stack, frag)
    zf, zs, counts = build(stack.shape, frag, ink, surf, reliable)
    tifffile.imwrite(LABELS_OUT / "surface.tif", surf, compression="zlib")
    tifffile.imwrite(LABELS_OUT / "surface_reliable.tif", reliable.astype(np.uint8), compression="zlib")

    lines = [
        f"fragment pixels: {frag.sum():,}; ink (2D outline): {ink.sum():,}",
        f"reliable surface: {reliable[frag].mean():.1%} of fragment, {reliable[ink].mean():.1%} of ink outline",
        f"surface layer (smoothed, fragment): median {np.median(surf[frag]):.0f}, "
        f"5-95% {np.percentile(surf[frag], 5):.0f}-{np.percentile(surf[frag], 95):.0f}",
        "voxels [not ink, ink, ignore]:",
        f"  flat : {counts[0].tolist()}",
        f"  shell: {counts[1].tolist()}",
        f"ink voxels shell/flat: {counts[1, 1] / counts[0, 1]:.3f}",
    ]
    print("\n".join(lines))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    figures(stack, surf, reliable, frag, zf, zs, out)
    print(f"Labels in {LABELS_OUT}; figures in {out.resolve()}")


if __name__ == "__main__":
    main()
