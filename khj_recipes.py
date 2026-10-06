"""Rebuild khj1222's three 3D ink-label recipes on an IR-imaged fragment.

Port of the label construction in khj1222/vesuvius-challenge (MIT),
tools/make_3d_labels.py and docs/11-12, so their recipes can be scored against
fragment ground truth with score_3d.py. These are *our rebuilds of their recipes on
Frag1*, not their published labels (which are for scroll segment w00).

  v2 plane    ink on one layer, the middle of the surface volume (z 32)
  v3 constant ink within +-half-width of a single centre: the fragment-wide median of v4
  v4 measured per 64 px cell: occlude 4-layer slabs of the model input, take how much the
              ink logit falls (ink-weighted mean per cell), centre each profile on its own
              mean, band centre = centroid of the positive part, half-width = FWHM in band
              units clamped to [2, 16]; unconvincing cells (peak < 0.2 logit or prominence
              < 1.5 sd) fall back to the median; 3x3 median filter on the cell grid; bilinear
              upsampling; voxel is ink if |z - centre| <= half-width and the 2D label is ink.

Differences from the original, forced by the setting:
  - the occlusion model is our step-4 3D U-Net trained on flat labels (outputs/step4/
    flat_seed0), whose input is layers 12-43; its 3D logit is reduced by max over depth;
  - blanked slabs are set to the global mean (0 after our normalisation) rather than the
    patch median;
  - no annotated "regions" on a fragment, so fallback is to the fragment-wide median.

Outputs: D:/DBBun/data/vesuvius/Frag1_3d_labels/labels_khj_v{2,3,4}.zarr (0/1/2 like ours),
         outputs/khj_recipes/{inkdepth.npz, summary.txt, cross_sections.png}

Usage:  python khj_recipes.py [--run flat_seed0]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
from scipy import ndimage

from common import IGNORE, INK, LABELS_OUT, NOT_INK, OUT, LazyStack, load_png_mask
from step4_train_compare import UNet3D, Z0, Z1, load_inputs

OCCLUDE = 4
CELL = 64
TILE = 256
MIN_CELL_INK = 64
REGULARIZE = 3
HW_FRACTION, HW_MIN, HW_MAX = 0.5, 2.0, 16.0
MIN_RESPONSE, MIN_PROMINENCE = 0.2, 1.5
EDGE_BAND = 3
BATCH = 3  # variants per forward pass (peak ~2.9 GB on an 8 GB GPU)


def estimate_depth(sensitivity, band_centers):
    """Same arithmetic as khj1222's estimate_depth(), centroid estimator."""
    profile = sensitivity - sensitivity.mean(axis=0, keepdims=True)
    peak_index = np.argmax(profile, axis=0)
    peak_value = np.take_along_axis(profile, peak_index[None], axis=0)[0]
    above = profile >= (peak_value * HW_FRACTION)[None]
    half_width = np.clip(above.sum(axis=0) * OCCLUDE / 2.0, HW_MIN, HW_MAX)
    prominence = peak_value / np.maximum(profile.std(axis=0), 1e-6)
    confident = (peak_value >= MIN_RESPONSE) & (prominence >= MIN_PROMINENCE)
    weights = np.clip(profile, 0.0, None)
    center = (weights * band_centers[:, None, None]).sum(axis=0) / np.maximum(weights.sum(axis=0), 1e-6)
    return center, half_width, confident


def weighted_median(values, weights):
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cum = np.cumsum(weights)
    return float(values[int(np.searchsorted(cum, cum[-1] / 2.0))])


def sample_grid(grid, H, W):
    rows = (np.arange(H, dtype=np.float32) + 0.5) / CELL - 0.5
    cols = (np.arange(W, dtype=np.float32) + 0.5) / CELL - 0.5
    out = np.empty((H, W), np.float32)
    for r in range(0, H, 512):  # in strips to bound memory
        rr = rows[r:r + 512]
        coords = np.stack(np.meshgrid(rr, cols, indexing="ij"))
        out[r:r + 512] = ndimage.map_coordinates(grid, coords, order=1, mode="nearest")
    return out


@torch.no_grad()
def measure(model_path, norm):
    vol, frag, mu, sd = load_inputs()
    if norm == "u8":  # step-5 models: per-fragment 0.5-99.5 percentile scaling to uint8, then (u8-127.5)/64
        lo, hi = np.percentile(vol[:, frag][:, ::997].astype(np.float32), [0.5, 99.5])
    ink = load_png_mask("inklabels.png") & frag
    H, W = frag.shape
    model = UNet3D().cuda()
    model.load_state_dict(torch.load(model_path, map_location="cuda"))
    model.eval()

    depth = Z1 - Z0
    starts = list(range(0, depth, OCCLUDE))
    band_centers = np.array([0.5 * (s + min(depth, s + OCCLUDE)) for s in starts], np.float32)
    gh, gw = -(-H // CELL), -(-W // CELL)
    cell_sum = np.zeros((len(starts), gh, gw), np.float32)
    cell_w = np.zeros((gh, gw), np.float32)
    tiles = [(y, x) for y in range(0, H - TILE + 1, TILE) for x in range(0, W - TILE + 1, TILE)
             if ink[y:y + TILE, x:x + TILE].any()]
    for n, (y, x) in enumerate(tiles):
        truth = ink[y:y + TILE, x:x + TILE].astype(np.float32)
        raw = torch.from_numpy(vol[:, y:y + TILE, x:x + TILE].astype(np.float32)).cuda()
        if norm == "u8":
            xb = (torch.floor(torch.clamp((raw - lo) * (255.0 / (hi - lo)), 0, 255)) - 127.5) / 64.0
        else:
            xb = (raw - mu) / sd
        batch = xb[None].repeat(len(starts) + 1, 1, 1, 1)
        for i, s in enumerate(starts, 1):
            batch[i, s:s + OCCLUDE] = 0.0
        logit = torch.empty((batch.shape[0], TILE, TILE), device="cuda")
        for b in range(0, batch.shape[0], BATCH):  # small batches: 9 at once overflows 8 GB
            with torch.autocast("cuda", dtype=torch.float16):
                logit[b:b + BATCH] = model(batch[b:b + BATCH, None]).float()[:, 0].amax(dim=1)
        sens = (logit[0:1] - logit[1:]).cpu().numpy()
        cs = sens * truth[None]
        k = TILE // CELL
        cell_sum[:, y // CELL:y // CELL + k, x // CELL:x // CELL + k] += cs.reshape(len(starts), k, CELL, k, CELL).sum(axis=(2, 4))
        cell_w[y // CELL:y // CELL + k, x // CELL:x // CELL + k] += truth.reshape(k, CELL, k, CELL).sum(axis=(1, 3))
        print(f"\rocclusion tiles {n + 1}/{len(tiles)}", end="", flush=True)
    print()

    mean_sens = cell_sum / np.maximum(cell_w, 1e-6)[None]
    center, half_width, confident = estimate_depth(mean_sens, band_centers)
    confident &= cell_w >= MIN_CELL_INK
    g_center = weighted_median(center[confident], cell_w[confident])
    g_width = weighted_median(half_width[confident], cell_w[confident])
    fc, fw = center.copy(), half_width.copy()
    fc[~confident], fw[~confident] = g_center, g_width
    fc = ndimage.median_filter(fc, size=REGULARIZE, mode="nearest")
    fw = ndimage.median_filter(fw, size=REGULARIZE, mode="nearest")
    stats = dict(cells_measured=int(confident.sum()), cells_with_ink=int((cell_w >= MIN_CELL_INK).sum()),
                 ink_share_measured=float(cell_w[confident].sum() / cell_w.sum()),
                 median_center=g_center + Z0, median_half_width=g_width,
                 center_p5_p95=[float(np.percentile(center[confident], 5) + Z0),
                                float(np.percentile(center[confident], 95) + Z0)])
    return frag, ink, fc + Z0, fw, g_center + Z0, g_width, stats


def write_labels(frag, ink, center_grid, width_grid, g_center, g_width):
    H, W = frag.shape
    Z = LazyStack().shape[0]
    reliable = np.ones_like(frag)  # their recipe does not use our surface reliability
    edge = ndimage.binary_dilation(ink, iterations=EDGE_BAND) & ~ndimage.binary_erosion(ink, iterations=EDGE_BAND)
    ignore2d = ~frag | edge
    c_full = sample_grid(center_grid, H, W)
    w_full = sample_grid(width_grid, H, W)
    z = np.arange(Z, dtype=np.float32)[:, None, None]
    kw = dict(shape=(Z, H, W), chunks=(Z, 256, 256), dtype="uint8", fill_value=IGNORE, overwrite=True)
    arrs = {v: zarr.create_array(store=str(LABELS_OUT / f"labels_khj_{v}.zarr"), **kw) for v in ("v2", "v3", "v4")}
    for r in range(0, H, 256):
        sl = slice(r, r + 256)
        ink2, ig2 = ink[sl] & reliable[sl], ignore2d[sl]
        bands = {
            "v2": np.broadcast_to(z == 32, (Z, *ink2.shape)),
            "v3": np.broadcast_to(np.abs(z - g_center) <= g_width, (Z, *ink2.shape)),
            "v4": np.abs(z - c_full[sl][None]) <= w_full[sl][None],
        }
        for v, band in bands.items():
            vol = np.where(band & ink2[None], INK, NOT_INK).astype(np.uint8)
            vol[:, ig2] = IGNORE
            arrs[v][:, sl] = vol
        print(f"\rlabel rows {min(r + 256, H)}/{H}", end="", flush=True)
    print()
    return c_full, w_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="flat_seed0", help="step-4 run whose model measures the depth")
    ap.add_argument("--model", default=None, help="explicit model.pt (e.g. a step-5 leave-one-out model)")
    ap.add_argument("--norm", choices=["zscore", "u8"], default="zscore", help="input normalisation of that model")
    ap.add_argument("--tag", default="", help="suffix for the output folder; with a tag no label volumes are written")
    a = ap.parse_args()
    out = OUT / ("khj_recipes" + (f"_{a.tag}" if a.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    model_path = a.model or (OUT / "step4" / a.run / "model.pt")
    frag, ink, cg, wg, g_center, g_width, stats = measure(model_path, a.norm)
    np.savez_compressed(out / "inkdepth.npz", center=cg, half_width=wg)
    if a.tag:
        c_full = sample_grid(cg, *frag.shape)
    else:
        c_full, w_full = write_labels(frag, ink, cg, wg, g_center, g_width)

    import tifffile
    surf = tifffile.imread(LABELS_OUT / "surface.tif").astype(np.float32)
    rel = tifffile.imread(LABELS_OUT / "surface_reliable.tif") > 0
    m = ink & rel
    off = (c_full - surf)[m]
    lines = [f"occlusion model: {model_path} ({a.norm})",
             *(f"{k}: {v}" for k, v in stats.items()),
             f"v3 constant band: centre {g_center:.2f}, half-width {g_width:.2f}",
             f"v4 centre minus exposed surface (ink px, reliable surface): median {np.median(off):+.2f}, "
             f"IQR {np.percentile(off, 25):+.2f} to {np.percentile(off, 75):+.2f}"]
    print("\n".join(lines))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")

    if a.tag:  # cross-section shows the written label volumes, which a tagged run does not write
        return

    # one cross-section through the row with most ink
    r = int(np.argmax(ink.sum(axis=1)))
    cols = np.flatnonzero(ink[r])
    c0 = int(np.clip(cols.mean() - 400, 0, ink.shape[1] - 800))
    cs = slice(c0, c0 + 800)
    ct = LazyStack().rows(r, r + 1)[:, 0, cs].astype(np.float32)
    fig, ax = plt.subplots(1, 3, figsize=(16, 2.8))
    for j, v in enumerate(("v2", "v3", "v4")):
        lab = zarr.open_array(str(LABELS_OUT / f"labels_khj_{v}.zarr"), mode="r")[:, r, cs]
        ov = np.zeros((*lab.shape, 4))
        ov[lab == INK] = (1, 0.2, 0.2, 0.6)
        ax[j].imshow(ct, cmap="gray", aspect="auto")
        ax[j].imshow(ov, aspect="auto")
        ax[j].plot(np.arange(800), surf[r, cs], color="yellow", lw=0.6)
        ax[j].set_title(f"khj {v} recipe on Frag1 - row {r}", fontsize=9)
    fig.suptitle("khj1222's label recipes rebuilt on Frag1 (red = ink, yellow = exposed surface)")
    fig.tight_layout()
    fig.savefig(out / "cross_sections.png", dpi=110)


if __name__ == "__main__":
    main()
