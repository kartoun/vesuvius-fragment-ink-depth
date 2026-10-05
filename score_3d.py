"""Score a 3D ink label or 3D ink prediction against an IR-imaged fragment.

Input: a volume in the fragment's surface-volume coordinates (Z, H, W), as .zarr, .npy
or a folder of per-layer .tif files. Values may be probabilities (float 0-1), uint8
(0-255), uint16, or binary labels. A volume that covers only some layers is placed with
--z-offset (or the zarr attribute "z_offset").

Reference (no model involved):
  - 2D ink map traced from the infrared photo (inklabels.png), fragment mask, ir.png
  - per-pixel exposed-surface layer and its reliability (surface.tif,
    surface_reliable.tif from step3_build_labels.py)

Scores, each with a 95% CI from a block bootstrap over 256x256 tiles:
  (a) ink map:       max over depth -> 2D map; ROC AUC, average precision, best F0.5
                     vs the IR-based outline (stroke edges ignored); Spearman vs IR darkness.
  (b) depth:         per ink column, offset of the prediction's peak (argmax) and
                     probability-weighted centroid from the exposed surface; share within
                     +-3 and +-5 layers.
  (c) ink vs surface: inside the surface band, mean value in ink columns vs clear-papyrus
                     columns. surface_share = clear / ink (1 = marks the surface only,
                     0 = ink only); band_auc = ROC AUC separating ink from clear columns
                     by their in-band mean.

Usage:
  python score_3d.py PRED [--z-offset N] [--rows R0:R1] [--label-value V] [--name NAME] [--out DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import zarr
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from common import DATA, LABELS_OUT, OUT, load_png_mask

Image.MAX_IMAGE_PIXELS = None
EDGE_BAND = 3        # px ignored either side of stroke outlines, for (a)
CLEAR_GAP = 20       # px between strokes and "clear papyrus", for (b)/(c)
CORE_ERODE = 3       # px removed inside strokes for "ink core", for (b)/(c)
BAND = (-4, 5)       # surface band, layers relative to surface (inclusive)
OFFSETS = np.arange(-15, 16)
TILE = 256
PER_TILE = 400       # pixels sampled per tile and class for (a)
N_BOOT = 300
ROWS = 256           # rows read per block


# ---------------------------------------------------------------- input
class Volume:
    """Row-block reader that returns float32 values in [0, 1]."""

    def __init__(self, path, z_offset=None, label_value=None):
        self.label_value = label_value
        path = Path(path)
        if path.suffix == ".zarr":
            self.arr = zarr.open_array(str(path), mode="r")
            attr = self.arr.attrs.get("z_offset", 0)
        elif path.suffix == ".npy":
            self.arr = np.load(path, mmap_mode="r")
            attr = 0
        elif path.is_dir():
            layers = sorted(path.glob("*.tif"))
            self.arr = np.stack([tifffile.imread(p) for p in layers])
            attr = 0
        else:
            raise ValueError(f"unsupported input {path}")
        self.z_offset = attr if z_offset is None else z_offset
        dt = np.dtype(self.arr.dtype)
        self.scale = {np.dtype("uint8"): 255.0, np.dtype("uint16"): 65535.0}.get(dt, 1.0)
        self.shape = self.arr.shape

    def rows(self, r0, r1):
        a = np.asarray(self.arr[:, r0:r1])
        if self.label_value is not None:  # label volume: only this value counts as ink
            return (a == self.label_value).astype(np.float32)
        return a.astype(np.float32) / self.scale


# ---------------------------------------------------------------- helpers
def best_f05(y, s):
    p, r, _ = precision_recall_curve(y, s)
    return float((1.25 * p * r / np.maximum(0.25 * p + r, 1e-9)).max())


def ci(values):
    v = np.asarray([x for x in values if np.isfinite(x)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else [None, None]


def boot(tiles, stat, rng):
    """Resample tiles with replacement and recompute stat on the pooled samples."""
    ids = np.unique(tiles)
    out = []
    for _ in range(N_BOOT):
        pick = rng.choice(ids, ids.size)
        counts = np.bincount(np.searchsorted(ids, pick), minlength=ids.size)
        w = counts[np.searchsorted(ids, tiles)]
        out.append(stat(w))
    return ci(out)


# ---------------------------------------------------------------- main
def score(args):
    rng = np.random.default_rng(0)
    vol = Volume(args.pred, args.z_offset, args.label_value)
    Zp, H, W = vol.shape
    frag = load_png_mask("mask.png", args.data)
    ink = load_png_mask("inklabels.png", args.data) & frag
    ir = np.array(Image.open(args.data / "ir.png"), dtype=np.float32)
    surf = tifffile.imread(args.ref / "surface.tif").astype(np.int32) - vol.z_offset
    reliable = tifffile.imread(args.ref / "surface_reliable.tif") > 0
    assert (H, W) == frag.shape, f"volume is {H}x{W}, fragment is {frag.shape}"

    region = np.zeros_like(frag)
    r0, r1 = args.rows
    region[r0:r1] = True
    region &= frag
    edge = ndimage.binary_dilation(ink, iterations=EDGE_BAND) & ~ndimage.binary_erosion(ink, iterations=EDGE_BAND)
    eval_a = region & ~edge
    core = ndimage.binary_erosion(ink, iterations=CORE_ERODE) & region & reliable
    clear = region & reliable & ~ndimage.binary_dilation(ink, iterations=CLEAR_GAP)
    tile_id = (np.arange(H)[:, None] // TILE) * (W // TILE + 1) + (np.arange(W)[None] // TILE)

    # stream over row blocks
    score2d = np.zeros((H, W), np.float32)
    argmax_off = np.full((H, W), np.nan, np.float32)
    centroid_off = np.full((H, W), np.nan, np.float32)
    prof_sum = {k: np.zeros(len(OFFSETS)) for k in ("ink", "clear")}
    prof_n = {k: np.zeros(len(OFFSETS)) for k in ("ink", "clear")}
    band_mean = np.full((H, W), np.nan, np.float32)
    b_lo, b_hi = BAND
    for a in range(max(r0, 0), min(r1, H), ROWS):
        b = min(a + ROWS, r1, H)
        if not region[a:b].any():
            continue
        v = vol.rows(a, b)                      # (Zp, rows, W)
        score2d[a:b] = v.max(axis=0)
        s = surf[a:b]
        z = np.arange(Zp)[:, None, None]
        rel = z - s[None]
        tot = v.sum(axis=0)
        argmax_off[a:b] = v.argmax(axis=0) - s
        with np.errstate(invalid="ignore", divide="ignore"):
            centroid_off[a:b] = (v * rel).sum(axis=0) / tot
        inb = (rel >= b_lo) & (rel <= b_hi)
        nb = inb.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            band_mean[a:b] = np.where(nb > 0, (v * inb).sum(axis=0) / nb, np.nan)
        for k, m in (("ink", core[a:b]), ("clear", clear[a:b])):
            if not m.any():
                continue
            ys, xs = np.nonzero(m)
            for j, o in enumerate(OFFSETS):
                zz = s[ys, xs] + o
                ok = (zz >= 0) & (zz < Zp)
                prof_sum[k][j] += v[zz[ok], ys[ok], xs[ok]].sum()
                prof_n[k][j] += ok.sum()
        print(f"\rscored rows {b - r0}/{r1 - r0}", end="", flush=True)
    print()

    # (a) ink map: stratified sample per tile and class
    pos, neg = [], []
    for cls, store in ((True, pos), (False, neg)):
        m = eval_a & (ink == cls)
        idx = rng.permutation(np.flatnonzero(m))
        t = tile_id.ravel()[idx]
        # group by tile (random order within tile), keep up to PER_TILE per tile
        srt = np.argsort(t, kind="stable")
        idx, t = idx[srt], t[srt]
        _, first = np.unique(t, return_index=True)
        bounds = list(first[1:]) + [idx.size]
        keep = np.concatenate([np.arange(f, min(f + PER_TILE, e)) for f, e in zip(first, bounds)])
        store.append(idx[keep])
    sel = np.concatenate(pos + neg)
    y = np.r_[np.ones(pos[0].size), np.zeros(neg[0].size)].astype(np.uint8)
    sc = score2d.ravel()[sel]
    dark = -ir.ravel()[sel]
    tiles = tile_id.ravel()[sel]

    def wauc(w):
        return roc_auc_score(y, sc, sample_weight=w) if (w * y).sum() and (w * (1 - y)).sum() else np.nan

    def wap(w):
        return average_precision_score(y, sc, sample_weight=w) if (w * y).sum() else np.nan

    res_a = dict(
        n_pixels=int(sel.size), n_ink=int(y.sum()),
        auc=float(roc_auc_score(y, sc)), auc_ci=boot(tiles, wauc, rng),
        ap=float(average_precision_score(y, sc)), ap_ci=boot(tiles, wap, rng),
        best_f05=best_f05(y, sc),
        spearman_ir_darkness=float(spearmanr(sc, dark).statistic),
    )

    # (b) depth of ink columns
    ci_idx = np.flatnonzero(core.ravel())
    am = argmax_off.ravel()[ci_idx]
    ce = centroid_off.ravel()[ci_idx]
    ct = tile_id.ravel()[ci_idx]
    ok = np.isfinite(ce)
    am, ce, ct = am[ok], ce[ok], ct[ok]

    def wshare(vals, k):
        return lambda w: float((w * (np.abs(vals) <= k)).sum() / max(w.sum(), 1))

    res_b = dict(
        n_columns=int(am.size),
        argmax_offset_median=float(np.median(am)), argmax_offset_iqr=[float(np.percentile(am, 25)), float(np.percentile(am, 75))],
        centroid_offset_median=float(np.median(ce)),
        share_argmax_within_3=float((np.abs(am) <= 3).mean()), share_argmax_within_3_ci=boot(ct, wshare(am, 3), rng),
        share_argmax_within_5=float((np.abs(am) <= 5).mean()), share_argmax_within_5_ci=boot(ct, wshare(am, 5), rng),
    )

    # (c) ink vs surface inside the band
    bi = band_mean.ravel()[np.flatnonzero(core.ravel())]
    bc = band_mean.ravel()[np.flatnonzero(clear.ravel())]
    ti = tile_id.ravel()[np.flatnonzero(core.ravel())]
    tc = tile_id.ravel()[np.flatnonzero(clear.ravel())]
    oki, okc = np.isfinite(bi), np.isfinite(bc)
    bi, ti, bc, tc = bi[oki], ti[oki], bc[okc], tc[okc]
    allt = np.r_[ti, tc]
    lab = np.r_[np.ones(bi.size), np.zeros(bc.size)]
    vals = np.r_[bi, bc]
    sub = rng.choice(vals.size, min(vals.size, 400_000), replace=False)

    def wshare_surface(w):
        wi, wc = w[: bi.size], w[bi.size:]
        mi = (wi * bi).sum() / max(wi.sum(), 1)
        mc = (wc * bc).sum() / max(wc.sum(), 1)
        return mc / mi if mi > 0 else np.nan

    def wband_auc(w):
        ws = w[sub]
        return roc_auc_score(lab[sub], vals[sub], sample_weight=ws) if (ws * lab[sub]).sum() else np.nan

    mi, mc = float(bi.mean()), float(bc.mean())
    res_c = dict(
        band=list(BAND), mean_in_band_ink=mi, mean_in_band_clear=mc,
        surface_share=mc / mi if mi > 0 else None, surface_share_ci=boot(allt, wshare_surface, rng),
        band_auc=float(roc_auc_score(lab[sub], vals[sub])), band_auc_ci=boot(allt, wband_auc, rng),
    )
    prof = {k: (prof_sum[k] / np.maximum(prof_n[k], 1)).tolist() for k in prof_sum}

    out = dict(name=args.name, input=str(args.pred), z_offset=int(vol.z_offset), rows=[r0, r1],
               a_ink_map=res_a, b_depth=res_b, c_ink_vs_surface=res_c,
               depth_profile=dict(offsets=OFFSETS.tolist(), **prof))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.name}.json").write_text(json.dumps(out, indent=1))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(OFFSETS, prof["ink"], color="C3", label="ink columns (IR-based)")
    ax[0].plot(OFFSETS, prof["clear"], color="C0", label="clear papyrus columns")
    ax[0].axvspan(*BAND, color="gray", alpha=0.15)
    ax[0].set(title="Mean value by depth from exposed surface", xlabel="layers from surface (+ = toward air)")
    ax[0].legend(fontsize=8)
    ax[1].hist(am, bins=np.arange(-30.5, 31), color="C3")
    ax[1].axvspan(*BAND, color="gray", alpha=0.15)
    ax[1].set(title="Peak depth in ink columns", xlabel="argmax layer - surface layer")
    fig.suptitle(f"{args.name}: AUC {res_a['auc']:.3f} | surface share {res_c['surface_share']:.2f} | "
                 f"peak within +-5: {res_b['share_argmax_within_5']:.0%}")
    fig.tight_layout()
    fig.savefig(args.out / f"{args.name}.png", dpi=120)
    plt.close(fig)

    print(f"(a) AUC {res_a['auc']:.3f} {fmt(res_a['auc_ci'])}  AP {res_a['ap']:.3f} {fmt(res_a['ap_ci'])}  "
          f"F0.5 {res_a['best_f05']:.3f}  rho(IR) {res_a['spearman_ir_darkness']:.3f}")
    print(f"(b) peak offset median {res_b['argmax_offset_median']:+.1f}  within +-5: "
          f"{res_b['share_argmax_within_5']:.1%} {fmt(res_b['share_argmax_within_5_ci'])}")
    print(f"(c) surface share {res_c['surface_share']:.3f} {fmt(res_c['surface_share_ci'])}  "
          f"band AUC {res_c['band_auc']:.3f} {fmt(res_c['band_auc_ci'])}")
    return out


def fmt(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]" if c[0] is not None else "[n/a]"


def parse_rows(s):
    a, b = s.split(":")
    return int(a), int(b)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred")
    ap.add_argument("--z-offset", type=int, default=None, help="layer index of the volume's first slice")
    ap.add_argument("--label-value", type=int, default=None,
                    help="treat input as a label volume; voxels equal to this value are ink (e.g. 1)")
    ap.add_argument("--rows", type=parse_rows, default=(0, 10**9), help="row range R0:R1 to score")
    ap.add_argument("--name", default=None)
    ap.add_argument("--data", type=Path, default=DATA, help="fragment folder (mask.png, inklabels.png, ir.png)")
    ap.add_argument("--ref", type=Path, default=LABELS_OUT, help="folder with surface.tif, surface_reliable.tif")
    ap.add_argument("--out", type=Path, default=OUT / "scores")
    a = ap.parse_args()
    a.name = a.name or Path(a.pred).stem
    a.rows = (a.rows[0], min(a.rows[1], 10**9))
    score(a)
