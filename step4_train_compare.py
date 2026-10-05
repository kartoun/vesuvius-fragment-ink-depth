"""Step 4: Does training on surface-shell 3D labels beat flat (copied-through) labels?

Trains the same small 3D U-Net on Frag1 with either label set and evaluates both on a
held-out band of rows (never seen in training, with a buffer around it).

Evaluation on the held-out band:
  - 2D ink map = max of predicted voxel probability over depth; scored against the 2D
    hand-traced ink outline (stroke-edge band ignored): ROC AUC, average precision, best F0.5.
  - Independent check: Spearman correlation of the 2D ink map with infrared darkness.
  - Depth profile: mean predicted probability by layer offset from the detected surface,
    in ink and non-ink columns (does the model put ink where the ink is?).

Usage:
  python step4_train_compare.py --labels shell --seed 0
  python step4_train_compare.py --labels flat  --seed 0
  python step4_train_compare.py --summarize
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from common import DATA, IGNORE, INK, LABELS_OUT, OUT, load_png_mask

Z0, Z1 = 12, 44            # input depth window (32 layers)
PATCH = (32, 64, 64)       # z, y, x
HOLDOUT_FRAC = 0.2         # share of ink rows held out
BUFFER = 96                # px rows excluded around the held-out band
EDGE_BAND = 3
OFFSETS = np.arange(-12, 13)


# ---------------------------------------------------------------- data
def load_inputs():
    paths = sorted((DATA / "surface_volume").glob("*.tif"))[Z0:Z1]
    vol = np.stack([tifffile.imread(p) for p in paths])
    frag = load_png_mask("mask.png")
    s = vol[:, frag][:, ::997].astype(np.float32)
    return vol, frag, float(s.mean()), float(s.std())


def split_rows(ink2d):
    """Held-out band = rows holding the middle HOLDOUT_FRAC of ink pixels."""
    cum = np.cumsum(ink2d.sum(axis=1)) / ink2d.sum()
    r0 = int(np.searchsorted(cum, 0.5 - HOLDOUT_FRAC / 2))
    r1 = int(np.searchsorted(cum, 0.5 + HOLDOUT_FRAC / 2))
    return r0, r1


# ---------------------------------------------------------------- model
def block(i, o):
    return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.InstanceNorm3d(o), nn.LeakyReLU(0.1, True),
                         nn.Conv3d(o, o, 3, padding=1), nn.InstanceNorm3d(o), nn.LeakyReLU(0.1, True))


class UNet3D(nn.Module):
    def __init__(self, c=16):
        super().__init__()
        self.e1, self.e2, self.e3 = block(1, c), block(c, 2 * c), block(2 * c, 4 * c)
        self.d2, self.d1 = block(6 * c, 2 * c), block(3 * c, c)
        self.head = nn.Conv3d(c, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool3d(e1, 2))
        e3 = self.e3(F.max_pool3d(e2, 2))
        d2 = self.d2(torch.cat([F.interpolate(e3, scale_factor=2), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2), e1], 1))
        return self.head(d1)


# ---------------------------------------------------------------- train
def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda"
    out = OUT / "step4" / f"{args.labels}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    vol, frag, mu, sd = load_inputs()
    lab = zarr.open_array(str(LABELS_OUT / f"labels_{args.labels}.zarr"), mode="r")[Z0:Z1]
    ink2d = load_png_mask("inklabels.png") & frag
    r0, r1 = split_rows(ink2d)
    H, W = frag.shape
    train_rows = np.ones(H, bool)
    train_rows[max(0, r0 - BUFFER):r1 + BUFFER] = False
    print(f"held-out rows {r0}-{r1}; normalisation mean {mu:.0f} sd {sd:.0f}")

    # patch centres: half on ink columns, half anywhere on the fragment (training rows only)
    pz, py, px = PATCH
    ok = frag.copy()
    ok[:, :px // 2] = ok[:, W - px // 2:] = False
    ok[:py // 2] = ok[H - py // 2:] = False
    ok &= train_rows[:, None]
    cand_any = np.flatnonzero(ok)
    reliable = tifffile.imread(LABELS_OUT / "surface_reliable.tif") > 0
    cand_ink = np.flatnonzero(ok & ink2d & reliable)

    model = UNet3D().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=args.steps)
    scaler = torch.amp.GradScaler()
    t0, log = time.time(), []
    for step in range(args.steps):
        pool = np.where(rng.random(args.batch) < 0.5, 1, 0)
        centres = [rng.choice(cand_ink if p else cand_any) for p in pool]
        xs, ys = [], []
        for c in centres:
            y, x = divmod(int(c), W)
            sy, sx = slice(y - py // 2, y + py // 2), slice(x - px // 2, x + px // 2)
            xb, yb = vol[:, sy, sx], lab[:, sy, sx]
            k = rng.integers(4)
            xb, yb = np.rot90(xb, k, (1, 2)), np.rot90(yb, k, (1, 2))
            if rng.random() < 0.5:
                xb, yb = xb[:, :, ::-1], yb[:, :, ::-1]
            xs.append(xb.copy())
            ys.append(yb.copy())
        x = (torch.from_numpy(np.stack(xs)[:, None].astype(np.float32)).to(dev) - mu) / sd
        y = torch.from_numpy(np.stack(ys)[:, None]).to(dev)
        valid = (y != IGNORE).float()
        with torch.autocast("cuda", dtype=torch.float16):
            logit = model(x)
        loss = (F.binary_cross_entropy_with_logits(logit.float(), (y == INK).float(), reduction="none")
                * valid).sum() / valid.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        log.append(loss.item())
        if (step + 1) % 250 == 0:
            print(f"step {step + 1}/{args.steps}  loss {np.mean(log[-250:]):.4f}  {time.time() - t0:.0f}s", flush=True)
    torch.save(model.state_dict(), out / "model.pt")

    evaluate(model, vol, frag, ink2d, mu, sd, r0, r1, out, args, train_secs=time.time() - t0, loss_log=log)


# ---------------------------------------------------------------- evaluate
@torch.no_grad()
def evaluate(model, vol, frag, ink2d, mu, sd, r0, r1, out, args, train_secs, loss_log):
    model.eval()
    dev = "cuda"
    H, W = frag.shape
    T, S = 256, 192  # tile, stride
    pr0, pr1 = max(0, r0 - 32), min(H, r1 + 32)
    acc = np.zeros((Z1 - Z0, pr1 - pr0, W), np.float32)
    cnt = np.zeros((pr1 - pr0, W), np.float32)
    ys = list(range(pr0, max(pr0 + 1, pr1 - T + 1), S)) + [max(pr0, pr1 - T)]
    xs_ = list(range(0, W - T + 1, S)) + [W - T]
    for y in sorted(set(ys)):
        for x in sorted(set(xs_)):
            if not frag[y:y + T, x:x + T].any():
                continue
            xb = torch.from_numpy(vol[:, y:y + T, x:x + T].astype(np.float32))[None, None].to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                p = torch.sigmoid(model((xb - mu) / sd).float())[0, 0].cpu().numpy()
            acc[:, y - pr0:y - pr0 + T, x:x + T] += p
            cnt[y - pr0:y - pr0 + T, x:x + T] += 1
    prob = acc / np.maximum(cnt, 1)[None]
    prob = prob[:, r0 - pr0:r1 - pr0]
    score2d = prob.max(axis=0)

    f, ink = frag[r0:r1], ink2d[r0:r1]
    edge = ndimage.binary_dilation(ink2d, iterations=EDGE_BAND) & ~ndimage.binary_erosion(ink2d, iterations=EDGE_BAND)
    m = f & ~edge[r0:r1]
    yt, ys_ = ink[m].astype(np.uint8), score2d[m]
    pr, rc, _ = precision_recall_curve(yt, ys_)
    f05 = (1.25 * pr * rc / np.maximum(0.25 * pr + rc, 1e-9)).max()
    ir = np.array(Image.open(DATA / "ir.png"), dtype=np.float32)[r0:r1]
    sub = np.flatnonzero(f.ravel())[::50]
    rho = spearmanr(score2d.ravel()[sub], -ir.ravel()[sub]).statistic

    # depth profile relative to surface
    surf = tifffile.imread(LABELS_OUT / "surface.tif")[r0:r1].astype(int) - Z0
    rel = tifffile.imread(LABELS_OUT / "surface_reliable.tif")[r0:r1] > 0
    prof = {}
    for name, cols in (("ink", m & ink & rel), ("not_ink", m & ~ink & rel)):
        idx = np.flatnonzero(cols.ravel())[::20]
        zz = surf.ravel()[idx][:, None] + OFFSETS[None]
        okz = (zz >= 0) & (zz < prob.shape[0])
        vals = prob.reshape(prob.shape[0], -1)[np.clip(zz, 0, prob.shape[0] - 1), idx[:, None]]
        prof[name] = [float(vals[okz[:, j], j].mean()) for j in range(len(OFFSETS))]
    in_shell = (OFFSETS >= -4) & (OFFSETS <= 5)
    ink_p = np.array(prof["ink"])
    shell_share = float(ink_p[in_shell].sum() / ink_p.sum())

    res = dict(labels=args.labels, seed=args.seed, steps=args.steps, holdout_rows=[r0, r1],
               auc=float(roc_auc_score(yt, ys_)), ap=float(average_precision_score(yt, ys_)),
               best_f05=float(f05), spearman_ir_darkness=float(rho),
               ink_prob_share_in_surface_band=shell_share,
               depth_profile=dict(offsets=OFFSETS.tolist(), **prof),
               final_loss=float(np.mean(loss_log[-250:])), train_seconds=round(train_secs))
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    Image.fromarray((score2d * 255).astype(np.uint8)).save(out / "holdout_ink_map.png")
    print(json.dumps({k: v for k, v in res.items() if k != "depth_profile"}, indent=1))


# ---------------------------------------------------------------- summary
def summarize():
    runs = [json.loads(p.read_text()) for p in sorted((OUT / "step4").glob("*/metrics.json"))]
    keys = ["auc", "ap", "best_f05", "spearman_ir_darkness", "ink_prob_share_in_surface_band"]
    lines = ["labels,seed," + ",".join(keys)]
    for r in runs:
        lines.append(f"{r['labels']},{r['seed']}," + ",".join(f"{r[k]:.4f}" for k in keys))
    for lbl in ("flat", "shell"):
        rs = [r for r in runs if r["labels"] == lbl]
        if rs:
            lines.append(f"{lbl},mean," + ",".join(f"{np.mean([r[k] for r in rs]):.4f}" for k in keys))
    txt = "\n".join(lines)
    print(txt)
    (OUT / "step4" / "summary.csv").write_text(txt + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for a, lbl in zip(ax, ("flat", "shell")):
        for r in (r for r in runs if r["labels"] == lbl):
            d = r["depth_profile"]
            a.plot(d["offsets"], d["ink"], color="C3", alpha=0.8, label=f"ink cols, seed {r['seed']}")
            a.plot(d["offsets"], d["not_ink"], color="C0", alpha=0.8, label=f"non-ink cols, seed {r['seed']}")
        a.axvspan(-4, 5, color="gray", alpha=0.15)
        a.set(title=f"Model trained on {lbl} labels", xlabel="layers from surface (+ = toward air)")
        a.legend(fontsize=7)
    ax[0].set_ylabel("mean predicted ink probability (held-out)")
    fig.suptitle("Where in depth does each model put ink? (shaded = surface band)")
    fig.tight_layout()
    fig.savefig(OUT / "step4" / "depth_profiles.png", dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", choices=["flat", "shell"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    summarize() if a.summarize else train(a)
