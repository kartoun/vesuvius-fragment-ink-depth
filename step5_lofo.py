"""Step 5: leave-one-fragment-out baseline for the benchmark.

For each fragment F in Frag1-Frag6 and each label type (flat, shell): train the step-4
3D U-Net on the other five fragments, predict all of F, and score it with score_3d.py.
This replaces the Frag1-only models as the benchmark's baseline.

Memory and speed: the D: drive is too slow for random patch reads, so layers 12-43 of
every fragment are loaded once into RAM as uint8 (per-fragment 0.5-99.5 percentile
scaling; ~12 GB for all six). Labels are built per patch from the 2D outline, the
surface map and its reliability, with exactly the rules of step3_build_labels.py.

Resumable: a fold/arm whose score file already exists is skipped, so the script can be
stopped and restarted. Expect ~35 min per fold/arm on a Quadro RTX 4000 (12 in total).

Outputs: D:/DBBun/data/vesuvius/<Frag>_predictions/lofo_<arm>.zarr,
         outputs/scores/<Frag>_lofo_<arm>.json, outputs/step5/<Frag>_<arm>/model.pt

Usage:  python step5_lofo.py [--arms flat shell] [--folds Frag1 ...] [--steps 4000]
"""
import argparse
import subprocess
import sys
import time

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
import zarr
from scipy import ndimage

from common import DATA_ROOT, IGNORE, INK, NOT_INK, OUT, frag_paths, load_png_mask
from step4_train_compare import UNet3D, Z0, Z1

FRAGS = ["Frag1", "Frag2", "Frag3", "Frag4", "Frag5", "Frag6"]
SHELL = (-4, 5)
EDGE_BAND = 3
PATCH = (Z1 - Z0, 64, 64)
T = 256
PRED_BATCH = 3


class Fragment:
    def __init__(self, name):
        self.name = name
        data, ref = frag_paths(name)
        frag = load_png_mask("mask.png", data)
        ink = load_png_mask("inklabels.png", data) & frag
        reliable = tifffile.imread(ref / "surface_reliable.tif") > 0
        self.surf = tifffile.imread(ref / "surface.tif").astype(np.int16) - Z0  # in window coordinates
        edge = ndimage.binary_dilation(ink, iterations=EDGE_BAND) & ~ndimage.binary_erosion(ink, iterations=EDGE_BAND)
        lab = np.where(ink, INK, NOT_INK).astype(np.uint8)
        lab[~frag | edge | (ink & ~reliable)] = IGNORE          # same rules as step 3
        self.lab2d, self.frag, self.ink = lab, frag, ink
        self.H, self.W = frag.shape

        paths = sorted((data / "surface_volume").glob("*.tif"))[Z0:Z1]
        layers = [tifffile.memmap(p, mode="r") for p in paths]
        sample = np.stack([m[frag][::997] for m in layers]).astype(np.float32)
        lo, hi = np.percentile(sample, [0.5, 99.5])
        self.vol = np.empty((len(layers), self.H, self.W), np.uint8)
        for i, m in enumerate(layers):
            for r in range(0, self.H, 1024):
                self.vol[i, r:r + 1024] = np.clip((m[r:r + 1024].astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255)
            print(f"\r  {name}: layer {i + 1}/{len(layers)}", end="", flush=True)
        print()
        ok = frag.copy()
        ok[:, :32] = ok[:, -32:] = False
        ok[:32] = ok[-32:] = False
        self.cand_any = np.flatnonzero(ok)
        self.cand_ink = np.flatnonzero(ok & ink & reliable)

    def patch(self, c, arm, rng):
        y, x = divmod(int(c), self.W)
        sy, sx = slice(y - 32, y + 32), slice(x - 32, x + 32)
        xb = self.vol[:, sy, sx]
        l2 = self.lab2d[sy, sx]
        if arm == "flat":
            yb = np.broadcast_to(l2, xb.shape)
        else:
            rel = np.arange(PATCH[0])[:, None, None] - self.surf[sy, sx][None]
            band = (rel >= SHELL[0]) & (rel <= SHELL[1])
            yb = np.where(band & (l2 == INK)[None], INK, NOT_INK).astype(np.uint8)
            yb[:, l2 == IGNORE] = IGNORE
        k = rng.integers(4)
        xb, yb = np.rot90(xb, k, (1, 2)), np.rot90(yb, k, (1, 2))
        if rng.random() < 0.5:
            xb, yb = xb[:, :, ::-1], yb[:, :, ::-1]
        return xb.copy(), yb.copy()


def to_input(u8):
    return (u8.astype(np.float32) - 127.5) / 64.0


def train(frags, test, arm, steps, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    train_frags = [f for f in frags if f.name != test]
    model = UNet3D().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    scaler = torch.amp.GradScaler()
    t0, log = time.time(), []
    for step in range(steps):
        xs, ys = [], []
        for _ in range(8):
            f = train_frags[rng.integers(len(train_frags))]          # equal weight per fragment
            c = rng.choice(f.cand_ink if rng.random() < 0.5 else f.cand_any)
            xb, yb = f.patch(c, arm, rng)
            xs.append(xb)
            ys.append(yb)
        x = torch.from_numpy(to_input(np.stack(xs))[:, None]).cuda()
        y = torch.from_numpy(np.stack(ys)[:, None]).cuda()
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
        if (step + 1) % 500 == 0:
            print(f"  [{test} {arm}] step {step + 1}/{steps} loss {np.mean(log[-500:]):.4f} {time.time() - t0:.0f}s", flush=True)
    return model


@torch.no_grad()
def predict(model, f, arm, tag, trained_on):
    model.eval()
    out = DATA_ROOT / f"{f.name}_predictions"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{tag}_{arm}.zarr"
    z = zarr.create_array(store=str(path), shape=(Z1 - Z0, f.H, f.W), chunks=(Z1 - Z0, T, T),
                          dtype="uint8", fill_value=0, overwrite=True)
    z.attrs.update(z_offset=Z0, trained_on=trained_on, arm=arm)
    for y in range(0, f.H, T):
        h = min(T, f.H - y)
        res = np.zeros((Z1 - Z0, h, f.W), np.uint8)
        xs = [x for x in range(0, f.W, T) if f.frag[y:y + h, x:x + T].any()]
        for i in range(0, len(xs), PRED_BATCH):
            group = xs[i:i + PRED_BATCH]
            tiles = np.full((len(group), Z1 - Z0, T, T), 127, np.uint8)
            for k, x in enumerate(group):
                w = min(T, f.W - x)
                tiles[k, :, :h, :w] = f.vol[:, y:y + h, x:x + w]
            xb = torch.from_numpy(to_input(tiles)[:, None]).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                p = torch.sigmoid(model(xb).float())[:, 0].cpu().numpy()
            for k, x in enumerate(group):
                w = min(T, f.W - x)
                res[:, :, x:x + w] = np.rint(p[k, :, :h, :w] * 255).astype(np.uint8)
        z[:, y:y + h] = res
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["flat", "shell"])
    ap.add_argument("--folds", nargs="+", default=FRAGS)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="+", default=None, help="load only these fragments (for quick tests)")
    ap.add_argument("--tag", default="lofo", help="name prefix for outputs (use e.g. test for quick tests)")
    a = ap.parse_args()

    todo = [(f, arm) for f in a.folds for arm in a.arms
            if not (OUT / "scores" / f"{f}_{a.tag}_{arm}.json").exists()]
    if not todo:
        print("all folds already scored")
        return
    print("loading fragments into RAM (uint8, layers %d-%d)" % (Z0, Z1 - 1))
    frags = [Fragment(n) for n in (a.only or FRAGS)]
    by_name = {f.name: f for f in frags}
    for test, arm in todo:
        t0 = time.time()
        print(f"== fold {test}, arm {arm}: train on {[f.name for f in frags if f.name != test]}", flush=True)
        model = train(frags, test, arm, a.steps, a.seed)
        mdir = OUT / "step5" / f"{test}_{a.tag}_{arm}"
        mdir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), mdir / "model.pt")
        path = predict(model, by_name[test], arm, a.tag, [f.name for f in frags if f.name != test])
        data, ref = frag_paths(test)
        subprocess.run([sys.executable, "-u", "score_3d.py", str(path), "--data", str(data), "--ref", str(ref),
                        "--name", f"{test}_{a.tag}_{arm}"], check=True, cwd=str(OUT.parent))
        print(f"== done {test} {arm} in {(time.time() - t0) / 60:.0f} min", flush=True)
        del model
        torch.cuda.empty_cache()
    subprocess.run([sys.executable, "summarize_scores.py"], check=True, cwd=str(OUT.parent))


if __name__ == "__main__":
    main()
