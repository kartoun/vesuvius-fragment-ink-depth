"""Run a trained step-4 model over a whole fragment and save its 3D ink probabilities.

For cross-fragment testing: models trained on Frag1 applied to Frag2-Frag6, which they
never saw. Input normalisation uses the target fragment's own mean and sd (scans differ
in energy and contrast). Tiles of 256 x 256 px without overlap; output is written strip
by strip, so memory stays small even for Frag2 (14830 x 9506).

Output: D:/DBBun/data/vesuvius/<Frag>_predictions/<run>.zarr (uint8 0-255, layers Z0..Z1,
attribute z_offset = Z0).

Usage:  python predict_fragment.py Frag2 [Frag3 ...] [--runs flat_seed0 shell_seed0]
"""
import argparse

import numpy as np
import tifffile
import torch
import zarr

from common import DATA_ROOT, OUT, frag_paths, load_png_mask
from step4_train_compare import UNet3D, Z0, Z1

T = 256
BATCH = 3


def layer_stats(layers, frag):
    s = np.stack([m[frag][::997] for m in layers]).astype(np.float32)
    return float(s.mean()), float(s.std())


@torch.no_grad()
def predict(frag_name, runs):
    data, _ = frag_paths(frag_name)
    frag = load_png_mask("mask.png", data)
    H, W = frag.shape
    paths = sorted((data / "surface_volume").glob("*.tif"))[Z0:Z1]
    layers = [tifffile.memmap(p, mode="r") for p in paths]
    mu, sd = layer_stats(layers, frag)
    models = {}
    for run in runs:
        m = UNet3D().cuda()
        m.load_state_dict(torch.load(OUT / "step4" / run / "model.pt", map_location="cuda"))
        models[run] = m.eval()
    out_dir = DATA_ROOT / f"{frag_name}_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    arrs = {run: zarr.create_array(store=str(out_dir / f"{run}.zarr"), shape=(Z1 - Z0, H, W),
                                   chunks=(Z1 - Z0, T, T), dtype="uint8", fill_value=0, overwrite=True)
            for run in runs}
    for a in arrs.values():
        a.attrs.update(z_offset=Z0, norm_mean=mu, norm_sd=sd)

    for y in range(0, H, T):
        h = min(T, H - y)
        strip = np.stack([m[y:y + h] for m in layers]).astype(np.float32)  # (Z, h, W)
        res = {run: np.zeros((Z1 - Z0, h, W), np.uint8) for run in runs}
        xs = [x for x in range(0, W, T) if frag[y:y + h, x:x + T].any()]
        for i in range(0, len(xs), BATCH):
            group = xs[i:i + BATCH]
            tiles = np.zeros((len(group), Z1 - Z0, T, T), np.float32)
            for k, x in enumerate(group):
                w = min(T, W - x)
                tiles[k, :, :h, :w] = (strip[:, :, x:x + w] - mu) / sd
            xb = torch.from_numpy(tiles)[:, None].cuda()
            for run, m in models.items():
                with torch.autocast("cuda", dtype=torch.float16):
                    p = torch.sigmoid(m(xb).float())[:, 0].cpu().numpy()
                for k, x in enumerate(group):
                    w = min(T, W - x)
                    res[run][:, :, x:x + w] = np.rint(p[k, :, :h, :w] * 255).astype(np.uint8)
        for run in runs:
            arrs[run][:, y:y + h] = res[run]
        print(f"\r{frag_name}: rows {y + h}/{H}", end="", flush=True)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fragments", nargs="+")
    ap.add_argument("--runs", nargs="+", default=["flat_seed0", "shell_seed0"])
    a = ap.parse_args()
    for f in a.fragments:
        predict(f, a.runs)


if __name__ == "__main__":
    main()
