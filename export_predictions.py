"""Export 3D ink-probability volumes from the step-4 models for the held-out rows.

Writes D:/DBBun/data/vesuvius/Frag1_predictions/<run>.zarr (uint8 0-255, shape
(Z1-Z0, H, W), attribute z_offset = Z0; rows outside the held-out band are 0).

Usage:  python export_predictions.py [run ...]   (default: every run in outputs/step4)
"""
import sys

import numpy as np
import torch
import zarr

from common import LABELS_OUT, OUT, load_png_mask
from step4_train_compare import UNet3D, Z0, Z1, load_inputs, split_rows

PRED_OUT = LABELS_OUT.parent / "Frag1_predictions"
T, S = 256, 192  # tile, stride


@torch.no_grad()
def predict(model, vol, frag, mu, sd, r0, r1):
    H, W = frag.shape
    pr0, pr1 = max(0, r0 - 32), min(H, r1 + 32)
    acc = np.zeros((Z1 - Z0, pr1 - pr0, W), np.float32)
    cnt = np.zeros((pr1 - pr0, W), np.float32)
    ys = sorted(set(list(range(pr0, max(pr0 + 1, pr1 - T + 1), S)) + [max(pr0, pr1 - T)]))
    xs = sorted(set(list(range(0, W - T + 1, S)) + [W - T]))
    for y in ys:
        for x in xs:
            if not frag[y:y + T, x:x + T].any():
                continue
            xb = torch.from_numpy(vol[:, y:y + T, x:x + T].astype(np.float32))[None, None].cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                p = torch.sigmoid(model((xb - mu) / sd).float())[0, 0].cpu().numpy()
            acc[:, y - pr0:y - pr0 + T, x:x + T] += p
            cnt[y - pr0:y - pr0 + T, x:x + T] += 1
    prob = acc / np.maximum(cnt, 1)[None]
    return prob[:, r0 - pr0:r1 - pr0]


def main():
    runs = sys.argv[1:] or sorted(p.parent.name for p in (OUT / "step4").glob("*/model.pt"))
    vol, frag, mu, sd = load_inputs()
    r0, r1 = split_rows(load_png_mask("inklabels.png") & frag)
    H, W = frag.shape
    PRED_OUT.mkdir(parents=True, exist_ok=True)
    for run in runs:
        model = UNet3D().cuda()
        model.load_state_dict(torch.load(OUT / "step4" / run / "model.pt", map_location="cuda"))
        model.eval()
        prob = predict(model, vol, frag, mu, sd, r0, r1)
        z = zarr.create_array(store=str(PRED_OUT / f"{run}.zarr"), shape=(Z1 - Z0, H, W),
                              chunks=(Z1 - Z0, 256, 256), dtype="uint8", fill_value=0, overwrite=True)
        z[:, r0:r1] = np.rint(prob * 255).astype(np.uint8)
        z.attrs.update(z_offset=Z0, rows=[r0, r1], source=f"outputs/step4/{run}/model.pt")
        print(f"{run}: rows {r0}-{r1} -> {PRED_OUT / (run + '.zarr')}")


if __name__ == "__main__":
    main()
