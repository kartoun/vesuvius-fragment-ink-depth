# Fragment-grounded 3D ink labels (Vesuvius Challenge, villa #192)

**Author:** Uri Kartoun ([github.com/kartoun](https://github.com/kartoun)) · **Status:** work in progress (October 2026) · **License:** MIT

This repo tests whether the IR-imaged **fragments** can serve as an independent reference for *where in depth* ink sits, the open question in [ScrollPrize/villa#192](https://github.com/ScrollPrize/villa/issues/192) ("Accurate 3d ink labels").

On scroll segments, every existing 3D ink label depends on an ink model or on a human reading of one. On a fragment we know where the ink is from the infrared photo of the exposed writing, and roughly where it sits in depth, because the writing is on the exposed surface. Neither needs a model.

> Not an official Vesuvius Challenge result. No text is read or claimed here.

## Related work (please read these first)

- **khj1222**, [vesuvius-challenge](https://github.com/khj1222/vesuvius-challenge) and the #192 thread. They trained on plane, constant-band and per-pixel measured-band 3D labels on PHercParis4 segments `w00` and `w02`. The measured band lost to the constant band on both. They also found that villa's flat mode max-pools label depth before the loss.
- **stantheman0128**: [ink3d depth validation](https://github.com/stantheman0128/vesuvius-ink3d-depth-validation) and [villa#1945](https://github.com/ScrollPrize/villa/pull/1945). Scores depths against an independent 1.129 µm rescan. This checks geometry, not ink identity.
- [villa#923](https://github.com/ScrollPrize/villa/pull/923): CT-gated labels with a surface window. [villa#1295](https://github.com/ScrollPrize/villa/pull/1295): closed because brightness-placed depths weren't validated against an independent 3D reference. [villa#1714](https://github.com/ScrollPrize/villa/pull/1714): a tool for painting ink layer by layer.
- Team caveat (pmh47, #192): intensity edges aren't always the recto surface, and ink can look "peeled away".

## Data

Fragment 1, PHercParis2Fr47, 54 keV exposed-surface volume: 65 layers of 8181 × 6330, plus `ir.png`, `inklabels.png` and `mask.png`. It comes from the EduceLab-Scrolls dataset via the Vesuvius Challenge open data (CC BY-NC 4.0). The data is not redistributed here; `download_frag1.py` fetches it (~6.4 GB).

> Data used in this work were obtained from the EduceLab-Scrolls dataset: Parsons, S., Parker, C. S., Chapman, C., Hayashida, M., & Seales, W. B. (2023). *EduceLab-Scrolls: Verifiable Recovery of Text from Herculaneum Papyri using X-ray CT.* arXiv. https://doi.org/10.48550/arXiv.2304.02084

## Pipeline

| Script | What it does |
|---|---|
| `download_frag1.py` | Fetch the Frag1 surface volume, IR photo, ink labels and mask |
| `step1_ink_depth_profile.py` | Ink vs clear-papyrus contrast per fixed layer |
| `step2_surface_aligned_profile.py` | The same, aligned to each pixel's papyrus→air transition |
| `step3_build_labels.py` | Surface map plus two 3D label volumes (Zarr; 0 = not ink, 1 = ink, 2 = ignore): **flat** (2D outline through all layers) and **surface band** (ink only from −4 to +5 layers around the surface). Both ignore the same voxels, so depth is the only difference. |
| `step4_train_compare.py` | Same small 3D U-Net trained on each label set; scored on a held-out band of rows |

Run order: `python download_frag1.py [out_dir]`, then steps 1–4. Edit `DATA` / `LABELS_OUT` in `common.py` to match your paths. Step 4 needs a CUDA GPU; it took about 27 min per run on a Quadro RTX 4000 (8 GB).

## Findings so far

1. **Fixed layers:** single-voxel separability of ink vs clear papyrus is weak, ROC AUC ≤ 0.56 at any layer ([figure](outputs/step1/layer_profile.png)). That's close to khj1222's 0.546 on w00.
2. **Aligned to the surface:**
   - The contrast concentrates within about −4…+5 layers of the papyrus→air transition. Ink columns are brighter 1–3 layers inside it (Cohen's d +0.27) and darker 2–5 layers outside it (d −0.42).
   - Per-column normalisation removes most of it, so much of the contrast looks like surface geometry rather than a separable ink layer ([figure](outputs/step2/aligned_profile.png)).
   - Brightness alone can't label ink depth.
3. **Surface detection** is reliable on ~61% of the fragment (68% of ink columns) with the simple steepest-drop rule ([figure](outputs/step3/surface_map.png), [cross-sections](outputs/step3/cross_sections.png)).
4. **Training comparison:** in progress, results to follow. In the first seed, the 2D ink-detection scores were similar for both label sets. The surface-band model put about 92% of its ink probability inside the surface band, versus about 40% for the flat model.

## Limits

- One fragment so far.
- The 2D labels are hand-traced, though from the IR photo rather than from model output.
- Fragments differ from scroll segments: flatter sheets and exposed writing.
- The surface is found with a simple intensity rule, which is exactly what pmh47 cautions about. "Peeled away" ink would appear as a depth offset; it should be measured, not assumed away.

## AI disclosure

Code and analyses were produced with an LLM assistant, Claude Code (model `claude-opus-5-5`), running on the author's machine under the author's direction. The author reviewed the code and the reported numbers.
