# Fragment-grounded 3D ink labels (Vesuvius Challenge, villa #192)

**Author:** Uri Kartoun ([github.com/kartoun](https://github.com/kartoun)) · **Status:** work in progress (October 2026). The first result is negative; see Findings. · **License:** MIT

This repo tests whether the IR-imaged **fragments** can serve as an independent reference for *where in depth* ink sits, the open question in [ScrollPrize/villa#192](https://github.com/ScrollPrize/villa/issues/192) ("Accurate 3d ink labels").

On scroll segments, every existing 3D ink label depends on an ink model or on a human reading of one. On a fragment we know where the ink is from the infrared photo of the exposed writing, and roughly where it sits in depth, because the writing is on the exposed surface. Neither needs a model.

> Not an official Vesuvius Challenge result. No text is read or claimed here.

## Related work (please read these first)

- **khj1222**, [vesuvius-challenge](https://github.com/khj1222/vesuvius-challenge) and the #192 thread. khj1222 has since checked our port of their recipe, updated their docs/12 with these numbers, and scored their own model with `score_3d.py` (see 9). They trained on plane, constant-band and per-pixel measured-band 3D labels on PHercParis4 segments `w00` and `w02`. The measured band lost to the constant band on both. They also found that villa's flat mode max-pools label depth before the loss.
- **stantheman0128**: [ink3d depth validation](https://github.com/stantheman0128/vesuvius-ink3d-depth-validation) and [villa#1945](https://github.com/ScrollPrize/villa/pull/1945). Scores depths against an independent 1.129 µm rescan. This checks geometry, not ink identity.
- [villa#923](https://github.com/ScrollPrize/villa/pull/923): CT-gated labels with a surface window. [villa#1295](https://github.com/ScrollPrize/villa/pull/1295): closed because brightness-placed depths weren't validated against an independent 3D reference. [villa#1714](https://github.com/ScrollPrize/villa/pull/1714): a tool for painting ink layer by layer.
- Team caveat (pmh47, #192): intensity edges aren't always the recto surface, and ink can look "peeled away".

## Data

Fragment 1, PHercParis2Fr47, 54 keV exposed-surface volume: 65 layers of 8181 × 6330, plus `ir.png`, `inklabels.png` and `mask.png`. It comes from the EduceLab-Scrolls dataset via the Vesuvius Challenge open data (CC BY-NC 4.0). The data is not redistributed here; `download_fragments.py` fetches it (~6.4 GB for Frag1; ~47 GB for Frag1–6).

> Data used in this work were obtained from the EduceLab-Scrolls dataset: Parsons, S., Parker, C. S., Chapman, C., Hayashida, M., & Seales, W. B. (2023). *EduceLab-Scrolls: Verifiable Recovery of Text from Herculaneum Papyri using X-ray CT.* arXiv. https://doi.org/10.48550/arXiv.2304.02084

## Pipeline

| Script | What it does |
|---|---|
| `download_fragments.py` | Fetch the surface volume, IR photo, ink labels and mask for Frag1–Frag6 |
| `step1_ink_depth_profile.py` | Ink vs clear-papyrus contrast per fixed layer |
| `step2_surface_aligned_profile.py` | The same, aligned to each pixel's papyrus→air transition |
| `step3_build_labels.py` | Surface map plus two 3D label volumes (Zarr; 0 = not ink, 1 = ink, 2 = ignore): **flat** (2D outline through all layers) and **surface band** (ink only from −4 to +5 layers around the surface). Both ignore the same voxels, so depth is the only difference. |
| `step4_train_compare.py` | Same small 3D U-Net trained on each label set; scored on a held-out band of rows |
| `khj_recipes.py` | khj1222's v2/v3/v4 label recipes rebuilt on a fragment, for scoring |
| `predict_fragment.py` | Run a trained model over a whole fragment (cross-fragment test) |
| `step5_lofo.py` | Leave-one-fragment-out baseline: train on five fragments, predict and score the sixth (resumable, ~7 h for 12 runs) |

Run order: `python download_fragments.py Frag1 [Frag2 ...]`, then steps 1–4. Set `DATA_ROOT` in `common.py` to your data folder; `step3_build_labels.py --frag FragN` builds the surface map and labels for any fragment. Step 4 needs a CUDA GPU; it took about 27 min per run on a Quadro RTX 4000 (8 GB).

## Findings so far

1. **Fixed layers:** single-voxel separability of ink vs clear papyrus is weak, ROC AUC ≤ 0.56 at any layer ([figure](outputs/step1/layer_profile.png)). That's close to khj1222's 0.546 on w00.
2. **Aligned to the surface:**
   - The contrast concentrates within about −4…+5 layers of the papyrus→air transition. Ink columns are brighter 1–3 layers inside it (Cohen's d +0.27) and darker 2–5 layers outside it (d −0.42).
   - Per-column normalisation removes most of it, so much of the contrast looks like surface geometry rather than a separable ink layer ([figure](outputs/step2/aligned_profile.png)).
   - Brightness alone can't label ink depth.
3. **Surface detection** is reliable on ~61% of the fragment (68% of ink columns) with the simple steepest-drop rule ([figure](outputs/step3/surface_map.png), [cross-sections](outputs/step3/cross_sections.png)).
4. **Training comparison** (same 3D U-Net, 4,000 steps, 2 seeds per arm). Held-out rows 3298–4432 never seen in training, with a 96-row buffer. The 2D map is the max over the supervised depth window, scored against the IR-based outline with stroke edges ignored ([summary.csv](outputs/step4/summary.csv), [depth profiles](outputs/step4/depth_profiles.png)):

   | labels | ROC AUC | avg. precision | best F0.5 | Spearman vs IR darkness | ink probability inside surface band |
   |---|---|---|---|---|---|
   | flat, mean of 2 seeds | **0.750** | **0.486** | **0.507** | **0.299** | 40% |
   | surface band, mean of 2 seeds | 0.725 | 0.475 | 0.484 | 0.283 | 92% |

   - The surface-band labels did **not** improve 2D ink detection; they were slightly worse on every metric in both seeds, though within noise (see 5). That agrees with khj1222's result on scroll segments, now on IR-grounded fragment labels.
   - They did confine predictions to the surface band. But the depth profiles show the band model gives high probability inside the band to **non-ink columns as well** (≈0.52 vs ≈0.63 for ink columns). What it mainly learned is *where the surface is*. That is the failure mode #192 warns about, so a surface band built from an intensity rule doesn't give "ink-only" 3D labels.
   - **Caveats:** the model is small and short-trained (AUC 0.75; khj1222's models are much stronger). There is one held-out band, two seeds, and one fragment.

5. **Scoring with confidence intervals** (`score_3d.py`, held-out rows, 95% CIs from a block bootstrap over 256 × 256 tiles):
   - The 2D gap between flat and surface-band models is **within noise**: AUC 0.75 [0.69, 0.81] for flat vs 0.72–0.73 [0.67, 0.78] for the band.
   - Both kinds of model give clear-papyrus columns about 81–83% of the in-band signal that ink columns get (*surface share*).
   - So the band labels changed *where* the model puts its output (99.8% of ink-column peaks within ±5 layers of the surface, vs 15% for flat), but not *how ink-specific* it is.
   - Full table: [outputs/scores/summary.md](outputs/scores/summary.md).

6. **khj1222's label recipes on a fragment** (`khj_recipes.py`). This is a port of the label construction in [khj1222/vesuvius-challenge](https://github.com/khj1222/vesuvius-challenge) (MIT): plane v2, constant band v3, and per-cell measured band v4 from occlusion profiling with the centroid estimator and their default thresholds.
   - **Differences from the original:** the occlusion model is our step-4 flat-label model, not their checkpoint; blanked slabs are set to the global mean; there are no annotated regions on a fragment, so the fallback is the fragment median.
   - **The port reproduces their headline geometry:** median band centre z 32.6, half-width 4.0 (theirs on w00: 32.5 and 4.0), with 68% of ink in confidently measured cells (theirs: 86%).
   - **Scored against the exposed surface** ([numbers](outputs/khj_recipes/depth_vs_surface.txt), [cross-section](outputs/khj_recipes/cross_sections.png)):

     | ink columns, reliable surface | band centre − surface: median | spread (MAD) | within ±3 layers |
     |---|---|---|---|
     | v3 constant | +3.6 | 2.0 | 46% |
     | v4 measured | +3.6 | 2.0 | 42% |

     corr(v4 centre, surface) = 0.16 over the whole fragment and 0.40 on the held-out rows.
   - **Reading:**
     - On this fragment the measured band moves per cell, but its movement tracks the independently located surface **no better than a constant band does**. That is one candidate explanation for why v4 lost to v3 in khj1222's training runs.
     - Both bands sit about 3.6 layers (~12 µm) on the air side of the detected surface. It could be ink on top of the fibres, a surface detector biased inward, or a model keying on the edge itself; we can't yet tell these apart.
   - **Limits:** our model rather than theirs, and one fragment.

7. **Cross-fragment test** (`predict_fragment.py`). The Frag1-trained models (seed 0) were applied to Frag2–Frag6, which they never saw, using each fragment's own normalisation, and scored on the whole fragment. 95% CIs are in [outputs/scores/summary.md](outputs/scores/summary.md).

   | fragment | AUC flat / band | peak within ±5 of surface, flat / band | surface share, flat / band |
   |---|---|---|---|
   | Frag1 (held-out rows) | 0.75 / 0.73 | 15% / 99.8% | 0.81 / 0.82 |
   | Frag2 | 0.59 / 0.61 | 35% / 97% | 0.89 / 0.84 |
   | Frag3 | 0.62 / 0.62 | 20% / 99.7% | 0.89 / 0.90 |
   | Frag4 (no aligned IR) | 0.59 / 0.58 | 13% / 99.7% | 0.96 / 0.97 |
   | Frag5 | 0.72 / 0.71 | 18% / 99.4% | 0.81 / 0.71 |
   | Frag6 | 0.71 / 0.68 | 18% / 99.4% | 0.85 / 0.86 |

   - A model trained on one fragment transfers poorly (AUC 0.58–0.72). Inside the surface band its output on unseen fragments is mostly surface (surface share 0.81–0.97). That is the risk #192 describes, measured on five independent fragments.
   - Surface-band labels reliably move predictions to the surface but don't make them consistently more ink-specific. Frag5 and Frag2 hint in that direction, with overlapping or barely separated CIs.
   - **Caveat:** one small, briefly trained model from a single fragment. A train-on-five / test-on-the-sixth round is the fair baseline and is next.

8. **Leave-one-fragment-out baseline** (`step5_lofo.py`). For each fragment, the same 3D U-Net (4,000 steps, seed 0) is trained on the other five and scored on the whole held-out fragment, once with flat labels and once with surface-band labels.

   | held-out fragment | AUC flat / band | surface share flat / band | peak within ±5 flat / band |
   |---|---|---|---|
   | Frag1 | 0.654 / 0.656 | 0.870 / 0.825 | 44% / 99% |
   | Frag2 | 0.577 / 0.604 | 0.934 [0.92, 0.95] / 0.866 [0.84, 0.90] | 35% / 97% |
   | Frag3 | 0.609 / 0.620 | 0.935 / 0.928 | 29% / 100% |
   | Frag4 | 0.566 / 0.577 | 0.972 / 0.969 | 17% / 100% |
   | Frag5 | 0.652 / 0.669 | 0.896 / 0.830 | 38% / 100% |
   | Frag6 | 0.689 / 0.658 | 0.888 / 0.887 | 46% / 99% |
   | **mean** | **0.625 / 0.631** | **0.916 / 0.884** | |

   - **Weak baseline.** Training on five fragments did not beat the Frag1-only models on unseen fragments. With this small, short-trained model, unseen fragments are read mostly as surface: surface share 0.83–0.97.
   - **Surface-band vs flat labels.** The band labels give a lower surface share in 4 of 6 folds (equal in 2) and a higher AUC in 5 of 6. The difference is outside the 95% CIs only on Frag2, and 5 of 6 in one direction is not significant on its own (two-sided sign test p = 0.22).
   - **Reading:** a small, consistent tilt toward more ink-specific predictions, not an established effect. This is the opposite sign to the single-fragment result in (4)–(5); only more folds, seeds or stronger models can settle it.
   - **Use as a baseline:** these 12 rows are the reference other methods can be compared against on the same folds.

9. **External entry: a scroll-trained model on Frag1** (reported by [khj1222](https://github.com/khj1222), included with permission). This is the `w00` occluder from [khj1222/vesuvius-challenge docs/11](https://github.com/khj1222/vesuvius-challenge/blob/main/docs/11_measured_3d_labels.md), trained on one PHerc. Paris 4 segment at 7.91 µm. khj1222 ran it themselves and scored it with `score_3d.py` (a) on the whole fragment. The checkpoint is not published; the source is their comments ([result](https://github.com/ScrollPrize/villa/issues/192#issuecomment-6037820818), [entry details](https://github.com/ScrollPrize/villa/issues/192#issuecomment-6075223909)).

   | variant (fixed before running) | (a) ROC AUC on Frag1 [95% CI] |
   |---|---|
   | native 3.24 µm | 0.521 [0.494, 0.556] |
   | resampled to 7.91 µm | 0.565 [0.523, 0.610] |
   | *post hoc:* layer order reversed, native / resampled | 0.578 / 0.553 |

   - **Context:** the same inference code gives AUC 0.95 on held-out regions of the segment it was trained on, so the low score comes from the model, not the pipeline. For comparison, the leave-one-fragment-out model trained on Frag2–6 scores 0.654 on Frag1.
   - **Reading:** a strong segment-trained model does not transfer to an IR-imaged fragment without adaptation.
   - **Consequence for this benchmark:** fragments are useful as a depth reference only with models that actually read ink on them, which today means fragment-trained or adapted ones.

## Scoring tool: `score_3d.py`

It scores any 3D ink prediction or 3D label volume in a fragment's surface-volume coordinates (`.zarr`, `.npy`, or a folder of per-layer `.tif`) against references that don't come from a model:

| Score | Question it answers | Reference |
|---|---|---|
| **(a) ink map** | Does the volume, flattened by max over depth, find the ink? Reports ROC AUC, average precision, best F0.5 and Spearman vs IR darkness. | IR-traced ink outline (stroke edges ignored); `ir.png` |
| **(b) depth** | Where in depth does it put ink? Reports the peak and centroid offset from the exposed surface, and the share of ink columns peaking within ±3 / ±5 layers. | Exposed-surface map (`surface.tif`, reliable pixels only) |
| **(c) ink vs surface** | Inside the surface band (−4…+5), is the signal ink-specific or does it just mark the surface? *Surface share* = clear-papyrus mean / ink mean (0 = ink only, 1 = surface only); *band AUC* separates ink columns from clear ones by their in-band mean. | IR-traced ink core vs clear papyrus ≥ 20 px from any stroke |

```bash
python score_3d.py my_prediction.zarr --z-offset 12 --rows 3298:4432 --name my_model
python score_3d.py labels.zarr --label-value 1 --name my_labels   # label volume: value 1 = ink
python summarize_scores.py
```

`export_predictions.py` writes the step-4 models' 3D predictions in this format. Score (c) is the one that needs a ground-truth ink map, which is why fragments are useful here: on scroll segments there's no model-independent way to say which columns are clear papyrus.

## Limits

- One fragment so far.
- The 2D labels are hand-traced, though from the IR photo rather than from model output.
- Fragments differ from scroll segments: flatter sheets and exposed writing.
- The surface is found with a simple intensity rule, which is exactly what pmh47 cautions about. "Peeled away" ink would appear as a depth offset; it should be measured, not assumed away.

## AI disclosure

Code and analyses were produced with an LLM assistant, Claude Code (model `claude-opus-5-5`), running on the author's machine under the author's direction. The author reviewed the code and the reported numbers.
