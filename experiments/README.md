# Quantitative Studies: Reconstruction + AI Segmentation

This directory upgrades the main application's two major AI/algorithm capabilities from **feature implementations** into **quantitative studies**, producing reproducible, patient-data-free figures and metrics:

- **Study I (reconstruction)** — uses the standard Shepp-Logan phantom to measure how `recon.py` reconstruction quality varies with dose / filter / algorithm.
- **Study II (AI segmentation)** — uses a ground-truth-labelled public CT (TotalSegmentator-CT-Lite) to measure the Dice of `organs.onnx`, and to *recover by measurement* its 25-class label mapping.

> The object under test is the production code itself — the reconstruction experiment directly `import recon`s and calls the same reconstruction functions the GUI uses; the segmentation experiment replicates `ai_engine`'s identical (step-by-step numerically equivalent) preprocessing and sliding-window inference, running the very same `organs.onnx` the GUI uses.

## Running

```bash
python experiments/recon_study.py           # Experiments A + B (fast, pure FBP)
python experiments/recon_study.py c          # Experiment C (needs to build the system matrix; slow the first time, then served from disk cache)
python experiments/recon_study.py a b c      # run all
```

Outputs are written to `experiments/results/`: one PNG figure plus one CSV of raw data per experiment.

> Reproduction dependencies (outside the App): `pip install -r experiments/requirements-experiments.txt` (matplotlib / nibabel / remotezip).

## Methods

- **Phantom**: `skimage` Shepp-Logan, rescaled to the target side length, normalised to `[0,1]`, and masked with a circular mask aligned to `radon(circle=True)`. The masked image is the ground truth (GT) — the sinogram encodes only the information inside the inscribed circle, so comparison is done inside the circle only.
- **Forward model**: `recon.compute_sinogram` (Radon transform, `circle=True`).
- **Dose proxy**: the number of projection angles `n_proj` (angular range fixed at 180°). Fewer projections ≈ lower dose.
- **Noise model (Experiment C)**: Beer–Lambert + Poisson photon statistics. Incident photons `I0`, transmitted counts `N ~ Poisson(I0·e^{-p})`, noisy projection `p' = -ln(N/I0)`. The smaller `I0`, the more noise. Fixed random seed, reproducible.
- **Metrics**: in-circle RMSE, NRMSE, SSIM (`data_range=1`), PSNR.

## Findings

### A — Dose–quality curve (FBP Ram-Lak, 256×256)
`exp_a_dose_quality.png`

RMSE falls monotonically from **0.222** at 15 views to **0.035** at 360 views; SSIM rises from **0.35** to **0.95**.
**Beyond ≈180 views it enters a clear region of diminishing returns** — adding further dose gives only a marginal quality improvement. This provides a quantitative basis for "enough is enough" dose selection.

### B — The optimal filter choice inverts with dose (256×256)
`exp_b_filters.png`

| n_proj | ramp (Ram-Lak) | shepp-logan | cosine | hamming | hann |
|---|---|---|---|---|---|
| 20 (sparse) | 0.176 | 0.167 | 0.154 | 0.145 | **0.143** |
| 180 (dense) | **0.037** | 0.040 | 0.048 | 0.054 | 0.055 |

**Key conclusion: there is no single "best" filter; the optimal choice depends on dose.**
At low dose / sparse angles, apodisation filters (hann/hamming) that suppress high-frequency noise win; as dose increases, the sharp Ram-Lak overtakes them on fidelity. **The crossover is at ≈45–60 views.**

### C — Analytic vs iterative under photon noise (64×64, I0=3×10⁴)
`exp_c_analytic_vs_iterative.png` · `exp_c_gallery.png`

| n_proj | FBP | DMR (least-squares) | ART | SIRT |
|---|---|---|---|---|
| 30 | 0.099 | 0.151 | **0.069** | 0.083 |
| 60 | 0.089 | **0.611** | **0.053** | 0.076 |
| 90 | 0.087 | 0.090 | **0.050** | 0.075 |

1. **ART is best throughout** — the non-negativity constraint plus per-ray (Kaczmarz) updates act as implicit regularisation, making it the most robust under noise.
2. **Naive least-squares (DMR) is unstable under noise**, with RMSE spiking to **0.611** at 60 views: here 64 detectors × 60 views ≈ 3,840 equations ≈ 4,096 unknowns, so the system is nearly square and its **condition number is worst**, amplifying noise most violently; the 30-view case (under-determined, minimum-norm solution) and the 90-view case (over-determined, with an averaging effect) are by contrast more stable. This is the classic behaviour of an ill-posed inverse problem, not an implementation defect.
3. **SIRT is steady and conservative** (~0.075), far more robust than DMR; the price of its smoothing is that it trails ART slightly.

The visual comparison in `exp_c_gallery.png` fully matches the RMSE: DMR is covered in salt-and-pepper noise, FBP shows streak artefacts, ART is the cleanest, SIRT is the smoothest.

## Reconstruction study — one-sentence summary

In low-dose CT, **the answer to "which algorithm/filter" changes with dose**: in the sparse, low-dose regime choose a constrained iterative method (ART) or an apodisation filter; in the ample-dose regime the analytic method (FBP + Ram-Lak) is sufficient and faster.

---

# Study II: Quantitative validation of AI segmentation (`seg_validate.py`)

## Motivation
The 25-class label→organ mapping of `organs.onnx` was long only **inferred** (no official dataset.json). This study objectively measures segmentation quality on a ground-truth-labelled public CT, and **recovers the mapping by measurement via a confusion matrix rather than guessing it**.

## Data (not in the repository — bring your own)
A single case of **TotalSegmentator-CT-Lite** (CC-BY-4.0, 1.5 mm isotropic, thorax–abdomen–pelvis coverage), extracting **only one case** (≈42 MB) from a 22 GB archive using HTTP Range requests:

```python
from remotezip import RemoteZip
base = "https://huggingface.co/datasets/YongchengYAO/TotalSegmentator-CT-Lite/resolve/main"
open("s0029_img.nii.gz","wb").write(RemoteZip(base+"/Images.zip").read("Images/s0029.nii.gz"))
open("s0029_msk.nii.gz","wb").write(RemoteZip(base+"/Masks.zip").read("Masks/s0029.nii.gz"))
```

```bash
python experiments/seg_validate.py s0029_img.nii.gz s0029_msk.nii.gz
```

## Methods
The image is normalised to RAS and then converted to the GUI's (Z,H,W) axis order → the same `ai_engine` preprocessing (clip [-1000,400], normalise) + DZ=32 sliding-window inference along z → per-label Dice/IoU against the ground truth, taking for each output label the ground-truth organ with the greatest overlap (the mapping is not assumed to be the identity but **measured**).

## Findings
`exp` outputs: `seg_confusion.png` (confusion heatmap) · `seg_dice.csv` · `seg_mapping.md`

1. **The mapping is measured to be an identity diagonal**: our#k → the k-th TotalSegmentator organ, matching one by one (this case's ground truth contains no prostate/kidney cyst, so labels 22/23 have no measured Dice; their identity is fixed by this mapping scheme). **This confirms the model = TotalSegmentator v2 `class_map_part_organs`** (24 organs + background, an nnU-Net v2 export), no longer "provenance unknown".
2. **Mean Dice ≈ 0.92 over the 21 organs present** (kidneys 0.98, lung lobes 0.96–0.99, small organs such as thyroid/gallbladder 0.79–0.82), consistent with TotalSegmentator's officially published level — **simultaneously validating that the GUI inference pipeline is correct**.
3. **Corrected historical mislabels**: `5` = **liver** (an earlier inference wrongly took it as "heart"; the model has no heart/aorta output — both live in another TS part, labels 51/52, outside 0–24); lung lobes `10,11` = **left**, `12,13,14` = **right** (the earlier left/right swap was a radiological-convention mirror artefact). `models/organ_labels_candidate.json` has been rewritten accordingly into the confirmed mapping.

| our# | Organ | Dice | | our# | Organ | Dice |
|---|---|---|---|---|---|---|
| 1 | Spleen | 0.97 | | 12 | Right lung upper lobe | 0.97 |
| 2 | Right kidney | 0.99 | | 13 | Right lung middle lobe | 0.96 |
| 3 | Left kidney | 0.98 | | 14 | Right lung lower lobe | 0.99 |
| 5 | Liver | 0.95 | | 16 | Trachea | 0.96 |
| 10 | Left lung upper lobe | 0.99 | | 18 | Small intestine | 0.91 |
| 11 | Left lung lower lobe | 0.99 | | 21 | Bladder | 0.87 |

## Segmentation study — one-sentence summary
No guessing — **a single ground-truth-labelled public CT pins down the model's identity, its label mapping, and pipeline correctness all at once**: organs.onnx is TotalSegmentator `class_map_part_organs`, mean Dice ≈ 0.92.

---

# Study III: Learned sparse-view reconstruction — what does it recover, and what does it invent? (`recon_dl.py`)

## Motivation

Study I measured the ceiling of the classical methods: error saturates beyond a certain number of views, and the optimal FBP filter *inverts* with dose. This study asks the follow-up question: can a small self-implemented network, used as an FBP post-processor, break the linear filter's built-in trade-off between **preserving detail** and **suppressing streaks** — and what does it cost?

**Why hallucination is a headline result, not an appendix.** Classical methods (FBP/ART) fail by producing **artefacts** — ugly, but a reader can see they are artefacts. Learned reconstruction fails by producing **hallucinations** — a clean image that looks like real anatomy where nothing was. Clinically these are entirely different failure modes, so "does it invent structure?" is measured directly rather than assumed away.

## Running

```bash
python experiments/recon_dl.py                  # everything (~55 min on an M-series MPS backend)
python experiments/recon_dl.py matrix           # view-count matrix only
python experiments/recon_dl.py halluc ood res   # the three probes (reuses saved weights)
python experiments/recon_dl.py plot             # redraw figures from the committed CSVs
```

Extra dependency: `torch` (see `requirements-experiments.txt`). **The App does not need it** — a model reaching the GUI would be exported to ONNX and served by the `onnxruntime` already in the main requirements.

## Methods

- **Data**: a *family* of random phantoms — random ellipses plus 0–3 small high-contrast "lesions" — generated in code with fixed seeds. A single fixed Shepp-Logan would simply be memorised by the network, so what you would then measure is recall, not reconstruction. **No patient data**; anyone re-running gets the same numbers.
- **Split**: train / val / test / pairing draw from four **non-overlapping seed bands**, so no phantom instance appears in two sets. Checkpoints are selected on the **validation** set only; the test set never participates in any selection.
- **Forward model**: `recon.compute_sinogram` → `recon.compute_fbp` — the same production functions the GUI calls.
- **Network**: a self-implemented residual U-Net (1.9 M params, no MONAI/nnU-Net). It predicts the *artefact* and subtracts it, because sparse-view streaks are sparse and high-frequency — far easier to learn than re-synthesising anatomy, and much less prone to inventing structure. The output layer is zero-initialised, so training starts exactly at "pass the FBP through unchanged", giving a clean zero baseline for "how much did the network change?".
- **Network input is ramp-FBP, not hann.** Hann has already filtered the high frequencies — and the detail with them — away at the filtering stage; that information is gone and the network cannot recover it. Ramp keeps the information but leaves streaks, and streaks are what can be learned.
- **Three-tier metrics** (any one tier alone is self-deceiving): global RMSE/SSIM; detail — lesion-contrast retention and bar-pattern modulation transfer (CTF); safety — background streak level and **false-structure rate**.

## Findings

### A — The view-count matrix (128², 80 test phantoms per cell)

| views | FBP-hann RMSE / lesion | FBP-ramp RMSE / lesion | **+CNN** RMSE / lesion |
|---|---|---|---|
| 15 | 0.0415 / 0.750 | 0.0525 / 0.867 | **0.0110 / 0.957** |
| 20 | 0.0339 / 0.758 | 0.0415 / 0.877 | **0.0084 / 0.974** |
| 30 | 0.0281 / 0.752 | 0.0291 / 0.869 | **0.0059 / 0.987** |
| 45 | 0.0264 / 0.750 | 0.0220 / 0.864 | **0.0042 / 0.993** |
| 60 | 0.0260 / 0.751 | 0.0196 / 0.867 | **0.0035 / 0.996** |

**The linear filters' lesion-contrast loss is independent of dose.** Hann sits at 0.750 and ramp at 0.867 across all five view counts — essentially flat. That 25% / 13% loss is therefore **not** caused by undersampling; it is the filter's intrinsic price, and no amount of extra dose buys it back. The CNN instead climbs from 0.957 to 0.996, converging on the ground truth as dose increases.

**Study I's filter inversion reproduces here, on a different phantom family.** At 15 views hann wins (0.0415 < 0.0525); by 45 views ramp wins (0.0220 < 0.0264); the crossover sits between 30 and 45 views. Study I found this on Shepp-Logan — an independent cross-check.

### B — Hallucination: paired phantoms, identical background, lesion present vs absent

| quantity | value |
|---|---|
| true lesion signal (the yardstick) | +0.3472 |
| FBP at the lesion-free site | +0.0017 |
| **network at the lesion-free site** | **+0.0028** |
| network at the true-lesion site | +0.3427 (**99% recovered**) |
| false-structure rate (> 20% / 30% / 50% of a true lesion) | **1.7% / 0% / 0%** |

The network's signal where nothing exists is the same order as FBP's own fluctuation. **It is not inventing structure.** This has to be measured with paired phantoms: the matrix's "background streak std" is a whole-background statistic, in which one isolated fake lesion is diluted by tens of thousands of pixels and cannot be seen at all.

### C — Out-of-distribution: is it generic de-streaking, or memorised shapes?

| test set | FBP RMSE | +CNN | reduction |
|---|---|---|---|
| squares (sharp corners, straight edges) | 0.0458 | 0.0084 | **81.6%** |
| polygons (non-convex) | 0.0348 | 0.0044 | **87.2%** |
| line gratings (high frequency) | 0.2093 | 0.1590 | **24.1%** |

In-distribution reduction 79.7% → out-of-distribution mean 64.3%, **ratio 0.81**. Not one of these shapes appears in training, yet on both shape families the gain **matches or exceeds** the in-distribution figure. What the network learned is a shape-independent de-streaking operation; the mean is dragged down solely by the high-frequency gratings.

### D — The resolution limit (bar patterns, `+CNN` vs `FBP-ramp`)

Judge by **|CTF − 1|**, not by "higher CTF is better": ramp overshoots at sharp edges (Gibbs ringing), so CTF can exceed 1, and that is distortion, not an advantage.

| period (px) | line width | \|CTF − 1\| FBP → CNN | RMSE reduction |
|---|---|---|---|
| 4 | 2 | 0.988 → **0.729** | 28.0% |
| 6 | 3 | 0.705 → **0.436** | 37.2% |
| 8 | 4 | 0.316 → **0.155** | 37.4% |
| 10 | 5 | 0.265 → **0.226** | 37.2% |
| 12–28 | 6–14 | ≈ level | 36.7–40.3% |

**4 of 8 frequencies land closer to ground truth, 4 are level, none is worse.** The gain concentrates at high frequency, where ramp's overshoot is worst (CTF 1.99 at a 4 px period — nearly double the true modulation) and the CNN corrects it. At low frequency FBP is already close to the truth, so there is little left to fix. RMSE reduction only drops off at the finest line width (2 px, 28.0%) — the sampling limit itself.

## Limitations (stated, not buried)

- **ART/SIRT are not in the matrix.** They need an explicit system matrix, and `lstsq` cost caps the usable size at about 64² — not comparable with this study's 128². Putting them in the same table would manufacture a false equivalence.
- **Phantoms, not anatomy.** The phantom family is richer than a single Shepp-Logan and the OOD sets probe shapes never trained on, but none of this is real CT. The transfer to clinical images is untested here.
- **One network size, one loss.** 1.9 M parameters trained with plain MSE. Whether a different capacity or a perceptual/adversarial loss would move the hallucination rate is not measured.
- **The 24.1% on gratings is a real weakness**, not a rounding artefact: at the sampling limit a post-processor cannot restore information the sparse projections never carried.

## Reconstruction-DL study — one-sentence summary
A 1.9 M-parameter self-implemented post-processor cuts sparse-view RMSE by **3–6×** versus the best linear filter and lifts lesion-contrast retention from a dose-independent **0.87 ceiling to 0.96–1.00** — and, measured rather than assumed, it does so **without inventing structure** (false-structure rate 1.7% at the loosest threshold, 0% beyond) and **without depending on the training shapes** (out-of-distribution gain ratio 0.81), with its only real limit at the sampling frequency itself.
