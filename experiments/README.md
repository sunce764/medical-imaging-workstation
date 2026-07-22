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
