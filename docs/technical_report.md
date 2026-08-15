# A Teaching CT Imaging Workstation with Quantitative Reconstruction and Segmentation Studies

**Technical Report**

---

## Abstract

We present a desktop CT imaging workstation built on PySide6/Qt6 that integrates clinical reading tools, an educational tomographic-reconstruction laboratory, and an AI multi-organ segmentation pipeline. Beyond the software engineering, we contribute two quantitative studies that turn the platform's built-in algorithms into measured findings. **Study I** characterises the dose–quality tradeoff of analytic and iterative reconstruction on the Shepp-Logan phantom: reconstruction error falls monotonically with projection count and saturates beyond ≈180 views; the optimal apodisation filter *inverts* with dose (smoothing filters win at sparse angles, the sharp Ram-Lak wins at dense angles, crossover ≈45–60 views); and under realistic photon noise a constrained iterative solver (ART) is the most robust while naive least-squares inversion becomes unstable near the square-system regime. **Study II** recovers the provenance of an undocumented ONNX segmentation model by measurement rather than metadata: running the model on a single ground-truth-labelled public CT and computing a label-overlap confusion matrix yields an identity diagonal, confirming the model is TotalSegmentator v2 `class_map_part_organs` (24 organs + background, nnU-Net v2), with a mean Dice ≈ 0.92 over the 21 organs present. The same experiment validated the inference pipeline and corrected two long-standing label errors. All experiments are scripted, reproducible, and use no patient data.

---

## 1. Introduction

Medical-image computing sits at the intersection of physics (image formation), numerical methods (reconstruction), and machine learning (segmentation). A common weakness of student projects in this area is that they *implement* known algorithms without *measuring* them: there is code, but no experiment, no metric, and no conclusion. This report addresses that gap. Starting from a functional CT workstation, we design two small but rigorous quantitative studies — one on reconstruction physics, one on AI-model evaluation — each producing defensible, figure-backed findings. Study I exercises the application's production reconstruction code directly; Study II runs the shipped ONNX model through a numerically identical copy of the application's preprocessing and sliding-window inference (§4.1). The measurements therefore characterise the shipped system.

## 2. Platform

The workstation (≈4,100 lines of application Python across 13 modules) provides DICOM loading with anatomical sorting, tri-planar MPR with linked cross-hairs, clinical window/level presets, measurement and annotation tools, dual-series follow-up comparison, and a background AI segmentation engine (`ai_engine.py`) wrapping an ONNX model with sliding-window inference. A separate **reconstruction laboratory** (`recon.py`) covers the Radon transform, back-projection (BP), filtered back-projection (FBP) with five apodisation filters, direct Fourier reconstruction (DFR), a direct matrix least-squares solve (DMR), and the algebraic iterative solvers ART (Kaczmarz) and SIRT. The forward Radon projector and the analytic inverses (BP, FBP) are built on scikit-image's `radon`/`iradon`; the DFR, DMR, ART and SIRT inverse solvers are implemented from first principles on top of that projector. The codebase is covered by a 262-check offscreen-Qt regression suite. Study I calls `recon.py` directly; Study II reuses the `ai_engine` preprocessing and sliding-window procedure (§4.1).

## 3. Study I — Dose–Quality Tradeoffs in CT Reconstruction

### 3.1 Methods

The test object is the Shepp-Logan phantom (a standard analytic CT benchmark), rescaled to the working resolution, normalised to `[0,1]`, and masked to the inscribed circle to match `radon(circle=True)`; the masked phantom is the ground truth, since the sinogram only encodes information inside that circle. Forward projection uses the workstation's `compute_sinogram` (Radon transform). The **dose proxy** is the number of projection angles over a fixed 180° arc — fewer views approximate lower dose. Reconstruction quality is reported as in-circle RMSE and SSIM (data range 1.0). For the analytic-vs-iterative comparison (§3.2c) we add a physically grounded **Beer–Lambert + Poisson photon-noise** model to the sinogram (incident photons `I₀`; transmitted counts `N ~ Poisson(I₀·e^{-p})`; noisy projection `p' = -ln(N/I₀)`), with a fixed random seed for reproducibility. Experiments are scripted in [`experiments/recon_study.py`](../experiments/recon_study.py).

### 3.2 Results

**(a) Dose–quality curve (FBP, Ram-Lak, 256×256).** RMSE falls monotonically from 0.222 at 15 views to 0.035 at 360 views, while SSIM rises from 0.35 to 0.95 (Figure 1). RMSE saturates beyond ≈180 views — doubling 180→360 views buys only a further 3.2% (0.0367→0.0355), while SSIM keeps climbing slowly (0.90→0.95). These diminishing returns give a quantitative basis for "enough is enough" acquisition planning.

![Figure 1: Dose–quality curve](../experiments/results/exp_a_dose_quality.png)
*Figure 1. In-circle RMSE (left) and SSIM (right) versus projection count for FBP with the Ram-Lak filter. RMSE saturates beyond ≈180 views; SSIM continues to rise slowly.*

**(b) The optimal filter inverts with dose (256×256).** Comparing five FBP filters (Figure 2), no single filter is best at all doses. At 20 views the smoothing Hann filter minimises RMSE (0.143 vs 0.176 for Ram-Lak); at 180 views the sharp Ram-Lak wins (0.037 vs 0.055 for Hann). The crossover lies at ≈45–60 views. The mechanism is a bias–variance tradeoff: at sparse angles, high-frequency streak noise dominates and apodisation helps; at dense angles, fidelity dominates and the ramp filter's sharpness is decisive.

![Figure 2: Filter comparison](../experiments/results/exp_b_filters.png)
*Figure 2. In-circle RMSE versus projection count for five FBP filters. The ranking of filters reverses between the sparse and dense regimes.*

**(c) Under photon noise, constrained iteration is the most robust (64×64, I₀=3×10⁴).** Table 1 and Figure 3 compare FBP, direct matrix inversion (DMR, least-squares), ART, and SIRT under realistic noise. ART achieves the lowest RMSE at every dose. Naive least-squares (DMR) is unstable: at 60 views its RMSE spikes to 0.611 — an order of magnitude worse than the others — because with 64 detectors × 60 views the linear system is nearly square (≈3,840 equations for 4,096 unknowns), so its condition number is worst there and noise is maximally amplified; the 30-view (under-determined, minimum-norm) and 90-view (over-determined, averaging) cases are far more stable. This is the classic ill-posedness of unregularised inversion, not an implementation defect.

| Views | FBP | DMR (least-squares) | ART | SIRT |
|------:|----:|----:|----:|----:|
| 30 | 0.099 | 0.151 | **0.069** | 0.083 |
| 60 | 0.089 | **0.611** | **0.053** | 0.076 |
| 90 | 0.087 | 0.090 | **0.050** | 0.075 |

*Table 1. In-circle RMSE under Poisson photon noise. Bold marks the best (ART) and the catastrophic DMR failure at the near-square regime.*

![Figure 3: Analytic vs iterative](../experiments/results/exp_c_analytic_vs_iterative.png)

![Figure 3b: Visual gallery](../experiments/results/exp_c_gallery.png)
*Figure 3. Top: RMSE by method and dose under photon noise. Bottom: visual comparison at 30 views — DMR shows salt-and-pepper noise amplification, FBP shows streak artefacts, ART is cleanest, SIRT is smoothest. Visual quality tracks the measured RMSE.*

## 4. Study II — Provenance Recovery and Dice Validation of an ONNX Segmentation Model

### 4.1 Motivation and Methods

The workstation ships a 25-class organ segmentation model (`organs.onnx`) whose training `dataset.json` — and therefore its label→organ mapping — was unavailable; earlier labels were *inferred* from anatomy on a chest-only scan and were partly wrong. We resolve this by **measurement**. From the public, CC-BY-4.0 TotalSegmentator-CT-Lite dataset we fetch a single thorax–abdomen–pelvis case (`s0029`, 1.5 mm isotropic) using HTTP range requests to extract ≈42 MB from a 22 GB archive without downloading it. The CT is reoriented to the application's axis convention, passed through the *exact* `ai_engine` preprocessing (clip to `[-1000,400]` HU, normalise) and DZ=32 sliding-window inference, and the argmax output is compared voxel-wise to the ground-truth labels. For each model output label we compute the Dice overlap against every ground-truth organ and take the best match — **discovering** the mapping rather than assuming it. Scripted in [`experiments/seg_validate.py`](../experiments/seg_validate.py).

### 4.2 Results

The confusion matrix is a clean identity diagonal (Figure 4): model label *k* maps to the *k*-th TotalSegmentator organ, one-to-one, with no off-diagonal mass. Two of the model's output labels have no counterpart in this subject's ground truth — label 22 (prostate, 2,674 voxels) and label 23 (left kidney cyst, 3 voxels) — so their identity rests on the recovered mapping rather than on a measured Dice; the remaining 21 are matched. This **confirms the model is TotalSegmentator v2 `class_map_part_organs`** (24 organs + background, an nnU-Net v2 PlainConvUNet export) — a provenance established by measurement, not metadata (the ONNX file carries no label→organ mapping or dataset metadata; its graph does retain the exporter string and nnU-Net layer names, which independently corroborate the architecture). The heart and aorta columns are empty, decisively refuting the earlier hypothesis that label 5 was the heart: this model has no heart/aorta output (those live in a different TotalSegmentator part, labels 51/52, outside the 0–24 range).

Over the 21 organs present, the **mean Dice is ≈0.92** (kidneys 0.98; lung lobes 0.96–0.99; small structures such as thyroid and gallbladder 0.79–0.82), consistent with TotalSegmentator's published performance — which simultaneously **validates the workstation's inference pipeline** as correct. The study also corrected two label errors now fixed in the code: label 5 is **liver** (not heart), and the lung lobes are 10,11 = **left** (upper/lower) and 12,13,14 = **right** (upper/middle/lower); the earlier left/right swap was a radiological-convention mirror artefact.

![Figure 4: Segmentation confusion matrix](../experiments/results/seg_confusion.png)
*Figure 4. Dice overlap between each `organs.onnx` output label (rows) and each TotalSegmentator ground-truth organ (columns). The identity diagonal recovers the label map; empty heart/aorta columns confirm the model's organ set.*

## 5. Discussion and Limitations

Study I is deliberately small-scale (a single analytic phantom, 2-D slices, systems up to 64×64 for the matrix methods), matching the workstation's educational scope; the qualitative conclusions (saturation, filter inversion, iterative robustness) are standard CT results here reproduced and quantified end-to-end on production code. Study II uses a single labelled case; because the confusion diagonal is unambiguous and the per-organ Dice is high and consistent with published numbers, one case suffices to establish provenance and validate the pipeline, but a multi-case evaluation would tighten the Dice estimates and their variance. Neither study makes the workstation a clinical device: it carries no regulatory clearance, and its de-identification is display-layer only.

## 6. Conclusion

Two compact studies convert a functional imaging workstation into measured science: a reconstruction dose–quality characterisation with a non-obvious filter-inversion finding and a photon-noise robustness comparison, and a provenance-recovery experiment that identifies an undocumented segmentation model, validates its pipeline at Dice ≈0.92, and corrects its labels — all reproducibly and without patient data. Both studies are directly extensible: Study I toward sparse-view/low-dose reconstruction research, Study II toward multi-case quantitative model evaluation.

## Reproducibility

All results are regenerated by:
```bash
python experiments/recon_study.py a b c      # Study I (Figures 1–3)
python experiments/seg_validate.py <img.nii.gz> <mask.nii.gz>   # Study II (Figure 4)
```
Environment: Python 3.10, PySide6 6.11, numpy 2.2, scipy 1.15, scikit-image 0.25, onnxruntime 1.23, nibabel 5.4. See [`experiments/README.md`](../experiments/README.md) for data access and full method details.

## References

1. J. Wasserthal et al., "TotalSegmentator: Robust Segmentation of 104 Anatomical Structures in CT Images," *Radiology: Artificial Intelligence*, 2023.
2. F. Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation," *Nature Methods*, 18:203–211, 2021.
3. L. A. Shepp and B. F. Logan, "The Fourier reconstruction of a head section," *IEEE Trans. Nuclear Science*, 21(3):21–43, 1974.
4. R. Gordon, R. Bender, and G. T. Herman, "Algebraic reconstruction techniques (ART) for three-dimensional electron microscopy and X-ray photography," *J. Theoretical Biology*, 29(3):471–481, 1970.
5. A. C. Kak and M. Slaney, *Principles of Computerized Tomographic Imaging*, IEEE Press, 1988.
