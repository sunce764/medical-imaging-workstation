<h1 align="center">Medical Imaging Workstation + Reconstruction Lab</h1>

<p align="center"><strong>CT reading · AI segmentation · reproducible reconstruction research</strong></p>

<p align="center"><strong>English</strong> · <a href="README.zh-CN.md">简体中文</a></p>

<p align="center">
  <a href="https://github.com/sunce764/medical-imaging-workstation/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/sunce764/medical-imaging-workstation/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.10" src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&amp;logoColor=white">
  <img alt="PySide6 / Qt6" src="https://img.shields.io/badge/GUI-PySide6%20%2F%20Qt6-41CD52?logo=qt&amp;logoColor=white">
  <a href="LICENSE"><img alt="Proprietary license" src="https://img.shields.io/badge/License-Proprietary-lightgrey"></a>
  <img alt="Teaching and research only" src="https://img.shields.io/badge/⚠️-teaching%2Fresearch%20·%20not%20a%20medical%20device-critical">
</p>

**3D multi-organ CT segmentation and tomographic reconstruction**, wrapped in a clinical-style DICOM workstation (PySide6/Qt6, CPU-only). Underneath the interface: the iterative solvers (DMR / ART / SIRT), direct-Fourier reconstruction and the analytic Shepp-Logan phantom are written from first principles; **two networks are trained from scratch** — a 1.9 M residual U-Net for sparse-view reconstruction and a 0.35 M 3D U-Net for lung-lobe segmentation; and the shipped 25-class ONNX segmenter, which arrived with no documentation of its origin, was identified and then validated against public ground truth across 297 CTs. **Four documented quantitative studies, two multi-case validations, and an ablation that changed the product** turn production code into measured, reproducible evidence.

> [!WARNING]
> **Teaching and research only.** This software is not a certified medical device and must not be used for clinical diagnosis. AI segmentation and organ measurements are automated estimates, not clinical findings.

![AI segmentation with per-organ confidence](docs/img/gui_confidence.png)

## At a glance

| Product | Runtime | Evidence | Data boundary |
|---|---|---|---|
| DICOM reader + AI segmentation + reconstruction lab | Python 3.10 · PySide6/Qt6 · CPU-only | Studies I–IV · 57-case lung-lobe and 20-case multi-organ validations · a spacing ablation that fixed a real defect | Synthetic phantoms and de-identified public research CTs; no PHI is committed |

## Interface

| AI multi-organ segmentation | Tri-planar MPR with linked cross-hairs |
|:---:|:---:|
| ![Axial segmentation](docs/img/gui_axial_segmentation.png) | ![Tri-planar MPR](docs/img/gui_mpr_triplanar.png) |

**Per-voxel confidence alongside every measurement.** Each organ row carries the model's softmax max-class probability and its 5th percentile — the low percentile is the revealing one, since errors concentrate at boundaries. Entries below 0.9 are flagged; in the run below the gallbladder (`conf 0.88 / p5 0.54`) is the one the model is least sure about, matching what the spacing ablation independently found to be the most fragile structure.

![AI segmentation with per-voxel confidence](docs/img/gui_confidence.png)

| Reconstruction lab, no data required | Model card: provenance and limits |
|:---:|:---:|
| ![Built-in phantom reconstruction](docs/img/gui_recon_phantom.png) | ![Model card](docs/img/gui_model_card_en.png) |

The **built-in Shepp-Logan phantom** makes the whole reconstruction pipeline usable with nothing imported — V3 is unfiltered back-projection (a blur), V4 the filtered version resolving the same phantom down to its smallest lesions. Because the phantom's ground truth is known analytically, the error view measures distance to the truth rather than to another reconstruction. The **model card** states how the model's identity was established by measurement, how far it has been validated, and what remains unmeasured; every number on it is read live from `experiments/results/`, so re-running an experiment updates the card.

> Screenshots use **TotalSegmentator-CT-Lite** (CC-BY-4.0), a de-identified public research dataset; no PHI is committed to this repository. The phantom and model-card views need no data at all.

## Core capabilities

| Area | What is included |
|---|---|
| **Clinical reading** | Anatomically sorted DICOM loading · tri-planar MPR with linked cross-hairs · 6 window presets + invert · slab MIP / MinIP / AIP on all three planes · 9 measurement and annotation tools · ellipse ROI statistics · four-corner PACS overlay · Cine playback · dual-series follow-up comparison |
| **AI segmentation** | Background sliding-window ONNX inference for 25 classes, including 5 lung lobes · tri-planar colour overlay and clickable legend · cursor HUD · per-organ statistics and CSV export · 3-D marching-cubes surface preview, shape features and STL export · brush/eraser editing with undo · per-voxel confidence reporting |
| **Reconstruction lab** | Built-in analytic Shepp-Logan phantom · Radon projection · BP / FBP with 5 filters / DFR · first-principles DMR, ART and SIRT solvers · error maps and RMSE · learned CNN post-processing with its training-view and input-filter constraints shown in the UI |
| **Safety and review** | Display-layer de-identification · persistent AI disclaimer · model card covering measured provenance and unmeasured limits · bilingual EN / 中文 interface |

## Implemented here, or called from a library

Stated explicitly, because "I built a CT reconstruction lab" means very different things depending on the answer.

| Component | How it exists here | Where |
|---|---|---|
| Radon projection · BP · FBP with 5 filters | **Called** — `skimage.transform.radon` / `iradon` | [`recon.py`](recon.py) |
| Direct Fourier reconstruction (DFR) | **Written from first principles** — central-slice theorem: 1-D FFT per projection, polar→Cartesian interpolation, 2-D inverse FFT. Includes a half-pixel correction for even sizes that took a real debugging pass to find | [`recon.py`](recon.py) |
| Shepp-Logan phantom | **Written from first principles** — ten analytic ellipses, not a bitmap from a library, so it renders at any resolution without interpolation loss | [`recon.py`](recon.py) |
| System matrix · DMR · ART · SIRT | **Written from first principles** — per-pixel matrix construction with caching; ART is Kaczmarz row-action with precomputed row norms | [`recon.py`](recon.py) |
| Sparse-view reconstruction CNN (1.9 M) | **Trained from scratch**, PyTorch, seed-fixed | [`recon_dl.py`](experiments/recon_dl.py) |
| Lung-lobe 3D U-Net (0.35 M) | **Trained from scratch**, patient-level split, seed-fixed | [`seg3d_train.py`](experiments/seg3d_train.py) |
| 25-class organ segmentation | **Third-party weights** (TotalSegmentator v2). The provenance recovery, label-map verification, and 20-case / 57-case validation are this project's work; the network is not | [`ai_engine.py`](ai_engine.py) |
| DICOM I/O · MPR geometry · quantification · registration | **Written here** on top of `pydicom` / `numpy` / `scipy` | [`main.py`](main.py), [`mpr_geometry.py`](mpr_geometry.py), [`quantify.py`](quantify.py) |

## Quick start

```bash
conda env create -f environment.yml     # create the Python 3.10 environment
conda activate dicom_gui
python main.py                           # start without loading data
python main.py --data /path/to/dicom_dir # or load a DICOM directory at launch
```

- **CPU-only; no GPU is required.** Full-volume AI inference takes about 100 seconds on the reference machine.
- Model weights (`models/organs.onnx.data`, 119 MB) are **not distributed in the repository**. Without them, segmentation falls back to a classical connected-component algorithm. See [Architecture → Model](docs/ARCHITECTURE.md#segmentation-model).

## Measured evidence

The experiments call the shipped production pipeline rather than a separate reimplementation. Studies I–III and the spacing ablation are documented in the [technical report](docs/technical_report.md); **Study IV is newer than that report** and lives, with every script and committed output, under [`experiments/`](experiments/README.md).

| Evidence track | Measured finding | Scope and boundary |
|---|---|---|
| **Study I — reconstruction dose–quality** | Error saturates beyond ≈180 views; the best FBP filter switches from smoothing at sparse angles to sharp Ram-Lak at dense angles; ART is the most robust tested solver under Poisson photon noise. | Analytic 2-D Shepp-Logan phantom; matrix methods are limited to ≈64×64. [Preprint](docs/preprint_recon.md) |
| **Study II — model provenance and Dice** | A label-overlap confusion matrix identifies the undocumented ONNX model as TotalSegmentator v2 `class_map_part_organs`, and corrects two label errors. Across **20 cases** the patient-level mean Dice is **0.909** (95% CI [0.889, 0.927]); the original single case at 0.922 was mildly optimistic but inside the interval. | Per-organ reliability varies far more than the aggregate implies: liver 0.982 and spleen 0.976 versus right upper lung lobe 0.773 and prostate 0.554 (present in only 7 cases). [`seg_multi.py`](experiments/seg_multi.py) |
| **Study III — learned sparse-view reconstruction** | A self-implemented 1.9 M-parameter residual U-Net reduces RMSE by **3–6×**, raises lesion-contrast retention from 0.87 to **0.96–1.00**, records a 1.7% false-structure rate and retains an 0.81 out-of-distribution gain ratio. | Noise-free synthetic projections; the hallucination rate is a favourable-condition lower bound and does not transfer to photon-starved low-dose CT. |
| **Study IV — compressing the segmenter, and the evaluation defect it exposed** | A 0.35 M-parameter 3D U-Net trained from scratch, measured against the shipped 31.2 M teacher. Scoring it surfaced a defect in the *evaluation* rather than the model: `InstanceNorm3d` normalises per sample over the spatial dims, and after HU rescaling air and zero-padding are the same value — so enlarging the inference tensor **destroys 99.3 % of predicted foreground without altering a single input voxel** (225,374 → 1,529). The same weights score **0.490 or 0.746** depending only on tensor size. Turning that suspicion on the shipped inference path — paired, all 24 organs, over the **59 of 61 test cases that contain at least one in-scope organ** — measures **+0.0133** [+0.0072, +0.0194] all-organ Dice for 1.18× time and +0.65 GB, improving **54 of 59** cases. | Five independent controls — negative controls, dose–response, zero-padding with content held identical, forward hooks on the norm layers, and a training-set control — each eliminate a different competing explanation. A 3-case pilot had advertised **+0.205** for that same change; the full split says **+0.013**, and the full-split run exists precisely to stop an outlier from becoming the headline. Student figures are validation-set only. [`seg3d_infer_bias.py`](experiments/seg3d_infer_bias.py) |
| **Ablation — the spacing contract** | The engine was skipping nnU-Net's mandatory resampling to the training spacing. Measured first (mean Dice 0.9219 → 0.7995 at twice the training spacing, small organs collapsing first and non-monotonically), then implemented. Across **20 paired cases** the same mismatched input recovers **0.684 → 0.840**, improving in **20/20** (Wilcoxon *p* = 1.9×10⁻⁶); inference on the bundled series drops from 100 s / 8.8 GB to **37 s / 3.0 GB**. | Only the coarser direction is testable on 32 GB; the finer side is argued from it being downsampling, not measured. Mask boundaries are now quantised to the 1.5 mm grid — structural accuracy up, pixel-level boundary precision down. [`seg_spacing.py`](experiments/seg_spacing.py) |
| **Follow-up validation — lung lobes** | Across 57 public CTs, five-lobe mean Dice is **0.8867** (95% CI **[0.859, 0.914]**); right-upper-lobe Dice is 0.727 versus 0.967 in the original single case. | Validates five lung lobes only. Independently corroborated: a separate 20-case run measures the same right upper lobe at 0.773 using a different script and draw of cases. [`seg3d_teacher.py`](experiments/seg3d_teacher.py) |

## Engineering and testing

- The original God-object is decomposed into **5 UI mixins + 9 Qt-free compute modules**.
- The full suite is **515 checks** (needs local research data); CI runs the **424 data-independent** ones, leaving interaction tests outside CI.
- Reconstruction tests assert numerical correctness, not merely finite output; DICOM loading is defensive against malformed metadata.

```bash
python tests/test_gui.py                     # full suite: 515 checks; requires local RIDER data
SKIP_REAL_DATA=1 python tests/test_gui.py    # 424 data-independent checks used by CI
ruff check .                                 # lint
coverage run tests/test_gui.py && coverage report
```

<details>
<summary><strong>Coverage detail</strong></summary>

Offscreen-Qt coverage is **89%** over 3300 statements. The nine Qt-free modules (`recon` 84%, `quantify` 100%, `segmentation` 86%, `mpr_geometry` 96%, `followup` 90%, `projection` 95%, `mesh3d` 96%, `registration` 98%, `model_card` 87%) are unit-tested independently. Synthesised mouse press/move/release sequences assert emitted signal payloads (`graphics_view` 91%). Pinning down the previously untested layers moved them substantially: reconstruction-lab UI scheduling `recon_lab` 44% → **89%**, annotation/segmentation editing `annotation_lab` 74% → **83%**, the mouse-interaction dispatcher `interaction.py` 64% → **98%**, follow-up comparison `compare_lab` 82% → **95%**. Writing those assertions surfaced three real defects that reading the code had not: a probe read-out that kept showing the previous value when the cursor left the volume, a model card that crashed on a truncated CSV, and annotations with numeric ids that could never render or be deleted. A green CI run therefore confirms the data-independent subset, not every local-data interaction test.

</details>

## Documentation

| Document | Language | Purpose |
|---|---|---|
| [Architecture](docs/ARCHITECTURE.md) | EN | Module map, God-object decomposition, segmentation-model provenance and AI-pipeline contract |
| [The spacing contract](docs/spacing_contract.md) | 中文 | A worked example: finding that the product violated the model's inference contract, quantifying the cost, fixing it, and stating what remains unmeasured |
| [User manual](docs/manual_en.md) · [中文](docs/manual_zh.md) · [PDF](docs/manual_zh.pdf) | EN · 中文 | Screenshot-led guide to every user-facing feature |
| [Technical report](docs/technical_report.md) | EN | Methods, figures and results for Studies I–III |
| [Preprint — Study I](docs/preprint_recon.md) | EN | Sparse-view / low-dose reconstruction in academic format |
| [Experiments](experiments/README.md) | EN | Reproducible scripts, figures and CSV outputs |
| [Changelog](CHANGELOG.md) | EN | Auditable defect-fix and review notes |
| [Third-party notices](THIRD_PARTY_NOTICES.md) | EN | Upstream-verified licenses for integrated components |

## Safety and known limits

- **Not a clinical device:** no regulatory clearance, clinical-validation dossier, audit trail or access control.
- **Display-layer de-identification only:** PHI is hidden on screen and in export filenames, but underlying DICOM tags and burned-in text are not scrubbed.
- **AI generalisation remains partly unmeasured:** 57 cases for lung lobes and 20 for the 21-organ aggregate are still small samples, all drawn from one public dataset (1.5 mm isotropic); other protocols and scanners are untested, and per-organ reliability varies far more than the aggregate (liver 0.98 vs prostate 0.55). Spacing resampling is now in place (see the evidence table), but its finer-spacing side remains argued rather than measured, and it is skipped for very large scan ranges.
- **Educational reconstruction scope:** DMR / ART matrix reconstruction is bounded to ≈64×64 by least-squares cost. Study III uses noise-free synthetic projections and cannot establish low-dose clinical performance.
- **Follow-up comparison is rigid, not deformable:** in-plane registration reduced shift-induced MAE from 321 HU to 13 HU in its test, but respiratory organ deformation is not corrected. Reported differences remain qualitative rather than clinical change measurements.

## License and copyright

© 2026 **Sheng Chao (盛超)** and **Lai Shengsheng (赖胜圣)**. All rights reserved.

The software is jointly owned by the two copyright holders above. A software-copyright registration application naming both holders has been submitted to the Copyright Protection Centre of China and is awaiting acceptance.

**What that filing covers, precisely.** The submitted materials are a snapshot dated **2026-07-08**, comprising the thirteen product modules listed in [`docs/build_source_pdf.py`](docs/build_source_pdf.py) — `main.py`, `ui_builder.py`, `interaction.py`, `recon_lab.py`, `compare_lab.py`, `annotation_lab.py`, `ai_engine.py`, `graphics_view.py`, `recon.py`, `quantify.py`, `segmentation.py`, `mpr_geometry.py`, `constants.py`. Two consequences worth stating rather than leaving to assumption:

- **The working tree has moved ahead of that snapshot**, and deliberately so — a filing is examined on the materials submitted, so those are frozen while the product keeps developing. Nothing here claims the current code *is* the registered version.
- **`experiments/` is not part of the registration.** The studies are research code that measures the product; they were never in scope, and none of their findings are covered by the filing.

This repository is provided for teaching, research and portfolio review only; **no license is granted for copying, modification or redistribution**. Contact the copyright holders for permission. Integrated third-party components retain their own licenses; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
