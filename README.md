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

A desktop **CT imaging workstation** built with PySide6/Qt6. It brings clinical-style DICOM reading, AI multi-organ segmentation and a tomographic-reconstruction teaching lab into one application. The repository also contains **three documented quantitative studies, two multi-case validations, and an ablation that changed the product**, turning production code into measured, reproducible evidence.

> [!WARNING]
> **Teaching and research only.** This software is not a certified medical device and must not be used for clinical diagnosis. AI segmentation and organ measurements are automated estimates, not clinical findings.

![AI segmentation with per-organ confidence](docs/img/gui_confidence.png)

## At a glance

| Product | Runtime | Evidence | Data boundary |
|---|---|---|---|
| DICOM reader + AI segmentation + reconstruction lab | Python 3.10 · PySide6/Qt6 · CPU-only | Studies I–III · 57-case lung-lobe and 20-case multi-organ validations · a spacing ablation that fixed a real defect | Synthetic phantoms and de-identified public research CTs; no PHI is committed |

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

The experiments call the shipped production pipeline rather than a separate reimplementation. Studies I–III and the spacing ablation are documented in the [technical report](docs/technical_report.md); scripts and committed outputs are indexed under [`experiments/`](experiments/README.md).

| Evidence track | Measured finding | Scope and boundary |
|---|---|---|
| **Study I — reconstruction dose–quality** | Error saturates beyond ≈180 views; the best FBP filter switches from smoothing at sparse angles to sharp Ram-Lak at dense angles; ART is the most robust tested solver under Poisson photon noise. | Analytic 2-D Shepp-Logan phantom; matrix methods are limited to ≈64×64. [Preprint](docs/preprint_recon.md) |
| **Study II — model provenance and Dice** | A label-overlap confusion matrix identifies the undocumented ONNX model as TotalSegmentator v2 `class_map_part_organs`, and corrects two label errors. Across **20 cases** the patient-level mean Dice is **0.909** (95% CI [0.889, 0.927]); the original single case at 0.922 was mildly optimistic but inside the interval. | Per-organ reliability varies far more than the aggregate implies: liver 0.982 and spleen 0.976 versus right upper lung lobe 0.773 and prostate 0.554 (present in only 7 cases). [`seg_multi.py`](experiments/seg_multi.py) |
| **Study III — learned sparse-view reconstruction** | A self-implemented 1.9 M-parameter residual U-Net reduces RMSE by **3–6×**, raises lesion-contrast retention from 0.87 to **0.96–1.00**, records a 1.7% false-structure rate and retains an 0.81 out-of-distribution gain ratio. | Noise-free synthetic projections; the hallucination rate is a favourable-condition lower bound and does not transfer to photon-starved low-dose CT. |
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

Offscreen-Qt coverage is **90%** over 3277 statements. The nine Qt-free modules (`recon` 84%, `quantify` 100%, `segmentation` 86%, `mpr_geometry` 96%, `followup` 90%, `projection` 95%, `mesh3d` 96%, `registration` 98%, `model_card` 73%) are unit-tested independently. Synthesised mouse press/move/release sequences assert emitted signal payloads (`graphics_view` 91%). Pinning down the previously untested layers moved them substantially: reconstruction-lab UI scheduling `recon_lab` 44% → **89%**, annotation/segmentation editing `annotation_lab` 74% → **84%**, the mouse-interaction dispatcher `interaction.py` 64% → **98%**, follow-up comparison `compare_lab` 82% → **95%**. Writing those assertions surfaced three real defects that reading the code had not: a probe read-out that kept showing the previous value when the cursor left the volume, a model card that crashed on a truncated CSV, and annotations with numeric ids that could never render or be deleted. A green CI run therefore confirms the data-independent subset, not every local-data interaction test.

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

The software is jointly owned by the two copyright holders above. A software-copyright registration application naming both holders has been submitted to the Copyright Protection Centre of China. This repository is provided for teaching, research and portfolio review only; **no license is granted for copying, modification or redistribution**. Contact the copyright holders for permission. Integrated third-party components retain their own licenses; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
