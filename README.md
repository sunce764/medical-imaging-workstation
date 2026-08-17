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

A desktop **CT imaging workstation** built with PySide6/Qt6. It brings clinical-style DICOM reading, AI multi-organ segmentation and a tomographic-reconstruction teaching lab into one application. The repository also contains **three documented quantitative studies plus a 57-case follow-up validation**, turning product code into measured, reproducible evidence.

> [!WARNING]
> **Teaching and research only.** This software is not a certified medical device and must not be used for clinical diagnosis. AI segmentation and organ measurements are automated estimates, not clinical findings.

![AI multi-organ segmentation overlay](docs/img/gui_axial_segmentation.png)

## At a glance

| Product | Runtime | Evidence | Data boundary |
|---|---|---|---|
| DICOM reader + AI segmentation + reconstruction lab | Python 3.10 · PySide6/Qt6 · CPU-only | Studies I–III + a 57-case lung-lobe validation | Synthetic phantoms and de-identified public research CTs; no PHI is committed |

## Interface

| AI multi-organ segmentation | Tri-planar MPR with linked cross-hairs |
|:---:|:---:|
| ![Axial segmentation](docs/img/gui_axial_segmentation.png) | ![Tri-planar MPR](docs/img/gui_mpr_triplanar.png) |

> The screenshots use **TotalSegmentator-CT-Lite** (CC-BY-4.0), a de-identified public research dataset with no PHI in this repository.

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

The experiments call the shipped production pipeline rather than a separate reimplementation. Studies I–III are documented in the [technical report](docs/technical_report.md); scripts and committed outputs are indexed under [`experiments/`](experiments/README.md).

| Evidence track | Measured finding | Scope and boundary |
|---|---|---|
| **Study I — reconstruction dose–quality** | Error saturates beyond ≈180 views; the best FBP filter switches from smoothing at sparse angles to sharp Ram-Lak at dense angles; ART is the most robust tested solver under Poisson photon noise. | Analytic 2-D Shepp-Logan phantom; matrix methods are limited to ≈64×64. [Preprint](docs/preprint_recon.md) |
| **Study II — model provenance and Dice** | A label-overlap confusion matrix identifies the undocumented ONNX model as TotalSegmentator v2 `class_map_part_organs`; mean Dice is ≈0.92 across 21 present organs and two label errors are corrected. | One ground-truth-labelled public CT (*n = 1*); provenance is supported, but population-level Dice is not. |
| **Study III — learned sparse-view reconstruction** | A self-implemented 1.9 M-parameter residual U-Net reduces RMSE by **3–6×**, raises lesion-contrast retention from 0.87 to **0.96–1.00**, records a 1.7% false-structure rate and retains an 0.81 out-of-distribution gain ratio. | Noise-free synthetic projections; the hallucination rate is a favourable-condition lower bound and does not transfer to photon-starved low-dose CT. |
| **Follow-up validation — lung lobes** | Across 57 public CTs, five-lobe mean Dice is **0.8867** (95% CI **[0.859, 0.914]**); right-upper-lobe Dice is 0.727 versus 0.967 in the original single case. | Validates five lung lobes only. The original 21-organ aggregate remains a single-case estimate. [`seg3d_eval.py`](experiments/seg3d_eval.py) · [`seg3d_report.py`](experiments/seg3d_report.py) |

## Engineering and testing

- The original God-object is decomposed into **5 UI mixins + 8 Qt-free compute modules**.
- CI runs **307 data-independent checks**; interaction tests that require local research data remain outside CI.
- Reconstruction tests assert numerical correctness, not merely finite output; DICOM loading is defensive against malformed metadata.

```bash
python tests/test_gui.py                     # full suite; requires local RIDER data
SKIP_REAL_DATA=1 python tests/test_gui.py    # 307 data-independent checks used by CI
ruff check .                                 # lint
coverage run tests/test_gui.py && coverage report
```

<details>
<summary><strong>Coverage detail</strong></summary>

Offscreen-Qt coverage is ≈79%. The eight Qt-free compute modules (`recon`, `quantify`, `segmentation`, `mpr_geometry`, `followup`, `projection`, `mesh3d`, `registration`) are unit-tested independently at 77–100%. Synthesised mouse press/move/release sequences assert emitted signal payloads (`graphics_view` 91%). Reconstruction-lab UI scheduling (`recon_lab` 44%) remains the least-covered layer. A green CI run therefore confirms the data-independent subset, not every local-data interaction test.

</details>

## Documentation

| Document | Language | Purpose |
|---|---|---|
| [Architecture](docs/ARCHITECTURE.md) | EN | Module map, God-object decomposition, segmentation-model provenance and AI-pipeline contract |
| [User manual](docs/manual_en.md) · [中文](docs/manual_zh.md) · [PDF](docs/manual_zh.pdf) | EN · 中文 | Screenshot-led guide to every user-facing feature |
| [Technical report](docs/technical_report.md) | EN | Methods, figures and results for Studies I–III |
| [Preprint — Study I](docs/preprint_recon.md) | EN | Sparse-view / low-dose reconstruction in academic format |
| [Experiments](experiments/README.md) | EN | Reproducible scripts, figures and CSV outputs |
| [Changelog](CHANGELOG.md) | EN | Auditable defect-fix and review notes |
| [Third-party notices](THIRD_PARTY_NOTICES.md) | EN | Upstream-verified licenses for integrated components |

## Safety and known limits

- **Not a clinical device:** no regulatory clearance, clinical-validation dossier, audit trail or access control.
- **Display-layer de-identification only:** PHI is hidden on screen and in export filenames, but underlying DICOM tags and burned-in text are not scrubbed.
- **AI generalisation remains partly unmeasured:** the 57-case extension covers only five lung lobes; the 21-organ aggregate remains *n = 1*. Input spacing is not resampled, so accuracy outside the measured 1.5 mm isotropic setting is unknown.
- **Educational reconstruction scope:** DMR / ART matrix reconstruction is bounded to ≈64×64 by least-squares cost. Study III uses noise-free synthetic projections and cannot establish low-dose clinical performance.
- **Follow-up comparison is rigid, not deformable:** in-plane registration reduced shift-induced MAE from 321 HU to 13 HU in its test, but respiratory organ deformation is not corrected. Reported differences remain qualitative rather than clinical change measurements.

## License and copyright

© 2026 **Sheng Chao (盛超)** and **Lai Shengsheng (赖胜圣)**. All rights reserved.

The software is jointly owned by the two copyright holders above. A software-copyright registration application naming both holders has been submitted to the Copyright Protection Centre of China. This repository is provided for teaching, research and portfolio review only; **no license is granted for copying, modification or redistribution**. Contact the copyright holders for permission. Integrated third-party components retain their own licenses; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
