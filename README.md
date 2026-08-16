# Medical Imaging Workstation + Reconstruction Lab

**English** · [简体中文](README.zh-CN.md)

[![CI](https://github.com/sunce764/medical-imaging-workstation/actions/workflows/ci.yml/badge.svg)](https://github.com/sunce764/medical-imaging-workstation/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![GUI](https://img.shields.io/badge/GUI-PySide6%20%2F%20Qt6-41CD52?logo=qt&logoColor=white)
[![License](https://img.shields.io/badge/License-Proprietary-lightgrey)](LICENSE)
![Not a medical device](https://img.shields.io/badge/⚠️-teaching%2Fresearch%20·%20not%20a%20medical%20device-critical)

A desktop **CT imaging workstation** (PySide6/Qt6) that unites a clinical DICOM reader, **AI multi-organ segmentation**, and a tomographic **reconstruction teaching lab** in one application. Beyond the software, the repository ships **two reproducible quantitative studies** that turn the built-in algorithms into measured findings.

> **Positioning.** A teaching / research tool — **not a certified medical device, not for clinical diagnosis.** AI segmentation and quantification are automated estimates for reference only.

![AI multi-organ segmentation overlay](docs/img/gui_axial_segmentation.png)

---

## Highlights

- **Clinical reader** — parallel DICOM loading with anatomical sort; tri-planar MPR with linked cross-hairs; 6 clinical window/level presets; measurement, annotation and ellipse-ROI densitometry; dual-series follow-up comparison with anatomical registration.
- **AI multi-organ segmentation** — background sliding-window ONNX inference (25 classes, incl. 5 lung lobes). The shipped model was undocumented; its provenance was **recovered by measurement** — identified as TotalSegmentator v2 `class_map_part_organs`, with **mean Dice ≈ 0.92** over 21 organs on one ground-truth case (*n = 1*).
- **Reconstruction lab (teaching)** — forward Radon and analytic inverses (BP, FBP with 5 apodisation filters) via scikit-image; **DFR, DMR, ART and SIRT inverse solvers implemented from first principles.**
- **Three quantitative studies** — a reconstruction dose–quality characterisation (with a non-obvious filter-inversion finding), an AI-model provenance/Dice validation, and a **learned sparse-view reconstruction study that measures hallucination rather than assuming it away**. All drive the production code, are fully reproducible, and use no patient data.
- **Engineered for review** — a God-object decomposed into 5 UI mixins + 8 Qt-free compute modules; a **325-check** offscreen-Qt regression suite with CI (reconstruction algorithms carry numerical-correctness assertions, not just "finite"); defensive DICOM handling.

## Screenshots

| AI multi-organ segmentation | Tri-planar MPR with linked cross-hairs |
|:---:|:---:|
| ![Axial segmentation](docs/img/gui_axial_segmentation.png) | ![Tri-planar MPR](docs/img/gui_mpr_triplanar.png) |

> Demo data is the public **TotalSegmentator-CT-Lite** (CC-BY-4.0) — no patient data, no PHI.

## Quick start

```bash
conda env create -f environment.yml     # creates the "dicom_gui" env (Python 3.10)
conda activate dicom_gui
python main.py                           # empty start (no data loaded)
python main.py --data /path/to/dicom_dir # or load a DICOM directory on launch
```

- **CPU-only; no GPU required.** Full-volume AI inference takes ≈ 100 s.
- Model weights (`models/organs.onnx.data`, 119 MB) are **not shipped**; without them, segmentation falls back to a classical connected-component algorithm. See [Architecture → Model](docs/ARCHITECTURE.md#segmentation-model).

## Features

| Area | Capabilities |
|---|---|
| **Clinical reading** | Anatomical-sorted DICOM loading · tri-planar MPR + linked cross-hairs · 6 window presets + invert · **slab projection (MIP / MinIP / AIP) on all three planes** · 9 measurement/annotation tools · ellipse ROI (mean±SD / min-max HU / area) · four-corner PACS overlay · Cine playback · dual-series follow-up comparison with difference quantification |
| **AI segmentation** | Auto background inference · colour overlay on all three planes with clickable legend · cursor HUD (HU / coords / organ) · per-organ quantification (volume mL, mean ± SD, median, p5–p95, min–max HU) with CSV export · **3D surface reconstruction (marching cubes) — drag-to-rotate preview, shape features, STL export** · brush/eraser editing with undo |
| **Reconstruction lab** | Radon projection (60–360°, 1–4× sampling) · BP / FBP (5 filters) / DFR · DMR (least-squares) / ART / SIRT with error maps + RMSE · **learned CNN post-processing (Study III), with its training view count and input-filter constraints surfaced in the UI** |
| **Compliance** | Display-layer de-identification · persistent AI disclaimer · bilingual (EN / 中文) UI toggle |

## Quantitative studies

All three studies exercise the shipped production code and use no patient data. See the [technical report](docs/technical_report.md) for methods, figures and results.

- **Study I — dose–quality tradeoffs in CT reconstruction.** On the Shepp-Logan phantom, error saturates beyond ≈ 180 views; the optimal FBP filter *inverts* with dose (smoothing filters win at sparse angles, sharp Ram-Lak at dense); under Poisson photon noise, constrained iteration (ART) is most robust while naive least-squares inversion destabilises near the square-system regime. Written up as a [preprint](docs/preprint_recon.md).
- **Study III — learned sparse-view reconstruction: what it recovers vs what it invents.** A self-implemented 1.9 M-parameter residual U-Net used as an FBP post-processor cuts sparse-view RMSE by **3–6×** against the best linear filter, and lifts lesion-contrast retention from a **dose-independent 0.87 ceiling** (the filter's intrinsic price — flat across 15–60 views) to **0.96–1.00**. Crucially, the two failure modes people usually assert away are measured: paired lesion-present/absent phantoms put the **false-structure rate at 1.7% (0% beyond a 30% threshold)**, and out-of-distribution shapes never trained on (sharp-cornered squares, non-convex polygons) retain an **0.81 gain ratio** — so the improvement is generic de-streaking, not memorised shapes. The one real limit is the sampling frequency itself.
- **Study II — provenance recovery & Dice validation.** Running the undocumented ONNX model on one ground-truth-labelled public CT and computing a label-overlap confusion matrix recovers the label map (identity diagonal) and yields mean Dice ≈ 0.92 over 21 organs — simultaneously validating the inference pipeline and correcting two label errors.

Reproduce via [`experiments/`](experiments/README.md) (scripts + figures + CSVs).

## Testing

```bash
python tests/test_gui.py                     # full suite: 325 checks (needs local RIDER data)
SKIP_REAL_DATA=1 python tests/test_gui.py    # data-independent subset (used by CI)
ruff check .                                 # lint
coverage run tests/test_gui.py && coverage report
```

Offscreen Qt; exit code 0 = all pass. Coverage ≈ 79% — the eight Qt-free compute modules (`recon` / `quantify` / `segmentation` / `mpr_geometry` / `followup` / `projection` / `mesh3d` / `registration`) are unit-tested in isolation at 77–100%, and mouse interaction is driven through synthesised press/move/release sequences with the emitted signals asserted (`graphics_view` 91%). The reconstruction-lab UI scheduling (`recon_lab` 44%) remains the least-covered layer. CI runs the data-independent subset on every push/PR — so a green CI is **not** the full 325 checks (interaction-layer tests need local data).

## Documentation

| Document | Lang | Contents |
|---|---|---|
| [Architecture](docs/ARCHITECTURE.md) | EN | Module layout, God-object decomposition, segmentation-model reverse-engineering, AI-pipeline contract |
| [User manual](docs/manual_en.md) · [中文](docs/manual_zh.md) · [PDF](docs/manual_zh.pdf) | EN · 中文 | Feature-by-feature guide with screenshots |
| [Technical report](docs/technical_report.md) | EN | Two quantitative studies — methods, figures, results |
| [Preprint (Study I)](docs/preprint_recon.md) | EN | Sparse-view / low-dose reconstruction, academic format |
| [Experiments](experiments/README.md) | EN | Reproducible scripts + figures + CSVs |
| [Changelog](CHANGELOG.md) | EN | Defect-fix review notes |
| [Third-party notices](THIRD_PARTY_NOTICES.md) | EN | Licenses of integrated components (verified against upstream) |

## Limitations

- **Not a clinical device** — no regulatory clearance, no algorithm-validation dossier, no audit/access control.
- **De-identification is display-layer only** — hides PHI on screen and in export filenames, but does **not** scrub underlying DICOM tags or burned-in text.
- **Dice ≈ 0.92 rests on a single labelled case (*n = 1*)** — the label mapping is verified, but the figure should not be read as a population estimate.
- Matrix reconstruction (DMR/ART) is bounded to ≈ 64×64 by `lstsq` cost (teaching scope); AI/reconstruction/annotation act on the current primary series.
- **Follow-up comparison offers in-plane rigid registration, but no deformable registration.** The two series are first aligned by anatomical z-position; ticking "Register" then applies an in-plane rigid transform (phase-correlation translation plus a rotation search — measured to cut the MAE caused by a whole-image shift from 321 HU to 13 HU) before reporting Δ mean, MAE, RMSE and the difference map. **Respiratory organ deformation is not corrected**: a rigid transform only compensates posture and leaves internal anatomy unchanged. The reported difference therefore remains a qualitative indicator, not a clinical change measurement. Organ-level volume change is not available, since the prior series carries no segmentation.

## License · Copyright

© 2026 **Sheng Chao (盛超)** and **Lai Shengsheng (赖胜圣)**. All rights reserved.

Jointly owned by the two copyright holders above, as registered with the Copyright Protection Centre of China. This repository is provided for teaching / research and portfolio review only; **no license** is granted for copying, modification or redistribution — contact the copyright holders. Integrated third-party components remain under their own licenses — see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
