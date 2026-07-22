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
- **Two quantitative studies** — a reconstruction dose–quality characterisation (with a non-obvious filter-inversion finding) and an AI-model provenance/Dice validation, both driving the production code, fully reproducible, using no patient data.
- **Engineered for review** — a God-object decomposed into 5 UI mixins + 4 Qt-free compute modules; a **155-check** offscreen-Qt regression suite with CI (reconstruction algorithms carry numerical-correctness assertions, not just "finite"); defensive DICOM handling.

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
| **Clinical reading** | Anatomical-sorted DICOM loading · tri-planar MPR + linked cross-hairs · 6 window presets + invert · 9 measurement/annotation tools · ellipse ROI (mean±SD / min-max HU / area) · four-corner PACS overlay · Cine playback · dual-series follow-up comparison |
| **AI segmentation** | Auto background inference · colour overlay with clickable legend · cursor HUD (HU / coords / organ) · per-organ quantification (volume mL / mean HU) with CSV export · brush/eraser editing with undo |
| **Reconstruction lab** | Radon projection (60–360°, 1–4× sampling) · BP / FBP (5 filters) / DFR · DMR (least-squares) / ART / SIRT with error maps + RMSE |
| **Compliance** | Display-layer de-identification · persistent AI disclaimer · bilingual (EN / 中文) UI toggle |

## Quantitative studies

Both studies exercise the shipped production code and use no patient data. See the [technical report](docs/technical_report.md) for methods, figures and results.

- **Study I — dose–quality tradeoffs in CT reconstruction.** On the Shepp-Logan phantom, error saturates beyond ≈ 180 views; the optimal FBP filter *inverts* with dose (smoothing filters win at sparse angles, sharp Ram-Lak at dense); under Poisson photon noise, constrained iteration (ART) is most robust while naive least-squares inversion destabilises near the square-system regime. Written up as a [preprint](docs/preprint_recon.md).
- **Study II — provenance recovery & Dice validation.** Running the undocumented ONNX model on one ground-truth-labelled public CT and computing a label-overlap confusion matrix recovers the label map (identity diagonal) and yields mean Dice ≈ 0.92 over 21 organs — simultaneously validating the inference pipeline and correcting two label errors.

Reproduce via [`experiments/`](experiments/README.md) (scripts + figures + CSVs).

## Testing

```bash
python tests/test_gui.py                     # full suite: 155 checks (needs local RIDER data)
SKIP_REAL_DATA=1 python tests/test_gui.py    # data-independent subset (used by CI)
ruff check .                                 # lint
coverage run tests/test_gui.py && coverage report
```

Offscreen Qt; exit code 0 = all pass. Coverage ≈ 70%; the four Qt-free compute modules (`recon` / `quantify` / `segmentation` / `mpr_geometry`) are unit-tested in isolation. CI runs the data-independent subset on every push/PR — so a green CI is **not** the full 155 checks (interaction-layer tests need local data).

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
- Matrix reconstruction (DMR/ART) is bounded to ≈ 64×64 by `lstsq` cost (teaching scope); AI/reconstruction/annotation act on the current primary series; AI overlay is axial-only.

## License · Copyright

© 2026 **Sheng Chao (盛超)** and **Lai Shengsheng (赖胜圣)**. All rights reserved.

Jointly owned by the two copyright holders above, as registered with the Copyright Protection Centre of China. This repository is provided for teaching / research and portfolio review only; **no license** is granted for copying, modification or redistribution — contact the copyright holders. Integrated third-party components remain under their own licenses — see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
