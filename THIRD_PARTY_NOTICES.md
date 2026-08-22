# Third-Party Notices

This file faithfully records the third-party works this project integrates or redistributes, together with
their **upstream-declared licenses**, and gives verifiable primary-source URLs. Each entry was verified on
2026-07-17 by reading the upstream `LICENSE` text / PyPI metadata directly.

> **Disclaimer**: This file states the **fact** of "what upstream declared," with primary sources attached for
> cross-checking; it is **not legal advice**. Before public release, commercial use, or redistribution, confirm
> compliance yourself (or through a legal professional).
> Anything marked **to be confirmed** here is something the primary source cannot settle conclusively — **better
> to flag the uncertainty than to presume.**

---

## 1. AI Segmentation Model (this repository **does redistribute** its computation graph)

This repository git-tracks and distributes `models/organs.onnx` (45 KB, **computation graph only, no weights**).
The 119 MB weights `models/organs.onnx.data` are **not** distributed with this repository (see `README.md` →
"Model" for how to obtain them).

| Item | License declared upstream | Primary source |
|---|---|---|
| **TotalSegmentator** (code) | **Apache-2.0** | [LICENSE text](https://raw.githubusercontent.com/wasserth/TotalSegmentator/master/LICENSE) |
| **TotalSegmentator weights — `total` task** (includes Task 291 `class_map_part_organs`, the task whose label scheme this model's outputs were measured to match; see `docs/ARCHITECTURE.md` → Getting the weights for why that is an inference about the weights' origin rather than a verified one) | **Apache-2.0** | [README "Subtasks" section](https://github.com/wasserth/TotalSegmentator): this task is listed under "Openly available for any usage (Apache-2.0 license)"; the weights are hosted in that Apache-2.0 repository's own [v2.0.0-weights release](https://github.com/wasserth/TotalSegmentator/releases) |
| **nnU-Net v2** (the architecture source of this model) | **Apache-2.0** | [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) |

**Upstream details worth knowing (to avoid over-extrapolation)**: TotalSegmentator is **licensed per task**, not
uniformly across the whole repository. Beyond the `total` task in the table above (Apache-2.0), several other
tasks "require a separate license (free for non-commercial use; contact upstream for commercial use)," and
`brain_aneurysm` is further CC-BY-NC-4.0 with no commercial license. **This project uses only Task 291 of the
`total` task**, which falls in the Apache-2.0 tier. **If the model is ever swapped out, the license tier of that
task must be re-verified; do not extrapolate from this entry.**

**Redistribution obligations (per Apache-2.0 §4)**:
- **§4(a) requires providing recipients with a copy of the Apache-2.0 license** — the complete, unmodified
  text is included in this repository at [`licenses/APACHE-2.0.txt`](licenses/APACHE-2.0.txt), so the
  obligation is met by the distribution itself rather than by an external link. (Canonical copy:
  <https://www.apache.org/licenses/LICENSE-2.0>. Earlier revisions of this file offered only the link,
  which is weaker than what §4(a) asks for.)
- **§4(b) statement of modification**: `models/organs.onnx` is **not the original upstream file**, but a
  computation graph exported from the upstream weights via `torch.onnx` (the exporter string is
  `pytorch 2.11.0`). This modification is hereby stated.
- **§4(d) NOTICE pass-through obligation: not triggered** — this clause is premised on the upstream Work
  carrying a NOTICE file, whereas the repository roots of both TotalSegmentator and nnU-Net **contain no NOTICE
  file** (verified via the GitHub Contents API). This project therefore **asserts — and fabricates — no** NOTICE
  obligation.
- **Academic citation** (requested by the upstream READMEs; an academic convention rather than a license
  requirement):
  - Wasserthal J. et al. *TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images.*
    Radiology: Artificial Intelligence, 2023.
  - Isensee F. et al. *nnU-Net: a self-configuring method for deep learning-based biomedical image
    segmentation.* Nature Methods, 2021.

**To be confirmed**: whether the 45 KB **weight-free computation graph** constitutes a *Derivative Work* of the
upstream weights (Apache-2.0 §1) cannot be settled from the primary source — §1 also provides that "Derivative
Works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of,
the Work," whereas this graph carries only the layer-naming structure and no weight values. This project handles
it **conservatively, treating it as if it does constitute one** (i.e. discharging the §4 obligations above),
while noting honestly that this characterization itself remains unsettled.

---

## 2. Datasets (**none** are distributed with this repository)

| Data | License | Use | Primary source |
|---|---|---|---|
| **TotalSegmentator-CT-Lite** (single case `s0029`) | **CC-BY-4.0** | segmentation validation in `experiments/seg_validate.py` | [HuggingFace: YongchengYAO/TotalSegmentator-CT-Lite](https://huggingface.co/datasets/YongchengYAO/TotalSegmentator-CT-Lite) |
| **TotalSegmentator upstream original dataset** (the parent of CT-Lite) | **CC-BY-4.0** | same as above (indirectly) | [Zenodo 10047292](https://zenodo.org/records/10047292) |
| **RIDER Lung CT** (local `肺癌/`) | TCIA public dataset; **de-identified via CTP** (`PatientIdentityRemoved=YES`) | local data for the full regression suite | [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/) |

**Attribution requirement under CC-BY-4.0**: figures produced using the above data (`seg_confusion.png` and
others under `experiments/results/`) already carry their source and license in `README.md`,
`docs/technical_report.md`, and `experiments/README.md`. **This repository redistributes none of the above data
itself.**

---

## 3. Runtime Dependencies (used only via `import`; their source is **not** vendored)

Versions are pinned by `requirements.txt` / `experiments/requirements-experiments.txt`.

| Component | Version | License declared upstream | Primary source |
|---|---|---|---|
| **PySide6** (Qt for Python) | 6.11.0 | **LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only** (licensee's choice; a separate commercial license is also offered by The Qt Company). This project uses **LGPL-3.0-only**. | [PyPI](https://pypi.org/project/PySide6/) · [Qt licensing](https://www.qt.io/licensing/) |
| **shiboken6** (the PySide6 binding runtime, installed automatically with PySide6) | 6.11.0 | same as PySide6 | [PyPI](https://pypi.org/project/shiboken6/) |
| **pydicom** | 3.0.2 | **MIT** | [PyPI](https://pypi.org/project/pydicom/) |
| **NumPy** | 2.2.6 | **BSD-3-Clause** | [PyPI](https://pypi.org/project/numpy/) |
| **SciPy** | 1.15.3 | **BSD-3-Clause** | [PyPI](https://pypi.org/project/scipy/) |
| **scikit-image** | 0.25.2 | **BSD-3-Clause** (primary); **not a single license** — also includes specific files covered by BSD-2-Clause and MIT | [PyPI](https://pypi.org/project/scikit-image/) |
| **ONNX Runtime** | 1.23.2 | **MIT** | [PyPI](https://pypi.org/project/onnxruntime/) |
| **NiBabel** | 5.4.2 | **MIT** (core package); its `COPYING` additionally covers bundled components under BSD-3-Clause / PDDL-1.0 / a custom permissive license | [PyPI](https://pypi.org/project/nibabel/) |
| **Matplotlib** | 3.10.8 | **Matplotlib License Agreement** (its own license, adapted from the PSF license, BSD-compatible). **Note: there is no corresponding SPDX identifier; do not label it as `PSF-2.0`.** | [PyPI](https://pypi.org/project/matplotlib/) |
| **remotezip** | 0.12.3 | **MIT** | [PyPI](https://pypi.org/project/remotezip/) |

Copyright and licenses of the above components remain with their respective authors; this project has not
modified their source and does not distribute their code or binaries with the repository.

**LGPL scope, stated for the form this repository actually takes.** Earlier revisions of this file left the
question open and told the reader to settle it "before publishing the repository". The repository has since been
published, so leaving a pre-publication gate standing would misdescribe the situation. What can be stated as
fact, and what remains a legal judgment, are separated here.

*What is observable about the current publication form.* This repository distributes its own Python source only. It
`import`s PySide6 at runtime, and **contains no Qt or PySide6 binaries, no statically linked artefacts, and no
bundled or frozen executable** — a reader obtains PySide6 themselves from PyPI under the licence of their
choosing. LGPL-3.0 obligations attach to *conveying* the library or a combined work containing it; this
repository conveys neither. **Our reading — this project's understanding, not legal advice — is therefore that
publishing in this form triggers no LGPL distribution obligation**, and that is why no LGPL licence text is
reproduced here, in contrast to `licenses/APACHE-2.0.txt`, which is reproduced because we read that obligation
as attaching. Whether a licence obligation attaches is a question of legal application, not a fact this file can
settle; only the sentence about what the repository does and does not contain is a matter of observation.

*Open, and deliberately not answered here.* If this project were ever shipped as a bundle that includes Qt —
a PyInstaller/py2app build, a wheel with binaries, an installer — then LGPL-3.0 §4 would apply to that bundle,
and its requirements (notably that the recipient be able to relink or replace the Qt library, plus the
accompanying notices and licence text) would have to be reconciled with this repository's `LICENSE`, which is
review-only and proprietary in style. Our reading is that §4 is designed to permit exactly that combination —
proprietary application code over a replaceable LGPL library — but **this is a legal question and nothing here
is legal advice**; it would need counsel before any binary distribution. No such distribution exists or is
planned, so the question is recorded as a known boundary of the current form, not as an unmet condition.

---

## 4. This Project Itself

The copyright in this project's own code is stated in [`LICENSE`](LICENSE): © 2026 Sheng Chao (盛超) and
Lai Shengsheng (赖胜圣), jointly held by the two co-copyright-holders.

---

*This file was generated by cross-checking the code against upstream primary sources; last verified: 2026-07-17.
Upstream licenses may change over time; when re-checking, defer to the upstream `LICENSE` text as it stands at
that time.*
