# Changelog · Code-Review Notes

This file collects one systematic round of defect investigation (2026-07) on the **Medical Imaging Workstation Pro + Reconstruction Lab**.

## Investigation method

Every issue followed the same disciplined loop, never guesswork:

1. **Hypothesis → real reproduction**: using real data or a deliberately malformed DICOM/project file, run it for real under offscreen Qt to first prove that the issue genuinely triggers (crash / mis-ordering / residual state), then touch the code.
2. **Fix → re-test**: after the change, verify with the same reproduction script that the issue is gone.
3. **Regression lock-in**: every issue is written into `tests/test_gui.py` to prevent regressions.
4. **One issue per commit**: before committing, check that no PHI / large files slipped in (`肺癌/`, `*.dcm`, `organs.onnx.data` are all .gitignore'd).

The regression suite grew from an initial ~10 checks to **64 checks (20 test functions)**; `python tests/test_gui.py` exit code 0 = all pass (later engineering work raised this to **102 checks**, see the section below).

---

## Fixed defects

### Crashes

| Defect | Trigger scenario | Commit |
|------|----------|------|
| Undo out-of-bounds on case switch | After switching to a smaller case, `Ctrl+Z` undo of segmentation makes the old slice index go out of bounds → `IndexError` | `48cf8e0` |
| System-matrix build exception freezes the UI | When `build_system_matrix` raises, the modal progress dialog never closes → UI freezes dead, and the exception also propagates up into the button slot | `22219b2` |
| Mixed-shape DICOM load crash | Slices within one series have inconsistent matrix sizes, or a completely missing `SeriesInstanceUID` lumps multiple series into one group → `np.array` stacking `ValueError` | `7f1ff72` |
| Null numeric-tag crash | The `getattr` default only takes effect when the tag is absent; a malformed DICOM leaves RescaleSlope/PixelSpacing/SliceThickness empty (`None`) → `float(None)` crashes and the series won't open | `6175a46` |
| Multi-frame / corrupt DICOM crash | A multi-frame single file has a 3D `pixel_array` that stacks into 4D and crashes on unpacking; truncated PixelData / missing codec → one bad slice brings down the whole volume | `654023b` |
| Malformed annotation JSON freezes reading | Loading a project with missing fields / empty points / wrong-length rect → `_render_annotations` crashes on every refresh = reading frozen | `bfbab63` |

### Security / privacy

| Defect | Trigger scenario | Commit |
|------|----------|------|
| Export-filename path traversal | `PatientID="../PWNED"` is concatenated straight into the save path → files written outside `Exported_Lesions`; if it contains `/`, it fails silently and loses annotations | `a7c92d6` |
| De-identification leaks prior study date | The dual-series comparison V2 title still shows the prior `StudyDate` in de-identified mode | `48cf8e0` |

### Interaction / state consistency

| Defect | Commit |
|------|------|
| Slice slider dead in reconstruction mode (`on_slice_changed` refreshed only the non-reconstruction state) | `4fddc11` |
| Chained source image `_last_recon_img` lingers after a slice change | `4fddc11` |
| Segmentation undo stack lingers after reset | `48cf8e0` |
| Cine playback not stopped on case switch | `48cf8e0` |
| DICOM sort key mixed float (z-coordinate) / int (instance number), scrambling anatomical order → changed to a series-level unified decision | `dab0f44` |
| Closing the window did not cancel background AI inference (8.8 GB/100 s lingering + the completion callback fires on an already-torn-down window → RuntimeError) | `e715d57` |
| Scope checkboxes not translated after a language switch (Chinese left over in English mode) | `85bc022` |
| Legend and mask overlay inconsistent in show/hide (legend still lists organs after Anno is turned off) | `bd33a9f` |

### Hardening (aligning with existing defensive conventions / consistency)

| Hardening | Commit |
|------|------|
| DMR/ART/SIRT reconstruction outputs pass through `_finite_clip` to guarantee finiteness, aligning with DFR's `nan_to_num` convention (a degenerate sinogram no longer produces a NaN black image + NaN RMSE) | `ecf390b` |
| `recon.build_system_matrix`'s `_mp.cpu_count()` → `os.cpu_count() or 4`, guarding against `NotImplementedError` on extreme platforms | `a7c92d6` |

### Distilled defensive utilities

All subsequent DICOM/filename handling should go through (`main.py` MedicalViewer):

- **`_dcm_float(ds, tag, default, idx=None)`** — safely reads a numeric DICOM tag (absent / empty / non-numeric all fall back uniformly, ruling out `float(None)`).
- **`_safe_name(s, fallback)`** — sanitizes a patient identifier into a safe filename fragment (strips path separators and `..`, ruling out path traversal; applied consistently on both the save and load sides, round-trip consistent).
- **`_valid_anno(a)`** — validates annotation structure by type, filtering out malformed / old-version entries at load time.

`_read_dicom_dir` triple disk-read hardening: pick the series with the most slices → keep the majority shape by `(Rows,Columns)` → series-level sort key.

---

## Engineering & architectural decoupling (2026-07)

A round of engineering-maturity and architecture improvements beyond the defect investigation. Each step preserves **zero behavioral regression** (full regression all-pass + the data-independent CI subset), one step per commit, checked for no PHI before committing.

### Engineering (5 items)

| Item | Content | Commit |
|------|------|------|
| CI + packaging | Added `pyproject.toml` (metadata/dependencies/tool config) + GitHub Actions (push/PR runs ruff + the data-independent test subset, offscreen Qt, needing no real data or the 119 MB weights); split out a `SKIP_REAL_DATA` subset of the tests so CI can run | `887e2f2` |
| ruff + type annotations | Configured ruff (ignoring the deliberate compact single-line style, focusing on real issues) + fixed all genuine lint; added full type annotations to `recon.py`/`ai_engine.py` | `029b572` |
| Table-driven i18n | `update_language` changed from a ~110-line wall of ternary `setText` calls to a `(widget, English, Chinese)` table + a `_retranslate_combo` helper, curing the risk of missed translations at the root | `6b0530b` |
| De-hardcoded entry point | Removed the startup hardcoded auto-load of `肺癌/` (a PHI-leak surface); switched to a `--data DIR` CLI argument (argparse entry `main()`), empty by default | `9d4ff0b` |
| Coverage quantification | coverage wired into pyproject + CI; full-suite coverage **≈66%** | `1486efb` |

### Architectural decoupling (3 blocks, 4 Qt-free pure-compute modules in total)

Addressing the weakness of "compute cores tangled inside the God object, impossible to unit-test in isolation", extracted along the same pattern: **pure logic → Qt-free standalone module → mixin/thread reduced to a thin wrapper → isolated unit tests with synthetic data (into the CI subset)**.

| Module | Extracted from | Logic | Isolated unit test | Commit |
|------|----------|------|----------|------|
| `quantify.py` | `AnnotationMixin` | Organ quantification (volume mL / mean HU) | `test_quantify` (100% coverage) | `ed47ab6` |
| `segmentation.py` | `AutoAIEngineThread` | AI mathematical fallback (lung connected-component segmentation) | `test_lung_fallback` | `e2a9857` |
| `mpr_geometry.py` | Consolidated coordinate conventions previously scattered across three places | MPR coordinate conversion (hover↔voxel↔crosshair) + dual-series z registration | `test_mpr_geometry` | `33fec02` |

(`recon.py` was the earliest precedent: the reconstruction algorithms had no Qt dependency to begin with, so the lab scripts can `import` it directly.)

Regression suite **64 → 102 checks**; GitHub CI green 9 times in a row.

---

## Documentation fidelity calibration (2026-07)

Before tidying the repository into a presentable state, a round of **fidelity calibration** was done across all documentation — on the principle that "every number and claim in the docs must match the actual state of the code, without exaggeration or fabrication". Aligned item by item against measured values:

| Calibration item | Original wording | Changed to (measured) |
|--------|--------|--------------|
| Code size | ~4,700 lines | ~4,100 lines of application code / 13 modules |
| Regression suite | 80 items / 80-check | 102 checks |
| Coverage | ≈66% | ≈67% (`ai_engine` 87% / `main` 82%) |
| Directory structure | Missing 3 new modules | Added `quantify`/`segmentation`/`mpr_geometry`, grouped as "UI layer / pure compute / resources" |

In addition:

- **Packaging converged to honesty**: under the flat module layout, the wheel does not include resources such as `style.qss`/the model, so `pip install` yields a degraded, resource-less application; therefore the over-promise of `[project.scripts]` was removed, making clear that `pyproject.toml` serves project metadata/dependencies/tool config and that the application runs via `python main.py`.
- **Authorship claim de-solo'd**: the README's "written solely by me" → "designed, led in development, and validated by me". The code is AI-assisted development; the commit history faithfully retains the `Co-Authored-By` attribution and makes no "independent/solo authorship" claim.
- Added `LICENSE` (all rights reserved, consistent with the software-copyright position) + bilingual EN/中文 navigation in the README; third-party component attributions verified one by one.

Principle: **better to understate than to distort.** This project's strongest asset is "verifiable honesty", and any padding backfires on it.

---

## Known limitations (recorded faithfully, unfixed)

- **MPR anisotropy uncorrected**: coronal/sagittal planes are displayed 1:1 by pixel; when slice thickness ≠ in-plane pixel spacing, the geometric proportions are distorted. A fix would require reworking the "scene coordinate = voxel index" mapping that runs through hover/measurement/cross-hairs — a non-surgical change whose risk outweighs its benefit, so it is recorded as a limitation. Caliper measurements use real mm, so **the measured values are correct**; only the displayed proportions do not match anatomy.
- **AI mask overlay is axial-only**: coronal/sagittal planes do not display the organ-segmentation overlay.

See "Limitations" in `README.md` for details.

---

## Positioning statement

This software is an **imaging teaching / research tool** — **not a certified medical device, and not for clinical diagnosis.** The fixes above improve software robustness and data safety; they do not constitute any clinical-compliance certification.
