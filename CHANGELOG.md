# Changelog · Code-Review Notes

This file collects the systematic rounds of defect investigation on the **Medical Imaging Workstation Pro + Reconstruction Lab** — a robustness round (2026-07) and a correctness round (2026-08).

## Annotation text grew with the zoom until it covered what it described (2026-08-27)

`QGraphicsTextItem` defines its point size in *scene* coordinates, so ROI statistics and ruler
labels scaled with the view. A 512² series filling the window sits at roughly 2.5–3.5×, which
turned a 10 pt label into the equivalent of 35 pt: the readout covered the very anatomy it was
measuring and ran past the right edge of the image. Both labels now set
`ItemIgnoresTransformations`, pinning the glyphs to screen pixels; zoom moves the anchor and
nothing else.

The overflow test that decided whether to flip the text to the other side of the ROI compared
against a hard-coded `95` scene units — a figure estimated at one particular zoom. It is now
derived from `QFontMetricsF` and divided by the current scale, because the label's screen size is
fixed while the image bounds are in scene units, so the two can only be compared after converting.

The first version of the regression test measured the wrong quantity: for an item with
`ItemIgnoresTransformations`, `sceneBoundingRect()` is itself a function of the view transform,
so it reported a 5× difference where there was none. `boundingRect()` is the one whose item
coordinates *are* screen pixels. The second version then passed against a mutation that restored
the hard-coded constant, because it only exercised magnification — where an oversized constant is
accidentally conservative. Adding a 0.5× case fixed that: shrinking makes the label *wider* in
scene units than the constant claims, so the bounds check wrongly says it fits. All three
mutations now bite — removing the flag, removing the edge flip, and restoring the constant.

## The 3D preview dialog printed its only numbers in near-invisible text (2026-08-27)

`QDialog` is a top-level window: it does not inherit the stylesheet the app sets on the
`MedicalViewer` instance (`ui_builder.py`), so its background is the system light grey. It *does*
inherit the parent's palette, whose foreground was chosen for the dark main window. The mesh
preview then styled its labels with the dark-theme ramp — `#C9D1D9` for the statistics line,
`#8B949E` for the two captions. Measured against the actual dialog background, the statistics line
came out at roughly **1.4:1**, and that line is the dialog's *only* quantitative output: surface
area, volume, sphericity, face count. The "View" label had no explicit colour at all and inherited
the dark palette's foreground, landing at **2.03:1**.

Rather than patching each label, the two self-built dialogs now declare their own foreground once
(`QDialog, QLabel { color: … }`), so a label added later is readable by default; the two muted
captions keep a dimmer but still-compliant tone. Worst measured contrast in the dialog is now
**5.56:1**, above the 4.5:1 WCAG AA threshold for body text.

The regression test opens the real dialog, reads each label's effective foreground and the
dialog's actual background, and computes the WCAG ratio — no source-string matching, so any
future colour that fails on that background is caught regardless of how it is spelled. Three
mutations confirm it bites (restoring either dark-ramp colour, or dropping the dialog-level
declaration). The first version of that test passed while measuring nothing: `QLabel.pixmap()`
returns an empty `QPixmap` rather than `None` in PySide6, so the `is not None` guard skipped every
label and the contrast check kept its initial sentinel. It now asserts how many labels were
actually measured, which is what turned it from decoration into a check.

## AI segmentation was mirrored on the product's own path, and no test could see it (2026-08-27)

`organs.onnx` comes from TotalSegmentator/nnU-Net, whose volumes are normalised to **RAS** —
the two in-plane axes run towards the patient's Right and Anterior. The product's `volume_hu`
comes from DICOM, and AI only runs when canonical orientation holds, i.e.
`ImageOrientationPatient = [1,0,0,0,1,0]`: columns towards **Left**, rows towards **Posterior**.
Both in-plane axes are inverted between the two conventions. `ai_engine` flipped neither.

**Measured against CT-Lite ground truth on one case**, the cost of that was not a slight loss
of accuracy but systematic mislabelling:

| | liver | lung UL(L) | lung LL(L) | lung UL(R) | lung ML(R) | lung LL(R) |
|---|---|---|---|---|---|---|
| before (no flip) | 0.181 | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** |
| flipping only L/R | 0.880 | 0.223 | 0.173 | 0.000 | 0.006 | 0.119 |
| **after (both axes)** | **0.965** | **0.979** | **0.989** | **0.792** | **0.987** | **0.992** |

Paired organs were swapped outright — the product labelled the patient's right lung as the left
one. Spleen and both kidneys land at 0.961 / 0.987 / 0.983 after the fix. The intermediate row is
kept deliberately: flipping only left/right looks like a fix (merged-lung Dice 0.977) while the
lobes stay scrambled, because the fissures are oblique and the anterior/posterior axis was still
reversed.

**Why every existing test passed.** None of the segmentation evidence goes through DICOM.
`seg_validate` and `seg3d_teacher` read NIfTI and normalise to RAS themselves; `516b7cb`, which
established the label mapping, ran on that path — its conclusion was right and remains right, it
simply never covered the path the product actually runs. Worse, `seg_multi` and `seg_spacing`
*do* call `ai_engine` at runtime, but they feed it `load_zhw` output, which is already RAS — so
they too were measuring the correct orientation and reported healthy Dice. The defect sat exactly
in the gap between the two.

That gap is now closed by making the convention explicit instead of assumed: `AutoAIEngineThread`
takes `inplane_axes`, `'lps'` (the product's default, flipped in and out as a pair) or `'ras'`
(already model orientation, passed through untouched), and rejects anything else at construction
rather than silently guessing. The three experiment call sites now declare `'ras'`, so their
behaviour — and the committed CSVs produced on it — are unchanged.

**Cached masks needed their own guard.** A mask written before this fix is mirrored in-plane, yet
its SeriesInstanceUID, shape and geometry fingerprint are all unaffected, so the three existing
guards would restore it without complaint. `axis_contract` is now recorded in the `.npz` and
checked by the pure function `mask_axis_contract_ok`; a cache without it fails closed. The one
cache present locally predates fingerprints entirely and was already being rejected.

The regression test drives `_run_body` with a direction-sensitive synthetic volume and a stub
model, asserting the volume in, the labels out and the confidence map are all flipped as a set.
Three mutations confirm it bites: degrading the flip to identity (5 failures), flipping only
left/right (5), and dropping the flip on the way back (1).

## The HU gate was right; the local data was under-declared (2026-08-27)

The local RIDER series loaded as viewer-only: no CT presets, no ROI quantification, no AI, no
3D tracking, no follow-up comparison. This had been recorded in six places as correct behaviour
and locked by tests, and re-measuring it confirmed the gate is right. Every one of the 233 slices
is `DERIVED\SECONDARY\PROCESSED` with no `RescaleType`, and DICOM PS3.3 C.8.2 lets an omitted
Rescale Type imply HU only for an `ORIGINAL` classic CT image. `_slice_has_standard_hu` fails
closed, exactly as specified.

**But the values are standard HU, and that is measurable.** Reconstructing the histogram over a
24-slice sample puts the air peak at **-1025 HU** and the soft-tissue peak at **-5 HU**, with the
padding-excluded range spanning **-1024..3071** — precisely the 12-bit signed CT interval, with
`slope=1 / intercept=-1024` identical across all 233 slices. The unit was never in doubt
numerically; only the declaration was missing.

**The fix belongs on the data side, and the product is unchanged.** `dicom_geometry` already
accepts an explicit `RescaleType=HU` regardless of `ImageType` — a `DERIVED` image may legitimately
declare its units, and the existing test suite already covered that case. So `tools/declare_rider_hu.py`
verifies the values against physical anchors and writes a derived copy that adds exactly one tag.
`ImageType` stays `DERIVED`: that is true of the series, and rewriting it to `ORIGINAL` would be the
actual falsification. Not one line of product code changed; the copy reports
`hu_calibrated=True` with all four geometry contracts green.

**One difference is not ours and is disclosed rather than glossed.** pydicom does not write retired
Group Length elements (`filewriter.py`: `if tag.element == 0 and tag.group > 6: continue`, citing
PS3.5 §7.2), so the copy loses 7 such elements per file, 1631 in total. Dropping them is also the
only coherent option — adding an element to group `0028` invalidates the old `(0028,0000)` length,
so preserving it would produce a self-contradictory file. Group Length carries no clinical or
geometric meaning and no reader depends on it.

The first version of the script only checked `PixelData`, `ImageType` and `RescaleType`, which is
how the Group Length difference went unnoticed until a separate comparison surfaced it. It now
asserts the **entire** tag difference per file — exactly one added tag, removals restricted to Group
Length, zero value changes — and a five-way mutation test confirms the assertion catches a changed
`WindowCenter`, an extra tag, a deleted `SliceThickness` and a forged `ImageType` rather than merely
passing. The copy is gitignored and no test reads it: the suite still loads the original `肺癌/`, so the copy
itself moves no count; the 13 checks this round adds over the tool's own assertions do. (The
suite totals are stated once, in the README, rather than repeated here.) Loading the copy does make the series
AI-eligible, which starts inference automatically — worth knowing before opening it on a CPU-only
machine.

## ASD-POCS reaches the GUI, and its iteration list had to differ (2026-08-26)

The TV baseline added earlier lived in `recon.py` and was called only by
`experiments/recon_tv.py` and the tests. Five documents had been amended to disclose that —
"ASD-POCS is implemented in that module but currently has no GUI entry, so its measurements
characterise an experiment-only solver rather than a user-exposed feature." That disclosure was
honest, but it opened an exception in the one claim this project's reconstruction studies rest
on: that the object under test is the code users actually run. The exception is now closed by
wiring the solver in rather than by keeping the caveat, and all five disclosures are rewritten.

**The iteration dropdown could not be reused, and measurement is why.** It offered 10 / 20 / 50,
chosen for ART and SIRT. One ASD-POCS iteration is a relaxed ART sweep *plus* `n_grad` TV
steepest-descent steps, so it converges an order of magnitude slower. Measured **on the laboratory's own default
path** — `shepp_logan(256)` through `prepare_small_image` to 32×32, 180°×1×, noise-free, i.e. the
default entries of `cb_matrix_size` and `combo_oversample`:

| | FBP | 10 | 20 | 50 | 100 | 150 | 300 |
|---|---|---|---|---|---|---|---|
| in-circle RMSE | 0.0995 | **0.1460** | **0.1336** | 0.0672 | 0.0142 | 0.0034 | 0.0003 |

Cost is linear in rounds: ≈8.8 ms/round at the 32×32 default (300 rounds ≈2.7 s) and ≈37 ms/round
at 64×64. Absolute seconds vary with machine and load and are given as an order of magnitude only;
an earlier revision of this entry printed a per-configuration wall-clock row whose implied per-round
cost swung 13% between adjacent rows, which cannot be real.

**An earlier revision of this table was measured in the wrong frame.** The numbers were taken from
`experiments.recon_study.get_phantom` at 64×64 — the study phantom, at a size the user has to
select — while the argument they support is about a GUI feature. The tier boundary is unchanged
under the corrected measurement (50 is still the first tier that beats FBP, at 32×32 and at 64×64
alike), but evidence for a claim about the shipped path has to come from that path.

At the two shortest settings a correct implementation is **worse than FBP**, and 50 is the first
that beats it. Shipping the shared list would have made the reconstruction laboratory — whose
entire purpose is comparing algorithms — display ASD-POCS as the worst of them. The iteration
options are therefore bound to the method (`ReconLabMixin.ITER_OPTIONS`), ASD-POCS getting
50 / 100 / 150 / 300 with 150 as default, and `test_recon_iter_options_contract` locks that: the
options table must match the method dropdown parsed out of `ui_builder.py` item for item, and
ASD-POCS's minimum may not drop below 50. Both halves matter — an entry missing from the table
silently falls back to ART's list rather than failing.

Full suite 758 → **784**, `SKIP_REAL_DATA=1` subset 667 → **693**, both exit 0, ruff clean.

## Pre-commit local contract snapshot (2026-08-26)

These changes were verified in the 2026-08-26 pre-commit local freeze-candidate snapshot. As of that snapshot they had not been committed, pushed, released, or covered by remote CI.

- **Coronal and sagittal views were displayed with the head–foot axis inverted; they are now superior-at-top.** `mpr_geometry.hover_to_voxel` / `voxel_to_crosshair` map the view's vertical pixel as `z = Z - 1 - py` instead of `z = py`, the clinical renderer applies `np.flipud` to both the HU plane and the mask overlay, and `interaction.py`'s hover read-out and cross-hair placement use the same convention, so the three stay consistent. This is a user-visible change to what the two reformatted planes look like — axial is unaffected. `test_dicom_landmark_orientation` pins it by driving an asymmetric bright landmark through the synthetic DICOM loader and the real render path and asserting all six of A/P/L/R/S/I. **This change shipped in the commit that introduced the geometry contract without being recorded in that commit's message or here; the entry is added retroactively.**
- The supported DICOM contract is now classic single-frame CT only. Enhanced CT, non-CT, and multi-frame input fail closed before pixel decoding. The historical multi-frame row below records the earlier crash fix, not the current support policy.
- Spatial geometry is accepted only when every slice independently has finite, unit-length, orthogonal IOP direction cosines and the series-level IOP/IPP, PixelSpacing, and projected slice positions prove the required capability. Slice spacing no longer falls back to `SpacingBetweenSlices`, `SliceThickness`, or a 1 mm default; MPR, physical quantification, mesh, AI, and comparison fail closed independently when their contracts are not met.
- Standard HU is all-or-nothing per series: every retained slice needs finite slope/intercept plus either explicit `RescaleType=HU` or the classic CT `ORIGINAL` / non-`LOCALIZER` / non-multi-energy guarantee. `DERIVED` images without explicit HU, unknown/non-HU units, mixed-unit series, and the unsupported multi-energy contract remain raw-value viewer-only.
- Capability loss now clears stale execution state, not just enabled flags: invalid HU resets named CT presets and the renderer independently ignores disabled preset text; invalid in-plane spacing cancels any ruler preview and synchronises the Ruler button, active tool, and every view back to Pointer. Irregular z spacing alone does not disable a still-valid axial 2-D ruler.
- A successful series change now clears the previous series' HU probe only after the new volume has been accepted, then rebuilds the centre-voxel HUD with the new series' unit. A directory/decode failure that retains the old series also retains its probe and HUD.
- Project persistence now distinguishes an AI-pending all-zero placeholder from a confirmed global mask clear. Placeholders do not create false cache hits; a confirmed clear persists a provenance-bound zero NPZ, invalidates older AI callbacks, survives reload, is cancelled by Ctrl+Z, and retains its pending intent after save failure. This contract covers the confirmed global clear action, not a mask erased voxel-by-voxel to zero.
- Mask and annotation restoration now requires a geometry fingerprint in addition to UID and shape; legacy cache entries without the fingerprint are rejected.
- Project persistence validates non-empty UID/fingerprint and mask shape before opening targets, serialises JSON/NPZ to temporary sibling files, and uses per-target `os.replace`. Failures return `False` without a success message or target truncation; no cross-file transaction atomicity is claimed.
- De-ID shows `ANON` on screen and uses a per-load random `ANON-…` alias plus collision-safe suffixes for explicit exports, but explicitly does not anonymise source DICOM tags, burned-in pixel text, or internal project/cache identifiers.

## Investigation method

Every issue followed the same disciplined loop, never guesswork:

1. **Hypothesis → real reproduction**: using real data or a deliberately malformed DICOM/project file, run it for real under offscreen Qt to first prove that the issue genuinely triggers (crash / mis-ordering / residual state), then touch the code.
2. **Fix → re-test**: after the change, verify with the same reproduction script that the issue is gone.
3. **Regression lock-in**: every issue is written into `tests/test_gui.py` to prevent regressions.
4. **One issue per commit**: before committing, check that no PHI / large files slipped in (`肺癌/`, `*.dcm`, `organs.onnx.data` are all .gitignore'd).

The regression suite grew from an initial ~10 checks to **64 checks (20 test functions)**; `python tests/test_gui.py` exit code 0 = all pass (later engineering work raised this to **102 checks**, and the 2026-08 round to **515**, see the sections below).

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
- **Authorship wording corrected**: the README no longer uses a solo-authorship formulation; it now describes the individual project role without conflating that role with the jointly owned copyright status.
- Added `LICENSE` (all rights reserved, consistent with the software-copyright position) + bilingual EN/中文 navigation in the README; third-party component attributions verified one by one.

Principle: **better to understate than to distort.** This project's strongest asset is "verifiable honesty", and any padding backfires on it.

---

## Correctness round (2026-08)

A second round, run under the same loop: reproduce first, then fix, then lock in with a regression check. Grouped by how each defect was found, because that turned out to be the more useful classification.

### Found by measuring, not by reading

- **The inference engine skipped nnU-Net's spacing resampling.** `organs.onnx` is an nnU-Net v2 export whose inference contract begins by resampling to the training spacing (1.5 mm isotropic); the engine fed each series at its native spacing (`grep resample|zoom|spacing` returned nothing, and the ONNX graph has no `Resize` op). Every Dice figure the project had published was measured at exactly 1.5 mm — the one condition where the mismatch is zero — so accuracy elsewhere was *unmeasured*, not merely lower. Quantified first (mean Dice 0.9219 → 0.7995 at twice the training spacing, small organs collapsing first and non-monotonically), then implemented. Validated across **20 paired cases**: 0.684 → 0.840, improving in 20/20, Wilcoxon *p* = 1.9×10⁻⁶. Inference on the bundled series dropped from 100 s / 8.8 GB to 37 s / 3.0 GB. The step is not free: mask boundaries are decided on the 1.5 mm grid and become stair-stepped when mapped back to a finer original — this is stated in the UI, the model card and the manual.
- **3-D tracking silently destroyed the AI segmentation.** `handle_3d_track_requested` assigned to `volume_mask` wholesale, so one tracking action erased all 24 organ labels — and `save_project` then persisted the result. Evidence was on disk, not in the code: the cached mask held 3,248,369 voxels of which 100% were the manual-tracking label, with no organ remaining. Tracking now writes only its own layer, and both it and "clear mask" push a whole-volume undo snapshot; clearing additionally requires confirmation that states what will be lost.
- **`SliceThickness` was used where slice spacing was meant.** Detector collimation is not the reconstruction interval; under overlapping reconstruction the two differ by a factor of two, which would scale the z axis wrongly. The bundled series happens to have both equal to 1.25 mm, so this could never surface locally. The 2026-08 correction first derived spacing from consecutive positions with metadata fallbacks; the 2026-08-26 pre-commit snapshot contract above is stricter and accepts z spacing only from finite, unique, uniformly spaced patient-space projections, with no `SpacingBetweenSlices` / `SliceThickness` fallback.

### Found by writing assertions

Three defects surfaced only because a test asked "what happens when this fails?" — none were visible by reading the code.

- **The probe read-out kept a stale value.** The whole body of `measure_hu` sat inside `try/except: pass`, so an out-of-range coordinate left the label showing the *previous* reading, coordinates included. It looked exactly like a valid measurement. For a number read off for interpretation, a stale display is worse than a blank one; it now clears.
- **The model card crashed on a damaged CSV.** Two layers: `csv.DictReader` yields `None` for a missing column and `float(None)` raises `TypeError`, not `ValueError`; and a file containing NUL bytes — the typical shape of a truncated write — makes the csv module raise its own `csv.Error`, which is not any builtin type. Either one propagated to the UI. The card's entire value rests on being trustworthy, so all eight read paths were hardened.
- **Annotations with numeric ids could never render or be deleted.** The id travels through `setToolTip` (which accepts only `str`) and comes back through `annotation_deleted = Signal(str)`, but `_valid_anno` checked only for the key's presence. A numeric id passed validation, persisted to disk, then threw `TypeError` inside the render layer's exception guard — invisible, undeletable, and one console warning per refresh. Normalised at both entry points rather than patched at the render site.

### Statistics and honesty

- **A confidence-interval overlap test was replacing a paired one.** Teacher and student run on the same cases, so the comparison is paired; judging by whether two bootstrap CIs overlap is a classic false negative. Replaced with a paired bootstrap CI plus Wilcoxon signed-rank. A constructed scenario reproduces the failure: two overlapping CIs whose paired difference interval lies entirely below zero at *p* = 1.6×10⁻¹¹.
- **The 21-organ Dice was still n = 1.** Now measured over 20 cases: patient-level mean **0.909, 95% CI [0.889, 0.927]**, the original single case at 0.922 sitting inside the interval on the optimistic side. Per-organ reliability spans 0.43 — liver 0.982 against prostate 0.554 — which the aggregate hides entirely. The right upper lung lobe at 0.773 is independently corroborated by a separate study measuring 0.727 on a different draw of cases.
- **Single cases mislead in both directions.** Three instances now: lung lobes 0.956–0.991 → 0.887 over 57 cases (optimistic), the spacing fix +0.064 → +0.155 over 20 cases (pessimistic by 2.4×), the 21-organ figure 0.922 → 0.909 (mildly optimistic). The lesson recorded in the technical report is not that single cases flatter, but that the direction of the bias cannot be known in advance.
- **Study III's noise-free condition was undeclared.** The learned-reconstruction study reports **1.67%** at the 20%-of-lesion threshold across 60 noise-free paired phantoms, and 0% at 30%/50%. Photon noise was not tested, so the result is now labelled neither an upper nor a lower bound for low-dose CT; direction and magnitude remain unmeasured, and low SNR is not claimed as the dominant driver.
- **The 25-class palette contradicted the measured label map.** Fifteen of sixteen colour comments named the wrong organ, the lung lobes were coloured left-for-right, and seven classes had no colour at all — rendered as one shared grey despite right kidney 0.985 and left kidney 0.977 being among the best-segmented structures. All 24 classes now have distinguishable colours (minimum pairwise distance 12 → 43).

### Testing

The suite grew from 325 to **515 checks** (424 data-independent, run in CI). Coverage 79% → **89%**. The layers that had never been exercised moved most: `recon_lab` 44% → 89%, `annotation_lab` 74% → 84%, `interaction` 64% → 79%. Matrix reconstruction is tested through a substituted system matrix — building a real one costs O(n²) Radon transforms and the cached 32² matrix is 23 MB, which CI does not have — since numerical correctness is already covered at the pure-function layer.

---

## Evaluation-path round (2026-08)

A third round, triggered by a number that did not add up rather than by a suspected defect. Its outcome revises conclusions the project had already committed to this repository, so the retractions are recorded alongside the findings. ("Published" in this file means committed here; nothing in this project has been peer-reviewed or published in a venue.)

### Found by refusing to accept a gap

- **A model–inference-path interaction suppressed the student's foreground.** Training reported `val patch-Dice` 0.8186 while whole-volume scoring gave 0.4903; training-size sliding takes the identical weights to **0.7457**. Zero-padding with content held fixed removes 99.3% of predicted foreground. The evidence points to tensor extent / zero-padding × `InstanceNorm3d` × fixed-size/no-augmentation training, and targeted controls make several alternatives inconsistent with the observation. No replacement-normalisation experiment was run, however, so `InstanceNorm3d` is a supported mechanism rather than a uniquely established root cause; this is not “evaluation bad, model fine.”
- **This retracts a published causal explanation.** The compression study had attributed the student's failure — five-lobe Dice 0.062, with three lobes receiving zero predicted voxels in *every* case they appear in — to a receptive-field ceiling, supported by two controls on capacity and ERF. Both controls ran at 1,200 optimiser steps against nnU-Net's 250,000; at 28× the budget the same architecture reaches 0.490 and all five lobes appear, so both controls measured undertrained models. The "three lobes never predicted" observation is itself largely the padding artefact above. The original section is kept verbatim in `experiments/README.md` with the retraction stated at its head, because the reasoning that produced the wrong conclusion is part of the result.

### Found by re-running at scale

- **A three-case pilot overstated a separate product z-seam finding by an order of magnitude.** This A/B does not reproduce the student's input-size collapse: it compares the **then-shipped** teacher z-block/per-block-`argmax` path (pre-`2a50e37`) against 25% z-overlap with logit accumulation. On three cases the gain appeared as high as **+0.205** Dice; over the full test split — 24 organs, paired, 59 of 61 cases carrying at least one in-scope organ — it is **+0.0133** [+0.0072, +0.0194], improving 54 of 59, for 1.18× wall-clock and +0.65 GB (that memory figure measured but never archived). On lung lobes alone the interval crosses zero. Both figures remain because the full-split run exists precisely to stop the outlier from becoming the headline.
- **The re-implementation was checked against the published baseline before being trusted.** The reproduction of the then-shipped path scores 0.8867 over 234 lobe instances — identical to the previously published teacher baseline to four decimals (−0.0000).

### Mistakes made in this round

Recorded because a defect log that only lists other people's defects is not a defect log.

- **Diagnostic artefacts were named by architecture alone**, so re-running the same model at a longer training budget silently overwrote the earlier results. Recovered from git. Artefact names now carry both the step count and the inference path, since the same weights differ by 0.25 Dice between the two paths.
- **`ru_maxrss` was used as a per-case peak a second time**, in code written *after* the earlier round had already documented that it is a process-lifetime high-water mark and monotonically non-decreasing. The benchmark now runs one configuration per process, which removes the ambiguity structurally rather than relying on remembering.
- **The 1,200-step weights were deleted before the path interaction was found**, so that budget can never be re-scored on the sliding path. The budget-versus-path decomposition of the original 0.062 is permanently unrecoverable and is stated as such.
- **A streaming rewrite introduced a double-counting bug that three of four test cases did not catch.** Fusing overlapped logits, the fused result was written back into the same array before being cached, so the block-before-last was counted twice. Only the case whose final two blocks sit 2 slices apart exposed it. Caught by checking the streaming implementation voxel-for-voxel against the full-accumulation version — a check that existed only because the rewrite touched numerical output.

---

## Post-publication audit round (2026-08)

Run after the repository went public, from six reader perspectives (first-time cloner,
researcher reproducing a study, algorithm interviewer, legal reviewer, non-technical
screener, hostile reviewer) and then a full evidence-chain pass.

### Numbers re-derived independently, not re-read

Every headline figure was recomputed from the committed per-case CSVs by code written
fresh for the audit, rather than by re-running the project's own scripts — so an error
inside those scripts could still surface. All of them held: teacher 0.8867, student
0.4367 / 0.7667, paired −0.4500 over n=234 (Wilcoxon 3.7e-39), right-upper-lobe
0.727 / 0.5459, the A/B gain +0.0133 [+0.0072, +0.0194] with 54 of 59 improving at 1.184×
(A and B differ in both final-window handling and overlap, so this is not overlap alone),
21-organ 0.9090 [0.889, 0.927] over 20 cases, spacing 0.6845 → 0.8399 with 20/20
improving at p=1.91e-06, and the zero-padding control's 225,374 → 1,529 foreground
voxels. Parameter counts were recomputed from the ONNX graphs: 31,194,809 and 1,927,841,
matching the stated 31.2 M and 1.9 M. The split was re-run: 207 / 29 / 61, assertions
passing. The full suite was re-run: 515 checks, all passing.

### A description that contradicted the code it described

The README explained streaming z-fusion with "no z position is ever covered by more
than two blocks — so only an 8-slice tail needs to be retained". The code says the
opposite, in a comment written when the defect was fixed: the final block is clamped to
the volume edge, so its gap from the previous one can fall below the stride and some z
positions are covered by **three**. Assuming two was precisely the assumption that
caused the double-counting defect, and the implementation retains the raw logits of the
two most recent blocks — not an 8-slice tail. The README had gone on repeating the
refuted version. Rewritten to state the edge case, which is also the more convincing
version of the story.

### A number with no artefact behind it

`8.44 → 9.09 GB` was measured, but `bench` only ever printed the peak to the terminal;
nothing in `results/` backs it up, so no third party can check it without re-running
59 cases twice. `bench` now appends to `seg3d_infer_bias_bench_peak.csv`, and the
committed run is labelled measured-but-unarchived, since re-running would overwrite
evidence cited elsewhere.

### An evaluation-scope question that had never been written down

Auditing every artefact against the split showed the 57- and 59-case rows are entirely
within `test`, while the 20-case rows for Study II and the spacing ablation sit mostly
in `train`. That is not a leak — those two lines measure third-party weights that
predate the split — but it forbids one specific comparison (0.909 against any student
number, since the student trained on 16 of those cases), and that was nowhere stated.
Now documented as a table in `experiments/README.md`.

### Also fixed

`experiments/requirements-experiments.txt` was missing `onnx==1.21.0`, which
`seg3d_bench.py` imports and which is a different package from `onnxruntime`; the same
file credited torch to Study III when Study IV is its main user. `LICENSE` pointed at a
README section that does not exist. `README.zh-CN.md` had drifted from the English
version, carrying eight numbers the English text had already devolved to the detailed
documents.

### Mistakes made in this round

Three findings were announced before checking the material that was already on disk,
and all three were wrong: `matplotlib` / `nibabel` / `torch` were reported as
undeclared when `experiments/requirements-experiments.txt` had them pinned all along
(the earlier link scan had covered seven file extensions but not `.txt`); `markdown`
was reported as undeclared when line 7 of `build_manual_pdf.py` declares it; and
`seg_multi.csv` was read as 21 cases when its last row is a summary line. A fourth
error was a mis-read of the wrong log file, nearly reporting the full suite as 1 check
instead of 515. The pattern is identical each time — concluding before reading material
that was already at hand, at zero cost.

## Documentation-truth lock-in (2026-08)

`tests/test_doc_code_consistency` (CI subset) turns documentation claims about the code into
assertions: a documented claim about the training seed must match an **AST-detected** call in
both directions; no document may carry an equivalence claim that no re-run supports; Study III
may not be labelled seed-fixed while its published artefacts predate the seeding; and the
module count claimed in `ARCHITECTURE.md` must equal the entries it actually lists. A mutation
self-check is part of it — commenting out the real call must flip the AST verdict while a plain
string search still matches. Scope is deliberately narrow: it covers the wordings currently in
use rather than every possible phrasing, and it does not prove semantic agreement.

`main_run` dispatches to two **separately hand-maintained** call lists (data-independent /
full). The full list does currently contain every data-independent test, but it does not
*inherit* them — a test added to one list silently skips the other until both are edited.

## Withdrawal-propagation round, and the TV baseline (2026-08)

**A withdrawal that only reached three of five files.** Commit `68abee8` withdrew two Study I
conclusions — "ART is the most robust solver" and "error flattens beyond ≈180 views, so the dose
is sufficient" — but `git show --stat 68abee8 -- docs/technical_report.md` is *2 insertions,
1 deletion*: a reference and a test count. The withdrawal itself never entered that file, which
kept asserting both conclusions in its abstract, a section heading, a table caption and a figure
caption. One of those sentences — "ART achieves the lowest RMSE at every tested dose **and
iteration count**" — was by then not merely stale but false, since `recon_stopping.py` sweeps
iteration count and reverses the ranking. The preprint abstract carried the withdrawal marker for
the ART claim and not for the dose claim; `experiments/README.md` withdrew the dose claim and
restated it in the same sentence. The repository was public throughout.

**The regression assertion that was supposed to prevent exactly this was empty.** Its blacklist
matched exact phrases (`ART is the most robust`), while the surviving text read
`(ART) is the most robust` (a closing parenthesis in between), `constrained iteration is the most
robust` (the subject is not ART at all), and `achieves the lowest RMSE`. Measured: **0 hits, green**.
Rewritten as semantic categories it went **red on 7 lines** before the fix and green after. Two more
were then found *in the same round* by the same mechanism — `ART is **the** cleanest` (an article
defeats `ART is cleanest`) and a one-sentence summary still recommending "choose a constrained
iterative method (ART)". The exemption rule for the dose claim had to be tightened separately:
allowing any line that mentions the metric floor would have exempted the self-contradicting
sentence via its own first half.

**A dangling cross-reference, and then the baseline it pointed at.** `recon_dl.py` stated "this
study currently lacks a TV baseline, see the README limitations" — and no markdown file in the
repository contained the words *total variation*, *全变差* or *TV baseline*. The reference resolved
to nothing. Rather than write the missing limitation, the gap was closed: `recon.compute_asdpocs`
implements ASD-POCS (Sidky & Pan 2008 §2.4.2) and `experiments/recon_tv.py` sweeps it against the
same phantom, noise realisations and system matrices as `recon_stopping.py`, all solvers
oracle-stopped.

The result reverses the assumption that motivated the check. TV was expected to be worthless at
this study's noise level (η≈0.9%); it halves the best solver's error there (+45.1% to +54.7% over
SIRT's own optimum). The advantage is monotone in SNR and by η≈9% it has **turned negative at 60 and 90 views**
(−0.8%, −10.0%) while still leading at 30 (+6.4%) — an earlier revision of the two READMEs said
flatly "loses by η≈9%", which the CSV does not support. SSIM also
separates the two in the opposite direction (0.519 against 0.687). A TV-adversarial phantom was
run to deflate the result and did not (+45.7% to +56.4%). Two limits are carried explicitly:
`n_iter` does not transfer from this repository's other solvers — taken as 20 by analogy with
ART=5 / SIRT=100, ASD-POCS is *worse than FBP* — and the inverse crime bears on a matrix method
inverting the exact generating operator harder than on any other result here.

**Two assertions in the new test were written before being measured, and both were wrong.**
"ASD-POCS output has lower total variation than ART" and "total variation decreases monotonically
with α" were asserted from the algorithm's description; measured, both fail — at 20 iterations on a
12×12 system ASD-POCS has not converged and TV only adds error. A third, scale covariance at the
public API, deviates by 24% because `_finite_clip` clips to the absolute range [0,1], which is not
scale-covariant; the property was being tested at the wrong layer. What replaced them: `_tv_grad`
against finite differences (4.19e-09), degree-0 homogeneity of `_tv_grad` (the property that makes
α dimensionless and therefore portable), and — the strongest — that `a=0, beta_red=1` makes
`compute_asdpocs` **bit-identical** to `compute_art`, which locks the POCS step, the α scaling of
`dtvg`, and the fact that `f_res` is returned pre-TV rather than post-TV, in one assertion.

## Known limitations

Both entries that stood here — "MPR anisotropy uncorrected" and "AI mask overlay is axial-only" —
have since been fixed, and are removed rather than left standing as false self-criticism.
Anisotropic planes are rescaled by `graphics_view._apply_aniso_fit` (a View transform only, so scene
coordinates still equal voxel indices and the hover/measurement/crosshair paths were never touched —
the rework this entry once called too risky turned out not to be needed). The AI organ overlay
renders in all three planes. Manual **annotation** overlay is still axial-only; that is a different
thing and is not claimed otherwise anywhere.

See **"Safety and known limits"** in `README.md` for the limitations that do still stand.

---

## Positioning statement

This software is an **imaging teaching / research tool** — **not a certified medical device, and not for clinical diagnosis.** The fixes above improve software robustness and data safety; they do not constitute any clinical-compliance certification.
