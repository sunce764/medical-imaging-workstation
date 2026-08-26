# Architecture

Technical reference for the module layout, the segmentation-model reverse-engineering, and the AI-inference contract. For a feature tour see the [user manual](manual_zh.md); for the quantitative studies see the [technical report](technical_report.md).

## Module layout

The main window is a `MedicalViewer` **God object** decomposed into five UI mixins plus ten Qt-free compute modules that are unit-tested in isolation. The packaging inventory is the 19 top-level modules declared by `pyproject.toml`; `constants.py` is Qt-free but is a data table rather than a compute module.

```
main.py            MedicalViewer + entry point (--data load, clinical render, W/L, tools, layout, AI scheduling, i18n, keyboard nav)
—— UI layer (mixins merged into MedicalViewer) ——
ui_builder.py      UiBuilderMixin    three-column layout and all widget construction
interaction.py     InteractionMixin  Cine playback + MPR linkage / navigation
recon_lab.py       ReconLabMixin     reconstruction-lab UI scheduling (projection / BP / FBP / DFR / DMR / ART / SIRT)
compare_lab.py     CompareMixin      dual-series follow-up comparison (anatomical registration / linkage)
annotation_lab.py  AnnotationMixin   annotation, mask editing, organ quantification, project persistence
ai_engine.py       AutoAIEngineThread  background AI inference (sliding window + signal callbacks)
graphics_view.py   MedicalGraphicsView  interactive image view + ROIGraphicsItem
—— Qt-free compute modules (unit-tested without the main window) ——
recon.py           reconstruction algorithms (Radon / BP / FBP / DFR / DMR / ART / SIRT)
quantify.py        organ quantification (volume mL + seven HU statistics per organ)
segmentation.py    classical fallback segmentation (lung connected-components)
mpr_geometry.py    MPR coordinate mapping + dual-series z-registration
dicom_geometry.py  classic CT HU-unit proof + patient-space geometry/order/fingerprint contracts
followup.py        follow-up comparison metrics (HU difference map + per-slice statistics)
projection.py      slab projection (MIP / MinIP / AIP) across the three planes
mesh3d.py          organ surface reconstruction (marching cubes), shape features, numpy renderer (drives the drag-to-rotate preview), STL export
registration.py    2-D rigid registration (phase correlation + rotation search) with an NCC safety valve
model_card.py      model card: reads experiments/results/ live and renders provenance, validated scope and unmeasured limits
constants.py       tool / plane constants + multi-organ palette
—— resources ——
style.qss          dark theme
models/organs.onnx segmentation model graph (external weights not committed — see below)
```

### Design: why the compute modules are Qt-free

Anything numerically testable is factored out of the Qt widgets into a pure module (`recon` / `quantify` / `segmentation` / `mpr_geometry` / `dicom_geometry` / `followup` / `projection` / `mesh3d` / `registration` / `model_card`), so it can be exercised with synthetic data in the data-independent test subset — no display, no real DICOM, no 119 MB weights. New testable logic follows the same pattern rather than being buried in a Qt- or data-dependent path.

## Segmentation model

`models/organs.onnx` is a 45 KB **graph only**; the 119 MB weights (`organs.onnx.data`) are not committed.

### Reverse-engineered architecture

The architecture was confirmed from the ONNX tensor names and structure:

- **nnU-Net v2 `PlainConvUNet`** (3-D full resolution) — names `decoder.encoder.stages.*` + `decoder.seg_layers.*` (nnU-Net v2 deep-supervision head).
- 6-level encoder channels `[32, 64, 128, 256, 320, 320]` (`max_features = 320`, nnU-Net default); 5 downsamplings (hence inputs are padded to multiples of 2⁵ = 32).
- InstanceNorm + LeakyReLU; 25 classes (24 organs + background); exported by PyTorch 2.11 `torch.onnx`.

### Provenance — measured, not assumed

The label→organ mapping was originally undocumented (no `dataset.json`). It was **recovered by measurement**: the model was run on one ground-truth-labelled public CT (TotalSegmentator-CT-Lite, 1.5 mm isotropic) and a label-overlap confusion matrix was computed against the ground truth. This yields an **identity diagonal** — model label *k* → the *k*-th organ — which measures the label scheme to be **TotalSegmentator v2 `class_map_part_organs`** (24 organs + background). Two claims are worth keeping apart here: the *label mapping* is a measurement, and the code depends only on it; that these weights are that exact upstream release is an inference drawn from the mapping, the nnU-Net v2 layer names and the exporter string, and it is strong but not a cryptographic identification — see [Getting the weights](#getting-the-weights). Over the 21 organs present, mean **Dice ≈ 0.92** on that case. A later 20-case run puts the patient-level mean at **0.909, 95% CI [0.889, 0.927]** — the single case was mildly optimistic but within the interval. Per-organ reliability varies far more than the aggregate suggests: liver 0.982 and spleen 0.976 at one end, right upper lung lobe 0.773 and prostate 0.554 at the other. See [`experiments/seg_multi.py`](../experiments/README.md). The experiment also corrected two earlier label errors: label 5 is **liver** (not heart), and the lung lobes are `10,11` = **left** and `12,13,14` = **right**. Reproduce via [`experiments/seg_validate.py`](../experiments/README.md).

### Inference contract

- **Input** `[1, 1, D, H, W]`, each spatial dim padded to a multiple of 32; HU clipped to `[-1000, 400]` and normalised to `[0, 1]`.
- **Output** `[1, 25, D, H, W]` logits — take `argmax` over the class axis (not a threshold).
- **Sliding window** along z in blocks of 32; must be run on the **full x-y frame** (a 256 centre-crop destroys global context and mislabels lung as background).
- Full-volume inference is CPU-only. On **the one local RIDER series this was measured on** (233 slices, 0.713 mm in-plane), it takes **≈ 37 s at ≈ 3.0 GB peak** since the spacing contract was implemented (`ai_engine.TARGET_SPACING = 1.5`); before that fix the same series took ≈ 100 s at ≈ 8.8 GB, which is the *before* side quoted in the README's evidence table. Both are single-series measurements, not a per-volume average — after resampling, cost tracks the scanned field of view rather than the acquisition spacing.

### Getting the weights

The repository ships only the graph, so without `organs.onnx.data` the model cannot load and segmentation falls back to the classical algorithm in `segmentation.py`. To obtain the weights: install [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) (v2, `class_map_part_organs` task), take its nnU-Net weights, and export to ONNX external-data format with `torch.onnx`; the resulting `.data` must sit next to the `.onnx`. Upstream licensing is summarised in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

**What is recorded, and what is not.** [`models/CHECKSUMS.sha256`](../models/CHECKSUMS.sha256) lists SHA-256 digests for the two ONNX pairs and the six PyTorch checkpoints Studies III and IV trained; it verifies from the repository root with `shasum -a 256 -c models/CHECKSUMS.sha256`. Two limits on what that buys, both stated in the file itself: the digests were taken when the file was written, **not** when the results were produced, so they identify the current bytes rather than proving which bytes produced a past number; and they cover only the artefacts listed there. What the ONNX graph itself carries — and therefore what anyone can re-read from the committed 44 KB file — is:

| Property | `organs.onnx` | `recon_dl_v20.onnx` |
|---|---|---|
| SHA-256 (graph, committed) | `360b7b97…61809e` | `b4dd27e0…f6921c` |
| SHA-256 (`.data`, **not** committed) | `0b96b9d1…d312ce` (124,878,848 B) | `76a183ea…7a6e55` (7,733,248 B) |
| Exporter | `pytorch` 2.11.0 | `pytorch` 2.11.0 |
| ONNX IR / opset | 10 / 16 | 10 / 17 |
| Input | `input_image` `[1, 1, depth, height, width]` | `fbp` `[n, 1, h, w]` |
| Output | `output_mask` `[1, 25, …]`, spatial dims padded up to a multiple of 32 | `recon` `[n, 1, h, w]` |

Three things are **not** recorded, and no amount of re-reading the file recovers them: the exact upstream TotalSegmentator release tag or checkpoint filename the weights were taken from, the export script and its arguments, and any digest published by upstream to compare against. The export predates this repository's provenance discipline; `experiments/seg3d_data.py`, which pins a dataset commit and checksums every file it downloads, is what that discipline looks like once it existed. The practical consequence is stated plainly: a third party who exports their own weights from upstream can reasonably expect a functionally equivalent model, but **cannot verify byte-identity with the blob these numbers came from** unless they obtain that blob itself and match the digest above. The `recon_dl_v20` pair has the mirror-image problem — it was trained here, but before `train_one` pinned the PyTorch RNG, so re-training is not expected to reproduce those bytes either.

## AI inference threading

Inference runs on a background `threading.Thread`. UI updates are delivered to the main thread **exclusively via Qt signals** (`QueuedConnection`) — never `QTimer.singleShot`, which attaches to the worker thread that has no Qt event loop and would never fire. The signal carrier (`_AISignals`) is deliberately created **without a parent**: giving it a parent would let Qt delete it during teardown while the worker is still emitting, turning an occasional race into a guaranteed crash. On ONNX failure or a missing model file, the engine degrades to the classical fallback; teardown-time races are classified separately from genuine inference errors so the log stays truthful.

## Persistence

Saving a project writes annotations and the AI mask under `Exported_Lesions/`. The mask cache path (`{safe PatientID}_mask.npz`) uses the same filename-safe PatientID transformation as the rest of project persistence. Automatic restoration requires all three provenance checks to match: `SeriesInstanceUID`, mask/volume shape, and a geometry/order fingerprint. That fingerprint binds the ordered `SOPInstanceUID` sequence, IOP/IPP, projected patient-space slice positions, `PixelSpacing`, and volume shape. A same-patient follow-up series, reordered stack, or changed geometry is therefore refused rather than silently reused; legacy cache entries without the fingerprint fail closed.
