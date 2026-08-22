# Architecture

Technical reference for the module layout, the segmentation-model reverse-engineering, and the AI-inference contract. For a feature tour see the [user manual](manual_zh.md); for the quantitative studies see the [technical report](technical_report.md).

## Module layout

The main window is a `MedicalViewer` **God object** decomposed into five UI mixins plus nine Qt-free compute modules that are unit-tested in isolation.

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
followup.py        follow-up comparison metrics (HU difference map + per-slice statistics)
projection.py      slab projection (MIP / MinIP / AIP) across the three planes
mesh3d.py          organ surface reconstruction (marching cubes), shape features, numpy renderer (drives the drag-to-rotate preview), STL export
registration.py    2-D rigid registration (phase correlation + rotation search) with an NCC safety valve
constants.py       tool / plane constants + multi-organ palette
—— resources ——
style.qss          dark theme
models/organs.onnx segmentation model graph (external weights not committed — see below)
```

### Design: why the compute modules are Qt-free

Anything numerically testable is factored out of the Qt widgets into a pure module (`recon` / `quantify` / `segmentation` / `mpr_geometry` / `followup` / `projection` / `mesh3d` / `registration`), so it can be exercised with synthetic data in the data-independent test subset — no display, no real DICOM, no 119 MB weights. New testable logic follows the same pattern rather than being buried in a Qt- or data-dependent path.

## Segmentation model

`models/organs.onnx` is a 45 KB **graph only**; the 119 MB weights (`organs.onnx.data`) are not committed.

### Reverse-engineered architecture

The architecture was confirmed from the ONNX tensor names and structure:

- **nnU-Net v2 `PlainConvUNet`** (3-D full resolution) — names `decoder.encoder.stages.*` + `decoder.seg_layers.*` (nnU-Net v2 deep-supervision head).
- 6-level encoder channels `[32, 64, 128, 256, 320, 320]` (`max_features = 320`, nnU-Net default); 5 downsamplings (hence inputs are padded to multiples of 2⁵ = 32).
- InstanceNorm + LeakyReLU; 25 classes (24 organs + background); exported by PyTorch 2.11 `torch.onnx`.

### Provenance — measured, not assumed

The label→organ mapping was originally undocumented (no `dataset.json`). It was **recovered by measurement**: the model was run on one ground-truth-labelled public CT (TotalSegmentator-CT-Lite, 1.5 mm isotropic) and a label-overlap confusion matrix was computed against the ground truth. This yields an **identity diagonal** — model label *k* → the *k*-th organ — establishing the model as **TotalSegmentator v2 `class_map_part_organs`** (24 organs + background). Over the 21 organs present, mean **Dice ≈ 0.92** on that case. A later 20-case run puts the patient-level mean at **0.909, 95% CI [0.889, 0.927]** — the single case was mildly optimistic but within the interval. Per-organ reliability varies far more than the aggregate suggests: liver 0.982 and spleen 0.976 at one end, right upper lung lobe 0.773 and prostate 0.554 at the other. See [`experiments/seg_multi.py`](../experiments/README.md). The experiment also corrected two earlier label errors: label 5 is **liver** (not heart), and the lung lobes are `10,11` = **left** and `12,13,14` = **right**. Reproduce via [`experiments/seg_validate.py`](../experiments/README.md).

### Inference contract

- **Input** `[1, 1, D, H, W]`, each spatial dim padded to a multiple of 32; HU clipped to `[-1000, 400]` and normalised to `[0, 1]`.
- **Output** `[1, 25, D, H, W]` logits — take `argmax` over the class axis (not a threshold).
- **Sliding window** along z in blocks of 32; must be run on the **full x-y frame** (a 256 centre-crop destroys global context and mislabels lung as background).
- Full-volume inference is CPU-only: **≈ 37 s at ≈ 3.0 GB peak** per volume since the spacing contract was implemented (`ai_engine.TARGET_SPACING = 1.5`). Before that fix it was ≈ 100 s at ≈ 8.8 GB — that older pair is still quoted in the evidence table as the *before* side of the ablation.

### Getting the weights

The repository ships only the graph, so without `organs.onnx.data` the model cannot load and segmentation falls back to the classical algorithm in `segmentation.py`. To obtain the weights: install [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) (v2, `class_map_part_organs` task), take its nnU-Net weights, and export to ONNX external-data format with `torch.onnx`; the resulting `.data` must sit next to the `.onnx`. Upstream licensing is summarised in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## AI inference threading

Inference runs on a background `threading.Thread`. UI updates are delivered to the main thread **exclusively via Qt signals** (`QueuedConnection`) — never `QTimer.singleShot`, which attaches to the worker thread that has no Qt event loop and would never fire. The signal carrier (`_AISignals`) is deliberately created **without a parent**: giving it a parent would let Qt delete it during teardown while the worker is still emitting, turning an occasional race into a guaranteed crash. On ONNX failure or a missing model file, the engine degrades to the classical fallback; teardown-time races are classified separately from genuine inference errors so the log stays truthful.

## Persistence

Saving a project writes annotations and the AI mask under `Exported_Lesions/`. The mask cache (`{PatientID}_mask.npz`) is keyed by `PatientID` **and** validated against `SeriesInstanceUID` and shape on reload — a same-patient follow-up series (same 512² shape, different series) is refused rather than silently reused, which would otherwise report the wrong organ volumes.
