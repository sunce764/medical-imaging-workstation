# Medical Imaging Workstation — User Manual

**English** · [简体中文](manual_zh.md)

> This manual is written for software version V1.0. All screenshots are demonstrated using the **public dataset TotalSegmentator-CT-Lite (CC-BY-4.0)** — **public, de-identified human CT**. These are images of real patients: what the dataset removes is *identifiability*, not the clinical origin, so no identifiable personal health information (PHI) is present, but "not patient data" would be the wrong claim. The patient-information panel is explicitly labelled as public data.

> **Version correspondence.** According to the project filing records, the manual **PDF used for the software-copyright registration application is the V1.0 snapshot**; V1.0 is defined by that PDF, which this document does not retroactively amend. This Markdown source has continued to be maintained since V1.0 and now documents features that V1.0 did not yet contain (spacing resampling before inference, per-voxel confidence, the model card, the reconstruction lab's built-in phantom). The resulting differences reflect normal source evolution, not a correction to the submitted snapshot. **One difference is a wording correction rather than a feature**: the PDF's cover still describes the screenshot data as "not patient data", which this Markdown has since corrected to "public, de-identified human CT". The PDF is deliberately left untouched while the copyright holders have not received a formal acceptance notice from CPCC.

---

## 1. Software Overview

**Software name**: Medical Imaging Workstation Software
**Software version**: V1.0
**Introduction**: This software is a desktop CT medical imaging workstation built on PySide6 (Qt6), aimed at imaging teaching and research. It integrates three major parts — **clinical reading tools**, **AI multi-organ segmentation**, and a **CT tomographic-reconstruction teaching lab**. The software supports loading DICOM images, multi-planar reformation (MPR) reading, window width / window level adjustment, measurement and annotation, AI automatic organ segmentation and quantification, dual-series follow-up comparison, and a complete teaching demonstration from projection to reconstruction.
**Verified environment**: local macOS with Python 3.10; the data-independent suite had also passed historically on a GitHub Actions Ubuntu runner as of the snapshot below. Windows was not verified in that snapshot and no platform-compatibility claim is made for it. Dependencies include PySide6, pydicom, NumPy, SciPy, scikit-image, and ONNX Runtime.
**Development language**: Python.
**Software scale**: application code is split across UI mixins and Qt-free compute modules. A local run on 2026-08-26 recorded 854 full-suite checks and 762 `SKIP_REAL_DATA=1` checks; these are local results, not fresh-clone, coverage, or remote-CI evidence.
**Positioning statement**: This software is a **teaching / research tool for imaging**, **not a certified medical device, and must not be used for clinical diagnosis**; AI segmentation and quantification results are automated inferences, for reference only.

---

## 2. Operating Environment and Launch

In a Python environment with the dependencies configured, run the following from the software root directory:

```
python main.py                          # empty start (no data loaded)
python main.py --data <DICOM directory path>    # load the specified DICOM directory on launch
```

After the software starts, it enters the main interface. By default it **loads no data**, waiting for the user to load data from the interface, as shown below.

![Empty-start interface](img/manual_startup_empty.png)

*Figure 2-1  The empty-start interface (English-UI example). The toolbar is on the left, the image view area in the centre, and the control panel on the right.*

---

## 3. Main Interface Layout

The main interface is divided into three columns:

1. **Left toolbar**: nine measurement / annotation tool buttons arranged top to bottom (probe & pan, distance caliper, freehand pen, rectangle capture, lasso, 3D tracking, segmentation brush, segmentation eraser, ROI densitometry). After a tool is selected with a click, mouse actions on the image correspond to that tool's function.
2. **Central image view area**: composed of 1–4 image views, supporting single-, dual-, and quad-view layouts. Each view has, along its top, dropdown / checkbox controls for plane selection (axial / coronal / sagittal), window-level presets and overlay display.
3. **Right control panel**: at the top are the "Load DICOM directory" and "Save annotation project" buttons; below them are two tabs, "Clinical reading / Reconstruction lab," which carry the clinical-reading controls and the reconstruction-lab controls respectively.

The top tabs switch between the two working modes, **Clinical reading** and **Reconstruction lab**.

---

## 4. Loading DICOM Data

Click the **"Load DICOM directory"** button at the top of the right panel and choose a directory containing DICOM slices. The current loader supports **classic single-frame CT Image Storage**; non-CT, Enhanced CT, and multi-frame inputs are rejected before pixel decoding. For supported input, the software will:

- **Parallel disk reading**: read all DICOM files in the directory with multiple threads, speeding up loading of large series;
- **Multi-series handling**: multi-file input is grouped only when every slice has a `SeriesInstanceUID`; otherwise it fails closed rather than merging unknown series. The selected group is then filtered to the majority matrix shape;
- **Spatial sorting**: when every slice has finite, consistent `ImageOrientationPatient` and `ImagePositionPatient`, sort by the patient-space projection `dot(IPP, normal)`. If that geometry cannot be proved, fall back for the whole series to `InstanceNumber`, without claiming that this establishes anatomical order;
- **Intensity and unit proof**: every retained slice must have finite non-zero `RescaleSlope` plus finite `RescaleIntercept`, and must either declare `RescaleType=HU` or satisfy the classic CT standard guarantee (`ImageType` is `ORIGINAL`, not `LOCALIZER`, and not multi-energy). Otherwise the entire volume stays in raw stored values and CT presets, HU quantification, AI, and HU follow-up are disabled rather than mixing or inventing HU units. Multi-energy CT remains unsupported even when some such images may intrinsically represent HU;
- **Capability gating**: valid PixelSpacing, canonical orientation, and uniform projected-z geometry are tracked independently. mm/mm², MPR, mL/physical STL, AI, and follow-up are enabled only when their contracts are proved. Non-canonical or incomplete geometry remains viewer-only where safe, with no invented anatomical labels or physical units.

---

## 5. Clinical Reading

### 5.1 Slice Browsing and Navigation

The "Slice" slider on the right panel lets you browse layer by layer; you can also move the mouse into the image and use the scroll wheel to page through slices. The keyboard `↑ / ↓` and `PgUp / PgDn` keys also page through slices.

### 5.2 Window Width / Window Level Adjustment

The "Display control" area on the right provides two sliders, **WW (window width) / WL (window level)**, along with 6 clinical window-level preset buttons: **lung, mediastinum, bone, vessel, abdomen, brain**; there is also an **invert** checkbox. You can also hold the right mouse button and drag on the image to adjust window width / level in real time. The figure below shows the effect after switching to the **lung window**, displaying a chest axial slice.

![Clinical reading · lung window](img/manual_clinical_lung.png)

*Figure 5-1  Reading a chest axial slice under the lung window. The four corners overlay patient information (public data), window level, slice number, and anatomical orientation letters (A/P/R/L).*

### 5.3 Tri-planar MPR and Linked Cross-hairs

After enabling "MPR linkage," the quad-view layout can simultaneously display the three planes **axial, coronal, sagittal**. Moving the mouse in any view links the cross-hairs and slices of the other views to the same anatomical point; the anisotropic planes (coronal / sagittal) automatically correct their display aspect ratio according to anatomical proportions.

![Tri-planar MPR linkage](img/gui_mpr_triplanar.png)

*Figure 5-2  Tri-planar MPR + linked cross-hairs (lung window, AI lung-lobe colour overlay shown in the axial view).*

### 5.4 Layout Modes

The layout dropdown in the right "Display control" can switch between **single (1×1) / dual (1×2) / quad (2×2)** views.

### 5.5 Four-corner DICOM Information Overlay

When the "Information overlay" checkbox is enabled, the four corners of the image overlay patient information, window width / level, and slice number in PACS style, and anatomical orientation letters are marked at the image edges (A anterior / P posterior / R right / L left / S superior / I inferior).

### 5.6 Cine Playback and Keyboard Paging

The "Play" button starts Cine playback, automatically paging through slices continuously (bouncing back at the top / bottom, no wrap-around jump); the speed dropdown offers slow / medium / fast; clicking again pauses.

### 5.7 Slab Projection (MIP / MinIP / AIP)

The projection dropdown at the top of each view switches between four modes, with the spin box beside it setting the slab thickness in slices:

| Mode | Meaning | Typical use |
|---|---|---|
| **Slice** | Default, shows a single slice | Routine reading |
| **MIP** | Maximum intensity projection | **High-density** structures: lung nodules, vessels, bone |
| **MinIP** | Minimum intensity projection | **Low-density** structures: airways, emphysematous regions |
| **AIP** | Average intensity projection | Noise reduction, overall density distribution |

Projection runs along the normal of the current plane and is **supported on all three planes**. Selecting "Slice" disables the thickness box; in that state the displayed result is **pixel-for-pixel identical** to not using the projection feature at all.

> Why slab rather than whole-volume projection: clinical practice uses slab MIP (typically 5–20 mm). Collapsing the entire volume into one image superimposes unrelated anatomy and obscures the target instead of revealing it. Thickness is converted to millimetres per plane — axial uses slice thickness along z, coronal/sagittal use pixel spacing along the in-plane axis, since the two carry different physical scales.

---

## 6. Measurement and Annotation Tools

The nine tools in the left toolbar operate on the axial image after selection:

1. **Probe & pan**: click to read the HU value and coordinates at that point; drag to pan the view.
2. **Distance caliper**: drag out a straight line to measure the physical distance between two points (mm, converted by pixel spacing).
3. **Freehand pen**: draw annotation lines freely on the image.
4. **Rectangle capture**: box-select a rectangular ROI, compute the region's area and mean HU, and optionally export the cropped image and a CSV.
5. **Lasso**: draw a polygonal ROI to generate a segmentation mask.
6. **3D tracking**: box-select an ROI on one slice, extract its HU statistics, track HU-similar connected structures throughout the whole 3-D volume, and generate a 3-D mask.
7. **Segmentation brush**: paint on the current axial slice to add the strokes into the segmentation mask (an optional target organ can be chosen, and painted-in strokes count toward that organ's quantification), used to correct AI omissions.
8. **Segmentation eraser**: erase the mask where it was painted (can remove AI mis-segmentations).
9. **ROI densitometry**: drag out an elliptical ROI and read the interior mean ± SD / min-max HU / area; the ellipse can be dragged, resized, and deleted.

Annotations support two ownership modes, **slice-specific** and **global (all-slices)**, and can be persistently saved together with the segmentation masks as a project JSON via "Save annotation project." Segmentation editing supports `Ctrl+Z` undo.

---

## 7. AI Multi-organ Segmentation

### 7.1 Automatic Inference

After DICOM data is loaded, the software automatically calls the segmentation model (`models/organs.onnx`, 25 thoracoabdominal organ classes including 5 lung lobes) in a background thread to perform whole-volume sliding-window inference; the "Automated AI engine" area on the right displays the inference progress in real time; inference does not block interface operations. When there is no model file or inference fails, it falls back to a purely mathematical connected-component lung-segmentation algorithm.

> **Spacing resampling before inference.** The model (nnU-Net v2) requires the volume to be resampled to its training voxel spacing (1.5 mm isotropic) first; the software does this automatically and says so in the status line. Skipping it has a measured cost: at twice the training spacing, mean Dice falls from 0.922 to 0.799, with small organs failing first. The step is not free either — mask boundaries are quantised to the 1.5 mm grid and appear stair-stepped when mapped back to a finer original resolution: **structural accuracy up, pixel-level boundary precision down**. Resampling is skipped when the series is already near 1.5 mm, or when the scan range is so large that resampling would exceed the memory limit.

### 7.2 Result Overlay and Legend

After inference completes, the segmentation result is overlaid on the image as a colour semi-transparent mask, **displayed on all three planes — axial, coronal and sagittal** (mask and image are taken as corresponding slices of the same 3-D array, so they align pixel for pixel); the legend on the right lists each detected organ and its colour, and **clicking a legend entry toggles that organ's visibility**.

![AI multi-organ segmentation overlay and organ quantification](img/gui_axial_segmentation.png)

*Figure 7-1  AI multi-organ segmentation result overlay (liver, spleen, kidney, stomach, lung lobes, etc.); the right legend includes each organ's volume / HU and a disclaimer.*

### 7.3 Cursor HUD

As the mouse moves over the image, the interface displays in real time the coordinates and HU value at the cursor, along with the name of the organ it lies in (if that voxel belongs to a segmented organ).

### 7.4 Organ Quantification Panel and CSV Export

The "Automated AI engine" area lists each detected organ's **volume (mL) and mean HU ± SD** (in descending order of volume). Clicking **"Export quantification CSV"** exports the quantification results to a CSV file (UTF-8-SIG encoding, so Excel displays Chinese correctly) carrying seven HU statistics per organ — mean, SD, median, 5th and 95th percentiles, minimum and maximum — alongside the volume, with the AI disclaimer embedded in the CSV.

> Why not the mean alone: a mean says nothing about how dispersed the density is inside an organ, and that dispersion is what tells you whether the segmentation has absorbed neighbouring tissue — and is a precondition for any statistical comparison. The 5th/95th percentiles are more robust to single-voxel noise than the extremes.

The panel also reports each organ's **confidence** (the model's softmax max-class probability) together with its **5th percentile**: the mean is pulled up by the large confident interior of an organ, whereas segmentation errors concentrate at boundaries, so the low percentile is the more revealing number; entries below 0.9 are flagged in orange. If an organ has been edited with the brush or 3D tracking, a **model-decided share** is shown as well — hand-edited voxels are excluded from the confidence statistics, because their stored value is the model's judgement about *the label that was there before the edit*, which says nothing about the current one. The manual tracking layer reports no confidence at all, since the model never judged it.

The **"Model card: provenance & limits"** button sets out how the model's origin was established by measurement, how far it has been validated, and what its known limits are. Every number on the card is read live from the experiment outputs under `experiments/results/`, so re-running an experiment updates the card.

### 7.5 3D Surface Reconstruction

Click **"3D Surface Preview"** and the software reconstructs a 3-D surface for **the organ currently selected as the brush target**, opening a dialog with a **drag-to-rotate** 3-D view together with **surface area, volume, sphericity and face count**. From there the mesh can be **exported as STL** (ASCII, millimetre units, ready for 3D printing or external software).

**Interaction**: press and drag on the image to rotate — horizontal motion changes azimuth, vertical motion changes elevation (0.5°/px); the six buttons on the right (Ant / Post / Left / Right / Sup / Oblique) jump to standard views; the current angles are shown live below the image. Elevation is clamped to ±89° to avoid gimbal lock, where the view direction aligns with the rotation axis, azimuth loses meaning and the image flips abruptly.

The pipeline is **isosurface extraction (marching cubes) → Taubin smoothing → vertex-clustering decimation**, matching the surface-model workflow used by 3D Slicer. Rendering is implemented in pure numpy (orthographic projection + Lambert shading + painter's-algorithm depth sorting), with no dependency on OpenGL or VTK, so no GPU is required. The trade-off is no perspective and no shadows.

> **How the rotation stays responsive**: a full-mesh frame takes ≈ 114 ms (measured, 360 px view, 4,615 faces) — driving the mouse with that is visibly choppy. The dialog therefore **drops quality while dragging and restores it on release**: during a drag it renders a further-decimated mesh (measured on a real organ: 6,798 → 1,984 faces, ≈ 48 ms/frame), and the instant the button is released it repaints one frame from the full mesh — so **what you see at rest is always full precision**. The coarse mesh affects the drag preview only; **shape features and STL export always use the full mesh** (the coarse mesh is 1.6% off in volume, which would corrupt quantification).

> **How spacing resampling propagates into shape features (measured)**: inference now resamples the volume to the model's training spacing (1.5 mm), so mask boundaries are quantised to that grid. On an analytic sphere (R = 20 mm, native 0.713 mm grid, 10 smoothing iterations) this moves the surface-area error from +0.26% to **+1.95%**, the volume error from −0.18% to **−0.97%**, and sphericity from 0.9962 to 0.9745. Taubin smoothing absorbs most of the staircase so the magnitude stays small, but anyone using shape features for quantitative comparison should know the figure includes this term.

> **Why smoothing matters**: marching cubes alone leaves a voxel staircase, which inflates surface area. Measured on an analytic sphere (R = 20, spacing 1 mm): without smoothing the surface area is **+9.3%** high and sphericity is 0.915; after 10 Taubin iterations these become **+1.2%** and 0.988, while **volume shifts by only +0.08%**. Taubin alternates a positive and a negative pass so the shrinkage cancels — plain Laplacian smoothing would steadily shrink the mesh and corrupt the volume measurement. Decimation (roughly halving the face count by default) halves render time at a volume error on the order of 0.1%.

### 7.6 Segmentation Editing

Using the "Segmentation brush / Segmentation eraser" tools, you can manually add to or erase the AI segmentation result; when painting in, a target organ can be specified (its quantification updates accordingly). All edits support `Ctrl+Z` undo.

---

## 8. Dual-series Follow-up Comparison

Click **"Load comparison series"** and select the prior DICOM directory. Dual-view comparison is entered only when both series satisfy HU calibration, canonical orientation, valid in-plane spacing, and uniform projected-z geometry. Slice correspondence uses each series' `dot(IPP, normal)` patient-space positions; missing or irregular geometry is rejected explicitly rather than presented as anatomical correspondence through index-ratio fallback. When de-identification is enabled, the prior examination date in the comparison title is hidden.

### 8.1 Difference quantification

After registration, the V2 title bar reports the HU difference for the current slice in real time: **Δ mean** (current − prior; positive means denser now), **mean absolute difference**, and **RMSE**. Switching to the quad-view layout shows a **difference map** in V3: warm colours where density has increased, cool where it has decreased, transparent where there is no change.

When the two series have different matrix sizes, the software **states plainly that they are not comparable rather than force-resampling them** — interpolation would introduce error and create the illusion of a valid comparison.

### 8.2 In-plane rigid registration

The **"Register"** checkbox in the top toolbar (**enabled only in comparison mode**; disabled until a prior series is loaded) rigidly aligns the prior slice to the current one before comparing: translation is estimated by **phase correlation**, then rotation is searched within ±6° in 0.5° steps, keeping whichever maximises normalised cross-correlation (NCC). The title bar reports the estimated angle, translation and the NCC before/after.

- **Safety valve**: if NCC does not improve (mismatched levels, anatomy changed too much), the transform is **rejected** and the title says so — better no registration than an alignment that makes things worse.
- **Measured effect**: for a prior series shifted by (12, −9), mean absolute difference drops from **321 HU to 13 HU**, NCC 0.85 → 0.99.

> **Important limitation**: rigid registration corrects **posture only** (translation + rotation); there is **no deformable registration**, so respiratory organ deformation is not corrected. Choosing rigid over deformable is deliberate — deformable registration absorbs breathing motion but also warps away **genuine lesion change**, which defeats the purpose of a follow-up comparison. The reported difference therefore **remains a qualitative indicator, not a clinical measurement of lesion change**. In addition, the prior series is not run through AI inference, so **organ-level volume change is not provided**.

---

## 9. Reconstruction Lab (CT Tomographic-reconstruction Teaching)

Click the top tab to switch to **"Reconstruction lab."** This module takes the current slice as its subject and fully demonstrates the process from X-ray projection to image reconstruction. Within it, the forward Radon projection and the analytic inverses (BP / FBP) are built on scikit-image's `radon` / `iradon`; the five inverse-solver algorithms DFR, DMR, ART, SIRT and ASD-POCS are implemented as self-contained numerical code in this project.

![Reconstruction lab](img/manual_recon_lab.png)

*Figure 9-1  The reconstruction lab quad view: V1 the real slice, V2 the projection sinogram, V3 the unfiltered back-projection (blurry), V4 the filtered back-projection FBP (sharp); on the right are the projection / algorithm controls and performance monitoring.*

### 9.1 Built-in Phantom (No Data Required)

Clicking **"Load Shepp-Logan phantom"** makes the entire reconstruction lab usable with no DICOM loaded at all. The phantom is generated analytically from ten superposed ellipses (Toft's revised parameters, the same convention as Study I), so it can be produced at any resolution without interpolation blur.

Its decisive advantage over a real slice is that **the ground truth is known**: the V3 error map then measures the distance between the reconstruction and the truth. For a real slice the "ground truth" is only the original image, which already carries noise and reconstruction artefacts of its own, so the error map measures the distance to *that* — not to the truth. Clicking again unloads the phantom and clears the sinogram and reconstructions derived from it, so a phantom sinogram is never left paired with a real-data reference image.

### 9.1.1 Projection Generation (Radon Transform)

In the "X-ray projection generation" area on the right, select the **angular range (60° / 120° / 180° / 360°)** and the **sampling density (standard 1× / high 2× / ultra 4×)**, then click **"Emit rays to generate sinogram"** to perform the Radon transform on the current slice and generate the projection sinogram, shown in V2.

### 9.2 Analytic Reconstruction

The "Image reconstruction algorithms" area provides:

- **Direct Fourier reconstruction (DFR)**: reconstructs directly from the sinogram based on the Fourier central-slice theorem;
- **Back-projection (BP, unfiltered)**: pure back-projection, with a blurry result (star-shaped artefacts), shown in V3 for comparison;
- **Filtered back-projection (FBP)**: with a choice of 5 filters (Ram-Lak / Shepp-Logan / Cosine / Hamming / Hann), the result shown in V4.

### 9.3 Matrix / Iterative Reconstruction

In the "Matrix Recon & Iterative" area, select the **image size (16/32/64)**, **iterative method (ART / SIRT)**, and **iteration count**; it provides:

- **Direct matrix reconstruction (DMR)**: solves the projection system of equations by least squares;
- **ART / SIRT / ASD-POCS iterative reconstruction**: algebraic and TV-regularised iterative reconstruction, with an error map and RMSE.

### 9.4 Deep-Learning Reconstruction (CNN post-processing)

Click **"DL Recon (CNN post-processing)"** to remove sparse-view FBP streak artefacts with a self-implemented residual U-Net. **V3 shows the network's input (ramp-FBP) and V4 its output**, so what the network actually changed can be compared directly.

Method and quantitative results are in the [experiments guide](../experiments/README.md) and Study III of the [technical report](technical_report.md): on a random phantom family, RMSE is 3–6× lower than the best linear filter and lesion-contrast retention rises from 0.87 to 0.96–1.00. Across **60 noise-free synthetic paired phantoms**, the false-structure rate is **1.67%** at the 20%-of-lesion threshold and **0%** at 30% and 50%. Photon noise was not applied, so 1.67% is neither an upper nor a lower bound for low-dose CT; both direction and magnitude are unmeasured, and low SNR is not claimed as the dominant driver.

> **Three limitations are stated in the UI itself, not just in the docs:**
> - **The model was trained at 20 views.** When the current view count differs, the V4 title is tagged "⚠ view mismatch" — results degrade at other view counts, and the software does not pretend otherwise.
> - **The input is forced to Ram-Lak (ramp) FBP**, regardless of the filter dropdown above. Smoothing filters (Hann and friends) have already discarded the high frequencies — and the detail with them — at the filtering stage, and the network cannot recover what is gone; ramp keeps the information but leaves streaks, and streaks are what can be learned.
> - **With the model or onnxruntime missing, the button stays disabled** and the tooltip explains why and how to obtain it. The repository ships only the 20 KB `.onnx` graph; the 7.7 MB weights (`.onnx.data`) must be trained and exported locally — the same convention as `organs.onnx`, and the two files must sit in the same directory.

### 9.5 Performance Monitoring

The "Algorithm performance monitoring" area displays in real time the running time of each reconstruction algorithm (as in Figure 9-1, "FBP (ram-lak) time: 254.9 ms").

---

## 10. Compliance and De-identification

- **De-identification switch**: after "De-ID" is enabled, on-screen identity is shown as `ANON`; explicit export filenames use a random `ANON-…` alias generated for the current load and add a suffix on collision, so repeated exports do not silently overwrite one another. This is not a DICOM anonymizer: it does not rewrite source DICOM tags or remove burned-in pixel text. Internal project JSON and mask caches retain patient/series identifiers and a geometry fingerprint for matching, and the UI warns again when saving them.
- **Project-state persistence**: an AI-pending all-zero mask is only a placeholder and does not create a cache hit. Confirming the global clear action records an explicit provenance-bound empty mask on the next successful save, so an older non-zero cache cannot reappear after reopening; Ctrl+Z before saving restores the non-zero mask. Voxel-by-voxel erasing to zero is not treated as this global-clear action.
- **AI disclaimer**: the AI panel permanently displays a disclaimer, and the exported quantification CSV also embeds that disclaimer.

---

## 11. Bilingual (Chinese / English) Toggle

The language button in the top-right corner of the interface toggles between **Chinese / English** with one click, and all persistent widget text (tools, buttons, panel titles, dropdown items, hover tooltips, status text, etc.) is re-translated accordingly. The figure below shows the English-UI example.

![English interface](img/manual_english_ui.png)

*Figure 11-1  The English interface (mediastinum window, abdominal axial slice).*

---

## 12. Annotation Project Persistence

Click **"Save annotation project"** on the right to save the current annotations and segmentation masks as a project JSON file; when the same patient's data is loaded again, the last-saved annotations and segmentation are automatically restored, with no need to re-run inference.

---

*(End of manual)*
