# Changelog · Code-Review Notes

This file collects the systematic rounds of defect investigation on the **Medical Imaging Workstation Pro + Reconstruction Lab** — a robustness round (2026-07) and a correctness round (2026-08).

**Version 1.1.0** (annotated tag `v1.1.0`, 2026-08-28) is the earlier release snapshot. The unreleased changes below are subsequent local work. The exact-tag CI recorded in the historical sections covers that tag, not subsequent changes; neither fact says anything about the copyright registration, which remains under review. The earlier `v1.0-copyright` tag marks the separate snapshot submitted for that registration.

## Unreleased: functional and UI state audit (2026-09-04)

Reset and valid manual mask edits now cancel stale AI results instead of allowing delayed callbacks
to restore cleared masks or overwrite edits. In-progress annotations survive ordinary redraws and
are cancelled when their slice, tool, plane, series or mode changes. Unsupported planes explain the
restriction before drawing starts. Loading another series invalidates reconstruction products even
when its dimensions and slice index match; removing the last phantom clears the reference image.

Clinical and reconstruction reference views share the CT preview transform. Reset restores inversion
and slab settings, empty-source controls reflect their actual capabilities, and PNG write failures
are reported truthfully. AI and reconstruction cancellation status can wrap. Model-card and mesh
dialogs explicitly set both background and foreground, fixing unreadable text on dark macOS systems.

The local candidate adds 31 checks to the full suite (1121/1121, exit 0). Validation includes actual
CPU ONNX inference, actual 16² matrix backends, native macOS controls and dialogs, and failing
regressions before each fix. Scope, limitations, commands, fingerprints and results are maintained
in the [functional audit](docs/gui_functional_audit.md). This entry records local verification,
not a push or a remote CI result.

## Unreleased: use CT window previews directly from the original directory (2026-09-04)

The previous fix explained disabled presets but still required a separately HU-declared copy.
That was an unnecessary user workflow for a display operation. CT window previews now have a
separate capability: supported classic CT missing a unit declaration can use a uniform positive
finite slope/intercept for display, without changing its raw array or HU-analysis capability.
Global presets, independent view presets, manual WW/WL, MPR and positive-affine slab projections
share that display transform. The renderer transforms only the displayed plane, avoiding another
full-volume allocation. Both the sidebar and image overlay identify unconfirmed-unit previews.
No file copy, tag edit, unit-confirmation dialog or AI startup is required for these previews.

Explicit non-HU units, localizers, multi-energy images, unsupported modality LUTs and inconsistent
or invalid transforms still use raw-value sliders. HU measurements, AI and follow-up retain the
existing stricter contract. The optional declaration tool is no longer the recommended route for
display; README wording now describes its histogram checks as plausibility, not independent proof.

Local candidate based on `ed723b2`: the new UI regression failed **12/15** checks before the fix
(exit 1). The expanded targeted run passed **79/79** (exit 0), covering all six preset pixels,
independent presets, three-plane orientation, max/min/mean slabs, manual adjustments, reset,
reload, failed load, language and mode changes, plus rejected metadata combinations. A read-only
load of the original 233-slice RIDER directory matched the independently calculated middle-slice
pixels for all six windows exactly; HU remained false and no model inference ran. Local temporary
logs: `/tmp/gui-0904-preview-red.log`, `/tmp/gui-0904-preview-targeted.log` and
`/tmp/gui-0904-preview-real.log`. These candidate checks do not establish remote CI results.

Final local validation in `dicom_gui`: full suite **1090/1090**, data-independent subset
**979/979**, Ruff and `git diff --check` passed (exit 0). The first full run was rejected
because the README change invalidated its documented diff digest; both README languages and
the digest in `docs/project_report_zh.md` were synchronized before the passing rerun. Final logs:
`/tmp/gui-0904-preview-full-verified.log` and `/tmp/gui-0904-preview-subset-verified.log`.
Native macOS (`cocoa`) was restarted on the original directory and confirmed 233 slices,
all six preset buttons enabled, HU false and no AI thread. The inspected native screenshot is
`/tmp/gui-0904-preview-native.png`; no dataset, saved mask, model, result or submitted PDF changed.

## Unreleased: window controls, dark sidebar and state transitions (2026-09-04)

Validation provenance: the checks below were recorded locally on the candidate based on
`02cd217`, before commit and push. They do not establish remote CI results.
The reported silent CT-preset buttons and light sidebar were reproduced with the original
233-slice RIDER input. Two different causes were involved: this input has no declared HU units,
so CT presets are intentionally unavailable, while the scroll content inherited the system's
light background. The HU-unit contract is unchanged. Unknown-unit data now receives a display
window based on its stored-value range (excluding padding), corresponding slider limits, visible
explanations beside the disabled presets, and distinct disabled styling. `windowing.py` contains
the pure display calculation; it neither changes voxels nor infers physical units.

The same investigation reproduced and corrected related state failures:

- Global preset buttons, slider edits and the first right-drag now clear visible views' independent
  presets before rendering. Clicking a preset with unchanged WW/WL still refreshes the image.
- Both sidebar tabs style their scroll content, viewport and scrollbar; the slab-thickness spinbox
  also follows the dark theme.
- Language changes and MPR's internal retranslation retain viewer-only status. Reset uses the
  current series' display default and synchronizes the MPR button caption.
- A successful series change clears the previous visible organ statistics and leaves a now-disabled
  ROI/crop/tracking tool. A failed load preserves the existing comparison session. Slice-slider
  signals are blocked while the new volume and its capabilities are being installed.
- Comparison mode disables clinical controls its renderer does not use. Reconstruction hides
  those controls; both modes also disable unsupported clinical annotation tools. Returning to
  clinical mode restores available tools and preserves noncanonical geometry restrictions.
- Expanding from one view to two/four renders newly visible views before fitting their images,
  preventing blank views and stale slices. Initial visibility is set synchronously, so a delayed
  startup callback cannot override a subsequently selected mode.

Regression evidence: before these fixes, the first two targeted runs recorded **14/15** and
**5/7** failed checks; a later layout probe recorded **4/4** failed checks. All three commands
returned exit 1. These were actual pixel/state/render checks, including unchanged slider values,
first mouse events, light system palette, and hidden-view transitions. The checks are registered
in both local full-data and `SKIP_REAL_DATA=1` runs. No dataset, saved mask, model, experimental
result, or submitted PDF was changed. The existing HU-declared RIDER copy was inspected with AI
startup stubbed; all six presets worked, but its old cache lacks the current axis contract and
is not a validated replacement cache. No real model inference was run.

An additional mode-tool probe failed **2/10** checks before the tool-state fix. During layout
integration, a full run had all assertion checks pass but caught a Qt-slot `IndexError` from the
old delayed startup callback; that run returned exit 1 and was rejected. Rendering and initial
visibility are now synchronous, with only size fitting deferred to the next layout cycle.

Validation in the local `dicom_gui` environment (all commands returned exit 0):

| Scope | Command / observation | Result |
|---|---|---|
| Local full-data | `python -u tests/test_gui.py` | 1048 / 1048 checks passed; no uncaught Qt exceptions |
| Local data-independent | `SKIP_REAL_DATA=1 python -u tests/test_gui.py` | 937 / 937 checks passed; no uncaught Qt exceptions |
| Lint | `ruff check .` | Passed |
| Native macOS Qt | `QT_QPA_PLATFORM=cocoa`, original RIDER load, both tabs rendered | Both scroll backgrounds `#12141a`; raw shape `(233, 512, 512)`, HU capability false |

The full and subset logs are at `/tmp/gui-0904-full-verified.log` and
`/tmp/gui-0904-subset-verified.log`; native-render observations and screenshot are at
`/tmp/gui-0904-native.log` and `/tmp/gui-0904-native-after.png`. These are local temporary
artifacts, not committed evidence or remote CI. The original dataset remains raw-value
viewer-only; normal loading of the HU-declared copy would start AI if no valid cache is available.

Runtime/test source identity: `shasum -a 256 main.py ui_builder.py compare_lab.py recon_lab.py
style.qss windowing.py tests/test_gui.py | shasum -a 256` gives
`d8e0e7b779884ef6e7a2e2a01e952b2e5514756f022518537434b1f01e54b5f6`.

## Four public-claim boundaries tightened, and the emphasis gate extended to single asterisks (2026-08-30)

An external multi-axis review of the repository found, among a larger set of presentation
suggestions, four places where a public claim reached past what the surrounding text had
established. Only those were acted on; the presentation items were deliberately left alone rather
than widening the freeze.

**`experiments/README.md` summarised what its own table refuses to summarise.** The opening scope
note said flatly that "the segmentation experiments replicate `ai_engine`'s preprocessing and
sliding-window inference", while the per-producer table 106 lines below records that `seg_multi.py`
and `seg_spacing.py` *call* `ai_engine` at runtime and that their committed artifacts predate
`2a50e37`. The opening note now carries the same per-producer split the table does.

**The technical report's conclusions outran its own sections in three places.** It listed
"saturation, filter inversion, iterative robustness" as reproduced qualitative conclusions, when
§3.2(a) had reinterpreted the saturation as a discretisation floor and §3.2(c) had withdrawn the
ART-over-SIRT ranking. Note what is and is not withdrawn: the saturation *is* real and *is*
measured — what does not follow from it is a dose operating point. The ordering is a property of
the fixed iteration counts, not of the solvers. Separately, "validate the pipeline" and "validates
its pipeline" both appeared unqualified in a document whose §4.1 states that no segmentation
evidence ever ran on the product's DICOM (LPS) orientation; both now name the RAS path they were
measured on.

**Two README claims.** "The model's identity was established by measurement" overstated what the
same file calls a strong inference two sections later — what measurement established is the label
mapping; the step to a particular upstream release remains an inference. The identical sentence in
`docs/manual_en.md` was corrected with it, since fixing one and leaving the other is the failure
mode this project has recorded before. And the right-upper-lobe Dice of 0.727 sat in a row opening
"Across 57 public CTs" without its own denominator; recomputed from `seg3d_teacher_dice.csv`, that
lobe is present in 31 of the 57, and both READMEs now say so.

**The emphasis gate only ever checked `**`, and said so.** Its docstring named single asterisks as
out of scope — but a bare `*` leaks a literal asterisk into GitHub's output exactly as an unpaired
`**` does, and `docs/technical_report.md:93` carried one (`the *one of the smallest gains`, which
was also missing an article). The gate now runs the same flanking-and-pairing pass at both
delimiter lengths. Escaped `\*` footnote markers and list-item bullets are not delimiters and are
excluded: the repository's two escaped markers at `technical_report.md:85` and `:89` are correctly
passed, and were the only apparent hits before that exclusion was added. Calibrated against
`markdown-it-py` across all fifteen Markdown files — zero unpaired delimiters at either length.
Reverting the detector to `**`-only fails the new known-bad sample; writing the stray asterisk back
into the report fails the repository scan.

The preprint's Limitations list was also renumbered — its items ran (i) (ii) (iii) (iv) (vi) (vii)
(v), with (v) last.

No experiment was re-run, no result, model, PDF or product code was touched. Local counts move to
**1013** full-suite and **902** `SKIP_REAL_DATA=1` checks; the 19-check CI shortfall quoted earlier
is now bound to the commit it was measured at, since the local subset has since grown.

## The tagged release is the first one remote CI actually covers (2026-08-28)

`v1.1.0` is tagged at `5c5e80741e7290ca8eee430e82f29ee179d85fa0`, and GitHub Actions run
`33156906344` carries that exact 40-character `headSha` — not a short-SHA approximation. It reports
878 PASS / 0 FAIL, 87% coverage over 4,005 statements, and Ruff PASS, with both jobs green under
`workflow_dispatch` (the repository's `push` trigger still does not fire; every green run to date
has been dispatched by hand). The previous exact-SHA evidence was `2e9b700` at 520 PASS and 81%
coverage, and it covered nothing after itself.

**The 19-check shortfall against the local subset's 897 is structural, and it reproduces a number
measured independently.** A clean clone checked out earlier in the same round ran 848 against a
local 867 — also 19 — and that gap decomposes into eight weight-checksum checks and eleven
learned-reconstruction checks that need artifacts the repository does not distribute. Two
independent measurements agreeing on the same decomposition is stronger than either alone: nothing
is being silently skipped.

**Recording this in the repository contradicts the practice the README itself had recommended**,
which was to keep the run and `headSha` out of the tree so that documenting a CI result does not
create a commit the result cannot cover. That regress is real and does not go away here — the
commit carrying these very words is not covered by run `33156906344`. What changes the calculus is
that the evidence now attaches to an immutable tag rather than to a moving branch: `v1.1.0` stays
verifiable regardless of what documentation is written afterwards. The README now says so
explicitly instead of leaving the reader to notice.

## Bold that never rendered, and two gates for claims nothing was checking (2026-08-28)

**Sixteen bold spans on the public pages were broken, and had been for as long as they existed.**
`**非临床器械：**无监管认证` renders on GitHub as the literal characters `**非临床器械：**`, not as
bold. The cause is CommonMark's flanking rule: a closing `**` that sits immediately after a
punctuation character and immediately before a letter is not right-flanking, so it cannot close.
Full-width CJK punctuation — `：` and `」` — is punctuation by that definition, and Chinese text puts
it exactly there. The mirror case breaks openers: `**「加载 DICOM 目录」**` opens with `**` followed by
`「`, which is not left-flanking after a CJK letter. Six spans in `README.zh-CN.md` and ten in
`docs/manual_zh.md` failed this way — the Chinese landing page and the Chinese manual, i.e. the
first thing a Chinese-reading visitor sees. The fix moves the punctuation outside the emphasis
(`**非临床器械**：`, `「**加载 DICOM 目录**」`), which is also the better typography. Verified against
GitHub's own `/markdown` rendering endpoint, not only a local parser.

**A seventeenth was mine, and it was invisible precisely because it did not break.** While adding
the orientation caveat to `docs/technical_report.md` I left `orientation****.` — four asterisks.
That renders without a stray `**`, because the extra pair silently nested one strong span inside
another, quietly extending the emphasis over an entire preceding clause. A defect that renders
cleanly is worse than one that does not: nothing prompts anyone to look.

**Neither would have been caught by anything.** The wording gate reads Markdown as plain text and
never asked whether it renders. The new gate implements CommonMark 0.30 left/right-flanking plus
delimiter matching in pure stdlib — `markdown-it-py` is not a product dependency and cannot enter
the data-independent subset. Its first version was calibrated by agreeing with that parser
delimiter for delimiter on the two broken files (12 and 20), and an earlier draft of this entry
claimed agreement "everywhere else" as well. That was false: on `docs/technical_report.md` the
implementation reported 2 where the parser reported 0 — the four-asterisk case, which pairs
without leaking. The two never agreed everywhere, because they are not measuring the same thing.

**A review then showed the gate could be switched off by deleting one line.** Its coverage check
compared a hardcoded document list against a count derived from that same list — emptying the list
left the suite green with an unchanged total. The scan is now driven by `os.walk` with no list at
all, and the guard is that enumeration finds at least twelve files. The same review found the
detector's blind spots and one branch that no known-bad sample reached: consecutive blockquote
lines were treated as one block, so a bad delimiter inside a large quote containing tables and
lists could be paired off by an unrelated one, and table cells were not separated. Both are fixed
— blockquotes are unwrapped and re-split recursively, each table cell is its own inline context —
and three known-bad samples now exercise the previously untested branch. Validation was redone by
injecting both failure shapes into every real `**…**` across all fifteen Markdown files: 2842
injections, zero misses, zero false positives. The one apparent false positive turned out to be
the *oracle's* error, not the detector's: `MarkdownIt("commonmark")` does not parse tables, so it
paired emphasis across table rows. GitHub, which does parse them, leaks the delimiters exactly
where the detector said it would.

**A later GitHub-renderer check found a third failure shape that those mutations never generated.**
In `**HU …：**每一层`, the punctuation-adjacent delimiter cannot close but can open; later valid
strong spans can then absorb and balance it, leaving no unmatched `**` even though GitHub bolds a
long unintended range. A target-shaped known-bad first made the gate fail. The detector now reports
that opener immediately; affected punctuation in the README and manual now sits outside emphasis.

**The project's only cryptographic self-attestation now has a gate.** `docs/project_report_zh.md`
invites the reader to recompute a SHA-256 over the README diff against a fixed baseline. That value
had already gone stale once — the document says so itself — and the write-up's own recommendation
was that nothing enforced recomputation. It does now, and deliberately does not reimplement the
recipe: the baseline SHA and the command are parsed out of the document, the command's shape is
asserted to match what the test computes, and only then is the digest compared. Full suite only:
it needs the baseline object, and CI's `actions/checkout@v4` is a shallow clone.

**One ordering claim was false and a second was not.** "The gallbladder is the structure most
fragile to spacing ablation" does not survive recomputation from
`seg_spacing_per_organ.csv` — the left adrenal gland drops 0.346 against the gallbladder's 0.272,
so the gallbladder is second. Applying the same rule to `experiments/README.md`'s five largest
per-organ gains first produced a contradiction (every organ at `n=1` against a documented `n=52`),
which turned out to be a BOM in the CSV header collapsing the join key in the *checking* script,
not an error in the document: with the encoding handled, `+0.0479 / +0.0256 / +0.0247 / +0.0239 /
+0.0238` reproduces the documented five to three decimals, along with two losing organs and 54 of
59 cases improving. Superlatives are a claim class that recomputing individual numbers does not
test, since every number in the sentence can be right while the ranking is wrong.

**Turning that rule into a gate immediately found a third.** `experiments/README.md` said the right
upper lobe is "present in 31 of 57 cases against 50–53 for the others" — `lung_middle_lobe_right`
appears in 48, so the range is 48–53. The claim it supports (that lobe is both weakest and rarest)
survives; the bracket was simply wrong. The new gate recomputes three ordering claims from the
committed CSVs — which organ degrades most under spacing ablation, which lobe is weakest and
rarest, and the five largest per-organ gains — and compares them against the documents, parsing the
five-gain sentence out of the prose so that drift on either side fails. It also keeps a semantic
guard so the retracted "gallbladder is the most fragile" wording cannot return by paraphrase.
Reverting the bracket, reinstating the gallbladder wording, and changing one `n` each produce a
failure. The gate reads the CSVs with `utf-8-sig`: their headers carry a BOM, which had already
silently collapsed a join key once during checking and produced a contradiction that looked like a
documentation error but was not.

**The user-facing manuals had been missed by the orientation round entirely.** Both quote the
spacing ablation's 0.922 → 0.799 while containing not a single mention of `RAS` or `LPS`; so did
`docs/spacing_contract.md`'s ablation table, whose only orientation note sat beside a different
experiment several sections away. The existing wording rules are **line-level** co-occurrence
checks and cannot see this: the number and its qualifier are usually paragraphs apart. A
document-level rule now requires any Markdown quoting the segmentation figures to carry the
qualifier somewhere in the file, and it is enumeration-driven rather than list-driven.

**That rule immediately surfaced a tenth document no list contained.**
`experiments/results/seg_mapping.md` — the label-mapping evidence record — describes its own
inference as "GUI 同款" (the same as the GUI's) with no orientation qualifier, which is precisely
the claim shape the axis round retracted. It is frozen evidence under `experiments/results/` and is
not edited here; its qualifier lives in `experiments/README.md`'s directory-level scope note, and
the gate's exemption asserts that note actually exists rather than merely listing the file. The
same file was also absent from the wording gate's document list and has been added to it.

**Three gates were added to close what a clean clone showed was unguarded.** A clone of the
repository was checked out and run: nothing unexpected is missing (the gaps are the undistributed
weights, external experiment data, and the private agreements), the subset passes there, and the
867-vs-848 difference decomposes into eight checksum checks and eleven learned-reconstruction
checks that need artifacts the repository does not ship. What the clone did expose is that nothing
compared the product's third-party imports against `requirements.txt` — the existing inventory
checker only reconciles local modules. `shiboken6` is imported directly and is not declared; it
arrives as a hard dependency of PySide6, and the new gate accepts that only because the reason is
written at the import site, which it verifies.

**The most zero-grade property of all had no check at all.** This repository is public; a single
committed DICOM slice, weight blob, external dataset file or private agreement voids everything
else in it. Nothing verified that — `git ls-files` appeared zero times in the entire suite, and
`.gitignore` does not cover the case that matters, since it stops governing a file once that file
is tracked. The new gate reads the tracked file list and rejects five categories, with a synthetic
sample per category proving each one is live, and asserts a lower bound on the tracked count first:
if the listing ever comes back empty, every pattern match trivially passes. The repository is clean
today — 147 tracked files, zero hits in all five categories — but that was previously an unverified
belief rather than a measurement.

**The self-referential coverage check was not the only one.** After fixing the one the review
found, a scan for the shape — an assertion comparing a collection against the very list it was
built from — turned up two more, in the wording gate and in the retraction gate. Both computed
`len(texts) == len(docs)` where `texts` is built by iterating `docs`, so both would have passed on
an empty list, silently reducing the two gates that protect retracted scientific claims to zero
coverage. Both now assert a floor on the list size as well; emptying either list fails. The scan
also cleared the rest: the other eighteen multi-`len` assertions compare independently produced
quantities, and the twelve `check(True, …)` calls are the "reached this line without raising"
idiom, which is load-bearing because the runner records uncaught exceptions as failures.

Local counts move to **1008** full-suite and **897** `SKIP_REAL_DATA=1` checks.

## Three independent audits, and what they found the axis fix had left behind (2026-08-27)

Three read-only reviews ran in parallel over documentation, test effectiveness and freeze-readiness.
None of them said the project was ready. What they found was not fabrication — one auditor
recomputed roughly sixty quantitative claims from the committed CSVs and every one of them held —
but **disclosure that had not kept up with the code**, plus guards that were not guarding.

**The axis caveat existed in exactly one file.** After the in-plane axis fix, the scope note went
into `experiments/README.md` and nowhere else: `RAS`, `LPS` and `inplane_axes` appeared **zero
times** across the eight reader-facing documents, while both READMEs still said the experiments
measure "the shipped pipeline" and the technical report still said they "characterise the shipped
system". The note is now in seven documents and in `_load_saved_mask`'s docstring.

**The first two figures a reader sees showed the patient mirrored.** `gui_axial_segmentation.png`,
`gui_confidence.png` and `gui_mpr_triplanar.png` all predate the fix — liver on the `L` side,
spleen and stomach on `R`, spinous process toward `A` — and sat a few lines above a newly added
sentence inviting readers to check laterality by eye. All three are re-shot from the same case
(CT-Lite s0029). Two of them land on the same slices as before (`249/422` and `279/422`) with
per-organ volumes unchanged — mirroring never altered voxel counts, only which side they landed on.
The third, `gui_confidence.png`, does not: it previously showed a 126-slice run with 18 organs, and
now shows the same 422-slice run as the others, so its numbers legitimately differ. The confidence
sentence in both READMEs was rewritten accordingly — at `conf 0.91` the gallbladder is no longer
below the 0.9 flag threshold, so calling it "flagged" and "least certain" would have been false
twice over. **Two** organs fall below 0.9 in this run — prostate 0.82 and thyroid 0.37 — the
latter being three voxels (0.01 mL); the gallbladder's 0.59 is the lowest 5th percentile **among the
rows the panel has room to show**. That is what the sentence now says, and it is derived from the
inference artifacts rather than from reading the figure. Two earlier rewrites got it wrong in
different ways: the first asserted no organ fell below 0.9 while a figure two lines above showed one
that did; the second said exactly one did, having checked only the rows a screenshot happened to
display.

**The project's only cryptographic self-check was failing.** `project_report_zh.md` publishes a
SHA-256 of the READMEs' diff against a fixed baseline and requires it be recomputed whenever a
README changes — a rule the file records as having been broken once before. Seven README-touching
commits later it was broken again. Recomputed and verified equal.

**Four assertions were decoration.** The worst guarded a defect that had already happened once:
`volume_mask` overlay on coronal and sagittal planes. Its assertion compared `volume_mask[z].shape`
against `volume_hu[z].shape` and counted the block the test itself had just written — both true by
construction, touching no product code. Restoring the historical bug (`plane == AXIAL` only) left
the suite green. It now drives the product's own render and inspects `mask_item`, which is where
the overlay actually lands; the mutation is caught precisely on the two non-axial planes. The other
three: a HUD organ check satisfied by the slice number `'5'`, a recon title check whose second
disjunct was stale state from a previous iteration, and a crop-statistics check that verified the
strings `mm` and `HU` but never the numbers — pinned now to 225.00 mm² and 60.0 HU, values that
immediately caught the figure this entry's author had computed by hand.

Also: `dicom_geometry`'s non-finite rejection for IPP, IOP and PixelSpacing had no coverage at all
(NaN is a documented trap in this project, and `_finite_vector`'s `isfinite` guard could be deleted
with the suite still green) — three assertions added; the skipped 12-assertion block in
`test_dl_recon_guard` now announces itself instead of vanishing silently on CI; and a claimed
row-permutation control in `preprint_recon.md` was **removed rather than corrected**, because the
before/after conditioning numbers it quoted exist in no commit and no artifact — the control was
never scripted.

## Reimplementations of the inference pipeline are now checked against the product (2026-08-27)

Two experiment scripts reimplement `ai_engine`'s preprocessing and sliding window rather than call
it — `seg_validate.run_onnx` and `seg3d_teacher.run_onnx` — and the segmentation evidence is
produced on those, not on the product path. Nothing detected when a reimplementation drifted. Both
known divergences were found by hand and long after they appeared: the final-window pullback
(`2a50e37`), and the in-plane axis order, during which the product labelled the patient's right
lung as the left one while every test stayed green.

The check compares the **tensors handed to ONNX**, not the resulting labels. That is where drift
actually lives — normalisation, window slicing, padding, axis order — and it avoids a trap the
first attempt fell into: a synthetic volume fed to the real model segments almost nothing, so
comparing label maps passes trivially with both sides empty. (The first version of this test did
exactly that; it was caught by an assertion demanding the mirrored path produce more than
background.) Both sessions are replaced by recorders, so the test needs no weights and takes well under a
second. It runs in the full suite only: `seg_validate` imports `nibabel` and `matplotlib` at
module level and neither is a product dependency, so CI — which installs `requirements.txt` —
cannot import it. That was found by CI failing on the first push of this change, which is the
same class of mistake the test itself is about: something that works locally and does not
exist on the path that actually runs.

Two size regimes divide the work. With `Z` a multiple of 32 the tensors must match element-wise —
covering normalisation, slicing, padding and axis order at once. With `Z` not a multiple, the
mirrors still use the pre-pullback form, so a difference is expected; the test requires it to fall
**only** on the last block. That pins the declared divergence instead of merely tolerating it: if
it ever spreads to another block, the test fails. Mutations confirm both halves bite — changing
the product's normalisation divisor fails two assertions, reverting the pullback fails one.

## The suite now reports its own count, because the log loses lines (2026-08-27)

Test totals were obtained by counting `PASS` lines in the output. Two runs of CI on the *same*
commit `3f0aa24` disagreed: run `33042859794` printed 797 such lines and run `33043192428` printed
798. Both finished green, both ended in "全部通过", and neither recorded a FAIL — the missing entry
(`pix=1200 slope=1 intercept=0`) exercises nothing outside the repository, the pinned dependency
versions match `requirements.txt` exactly, and the two preceding CI runs printed all four cases. So
the discrepancy is in the log, not in what executed; a counting method built on grepping that log
inherits the noise.

`check()` now records each outcome and the runner prints
`CHECKS total= passed= failed=` before its verdict. That line is the count to quote. It also makes
a dropped line visible rather than silent: when the `PASS` lines in a log do not add up to the
number the runner reports, the log is the thing that is wrong.

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
