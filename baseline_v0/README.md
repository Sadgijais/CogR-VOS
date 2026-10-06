# CogR-VOS — V0 Baseline (Grounding-DINO + SAM2)

> **Status note (2026-10-05).** This file is the original V0 setup guide, written in a sandbox *before* any real run. Its statements that stages 1 and 2 are "NOT verified" or a "first draft", and the 66.2 target, are historical. V0 has since been run on the full Ref-DAVIS17 val split (J&F 50.48; 66.2 was deliberately not chased), and V1 (J&F 50.97), V2 (passive 50.97, gated 50.40) V3 (real VLM 50.58) and V4 (memory-grounded re-identification, 50.87 memory / 50.86 VLM, no gain) have been built on top of the same pipeline in this folder. For current results see the root `README.md` and `results/README.md`; the pre-registered plans are `V1_TUNING_PLAN.md`, `V2_TUNING_PLAN.md`, `V3_PLAN.md` and `V4_PLAN.md`.

This is the zero-shot backbone baseline from the CogR-VOS ablation ladder:
Grounding-DINO finds the target in frame 0, SAM2 propagates a mask to the
end of the video, and nothing else runs — no memory, no coherence
estimation, no VLM reasoning. Its job is to give every later contribution
(V1..V5) an honest, correctly-measured zero to be compared against.

**Target: 66.2 J&F on Ref-DAVIS17** (Grounded-SAM2, AL-Ref-SAM2 AAAI 2025
ablation table — the same two frozen models in the same arrangement, so
this is a like-for-like number, not a rough guess).

## What has actually been verified, and what hasn't — read this first

Being direct about this matters more than anything else in this file.

**Verified, right now, in the environment that wrote this code:**
`tools/test_fixture.py` builds a tiny synthetic dataset with hand-designed,
known-correct answers and runs it through the real stage 3 (J/F) and stage 4
(drift / recovery / false-absence) code. All 11 checks pass — see the
"Stage 3/4 are already verified" section below for the actual output. This
proves the *math* is right: IoU, the boundary F-measure, the DAVIS
first/last-frame exclusion, the drift-event definition, the
disappearance-then-recovery definition, and the false-absence count.

**NOT verified — could not be, in this sandbox:** stage 1 (Grounding-DINO)
and stage 2 (SAM2 propagation). This environment has no GPU and does not
have the Ref-DAVIS17 files on disk. Those two scripts are written carefully
against the documented HuggingFace Grounding-DINO API and the documented
SAM2 video-predictor API, but they are a **first draft to debug against
your real data**, not pre-tested code. Expect to fix small things on first
run — a path that doesn't match your download's folder names, an API
signature that shifted a SAM2 version, that kind of thing. This is normal
and does not mean the design is wrong.

Say this plainly if anyone asks how far along the implementation is: the
evaluation and diagnostic machinery is built and proven correct; the two
GPU-dependent stages are built and ready to debug, not yet run.

## 1. Where to actually run this

**Your own machine, in WSL2 Ubuntu** (per the project's hardware notes — a
laptop RTX 2050, 4 GB VRAM). Reasons: your 112 GB of datasets are already
there; SAM2's own install guide recommends WSL over native Windows; and a
baseline takes several rounds of debugging, which a temporary cloud session
is the wrong place to do it. This document (and all the code) was written
in a cloud sandbox with no GPU — it is meant to be copied onto your machine
and run there, not run from here.

Reserve a cloud GPU notebook (Kaggle T4, Colab, etc.) for later: full
Long-RVOS passes, or anything needing more than 4 GB (a larger SAM2
variant, the "always-on VLM" V5 comparison). Not needed for V0.

## 2. Environment setup (WSL2 Ubuntu)

```bash
# Python >= 3.10, torch >= 2.5.1 required by SAM2.
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Install a CUDA-matched torch build FIRST — check your driver's CUDA
# version with `nvidia-smi` and match it. Do not just `pip install torch`
# and hope; a mismatched build either silently falls back to CPU or fails
# to import. Example for CUDA 12.6:
pip install torch --index-url https://download.pytorch.org/whl/cu126

# SAM2 from source, with the optional CUDA extension DISABLED — building
# it is where most Windows/WSL installs fail, and skipping it only costs
# an optional hole-filling post-process step.
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && SAM2_BUILD_CUDA=0 pip install -e . && cd ..

# Download a SAM2.1 checkpoint (tiny — the only variant that fits 4 GB):
# see https://github.com/facebookresearch/sam2#model-checkpoints for the
# current download links. Save it somewhere and point
# config.yaml -> propagation.checkpoint at it.
```

Copy the Ref-DAVIS17 **480p** archive (not Full-Resolution — see below)
onto the WSL ext4 filesystem itself, not `/mnt/c/...`. Reading JPEGs through
the Windows filesystem bridge is several times slower and will look like
the model being slow when it isn't.

## 3. Point config.yaml at your data

Open `config.yaml` and edit the two paths marked `<-- EDIT`:

```yaml
dataset:
  root: /path/to/Ref-DAVIS17-480p
propagation:
  checkpoint: /path/to/checkpoints/sam2.1_hiera_tiny.pt
```

Expected folder layout (standard Ref-DAVIS17 480p release, plus a
referring-expression file placed alongside it):

```
Ref-DAVIS17-480p/
  JPEGImages/480p/<video>/00000.jpg, 00001.jpg, ...
  Annotations/480p/<video>/00000.png, ...   (palette PNGs)
  meta_expressions/meta_expressions.json
```

If your download uses different folder names, either rename to match this,
or edit the small number of path properties in `common/dataset.py`'s
`DavisLayout` class — that keeps the fix in one place instead of scattered
across four scripts.

## 4. Before running anything on real data: `inspect_dataset.py`

```bash
python tools/inspect_dataset.py --config config.yaml
```

This checks, against your actual files, the four things that are known to
silently produce a plausible-looking but wrong final number (see §8 below):
resolution, the `obj_id`-to-palette mapping, whether `meta_expressions.json`
even has an `annotator` field to be confused about, and basic frame-count
sanity. **Fix everything it flags before moving on.** It has not been run
against real Ref-DAVIS17 files in this sandbox (there are none here) — you
are the first real run of it, and that is expected.

### The one open decision it will surface: which annotator set

Two of the project's own earlier documents disagree:
`CogR-VOS_Dataset_Subset_Protocol.md` says annotator set **0**;
`CogR-VOS_Dataset_Audit_Findings.md` says annotator set **1**. Ref-DAVIS17
ships four expression sets (2 annotators x {first-frame, full-video}
description style). `inspect_dataset.py` will tell you what values are
actually in your file. **Pick one, write it into `config.yaml`'s
`dataset.annotator_set`, and state that choice explicitly in any table you
later report.** The code will not silently guess — `load_expressions()`
raises an error rather than defaulting if it finds an `annotator` field and
`annotator_set` is unset.

## 5. Run order

Stages 1 and 2 are **separate processes on purpose** — Grounding-DINO
(~2-3 GB) and SAM2 (~2.5-3 GB) cannot both be resident on a 4 GB card, so
stage 1 fully exits before stage 2 loads. Start with `dataset.tier: smoke`
in `config.yaml` (a handful of videos) before ever running `full`.

```bash
# Stage 1 — Grounding-DINO on frame 0 of every video/expression.
python stage1_grounding.py --config config.yaml
#   -> grounding/<video>.json   (top-5 boxes cached per expression)

# Stage 2 — SAM2 propagates each expression's top-1 box to the end.
python stage2_propagation.py --config config.yaml
#   -> predictions/<video>/<exp_id>/00000.png, ...

# Stage 3 — score every prediction against ground truth.
python stage3_evaluate.py --config config.yaml
#   -> results/per_video_results.csv, results/summary.csv

# Stage 4 — classify HOW each expression failed, not just how much.
python stage4_failure_analysis.py --config config.yaml
#   -> failure_analysis.csv

# Optional — look at the worst cases.
python visualize.py --config config.yaml --failure-mode identity_drift
#   -> visualizations/<video>/<exp_id>/overlay_*.png, contact_sheet.png
```

Once `smoke` runs cleanly end to end, switch `dataset.tier` to `ablation`,
then to `full` for the number you actually report. **Every number that goes
in a results table must come from the full 30-video val split** — the
subset tiers exist only to make debugging fast; see
`CogR-VOS_Dataset_Subset_Protocol.md`.

If you haven't built the frozen `smoke`/`ablation` subset manifests yet
(they live under a `subsets/` folder this scaffold does not include), set
`dataset.tier: full` in `config.yaml` for now — it will just run every
video, which is slower but always works.

## 6. Stage 3/4 are already verified — see for yourself

```bash
python tools/test_fixture.py
```

This was actually run while building this scaffold. Output:

```
Stage 3 (J/F) checks:
  [PASS] clean_video J: got 100.0, expected 100.0
  [PASS] clean_video F: got 100.0, expected 100.0
  [PASS] drift_video J: got 50.0, expected 50.0

Stage 4 (failure taxonomy) checks:
  [PASS] drift_video n_drift_events: got 1, expected 1
  [PASS] drift_video drift onset: got 20, expected 20
  [PASS] drift_video drift onto: got '2', expected '2'
  [PASS] gap_video n_reappearance_events: got 1, expected 1
  [PASS] gap_video recovery_rate: got 1.0, expected 1.0
  [PASS] gap_video mean_delay: got 2.0, expected 2.0
  [PASS] gap_video n_false_absence_frames: got 2, expected 2
  [PASS] gap_video n_drift_events (none expected): got 0, expected 0

11/11 checks passed.
```

Run it again yourself any time you touch `common/metrics.py` or
`common/failure_taxonomy.py` — it takes a few seconds and needs no GPU.

## 7. Reading your number once V0 finishes

| You get | What it probably means | First thing to check |
|---|---|---|
| ~64-68 | Working. Proceed. | Nothing — move on |
| ~50-60 | Real but degraded | Are you on `sam2.1-hiera-tiny`? Grounding thresholds too strict/loose? `exclude_first_last` actually on? |
| ~30-45 | Something structural | `obj_id`/palette mapping — are you scoring object 1's prediction against object 2's ground truth? Re-run `inspect_dataset.py`. |
| Under 20 | Plumbing | Mask orientation, resolution mismatch, empty predictions, wrong annotator set |
| Over 72 | **Be suspicious, not pleased** | V0 has no mechanism that could beat AL-Ref-SAM2 (74.2). Check for frame-0 leakage into scoring, `exclude_first_last` silently off, or the wrong annotator set inflating things. |

`stage3_evaluate.py` prints one of these warnings automatically if your
number lands outside the expected band.

## 8. Four ways to get a plausible-looking WRONG number (all guarded in code)

1. **Full-Resolution instead of 480p.** Every published Ref-DAVIS17 number
   is computed at 480p. `DavisLayout.check_resolution()` refuses to run if
   frames look like 1920x1080 — don't work around this warning, fix the
   dataset path instead.
2. **`exclude_first_last: false`.** The official DAVIS evaluator drops each
   object's first and last frame (frame 0 is a free-marks giveaway — it's
   the prompt you were given). This is `true` by default in
   `config.yaml`; leave it there for any number you intend to report.
3. **`obj_id` not matching the annotation palette.** Every frame silently
   scored against the wrong object. `tools/inspect_dataset.py` and
   `common/dataset.py:cross_check_obj_ids()` both check this; stage 1 also
   runs the check and prints a loud warning (not an error) if it fails.
4. **The wrong annotator set.** See §4 above — settle this before trusting
   any number.

## 9. What the failure taxonomy actually buys you

A single J&F number tells you the system is mediocre; it does not tell you
*why*, and the why is what determines whether CogR-VOS's contributions
(memory, coherence, event-driven reasoning, re-identification) will
actually help. `failure_analysis.csv` splits every expression into:

- `grounding_failure` — frame-0 box was already on the wrong object.
  **CogR-VOS cannot fix this** — it's upstream of memory and reasoning.
- `identity_drift` — box was right, the track slid onto a distractor later.
  **This is exactly what CogR-VOS is designed to fix.**
- `track_lost_no_distractor` — track collapsed with nothing obvious to
  blame it on. Coherence estimation should catch this even without a named
  distractor.
- `poor_mask_quality` — right object, sloppy edges. Not something
  memory/reasoning fixes — it's SAM2's ceiling.

Expect `recovery_rate` to be blank for almost every Ref-DAVIS17 expression.
The dataset audit measured only 4/61 objects vanishing and 3 returning,
across 2 of 30 videos — Ref-DAVIS17 exists here for no-regression evidence,
not to test recovery. The real evidence for the recovery hypothesis (H4)
lives on Long-RVOS. A blank recovery column on Ref-DAVIS17 is expected, not
a bug — don't spend time chasing it.

## 10. Directory reference

```
config.yaml                # every setting that affects the reported number
common/
  dataset.py                # Ref-DAVIS17 layout, resolution + obj_id guards
  metrics.py                # IoU, J, F, sequence scoring
  failure_taxonomy.py       # drift / recovery / false-absence definitions
  io_utils.py                # palette PNG reading (raw index, not colour!)
stage1_grounding.py         # GPU — Grounding-DINO, frame 0 only
stage2_propagation.py       # GPU — SAM2 video propagation
stage3_evaluate.py          # CPU — J & F scoring
stage4_failure_analysis.py  # CPU — drift/recovery/failure-mode taxonomy
visualize.py                 # CPU — overlays + contact sheets
tools/
  inspect_dataset.py         # run BEFORE stage 1, against your real data
  test_fixture.py            # run any time — proves stage 3/4 math is right
grounding/                  # stage 1 output (JSON per video)
predictions/                # stage 2 output (PNG masks)
results/                    # stage 3 output (per_video_results.csv, summary.csv)
failure_analysis.csv        # stage 4 output
visualizations/             # visualize.py output
```

## 11. Next, after V0 lands near 66

1. Settle the annotator-set decision (§4) and record it in the write-up.
2. Verify Ref-YouTube-VOS's `valid/Annotations/` before scoring anything
   against it — the audit found it looks like a submission-format
   repackage, not real ground truth.
3. Extend the profiler with an RLE decoder for MeViS and missing-file
   absence handling for Long-RVOS (both use different mask encodings than
   Ref-DAVIS17), then freeze those subsets the same way Ref-DAVIS17's were.
4. Start specifying exactly what the Tracklet Coherence Score computes —
   still the largest open design question for V1/V2.
