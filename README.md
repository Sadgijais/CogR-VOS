# CogR-VOS

**Cognitive Memory and Event-Driven Reasoning for Zero-Shot Referring Video Object Segmentation**

A training-free system that keeps track of *who the target is* across a video, notices when tracking stops being trustworthy, and calls an expensive reasoning model only at those moments.

> Status: V0 baseline complete (J&F 50.48 on Ref-DAVIS17 val). V1 (memory) in design.

---

## Problem

Given a video and a sentence such as *"the boy in the red shirt"*, produce a pixel mask of that target in every frame. The target may be occluded, leave the frame, come back, or stand among look-alike distractors.

Existing tracking and segmentation models propagate masks well but accumulate **identity drift**: the tracker's memory is written from its own predictions, so a slip onto a distractor reinforces itself, and nothing inside the model notices. Running a vision-language model (VLM) on every frame fixes this but is far too expensive. Every LLM/VLM-based R-VOS system to date reasons on a fixed schedule chosen before propagation begins (one pivot frame, one key segment) and never detects that propagation has since failed.

## Approach

Three frozen models (no training, no fine-tuning) plus control logic between them:

| Component | Role |
|---|---|
| Grounding-DINO | Text to candidate boxes (frame 0, and again during recovery) |
| SAM 2 | Box to mask, and mask propagation through the video |
| VLM (API) | Adjudicates between candidates; never produces pixels |

Four contributions sit on top of the backbones:

1. **Semantic-Temporal Target Memory (STTM):** a bounded store of the target's identity with six fields: semantic, appearance, motion, spatial, context, reliability. Writes are gated on track validity.
2. **Tracklet Coherence Score (TCS):** a cheap per-frame score `c_t = w1*mask_affinity + w2*motion + w3*appearance + w4*semantic`, read from cached embeddings so it needs no VLM call. Bands: HIGH continue, MEDIUM monitor, LOW recover.
3. **Event-driven VLM reasoning:** the VLM fires only on the LOW branch, never on the per-frame path.
4. **Memory-grounded re-identification:** after loss, retrieve strong memory states, propose candidates with Grounding-DINO, let the VLM compare them, resume propagation. If no valid candidate exists, emit an empty mask (target absent).

**Training-free claim, stated precisely:** all three backbones are frozen; the only free parameters are four coherence weights and two thresholds, tuned on Ref-DAVIS17 only and then frozen. They are never re-tuned on MeViS, Ref-YouTube-VOS or Long-RVOS.

## Hypotheses

| ID | Claim | Evidence |
|---|---|---|
| H1 | Semantic-temporal memory improves long-term identity preservation | Long-RVOS, MeViS |
| H2 | Explicit coherence estimation identifies unreliable tracking | Long-RVOS, reported as DRE and NRE |
| H3 | Event-driven invocation reduces VLM calls without losing quality | All datasets, VLM calls per frame |
| H4 | Memory-grounded re-ID improves recovery after disappearance | Long-RVOS recovery rate |

## Ablation ladder

| Variant | Memory | Coherence | VLM | Re-ID | Tests |
|---|---|---|---|---|---|
| V0 | - | - | - | - | Zero-shot backbone baseline |
| V1 | yes | - | - | - | H1 |
| V2 | yes | yes | - | - | H2 |
| V3 | yes | yes | event | - | H3 |
| V4 | yes | yes | event | yes | H4 and system value |
| V5 | yes | yes | every frame | yes | Upper bound (quality ceiling, full cost) |
| V6 | yes | yes | once at clip start | yes | Scheduled reasoning (AL-Ref-SAM 2 / VISA style) |

The central experiment is **V4 vs V6 at a matched VLM call budget**. If event-driven reasoning cannot beat one-shot reasoning at the same number of calls, the contribution is the memory, not the trigger.

## Current result: V0 baseline

Grounding-DINO on frame 0, SAM 2 propagation to the end, nothing else. Full Ref-DAVIS17 val split: 30 videos, 61 objects, 480p.

| Metric | Value |
|---|---|
| Mean J | 56.01 |
| Mean F | 44.94 |
| **Mean J&F** | **50.48** |
| Published Grounded-SAM 2 reference | 66.2 |

Failure breakdown (61 expressions):

| Mode | Count | Share |
|---|---|---|
| poor_mask_quality | 30 | 49% |
| grounding_failure | 16 | 26% |
| ok | 10 | 16% |
| track_lost_no_distractor | 3 | 5% |
| identity_drift | 2 | 3% |

**Reading it.** The gap to 66.2 is mostly a mask-quality ceiling: V0 uses `sam2.1-hiera-tiny`, the only variant that fits a 4 GB GPU, and F sits about 11 points below J (right area, rough edges). 50.48 is kept as the working baseline; V0 is deliberately not re-tuned to chase 66.2. Identity drift plus track-lost account for only 5 of 61 expressions, so Ref-DAVIS17 serves as no-regression evidence, and H1-H4 are decided on Long-RVOS.

The exact code that produced this number is tagged `v0-baseline-50.48`.

## Repository layout

```
CogR-VOS/
├── README.md
├── baseline_v0/        V0 code: staged pipeline, config, subsets, tools, cached grounding
├── results/            one folder per variant (v0_baseline/, later v1_..., v2_...)
├── literature/
│   ├── papers/         61 PDFs in 6 topic folders
│   └── notes/          corpus notes and comparison tables
```

## Running V0

Run everything from inside `baseline_v0/`, where `config.yaml` lives. Details and flags are in `baseline_v0/README.md`.

```
stage 1  stage1_grounding.py         Grounding-DINO on frame 0    -> grounding/<video>.json
stage 2  stage2_propagation.py       SAM 2 video propagation      -> predictions/
stage 3  stage3_evaluate.py          J and F                      -> results/*.csv
stage 4  stage4_failure_analysis.py  drift / recovery taxonomy    -> failure_analysis.csv
         visualize.py                overlays and contact sheets
```

Stages 1 and 2 run as separate processes because the two models cannot be resident together on a 4 GB GPU. Grounding output is cached (top-5 boxes per video) so later variants reuse one frontend pass.

**Environment notes**
- WSL2 Ubuntu, stock cu126 PyTorch wheel, Python 3.10 or newer.
- Install SAM 2 with `SAM2_BUILD_CUDA=0` (the optional CUDA extension is where most installs fail).
- Use HuggingFace `GroundingDinoForObjectDetection`, not the IDEA-Research repo (no custom CUDA op).
- Evaluate at **480p**, with each object's first and last frame excluded, as the official DAVIS evaluator does. Full-resolution numbers compare to nothing published.
- Keep the dataset on the ext4 filesystem inside WSL, not `/mnt/c`.
- If HTTPS calls reset inside WSL, run `sudo ip link set dev eth0 mtu 1200`.

Model and dataset weights are not stored in this repo (see `.gitignore`).

## Metrics

Beyond J, F and J&F, the project reports finer instruments so that gains can be attributed:

| Metric | Measures | Hypothesis |
|---|---|---|
| DRE | Frames where the tracker latched onto the wrong thing | H2 |
| NRE | Frames wrongly declared absent | H2, H4 |
| ADQ | Quality of genuine absence detection | H4 |
| Recovery rate | Re-acquired at IoU >= 0.5 within 10 frames of reappearance | H4 |
| VLM calls per frame | Reasoning sparsity | H3 |

Definitions used in `failure_analysis.csv`: an **ID drift event** is IoU on the target below 0.10 while it is visible, and IoU above 0.50 on some other annotated object (contiguous frames count as one event).

## Datasets

| Dataset | Job |
|---|---|
| Ref-DAVIS17 | Development and controlled evaluation; no-regression evidence |
| Ref-YouTube-VOS | General R-VOS performance |
| MeViS | Motion and temporal language reasoning |
| Long-RVOS | Occlusion, disappearance, reappearance, recovery (where H1, H2, H4 are decided) |

Reporting rule: every number in a results table comes from the full Ref-DAVIS17 val split. The frozen `smoke` (6 videos) and `ablation` (15 videos) subsets exist only to speed up the inner loop and are stratified from ground-truth masks toward identity loss and recovery.

## Literature

61 papers in `literature/papers/`, grouped by theme: referring VOS, Grounding-DINO, SAM 2 and tracking, memory-guided VOS, VLM-guided VOS, and event-driven reasoning. The closest training-free rival is AL-Ref-SAM 2 (74.2 J&F on Ref-DAVIS17, reasoning once per clip on a fixed schedule).

## Compute constraints

Development runs on a 4 GB RTX 2050 under WSL2. This shapes the design: no two large models co-resident, SAM 2 tiny only, an API-based VLM, and disk-cached stages. Full Long-RVOS runs and the V5 always-on comparison are planned for Kaggle T4 GPUs.

