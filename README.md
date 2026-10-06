# CogR-VOS

**Cognitive Memory and Event-Driven Reasoning for Zero-Shot Referring Video Object Segmentation**

A training-free system that keeps track of *who the target is* across a video, notices when tracking stops being trustworthy, and calls an expensive reasoning model only at those moments.

> Status: V0 baseline complete (J&F 50.48). V1 (Semantic-Temporal Target Memory) complete (J&F 50.97, a no-regression result). V2 (Tracklet Coherence Score) complete (passive 50.97, gated 50.40 is a negative result). V3 (event-driven VLM) built and run once with a real VLM (J&F 50.58): no accuracy gain on Ref-DAVIS17, about 52x fewer VLM calls than every-frame. V4 (re-identification) not started.

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

**Training-free claim, stated precisely:** all three backbones are frozen. The remaining hyperparameters (memory size, write thresholds, coherence weights, band thresholds) are set on Ref-DAVIS17 only and then frozen. They are never re-tuned on MeViS, Ref-YouTube-VOS or Long-RVOS.

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

## Current result: V1 (Semantic-Temporal Target Memory)

V0 plus a bounded identity memory (frame-0 anchor plus up to 3 curated entries) whose stored frames are fed back to SAM 2 as conditioning frames. Same 30 videos, 61 objects and backbones as V0. V1 reuses the V0 pipeline, so its code lives in `baseline_v0/` and its results in `results/v1_sttm/`.

| Metric | V0 | V1 |
|---|---|---|
| Mean J | 56.01 | 56.48 |
| Mean F | 44.94 | 45.46 |
| **Mean J&F** | **50.48** | **50.97** |
| Total inference time | 2126 s | 2794 s (+31%) |
| Peak GPU memory | 721 MB | 805 MB |

Failure-mode counts are identical to V0. This is a no-regression result, not evidence that memory improves segmentation: about 90% of the gain comes from five expressions, and Ref-DAVIS17 has almost no occlusion. H1 is decided on Long-RVOS or MeViS. In V1, only appearance, spatial, reliability and the visible/absent state affect decisions; the semantic, motion and context fields are stored and logged for V2 and V4. See `results/README.md` for details. Code is tagged `v1-sttm-jf50.97`.

## Current result: V2 (Tracklet Coherence Score)

V1 plus a per-frame score `c_t`, a weighted geometric mean of four consistency terms (mask, motion, appearance via DINOv2, semantic via CLIP) with equal, untuned weights. Frozen thresholds tau_low 0.81 and tau_high 0.92 (`baseline_v0/config_v2_final.yaml`, chosen on the 15-video ablation tier by a plan committed before running: `baseline_v0/V2_TUNING_PLAN.md`). Same 30 videos and 61 objects.

| Variant | Mean J | Mean F | Mean J&F |
|---|---|---|---|
| V0 | 56.01 | 44.94 | 50.48 |
| V1 | 56.48 | 45.46 | 50.97 |
| **V2 passive (score logged, memory untouched; ships as V2)** | 56.48 | 45.46 | **50.97** (masks identical to V1) |
| V2 gated (memory written only on HIGH frames) | 55.97 | 44.84 | 50.40 (negative result) |

TCS costs 2.7% of run time and raises peak GPU memory to 1384 MB. Gated V2 failed its pre-registered ship rule (J&F at least 50.47), so passive is V2.

**Honest reading.** The score is a weak failure detector on Ref-DAVIS17: mean IoU is 0.77 on HIGH frames and 0.78 on MEDIUM frames, 44% of alarms are false, and AUROC for degraded frames is 0.56 (the mask term alone scores 0.62, the semantic term 0.45, below chance). Thresholds tuned on the 15-video tier (AUROC 0.746) did not transfer to the full split. Grounding failures (16 of 61 expressions) are invisible to the score. Whether the score works as a trigger is decided on Long-RVOS or MeViS.

## Current result: V3 (event-driven VLM)

The V2 score and its events decide *when* to ask a VLM one small question about crops the tracker already produced ("is this still the same object?", or "is the target visible?" when the mask is empty). Events: disappear, reappear, sudden coherence drop, suspected drift, distractor confusion, with a 10-frame pause between calls. A confident "not the target" answer blanks the mask (abstain). V3 is run as an offline replay of the logged V2 passive run: the VLM never feeds back into SAM 2, memory or the score, so it can blank a wrong mask but cannot recover the target (that is V4). Design and disclosures: `baseline_v0/V3_PLAN.md`.

| Run (V2 passive source = 50.97) | VLM calls | Calls per tracked frame | Mean J&F |
|---|---|---|---|
| **Real VLM (Gemini gemini-3.1-flash-lite), events + abstain** | 76 | 0.019 | **50.58** |
| Ground-truth oracle, events + abstain (ceiling, not a result) | 76 | 0.019 | 48.64 |
| Oracle, periodic schedule, matched budget | 81 | 0.021 | 50.72 |
| Oracle, every frame | 3,923 | 1.000 | 46.29 |

(The oracle uses ground-truth labels, so it is only a ceiling. Its default rule, "not the target when IoU < 0.5", is pessimistic; a post-hoc identity oracle at IoU < 0.10, chosen after seeing results, gives 50.77, 51.08 and 50.58 for events, periodic and every frame.)

The real VLM answered 76 of 76 calls (no errors), took 3.86 s per call, and added about 10% to run time (3,140 s estimated vs 2,846 s). Its "not the same object" warnings were mostly right (8 of 11) but it caught only 8 of the 21 truly lost or absent frames. **No accuracy gain on Ref-DAVIS17**, with the oracle or the real VLM; the cost side works (about 52x fewer calls than every frame). The event trigger found bad frames about as often as random frames, in line with the weak coherence score, and did not beat a fixed schedule at the same call budget. This is a single run of one model on a dataset with almost no drift, so H3 is undecided and must be tested on Long-RVOS or MeViS.

## Repository layout

```
CogR-VOS/
├── README.md
├── baseline_v0/        V0 to V3 code (V1 to V3 are layers added on top of the V0 pipeline; folder name kept for history)
├── results/            V0 and V1 result folders (V2 and V3 result folders are large and are kept outside git; their numbers are in this README, V2_TUNING_PLAN.md and V3_PLAN.md)
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

## Running V2 and V3

Run from inside `baseline_v0/`. These commands are the scripts' own usage, run by the author on a 4 GB GPU; V3 needs no GPU.

```
bash tools/run_v2_full.sh passive      V2 passive (or: gated)       -> results_v2_passive/
V3_SOURCE=v2_passive bash tools/run_v3.sh oracle                  V3 with the ground-truth oracle (CPU)
export GEMINI_API_KEY=...              key stays in the environment, never in files
V3_SOURCE=v2_passive V3_VLM_MODEL=gemini-3.1-flash-lite bash tools/run_v3.sh gemini    V3 with a real VLM
bash tools/run_all_tests.sh            unit tests (CPU, no dataset, no network)
```

Answers from the VLM are cached on disk (`vlm_cache/`), so a rerun costs nothing and only failed calls are retried.

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


## V4: memory-grounded re-identification (Task 4) and the V0-V4 ablation (Task 5)
Result on Ref-DAVIS17 val (30 videos, 61 expressions, one run each): **no improvement over V2.**

| Version | J&F | VLM calls | Events recovered |
|---|---|---|---|
| V2 | 50.97 | 0 | 2 of 4 |
| V3 | 50.58 | 76 | 2 of 4 |
| V4-memory | 50.87 | 0 | 2 of 4 |
| V4-vlm | 50.86 | 78 | 2 of 4 |
| V4-memory_margin (post hoc) | 50.90 | 0 | 2 of 4 |

The choosers restart even when the tracker is already right, the VLM is equally confident when wrong, and V4 cannot say the target is absent. Ref-DAVIS17 has only 4 of 61 vanishing targets, so re-ID is decided on Long-RVOS. Details: `results/v4_reid/`, `baseline_v0/V4_PLAN.md`, `baseline_v0/tools/run_v4.sh`.
