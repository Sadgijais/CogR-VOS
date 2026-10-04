# V2 plan: Tracklet Coherence Score (written and committed BEFORE any tuning run)

V2 = V1 (memory, frozen settings from `config_v1_final.yaml`) + TCS. No VLM, no re-identification.
Code: `tcs.py`, `stage2_tcs.py`, `tools/tcs_analysis.py`. Config: `config_v2.yaml`.

## What TCS computes (per frame, per target, no VLM call)

c_t in [0,1] from four terms, each in [0,1]:

| term | meaning | source |
|---|---|---|
| mask | SAM 2 certainty (sigmoid of object score, mean mask probability) and shape agreement with the previous visible mask (centroid-aligned IoU) | SAM 2 outputs |
| motion | exp(-residual / motion_scale); residual = distance from the memory's predicted centroid, in object diameters | STTM motion field |
| appearance | DINOv2 cosine of the masked crop to the identity prototype | STTM appearance field |
| semantic | CLIP image-text cosine of the masked crop to the expression, divided by the same number at frame 0 (capped at 1) | CLIP, cached text embedding |

Combined by a **weighted geometric mean** (terms missing on a frame, e.g. motion right after a reappearance, are dropped and the
weights renormalised). Reason, shown by `tools/test_tcs.py::test_arithmetic_mean_hides_...`: a teleport or an identity swap
collapses ONE term while the other three stay high; an arithmetic mean of four terms stayed above the HIGH threshold, the
geometric mean did not. The arithmetic option stays in the code (`agg: arithmetic`) for the ablation.

Bands: HIGH (c >= tau_high) = stable; MEDIUM; LOW (c < tau_low). Stable = HIGH, uncertain = MEDIUM + LOW.
An absent target (empty or invisible mask) gets c = 0, band LOW, but `visible: false, state: absent`: absence is not logged as drift.

## Fixed in advance (not tuned)

* Weights: all four equal (1.0). No weight search.
* Aggregation: geometric mean. motion_scale 0.5, app_floor 0.0, eps 0.02.
* Memory settings: exactly `config_v1_final.yaml` (K 4, delta 5, anchor_proto). Nothing from V1 is re-tuned.

## The only thing tuned: the two band thresholds

Tier: ablation (`subsets/ref_davis17_ablation.json`, sha256 checked by `tools/run_v2_tune.sh`). Passive run (TCS measures, masks equal V1).
Rule, applied by `tools/tcs_analysis.py --suggest`, which refuses any video outside the ablation manifest:

1. Evaluation frames = frames where the target is present in the ground truth, in expressions whose frame-0 mask hits the target
   (IoU >= 0.10). Grounding failures are left out: TCS compares a track with its own anchor, so a track consistently on the wrong
   object cannot look incoherent. They are listed separately.
2. `tau_low` = threshold maximising Youden J (TPR - FPR) for flagging LOST frames (IoU on target < 0.10).
3. `tau_high` = threshold maximising Youden J for flagging DEGRADED frames (IoU on target < 0.50). If below `tau_low`, set equal to it.
4. If a class has fewer than 10 frames, that threshold keeps its placeholder (0.50 / 0.70) and the report says so.
5. Freeze: the tool writes `config_v2_final.yaml`; commit it before any full-split run. Never change it afterwards.

## V2's action and the rule for shipping it

Two full-split runs with the frozen thresholds:

* **passive** (`gate_writes: false`): TCS measures only. Masks must equal V1's (checked). Source of all TCS diagnostics.
* **gated** (`gate_writes: true`): a frame is written to memory only if its band is HIGH.

Gated V2 ships as "V2" only if (a) J&F >= V1's 50.97 - 0.5 and (b) identity-drift and track-lost counts do not rise
(V1: 2 and 3). Otherwise V2 is reported as the passive variant (coherence measured and logged, memory writes unchanged) and the
gated result is reported next to it as a negative result. Same stance as V1a/V1b.

## What will be reported (all from the full 30-video val split, 61 expressions)

J, F, J&F; failure-mode counts (stage 4); inference time and peak VRAM, with the TCS overhead on its own line;
share of frames per band; per-term and combined AUROC for lost and degraded frames; every failure run with whether and when
TCS flagged it; number of memory writes blocked by TCS.

## Limits stated before the results

* Ref-DAVIS17 has almost no drift or occlusion (2 identity-drift and 3 track-lost expressions in V1; 4/61 targets vanish).
  AUROC for LOST frames will rest on few frames; the tool prints a warning below 10. H2 is decided on Long-RVOS / MeViS, not here.
* V1 found SAM 2's mask confidence uniformly 0.94-0.97 on a fragile video and V1's similarity gate passed ~95% of wrong frames.
  The mask and appearance terms may therefore add little; the per-term AUROC table shows which terms earn their place.
* TCS cannot flag a grounding failure (the anchor itself is wrong) and cannot fix mask-quality errors (SAM 2 tiny ceiling).
* Thresholds come from a proxy ground truth on 15 videos. They are an operating point, not a calibration.
