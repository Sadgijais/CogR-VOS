# Results

One folder per ablation variant. Every number in a table comes from the full 30-video Ref-DAVIS17 val split (61 objects).

| Folder | Variant | Status | Mean J&F |
|---|---|---|---|
| `v0_baseline/` | Grounding-DINO + SAM 2, no memory, coherence or VLM | Done | 50.48 |
| `v1_sttm/` | + Semantic-Temporal Target Memory (read action on; anchor_proto, delta 5) | Done | 50.97 |
| `v2_tcs/` | + Tracklet Coherence Score (`passive/` ships as V2; `gated/` is a negative result, 50.40) | Done | 50.97 |
| `v3_event_vlm/` | + Event-driven VLM reasoning (`real_gemini/` = real VLM, one run; `oracle_*` = ground-truth oracle controls, ceilings only) | Done (one real run) | 50.58 |
| `v4_reid/` | + Memory-grounded re-identification (`memory/`, `vlm/`; `memory_margin/` is post hoc) | Done | 50.87 (memory), 50.86 (vlm) |
| `v5_always_on_vlm/` | Upper bound: VLM on every frame | Planned | |
| `v6_scheduled_vlm/` | Scheduled one-shot reasoning (rival) | Planned | |

Each folder holds `summary.csv`, `per_video_results.csv` and `failure_analysis.csv`. V2 and V3 folders have one subfolder per run and, in addition, their own logs (`tcs_per_expression.csv`, `TCS_ANALYSIS.txt`, `v3_summary.txt`, `vlm_log.csv`) and a `CHECKSUMS.sha256`. Large per-frame logs and predicted masks are not stored in git.

Baseline note: V0 uses sam2.1-hiera-tiny (forced by the 4 GB GPU). The published Grounded-SAM 2 reference is 66.2; the measured 50.48 is the working baseline.

## V1 (STTM) notes

- Frozen full-val result: J 56.48, F 45.46, J&F 50.97 (V0: 56.01 / 44.94 / 50.48). Failure-mode counts identical to V0 (30 / 16 / 10 / 3 / 2).
- Reading: no-regression check passed (+0.49 J&F, at the edge of the +-0.5 tolerance). About 90% of the net gain (~27 of ~30 J&F points summed over 61 expressions) comes from five expressions (loading/0_2 +11.2, lab-coat/0_1 +6.6, shooting/0_1 +4.4, kite-surf/0_2 +3.1, shooting/0_3 +2.0); without loading/0_2 the mean gain is about +0.3. lab-coat/0_1 is a grounding failure, so that gain is not a recovery. Not evidence that V1 improves segmentation.
- Cost: total inference +31% (2125.7 s -> 2793.5 s; 1031 -> 1362 ms/frame). The STTM code itself is 96.2 s (3.4%); the rest is most likely SAM 2 attending to extra conditioning frames (inferred, not measured). Peak VRAM 805 MB vs 721 MB.
- Settings were chosen on the ablation tier (15 videos, 34 expressions, sha256 f609ea27...cbe22) by a rule fixed in advance (`baseline_v0/V1_TUNING_PLAN.md`); grid results in `v1_sttm/tuning/`. Chosen: sim_mode=anchor_proto, delta=5, K=4.
- Known limits: the memory can only influence SAM 2 through conditioning frames; it cannot actively re-find a reappearing target. Ref-DAVIS17 has almost no real drift or occlusion, so H1/H2/H4 are decided on Long-RVOS or MeViS, not here. The write gate cannot catch a confident drift onto a look-alike (V2/V4 territory).
- The read action can move J in either direction on fragile videos (motocross-jump/0_2: -11.4 J under one rejected config).

## V2 (TCS) notes

- Shipped V2 is the passive run (score logged, memory untouched): J 56.48, F 45.46, J&F 50.97, masks identical to V1. Gated V2 (memory written only on HIGH frames): J 55.97, F 44.84, J&F 50.40, a negative result (14 expressions better, 23 worse, 24 same vs V1); it failed the pre-registered ship rule (J&F >= 50.47).
- Frozen thresholds tau_low 0.81 / tau_high 0.92 from the 15-video ablation tier by a rule fixed in advance (`baseline_v0/V2_TUNING_PLAN.md`). Term weights are equal and untuned.
- Cost: TCS code is 2.7% of run time; peak VRAM 1384 MB; total V2 time 2846 s.
- The score is a weak failure detector here: mean IoU 0.77 on HIGH vs 0.78 on MEDIUM frames; 44% false alarms; AUROC for degraded frames 0.56 (mask term alone 0.62, semantic term 0.45); the tier AUROC of 0.746 did not transfer. Grounding failures (16 of 61 expressions) are invisible to the score. Details in `v2_tcs/passive/TCS_ANALYSIS.txt`.

## V3 (event-driven VLM) notes

- Offline replay of the V2 passive run (J&F 50.97): the VLM never feeds back into SAM 2, memory or the score. Design, controls and disclosures: `baseline_v0/V3_PLAN.md`.
- Real VLM (`real_gemini/`, Gemini gemini-3.1-flash-lite, events + abstain): 76 calls (0.019 per tracked frame), J&F 50.58 (J 56.12, F 45.04), 140 frames blanked, 3.86 s per call, about +10% run time. No accuracy gain; its "not the same object" answers were right 8 of 11 times but it caught only 8 of 21 truly lost or absent frames. Single run, one model.
- Oracle controls use ground-truth labels, so they are ceilings, not results: events 48.64, periodic (matched budget, 81 calls) 50.72, every frame (3,923 calls) 46.29 with the pre-registered IoU < 0.5 rule; 50.77 / 51.08 / 50.58 with the post-hoc IoU < 0.10 identity oracle (folders ending `_iou0.10`, chosen after seeing results).
- H3 is undecided on Ref-DAVIS17 (almost no drift or occlusion); it must be tested on Long-RVOS or MeViS.

## V4 (re-identification) notes

- Live SAM 2 reruns of the expressions that can search (18 videos, 37 expressions with a candidate cache); the other videos keep their V2 passive masks. Design, controls and disclosures: `baseline_v0/V4_PLAN.md`, written before any V4 run. The safety variant (chooser off) reproduces V2 exactly (393 of 393 mask files on india and kite-surf).
- `memory/`: J 56.39, F 45.35, J&F 50.87, 0 VLM calls, 79 searches, 15 restarts (5 correct, 7 wrong, 3 weak), 10 searches with no good candidate. `vlm/`: J 56.38, F 45.34, J&F 50.86, 78 VLM calls (0.0199 per tracked frame), 78 searches, 27 restarts (6 correct, 16 wrong, 5 weak), 7 with no good candidate. Each folder has `summary.csv`, `reid_log.csv` (one row per search, with the label), `reid_summary.txt` and `recovery_summary.txt`. `tables/` holds the V0 to V4 ablation tables (`ablation_preregistered.md` and `ablation_with_posthoc.md`).
- `memory_margin/` is **post hoc** (added after seeing the results): restart only if the pick beats the tracker's mask score by 0.10 or more, or the tracker mask is empty. J&F 50.90, 9 restarts (3 correct, 4 wrong, 2 weak). Not a result; a diagnosis.
- Reading: no improvement over V2 (50.97). In each of memory and VLM, 7 restarts replaced a mask that was already good (memory: 2 better, 2 mildly worse, 3 destroyed; VLM: 5 destroyed, for example gold-fish 0.91 to 0). Gemini's confidence was 0.9 to 1.0 on every answer, right or wrong. V4 has no way to say "the target is absent": 3 of the 4 wrong restarts that remain after the margin gate are on absent targets.
- Ref-DAVIS17 has only 4 of 61 targets that vanish (3 reappear), and recovery is 2 of 4 events for every version, so re-identification is decided on Long-RVOS or MeViS, not here. Single runs, no significance test.
