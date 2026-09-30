# Results

One folder per ablation variant. Every number in a table comes from the full 30-video Ref-DAVIS17 val split (61 objects).

| Folder | Variant | Status | Mean J&F |
|---|---|---|---|
| `v0_baseline/` | Grounding-DINO + SAM 2, no memory, coherence or VLM | Done | 50.48 |
| `v1_sttm/` | + Semantic-Temporal Target Memory (read action on; anchor_proto, delta 5) | Done | 50.97 |
| `v2_tcs/` | + Tracklet Coherence Score | Planned | |
| `v3_event_vlm/` | + Event-driven VLM reasoning | Planned | |
| `v4_reid/` | + Memory-grounded re-identification | Planned | |
| `v5_always_on_vlm/` | Upper bound: VLM on every frame | Planned | |
| `v6_scheduled_vlm/` | Scheduled one-shot reasoning (rival) | Planned | |

Each folder holds `summary.csv`, `per_video_results.csv` and `failure_analysis.csv`.

Baseline note: V0 uses sam2.1-hiera-tiny (forced by the 4 GB GPU). The published Grounded-SAM 2 reference is 66.2; the measured 50.48 is the working baseline.

## V1 (STTM) notes

- Frozen full-val result: J 56.48, F 45.46, J&F 50.97 (V0: 56.01 / 44.94 / 50.48). Failure-mode counts identical to V0 (30 / 16 / 10 / 3 / 2).
- Reading: no-regression check passed (+0.49 J&F, at the edge of the +-0.5 tolerance). About 90% of the net gain (~27 of ~30 J&F points summed over 61 expressions) comes from five expressions (loading/0_2 +11.2, lab-coat/0_1 +6.6, shooting/0_1 +4.4, kite-surf/0_2 +3.1, shooting/0_3 +2.0); without loading/0_2 the mean gain is about +0.3. lab-coat/0_1 is a grounding failure, so that gain is not a recovery. Not evidence that V1 improves segmentation.
- Cost: total inference +31% (2125.7 s -> 2793.5 s; 1031 -> 1362 ms/frame). The STTM code itself is 96.2 s (3.4%); the rest is most likely SAM 2 attending to extra conditioning frames (inferred, not measured). Peak VRAM 805 MB vs 721 MB.
- Settings were chosen on the ablation tier (15 videos, 34 expressions, sha256 f609ea27...cbe22) by a rule fixed in advance (`baseline_v0/V1_TUNING_PLAN.md`); grid results in `v1_sttm/tuning/`. Chosen: sim_mode=anchor_proto, delta=5, K=4.
- Known limits: the memory can only influence SAM 2 through conditioning frames; it cannot actively re-find a reappearing target. Ref-DAVIS17 has almost no real drift or occlusion, so H1/H2/H4 are decided on Long-RVOS or MeViS, not here. The write gate cannot catch a confident drift onto a look-alike (V2/V4 territory).
- The read action can move J in either direction on fragile videos (motocross-jump/0_2: -11.4 J under one rejected config).
