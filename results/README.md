# Results

One folder per ablation variant. Every number in a table comes from the full 30-video Ref-DAVIS17 val split (61 objects).

| Folder | Variant | Status | Mean J&F |
|---|---|---|---|
| `v0_baseline/` | Grounding-DINO + SAM 2, no memory, coherence or VLM | Done | 50.48 |
| `v1_sttm/` | + Semantic-Temporal Target Memory | Planned | |
| `v2_tcs/` | + Tracklet Coherence Score | Planned | |
| `v3_event_vlm/` | + Event-driven VLM reasoning | Planned | |
| `v4_reid/` | + Memory-grounded re-identification | Planned | |
| `v5_always_on_vlm/` | Upper bound: VLM on every frame | Planned | |
| `v6_scheduled_vlm/` | Scheduled one-shot reasoning (rival) | Planned | |

Each folder holds `summary.csv`, `per_video_results.csv` and `failure_analysis.csv`.

Baseline note: V0 uses sam2.1-hiera-tiny (forced by the 4 GB GPU). The published Grounded-SAM 2 reference is 66.2; the measured 50.48 is the working baseline.
