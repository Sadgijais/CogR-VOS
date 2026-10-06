| Version | J | F | J&F | Inference time (s) | VLM calls | VLM calls / tracked frame | Recovery rate (events) | Recovery latency (frames) | Loss episodes recovered | NRE | False presence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 56.01 | 44.94 | 50.48 | 2125.7 | 0 | 0.0000 | 2 of 4 | 1.0 | 1 of 23 | 0.7% | 16.7% |
| V1 | 56.48 | 45.46 | 50.97 | 2793.5 | 0 | 0.0000 | 2 of 4 | 1.0 | 2 of 21 | 0.6% | 16.7% |
| V2 | 56.48 | 45.46 | 50.97 | 2846.0 | 0 | 0.0000 | 2 of 4 | 1.0 | 2 of 21 | 0.6% | 16.7% |
| V3 | 56.12 | 45.04 | 50.58 | 3139.6 | 76 | 0.0194 | 2 of 4 | 1.0 | 3 of 23 | 3.9% | 0.0% |
| V4-memory | 56.39 | 45.35 | 50.87 | 2851.0 | 0 | 0.0000 | 2 of 4 | 1.0 | 3 of 22 | 0.6% | 20.4% |
| V4-vlm | 56.38 | 45.34 | 50.86 | 3164.2 | 78 | 0.0199 | 2 of 4 | 1.0 | 4 of 22 | 0.6% | 20.4% |

Notes:
- V0 time: measured in results_timed
- V1 time: measured
- V2 time: measured
- V3 time: V2 time + sequential VLM latency
- V4-memory time: V2 total + measured extra (5.0 s of search + VLM waiting); separate wall-clock runs differ by ~15%, so they are not compared
- V4-vlm time: V2 total + measured extra (318.2 s of search + VLM waiting); separate wall-clock runs differ by ~15%, so they are not compared
- Recovery = back at IoU >= 0.5 within 10 frames of the first visible frame. Always read it with the event count: Ref-DAVIS17 has only a handful of reappearance events, so these are anecdotes, not statistics.
- NRE = visible but empty mask; false presence = absent but mask not empty. One run each, no significance test.
- V4 VLM answers served from the disk cache cost no live waiting time here, so the V4 time is a lower bound when answers were cached.
