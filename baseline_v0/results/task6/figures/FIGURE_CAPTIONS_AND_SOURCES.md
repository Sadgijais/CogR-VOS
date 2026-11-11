# CogR-VOS figures for Sir's item 1 (5 Nov 2026): captions and sources

All figures: selected subsets, single runs, not official benchmark results. Sentences of one video are not independent. Oracle starts are a diagnostic ceiling, not method performance. Numbers were copied from the project notes listed below; they were NOT recomputed from the CSV files in this step. Script: make_figs.py (all values are typed in at the top of each figure block).

## Figure 1: top-k candidate recall (fig1_topk_recall)
Caption: Share of sentences whose target box (IoU >= 0.5 with ground truth) is among the top-k Grounding-DINO boxes at frame 0. Top-1 finds the target for 37.7% (MeViS, n = 191) and 27.6% (Long-RVOS, n = 29) of the sentences visible at frame 0, while the whole top-5 pool contains it for 82.7% and 86.2%. 33 and 4 sentences have the target outside the pool.
Values: MeViS 37.7 / 68.1 / 77.5 / 81.2 / 82.7; Long-RVOS 27.6 / 48.3 / 62.1 / 75.9 / 86.2.
Source: CogR-VOS_Candidate_Analysis_Results_2026-10-27 (tools/candidate_analysis.py, md5 8525a5335ac70f25ea9384eb7fe2307d).

## Figure 2: selection arms (fig2_selection_arms)
Caption: Top row: J&F of the five frame-0 selection arms with the same V0 tracker, on MeViS and Long-RVOS; dashed lines are oracle starts (best cached candidate 52.8 / 61.2, ground-truth box 64.3 / 61.1). Bottom row: difference to A0 top-1 with 95% video-level bootstrap intervals (10,000 resamples, seed 0). No interval excludes zero; the best arm differs by dataset (A3-5 on MeViS, A1 on Long-RVOS). No arm meets the Amendment 6 claim rule.
Values: J&F MeViS 26.88 / 27.06 / 32.16 / 31.51 / 28.16; Long-RVOS 30.38 / 37.77 / 35.12 / 33.58 / 26.85. Differences [95% interval] MeViS A1 +0.19 [-4.43, +6.10], A3-5 +5.28 [-0.74, +12.21], A3-10 +4.63 [-2.49, +12.54], largest +1.28 [-3.31, +8.12]; Long-RVOS A1 +7.39 [-3.58, +20.19], A3-5 +4.73 [-2.70, +12.05], A3-10 +3.20 [-4.40, +10.90], largest -3.53 [-23.33, +17.14].
Sources: CogR-VOS_Cost_Table_and_Evidence_Register_2026-11-04 (Part 2), CogR-VOS_Largest_Box_and_Ladder_LongRVOS_2026-11-03, CogR-VOS_Oracle_Diagnostic_Results_2026-10-23 (oracle values).
Note: the A1 MeViS difference is +0.18 in some earlier notes and +0.19 in the interval table (rounding of the unrounded J&F values); the figure uses +0.19 with its interval.

## Figure 3: initial vs later failures (fig3_initial_vs_later_failures)
Caption: Failure runs of the A0 top-1 start (visible sentences; a failure run is 3 or more consecutive visible frames with IoU < 0.5), split into the first failure run of each sentence and later failure runs. On MeViS two thirds of all failure runs (127 of 190) are initial-grounding failures; on Long-RVOS the largest group is later re-identification (24 of 54, 44%) and identity drift is absent. 62% of the later MeViS runs come from one video (329909730c21).
Values (runs): MeViS first run init 127, prop 11, reid 0; later reid 11, drift 9, prop 32 (total 190). Long-RVOS first run init 21, prop 2, reid 3; later reid 24, drift 0, prop 4 (total 54).
Sources: CogR-VOS_Task3_Run_Views_2026-11-04 (table V1 and V3), CogR-VOS_Status_vs_Sir_Items_2026-11-05 (138 and 26 sentences).
Open: the 125 vs 127 MeViS count is not reconciled; the leave-one-video-out recount without 329909730c21 is not done. The "first reid" row on MeViS is 0 because the note's table lists no such row; sums check (138 first, 52 later).

## Figure 4: quality against compute (fig4_quality_vs_cost)
Caption: J&F against extra time relative to V0 (tracking time plus offline selection time where measured). Memory (V1), passive TCS (V2) and memory re-ID (V4) cost about +21 to +54% time and raise peak memory (V2 about 1.4 GB against about 0.7 to 0.8 GB for V0; V4 about 1.7 GB), without a J&F gain on either dataset. The A3 gain on MeViS is paid for with a SAM 2 look-ahead pass of about 4,046 s (about +91 to +96% of the V0 tracking time). Differences under about 15% are not interpreted (run-to-run spread, Amendment 8).
Values: tracking time (s) MeViS V0 4430.2, A1 4371.4, A3-5 4631.9, A3-10 4420.3, largest 4462.3, V1 5347.8, V2 5415.3, V4 5782.6; Long-RVOS V0 6391.3, A1 6244.3, A3-5 7037.6, A3-10 6781.9, largest 7406.0, V1 8027.5, V2 7926.9, V4 8833.0 (10 videos; +54.1% against V0 on the same 10 videos, as in the cost table). Selection time (s): A1 68.7 / 20.6, A3 4046 / 868 (one shared pass for N = 5 and N = 10, counted in full for each arm here). Largest-box selection time not measured (hollow marker).
Source: CogR-VOS_Cost_Table_and_Evidence_Register_2026-11-04 (Parts 1 and 2). V3 and V4+VLM are left out (one live video per dataset, no timing file).
Note: x positions for A1 and A3 were computed for this figure from the table values (tracking + selection) / V0; they are not stated in the notes.
