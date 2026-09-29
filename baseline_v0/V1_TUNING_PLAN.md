# V1 tuning plan (written and committed BEFORE any grid run)

Tier: ablation (subsets/ref_davis17_ablation.json, sha256 checked by the runner). Reference: V0 predictions/.
Diagnostic runs on motocross-jump (diag_a..e) were understanding only and are NOT part of this selection.

Grid (all else at defaults: K=4, tau_sim=0.5, a_min=200, merge_thr=0.95, half_life=30, tau_obj=0):
  g0: sim_mode=anchor_proto,   delta=5
  g1: sim_mode=proto_or_recent, delta=5
  g2: sim_mode=proto_or_recent, delta=10
  g3: sim_mode=proto_or_recent, delta=15

Selection rule (fixed in advance, applied by tools/grid_summary.py):
  1. Admissible = no expression with J change worse than -5.0 vs V0.
  2. Among admissible configs pick the highest mean J change.
  3. If that best mean is below -0.5, the read action ships OFF (V1a: memory maintained and logged, SAM 2 conditioning untouched).

After selection: freeze parameters, run the full 30-video val split once with the chosen setting, then official
stage3/stage4. Report V1a and V1b (if chosen) with per-video J, F, J&F, timing and failure modes.
Ref-DAVIS17 is a no-regression check only. Drift/recovery claims are decided on Long-RVOS or MeViS, not here.
