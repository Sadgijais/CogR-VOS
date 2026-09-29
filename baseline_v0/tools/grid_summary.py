"""Apply the pre-registered selection rule to compare_v0_v1.py outputs.
Usage (from baseline_v0/): python tools/grid_summary.py results_grid/compare_g0.txt results_grid/compare_g1.txt ...
Rule (see V1_TUNING_PLAN.md): pick the config with the highest mean J change vs V0 among configs with
no expression worse than -5.0 J. If the best such mean is < -0.5, the read action ships OFF (V1a)."""
import sys, re
from pathlib import Path

WORST_ALLOWED, MEAN_FLOOR = -5.0, -0.5
res = {}
for f in sys.argv[1:]:
    deltas = []
    for line in Path(f).read_text().splitlines():
        p = line.split()
        if len(p) >= 5 and "/" in p[0] and p[0] != "video/exp":
            try:
                deltas.append((p[0], float(p[3])))
            except ValueError:
                pass
    res[f] = deltas
print(f"{'config':<32}{'n':>4}{'mean dJ':>9}{'worst dJ':>10}  worst expression")
ok = {}
for f, d in res.items():
    if not d:
        print(f"{f:<32}  (no rows parsed)")
        continue
    mean = sum(x for _, x in d) / len(d)
    name, worst = min(d, key=lambda t: t[1])
    print(f"{Path(f).stem:<32}{len(d):>4}{mean:>9.2f}{worst:>10.1f}  {name}")
    if worst >= WORST_ALLOWED:
        ok[f] = mean
if not ok:
    print("\nVERDICT: every config has an expression worse than -5 J -> read action OFF (V1a).")
else:
    best = max(ok, key=ok.get)
    if ok[best] < MEAN_FLOOR:
        print(f"\nVERDICT: best admissible mean dJ = {ok[best]:.2f} < {MEAN_FLOOR} -> read action OFF (V1a).")
    else:
        print(f"\nVERDICT: choose {Path(best).stem} (mean dJ {ok[best]:+.2f}).")
