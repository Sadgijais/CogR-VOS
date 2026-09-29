"""Build the 15-video ablation manifest deterministically (no RNG) from ground-truth masks.

Run from baseline_v0/:   python tools/build_ablation_manifest.py
Reads : config.yaml (dataset.root), subsets/ref_davis17_smoke.json, ../results/v0_baseline/per_video_results.csv
Writes: subsets/ref_davis17_ablation.json  ({"videos": [...]}, same format as smoke/full)
        subsets/ref_davis17_ablation_selection.csv  (the numbers behind every choice)
        subsets/ref_davis17_ablation.json.sha256    (verify: cd subsets && sha256sum -c ref_davis17_ablation.json.sha256)

Rule: ablation = smoke (6) + 6 highest-stress remaining videos + 3 lowest-stress ("easy") videos.
Stress signals (per video/object, from GT masks): interior reappearance events, occlusion-proxy frames
(area < 0.55 x rolling-75th-percentile, border-touching frames excluded), co-occurring objects, scale change, motion.
"""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
N_TOTAL, N_STRESS = 15, 6


def find_anno_dir(root, video):
    for sub in ("Annotations/480p", "Annotations/Full-Resolution", "Annotations"):
        d = root / sub / video
        if d.is_dir():
            return d
    sys.exit(f"ABORT: no annotation folder for {video} under {root}")


def load_video(root, video):
    d = find_anno_dir(root, video)
    files = sorted(d.glob("*.png"))
    return np.stack([np.array(Image.open(f)) for f in files])      # (T,H,W) palette indices


def touches_border(m, px=2):
    return bool(m[:px].any() or m[-px:].any() or m[:, :px].any() or m[:, -px:].any())


def object_stats(arr, obj):
    T, H, W = arr.shape
    masks = arr == obj
    area = masks.reshape(T, -1).sum(1).astype(float)
    present = area > 0
    # interior reappearance events: absent run that is followed by presence, after having been present
    events, seen, gap = 0, False, False
    for p in present:
        if p and gap and seen:
            events += 1
        if p:
            seen, gap = True, False
        elif seen:
            gap = True
    # occlusion proxy
    occ = 0
    for t in range(T):
        if not present[t] or touches_border(masks[t]):
            continue
        lo, hi = max(0, t - 15), min(T, t + 16)
        base = np.percentile(area[lo:hi], 75)
        if base > 0 and area[t] < 0.55 * base:
            occ += 1
    la = np.log(area[present] + 1.0)
    scale_var = float(la.std()) if present.sum() > 1 else 0.0
    cents = [np.argwhere(masks[t]).mean(0) for t in range(T) if present[t]]
    diag = float(np.hypot(H, W))
    motion = float(np.mean([np.linalg.norm(cents[i + 1] - cents[i]) for i in range(len(cents) - 1)]) / diag) if len(cents) > 1 else 0.0
    return dict(events=events, occ_frac=occ / T, scale_var=scale_var, motion=motion)


def z(v):
    v = np.array(v, float)
    s = v.std()
    return np.clip((v - v.mean()) / s, -3, 3) if s > 0 else np.zeros_like(v)


def main():
    cfg = yaml.safe_load(open(HERE / "config.yaml"))
    root = Path(cfg["dataset"]["root"])
    smoke = json.load(open(HERE / "subsets/ref_davis17_smoke.json"))["videos"]
    rows = list(csv.DictReader(open(HERE.parent / "results/v0_baseline/per_video_results.csv")))
    per_obj, cache = [], {}
    for r in rows:
        v, obj = r["video"], int(r["obj_id"])
        if v not in cache:
            cache[v] = load_video(root, v)
        arr = cache[v]
        st = object_stats(arr, obj)
        n_other = len([o for o in np.unique(arr) if o not in (0, 255)])
        per_obj.append(dict(video=v, exp_id=r["exp_id"], obj=obj, n_frames=arr.shape[0], n_objects=n_other, **st))
    zs = {k: z([p[k] for p in per_obj]) for k in ("events", "occ_frac", "n_objects", "scale_var", "motion")}
    W = dict(events=3.0, occ_frac=2.0, n_objects=1.5, scale_var=0.5, motion=0.5)
    for i, p in enumerate(per_obj):
        p["stress"] = sum(W[k] * zs[k][i] for k in W)
    videos = sorted({p["video"] for p in per_obj})
    vs = {v: max(p["stress"] for p in per_obj if p["video"] == v) for v in videos}
    cost = {v: sum(p["n_frames"] for p in per_obj if p["video"] == v) for v in videos}
    rest = [v for v in videos if v not in smoke]
    hard = sorted(rest, key=lambda v: (-vs[v], v))[:N_STRESS]
    rest2 = [v for v in rest if v not in hard]
    easy = sorted(rest2, key=lambda v: (vs[v], v))[: N_TOTAL - len(smoke) - N_STRESS]
    chosen = list(smoke) + hard + easy
    tag = {**{v: "smoke" for v in smoke}, **{v: "stress" for v in hard}, **{v: "easy" for v in easy}}
    out = HERE / "subsets"
    (out / "ref_davis17_ablation.json").write_text(json.dumps({"videos": chosen}) + "\n")
    with open(out / "ref_davis17_ablation_selection.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "tier_role", "video_stress", "n_expr", "cost_frames_x_objs", "events", "occ_frac_max"])
        for v in sorted(videos, key=lambda v: (-vs[v], v)):
            ps = [p for p in per_obj if p["video"] == v]
            w.writerow([v, tag.get(v, "-"), f"{vs[v]:.2f}", len(ps), cost[v], sum(p["events"] for p in ps), f"{max(p['occ_frac'] for p in ps):.2f}"])
    h = hashlib.sha256((out / "ref_davis17_ablation.json").read_bytes()).hexdigest()
    (out / "ref_davis17_ablation.json.sha256").write_text(f"{h}  ref_davis17_ablation.json\n")
    n_expr = sum(1 for p in per_obj if p["video"] in chosen)
    print(f"ablation: {len(chosen)} videos, {n_expr} expressions, cost {sum(cost[v] for v in chosen)}")
    for v in chosen:
        print(f"  {tag[v]:<6} {v:<20} stress={vs[v]:6.2f}  events={sum(p['events'] for p in per_obj if p['video']==v)}")
    print("sha256", h)


if __name__ == "__main__":
    main()
