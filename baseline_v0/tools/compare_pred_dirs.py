#!/usr/bin/env python3
"""
Do two prediction folders hold the same masks? CPU only. The V4 SAFETY TEST: V4-off must reproduce the V2 passive masks
exactly, because until its first restart V4 runs the very same SAM 2 + memory + coherence loop.

    python tools/compare_pred_dirs.py predictions_v2_passive predictions_v4_off --video libby --video india

Compares every PNG that exists in the SECOND folder (so a run restricted to a few videos is compared on those videos only).
Prints, per video, how many mask files are identical and which differ (pixels changed, first frame). Exit code 0 only when
every compared mask is identical.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load(p: Path):
    return np.array(Image.open(p).convert("L")) > 0


def compare(a: Path, b: Path, videos=None):
    """-> {video: {"same": n, "diff": [(exp_id, frame_name, pixels_changed), ...], "missing": [paths]}}"""
    out = {}
    for vd in sorted(x for x in b.iterdir() if x.is_dir()):
        if videos and vd.name not in videos:
            continue
        r = out[vd.name] = {"same": 0, "diff": [], "missing": []}
        for pb in sorted(vd.rglob("*.png")):
            rel = pb.relative_to(b)
            pa = a / rel
            if not pa.exists():
                r["missing"].append(str(rel))
                continue
            ma, mb = load(pa), load(pb)
            if ma.shape == mb.shape and (ma == mb).all():
                r["same"] += 1
            else:
                r["diff"].append((rel.parts[1], pb.stem, int((ma != mb).sum()) if ma.shape == mb.shape else -1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--video", action="append", default=None)
    args = ap.parse_args()
    res = compare(args.reference, args.candidate, set(args.video) if args.video else None)
    if not res:
        raise SystemExit(f"nothing to compare under {args.candidate}/")
    bad = False
    for v, r in res.items():
        status = "IDENTICAL" if not r["diff"] and not r["missing"] else "DIFFERENT"
        bad |= status != "IDENTICAL"
        print(f"{v}: {status}  ({r['same']} identical, {len(r['diff'])} different, {len(r['missing'])} not in the reference)")
        for e, f, n in r["diff"][:5]:
            print(f"    {e}/{f}.png: {n} pixels differ")
    print("\nSAFETY TEST " + ("FAILED: the masks are not identical" if bad else "PASSED: every compared mask is identical"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
