#!/usr/bin/env python3
"""
Task 6 audit step B1 (CPU only, reads only, changes nothing): are two prediction folders really identical?
Compares every mask PNG of folder A with the file of the same relative path in folder B: first by file fingerprint (sha256),
and if the bytes differ, by the pixel values. Used to check the claim "V1 and V2 passive masks are identical".

  python -I tools/audit_mask_identity.py --a predictions_v1_final --b predictions_v2_passive --label "MeViS V1 against V2 passive"
"""
import argparse, hashlib, sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()

    A, B = Path(a.a).expanduser(), Path(a.b).expanduser()
    fa = {p.relative_to(A): p for p in A.rglob("*.png")}
    fb = {p.relative_to(B): p for p in B.rglob("*.png")}
    both = sorted(set(fa) & set(fb))
    only_a, only_b = sorted(set(fa) - set(fb)), sorted(set(fb) - set(fa))

    same_bytes = same_pixels = 0
    differ = []
    for k in both:
        if sha(fa[k]) == sha(fb[k]):
            same_bytes += 1
            continue
        x, y = np.asarray(Image.open(fa[k])), np.asarray(Image.open(fb[k]))
        if x.shape == y.shape and np.array_equal(x, y):
            same_pixels += 1
        else:
            differ.append(k)

    print(a.label)
    print(f"A = {A}  ({len(fa)} png files)")
    print(f"B = {B}  ({len(fb)} png files)")
    print(f"files in both: {len(both)}   only in A: {len(only_a)}   only in B: {len(only_b)}")
    print(f"identical file bytes: {same_bytes}   different bytes but identical pixels: {same_pixels}   different pixels: {len(differ)}")
    if differ:
        per = Counter("/".join(k.parts[:2]) for k in differ)
        print(f"sentences with at least one different mask: {len(per)}")
        for name, n in per.most_common(10):
            print(f"  {name}: {n} files differ")
    verdict = "IDENTICAL" if not differ and not only_a and not only_b else "NOT IDENTICAL"
    print(f"RESULT: {verdict}")


if __name__ == "__main__":
    sys.exit(main())
