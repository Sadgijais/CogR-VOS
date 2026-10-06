#!/usr/bin/env python3
"""CPU tests for tools/compare_pred_dirs.py."""
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_pred_dirs as C


def put(root, video, exp, frame, mask):
    p = root / video / exp / f"{frame:05d}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(p)


def test_identical_different_and_missing():
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "a", Path(td) / "b"
        m = np.zeros((10, 10), bool); m[2:5, 2:5] = True
        m2 = m.copy(); m2[0, 0] = True
        for f in range(3):
            put(a, "v1", "0_1", f, m); put(b, "v1", "0_1", f, m)
        put(a, "v2", "0_1", 0, m); put(b, "v2", "0_1", 0, m2)          # one pixel differs
        put(b, "v3", "0_1", 0, m)                                       # not in the reference
        r = C.compare(a, b)
        assert r["v1"] == {"same": 3, "diff": [], "missing": []}
        assert r["v2"]["diff"] == [("0_1", "00000", 1)] and r["v2"]["same"] == 0
        assert r["v3"]["missing"] == ["v3/0_1/00000.png"]
        assert list(C.compare(a, b, {"v1"})) == ["v1"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok   {t.__name__}")
    print(f"all {len(tests)} tests passed")
