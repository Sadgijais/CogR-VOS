"""Small, boring I/O helpers shared by every stage."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

# DAVIS/YouTube-VOS palette PNGs are "P" mode images: each pixel stores a
# small integer, the object id, and a colour table maps it to a colour for
# display. Reading the *displayed* colour instead of the raw index is
# silent-killer #3 in the working primer — always read with mode "P" and
# take the raw index array, never convert to RGB first.


def read_palette_mask(path: Path, obj_id: int) -> np.ndarray:
    """Boolean mask for one object id from a palette PNG."""
    img = Image.open(path)
    if img.mode != "P":
        img = img.convert("P")
    arr = np.array(img)
    return arr == obj_id


def read_binary_mask(path: Path) -> np.ndarray:
    """A plain binary mask PNG (e.g. one produced by stage 2) as bool."""
    img = Image.open(path).convert("L")
    arr = np.array(img)
    return arr > 0


def write_binary_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = (mask.astype(np.uint8) * 255)
    Image.fromarray(out, mode="L").save(path)


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def dump_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def unique_palette_ids(path: Path) -> set[int]:
    """All non-zero object ids actually present in one annotation PNG."""
    img = Image.open(path)
    if img.mode != "P":
        img = img.convert("P")
    arr = np.array(img)
    ids = set(np.unique(arr).tolist())
    ids.discard(0)
    return ids
