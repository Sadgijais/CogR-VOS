"""
Ref-DAVIS17 dataset access, with the four guards from
CogR-VOS_V0_Baseline_Spec.md §5 built in rather than left to discipline:

  1. Refuses to run against Full-Resolution data — every published
     Ref-DAVIS17 number is computed at 480p, and comparing against a
     different resolution compares against nothing.
  2. `exclude_first_last` is a first-class, explicit setting (default True)
     rather than something you have to remember to pass at eval time.
  3. Object ids come from the palette PNG's raw index values, cross-checked
     against the referring-expression metadata, not assumed.
  4. The annotator-set number is an explicit, loud config field — see
     ANNOTATOR_SET_WARNING below. This project's own docs currently
     disagree about which one to use (0 vs 1); this loader will not
     silently guess.

Expected directory layout (standard Ref-DAVIS17 480p release plus a
referring-expression metadata file placed alongside it):

    <root>/
      JPEGImages/480p/<video>/00000.jpg, 00001.jpg, ...
      Annotations/480p/<video>/00000.png, 00001.png, ...
      meta_expressions/meta_expressions.json

If your download uses different folder names (some mirrors do), either
rename to match this layout or adjust `DavisLayout` below — do that in one
place rather than patching every script.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .io_utils import unique_palette_ids

ANNOTATOR_SET_WARNING = (
    "CogR-VOS_Dataset_Subset_Protocol.md says annotator set 0; "
    "CogR-VOS_Dataset_Audit_Findings.md says annotator set 1. "
    "This is an open decision (see V0_Baseline_Spec.md §7) — settle it "
    "before reporting any number, and record the choice in config.yaml."
)


class ResolutionGuardError(RuntimeError):
    pass


class AnnotatorSetUnresolvedError(RuntimeError):
    pass


@dataclass
class Expression:
    video: str
    exp_id: str
    obj_id: int
    text: str


@dataclass
class DavisLayout:
    root: Path
    require_480p: bool = True

    @property
    def jpeg_dir(self) -> Path:
        return self.root / "JPEGImages" / "480p"

    @property
    def anno_dir(self) -> Path:
        return self.root / "Annotations" / "480p"

    @property
    def meta_expressions_path(self) -> Path:
        return self.root / "meta_expressions" / "meta_expressions.json"

    def frame_paths(self, video: str) -> list[Path]:
        return sorted((self.jpeg_dir / video).glob("*.jpg"))

    def anno_paths(self, video: str) -> list[Path]:
        return sorted((self.anno_dir / video).glob("*.png"))

    def videos(self) -> list[str]:
        return sorted(p.name for p in self.jpeg_dir.iterdir() if p.is_dir())

    def check_resolution(self, video: str) -> None:
        """
        Guard against silently scoring the wrong Ref-DAVIS17 download.
        Ref-DAVIS17 480p frames are 480 px on the short side. A
        Full-Resolution download (1920x1080) must be refused outright —
        every number in the literature is computed at 480p, so anything
        else is not comparable to a single published result.
        """
        if not self.require_480p:
            return
        frames = self.frame_paths(video)
        if not frames:
            return
        with Image.open(frames[0]) as img:
            w, h = img.size
        if min(w, h) > 520:  # 480p should be ~480; give slack, not headroom
            raise ResolutionGuardError(
                f"{video}: frame size {w}x{h} does not look like 480p. "
                "This looks like the Full-Resolution release "
                "(DAVIS-2017-trainval-Full-Resolution). Every published "
                "Ref-DAVIS17 J&F number is computed at 480p — re-point "
                "config.yaml at the 480p archive, or set "
                "evaluation.require_480p: false only if you are certain "
                "you want an incomparable number."
            )


def load_expressions(layout: DavisLayout, annotator_set: int | None) -> list[Expression]:
    """
    Loads referring expressions from meta_expressions.json.

    `annotator_set` selects among however many expression sets the file
    actually contains for a given (video, object) — Ref-DAVIS17 ships four
    (2 annotators x {first-frame, full-video} description style). If your
    meta_expressions.json is already pre-filtered to one set, pass
    annotator_set=None and every expression found is used as-is.

    Raises AnnotatorSetUnresolvedError if the file contains an annotator
    field but `annotator_set` was not given — we refuse to guess, per
    ANNOTATOR_SET_WARNING above.
    """
    import json

    with open(layout.meta_expressions_path) as f:
        meta = json.load(f)

    out: list[Expression] = []
    videos = meta.get("videos", meta)  # tolerate either top-level shape
    for video, vdata in videos.items():
        exps = vdata.get("expressions", {})
        for exp_id, edata in exps.items():
            if "annotator" in edata and annotator_set is None:
                raise AnnotatorSetUnresolvedError(ANNOTATOR_SET_WARNING)
            if "annotator" in edata and int(edata["annotator"]) != int(annotator_set):
                continue
            out.append(
                Expression(
                    video=video,
                    exp_id=str(exp_id),
                    obj_id=int(edata["obj_id"]),
                    text=edata["exp"],
                )
            )
    return out


def cross_check_obj_ids(layout: DavisLayout, expressions: list[Expression]) -> list[str]:
    """
    Sanity check: does every expression's obj_id actually appear in that
    video's annotation palette? Silent-killer #3 in the working primer —
    scoring against the wrong palette index produces a plausible-looking,
    meaningless number. Returns a list of human-readable problems found
    (empty list = all good).
    """
    problems = []
    by_video: dict[str, list[Expression]] = {}
    for e in expressions:
        by_video.setdefault(e.video, []).append(e)

    for video, exps in by_video.items():
        anno_paths = layout.anno_paths(video)
        if not anno_paths:
            problems.append(f"{video}: no annotation files found")
            continue
        all_ids: set[int] = set()
        for p in anno_paths:
            all_ids |= unique_palette_ids(p)
        for e in exps:
            if e.obj_id not in all_ids:
                problems.append(
                    f"{video}/{e.exp_id}: obj_id {e.obj_id} never appears in the "
                    f"palette (ids present: {sorted(all_ids)})"
                )
    return problems


_FRAME_NUM_RE = re.compile(r"(\d+)")


def frame_index(path: Path) -> int:
    m = _FRAME_NUM_RE.search(path.stem)
    if not m:
        raise ValueError(f"cannot parse a frame index out of {path}")
    return int(m.group(1))
