"""
Event trigger for V3 (event-driven VLM reasoning). Pure python, no GPU, no VLM.

Question it answers, once per tracked frame per target: "is this frame an IMPORTANT EVENT
that deserves one expensive VLM call?"  It reads only what the Tracklet Coherence Score (V2)
already logged (c_t, visible, state, distractor margin), so it costs nothing.

Events (priority order, first match wins on a frame):
  reappear         state == "reappeared": SAM 2 found something again after the target was absent
  disappear        visible -> absent transition
  coherence_drop   sudden drop: c fell by >= drop_delta below the recent maximum, OR the band
                   just entered LOW (c < tau_low) from a non-LOW visible frame
  distractor       the crop is closer to a cached distractor than to the identity prototype
                   (margin < margin_thresh)
  drift_suspect    LOW for persist_frames visible frames in a row (persistent disagreement)
  keepalive        optional: nothing has been checked for `keepalive` frames (0 = off)

Pacing (Event-VStream style): after a call, no new call for `refractory` frames; an optional
`max_calls` per target caps the cost. Events that occur inside the refractory window are
counted as `suppressed`, never queued.

Decisions here do NOT depend on what the VLM answers (the VLM only changes masks, not TCS),
so the whole schedule can be computed up front and the VLM calls run in parallel.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

EVENT_TYPES = ("reappear", "disappear", "coherence_drop", "distractor", "drift_suspect", "keepalive")


@dataclass
class EventParams:
    tau_low: float = 0.50          # taken from the V2 run's frozen thresholds by stage_events.py
    tau_high: float = 0.70
    drop_delta: float = 0.30       # sudden drop size (in c units)
    drop_window: int = 3           # compared with the max of the previous N visible frames
    persist_frames: int = 5        # consecutive LOW visible frames -> drift_suspect
    margin_thresh: float = 0.0     # margin below this -> distractor (margin is None = ignored)
    refractory: int = 10           # minimum frames between two calls on one target
    keepalive: int = 0             # force a check after this many quiet frames (0 = off)
    max_calls: int = 0             # per target (0 = unlimited)
    fire_on_disappear: bool = True


class EventTrigger:
    """One per target. Call step() once per frame, in order, starting at frame 1."""

    def __init__(self, p: EventParams):
        self.p = p
        self.prev_visible = True               # frame 0 is the prompt: the target is there
        self.prev_low = False
        self.hist: deque = deque(maxlen=p.drop_window)
        self.low_run = 0
        self.last_call = None                  # no call yet; frame 0 (the prompt) is the reference for keepalive
        self.calls = 0
        self.fired = {e: 0 for e in EVENT_TYPES}
        self.suppressed = 0
        self.candidates = 0                    # raw events before pacing

    def _raw_event(self, f: int, c: float, visible: bool, state: str, margin) -> str | None:
        p = self.p
        ev = None
        if visible:
            low = c < p.tau_low
            drop = len(self.hist) > 0 and (max(self.hist) - c) >= p.drop_delta
            entered_low = low and not self.prev_low and self.prev_visible
            self.low_run = self.low_run + 1 if low else 0
            if state == "reappeared":
                ev = "reappear"
            elif drop or entered_low:
                ev = "coherence_drop"
            elif margin is not None and margin < p.margin_thresh:
                ev = "distractor"
            elif self.low_run >= p.persist_frames:
                ev = "drift_suspect"
            self.hist.append(c)
            self.prev_low = low
        else:
            self.low_run = 0
            self.prev_low = False
            if self.prev_visible and p.fire_on_disappear:
                ev = "disappear"
        if ev is None and p.keepalive and visible and (f - (self.last_call or 0)) >= p.keepalive:
            ev = "keepalive"
        self.prev_visible = visible
        return ev

    def step(self, f: int, c: float, visible: bool, state: str, margin=None) -> str | None:
        ev = self._raw_event(f, c, visible, state, margin)
        if ev is None:
            return None
        self.candidates += 1
        in_refractory = self.last_call is not None and (f - self.last_call) < self.p.refractory
        if in_refractory or (self.p.max_calls and self.calls >= self.p.max_calls):
            self.suppressed += 1
            return None
        self.last_call = f
        self.calls += 1
        self.fired[ev] += 1
        return ev


class PeriodicTrigger:
    """Control schedule for the matched-budget comparison: one call every `period` frames
    (period=1 is the every-frame upper bound, V5). Ignores TCS completely."""

    def __init__(self, period: int):
        self.period = max(1, int(period))
        self.calls = 0
        self.suppressed = 0
        self.candidates = 0
        self.fired = {"periodic": 0}

    def step(self, f: int, c=None, visible=None, state=None, margin=None) -> str | None:
        if f % self.period != 0:
            return None
        self.candidates += 1
        self.calls += 1
        self.fired["periodic"] += 1
        return "periodic"
