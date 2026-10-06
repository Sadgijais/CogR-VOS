"""
Wiring smoke test for stage_reid.py with a FAKE SAM 2 (it can restart from a mask), FAKE torch and FAKE embedders on the
tiny synthetic video of test_stage2_tcs_smoke.py. No GPU, no dataset, no model downloads.

It proves the plumbing and the properties V4 depends on, NOT accuracy:
  * variant "off" reproduces the V2 passive masks exactly (the safety test, here on the fake)
  * variant "oracle" / "memory" / "vlm" restart on the true object after the tracker jumps onto a distractor (frame 20)
    and change nothing before that frame
  * a mask the tracker reports empty is never blanked or filled by V4 unless a restart is chosen
Real SAM 2 behaviour is only exercised by the safety test and the smoke run on your GPU.

Run (from baseline_v0/):   python tools/test_stage_reid_smoke.py
"""
import csv
import importlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import test_stage2_tcs_smoke as sm
import stage_candidates as C

H, W, N = sm.H, sm.W, sm.N
DISTRACTOR = sm.box_mask(110, 5, 140, 35)
EVENTS = {"drop_delta": 0.3, "drop_window": 3, "persist_frames": 5, "margin_thresh": 0.0, "refractory": 10,
          "keepalive": 0, "max_calls": 0, "fire_on_disappear": True}


def install(tmp):
    sm.install_fakes(tmp)
    sys.modules["torch"].from_numpy = lambda a: a

    class Pred:
        def __init__(self):
            self.override, self.restarts = None, []

        def init_state(self, **kw):
            self.override = None
            return {"output_dict_per_obj": {0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}}}

        def add_new_points_or_box(self, **kw):
            pass

        def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
            self.override = (frame_idx, np.asarray(mask, bool))
            self.restarts.append(frame_idx)

        def _mask(self, t):
            ov = self.override
            if ov is not None and t >= ov[0] and (t not in (30, 31) or ov[0] >= 30):
                return np.roll(ov[1], 2 * (t - ov[0]), axis=1)         # follows its latest prompt, 2 px / frame
            if t in (30, 31):
                return None                                           # the tracker loses the target there (a prompt at 30 fixes it)
            return sm.scripted_mask(t)

        def propagate_in_video(self, state, start_frame_idx=None):
            od = state["output_dict_per_obj"][0]
            for t in range(0 if start_frame_idx is None else start_frame_idx, N):
                m = self._mask(t)
                present = m is not None
                m = m if present else np.zeros((H, W), bool)
                out = {"object_score_logits": np.array([[6.0 if present else -5.0]])}
                (od["cond_frame_outputs"] if t == 0 else od["non_cond_frame_outputs"])[t] = out
                yield t, [1], [sm.FT(np.where(m, 5.0, -5.0).astype(np.float32))]

        def reset_state(self, state):
            pass

    holder = {}

    def build(*a, **k):
        holder["pred"] = Pred()
        return holder["pred"]

    sys.modules["sam2.build_sam"].build_sam2_video_predictor = build
    return holder


def build_world(tmp):
    cfg_p = sm.build_dataset(tmp, gate=False)
    # ground truth annotations (palette PNGs, object id 1 = the true target)
    ad = tmp / "DAVIS" / "Annotations" / "480p" / "vid"
    ad.mkdir(parents=True, exist_ok=True)
    for t in range(N):
        im = Image.new("P", (W, H))
        im.putpalette([0, 0, 0, 255, 0, 0] + [0, 0, 0] * 254)
        im.putdata(sm.true_mask(t).astype(np.uint8).flatten().tolist())
        im.save(ad / f"{t:05d}.png")
    # candidate cache: on every frame the true object and the distractor
    frames = list(range(1, N))
    boxes = [[[*np.array([sm.true_mask(t).nonzero()[1].min(), sm.true_mask(t).nonzero()[0].min(),
                          sm.true_mask(t).nonzero()[1].max() + 1, sm.true_mask(t).nonzero()[0].max() + 1], float), 0.9],
              [110.0, 5.0, 140.0, 35.0, 0.5]] for t in frames]
    packed = [[np.packbits(sm.true_mask(t).reshape(-1)), np.packbits(DISTRACTOR.reshape(-1))] for t in frames]
    C.save_store(tmp / "candidates" / "vid" / "0.npz", frames, boxes, packed, (H, W), 5)
    v3 = {"v3": {"events": EVENTS, "vlm": {"provider": "gemini", "cache_dir": str(tmp / "vlmc")}}}
    (tmp / "cfg_v3.yaml").write_text(yaml.safe_dump(v3))
    return cfg_p


def run_stage(mod_name, argv):
    sys.argv = [mod_name + ".py", *argv]
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    importlib.import_module(mod_name).main()


def masks_of(d):
    return {t: np.array(Image.open(d / "vid" / "0" / f"{t:05d}.png")) > 0 for t in range(N)}


def rows(p):
    return list(csv.DictReader(open(p)))


def run_v4(tmp, cfg_p, variant, holder):
    out, res = tmp / f"pred_{variant}", tmp / f"res_{variant}"
    run_stage("stage_reid", ["--config", str(cfg_p), "--v3-config", str(tmp / "cfg_v3.yaml"), "--variant", variant,
                             "--src-preds", str(tmp / "pred"), "--cand-dir", str(tmp / "candidates"),
                             "--pred-dir", str(out), "--results-dir", str(res)])
    return masks_of(out), rows(res / "reid_log.csv"), holder["pred"].restarts


def main():
    import vlm as V
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        holder = install(tmp)
        cfg_p = build_world(tmp)
        run_stage("stage2_tcs", ["--config", str(cfg_p)])                     # the V2 passive masks V4 builds on
        v2 = masks_of(tmp / "pred")
        assert not v2[20].any() == False and (v2[20] == DISTRACTOR).all()      # the fake tracker is on the distractor at 20

        # ---- off: byte-for-byte the V2 masks, but the searches are logged and graded
        m, log, restarts = run_v4(tmp, cfg_p, "off", holder)
        assert all((m[t] == v2[t]).all() for t in range(N)) and restarts == []
        assert log and all(r["action"] == "keep" and r["choice"] == "none" for r in log)
        assert any(r["frame"] == "20" and r["trigger"] == "coherence_drop" and r["label"] == "missed" for r in log), \
            [(r["frame"], r["trigger"], r["label"]) for r in log]
        print("ok   off == V2 passive, searches logged and graded")

        # ---- oracle: restarts on the true object at frame 20 and follows it from there
        m, log, restarts = run_v4(tmp, cfg_p, "oracle", holder)
        assert restarts[0] == 20, restarts
        assert all((m[t] == v2[t]).all() for t in range(0, 20))               # causal: nothing before the restart changed
        assert all((m[t] == sm.true_mask(t)).all() for t in range(20, N)), [t for t in range(20, N) if not (m[t] == sm.true_mask(t)).all()]
        assert restarts == [20, 30]                                           # the tracker also goes empty at 30 while the target is there
        assert not v2[30].any() and not v2[31].any()                          # (V2 stayed empty for those two frames)
        r20 = next(r for r in log if r["frame"] == "20")
        assert r20["action"] == "restart" and r20["label"] == "correct_restart" and r20["choice"] == "d0", r20
        print("ok   oracle restarts at the jump and only from there")

        # ---- memory: picks the true object because it looks like the target and sits where it was
        m, log, restarts = run_v4(tmp, cfg_p, "memory", holder)
        assert restarts == [20, 30] and all((m[t] == v2[t]).all() for t in range(0, 20))
        assert all((m[t] == sm.true_mask(t)).all() for t in range(20, N))
        print("ok   memory chooser recovers the target, twice")

        # ---- vlm (scripted): answers A (= best by memory score) with confidence 0.9; counts calls
        calls = {"n": 0, "images": []}

        def script(images, prompt):
            calls["n"] += 1
            calls["images"].append(len(images))
            return '{"choice": "A", "confidence": 0.9, "reason": "same blob"}'

        orig = V.make_client
        V.make_client = lambda vcfg: V.CachedVLM(V.MockVLM(script), tmp / "vlmcache")
        try:
            m, log, restarts = run_v4(tmp, cfg_p, "vlm", holder)
        finally:
            V.make_client = orig
        assert restarts == [20, 30] and all((m[t] == v2[t]).all() for t in range(0, 20))
        assert all((m[t] == sm.true_mask(t)).all() for t in range(20, N))
        assert calls["n"] >= 1 and all(n >= 3 for n in calls["images"])        # start crop + memory crop + >= 1 candidate
        assert sum(r["action"] == "restart" for r in log) == 2                  # (the 2nd identical question may come from the cache)
        row = next(r for r in log if r["frame"] == "20")
        assert row["vlm_letter"] == "A" and row["action"] == "restart" and row["vlm_error"] == ""
        summ = (tmp / "res_vlm" / "reid_summary.txt").read_text()
        assert "vlm_calls:" in summ and "restarts: 2" in summ, summ
        print("ok   vlm variant asks, restarts and counts its calls")

        # ---- vlm that always errors: nothing restarts, nothing crashes, masks equal V2
        def boom(images, prompt):
            raise RuntimeError("quota exceeded")

        V.make_client = lambda vcfg: V.CachedVLM(V.MockVLM(boom), tmp / "vlmcache2")
        try:
            m, log, restarts = run_v4(tmp, cfg_p, "vlm", holder)
        finally:
            V.make_client = orig
        assert restarts == [] and all((m[t] == v2[t]).all() for t in range(N))
        assert any("quota exceeded" in r["vlm_error"] for r in log)
        print("ok   a failing VLM changes nothing")

        # ---- a gated config is refused (V4 builds on V2 passive)
        gated = sm.build_dataset(tmp, gate=True)
        try:
            run_stage("stage_reid", ["--config", str(gated), "--v3-config", str(tmp / "cfg_v3.yaml"), "--variant", "off",
                                     "--pred-dir", str(tmp / "x"), "--results-dir", str(tmp / "y")])
        except SystemExit as e:
            assert "PASSIVE" in str(e)
        else:
            raise AssertionError("expected SystemExit")
        print("ok   gated config refused")
    print("all stage_reid smoke checks passed")


if __name__ == "__main__":
    main()
