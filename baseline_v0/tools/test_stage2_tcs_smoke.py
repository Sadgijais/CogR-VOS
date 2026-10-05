"""
Wiring smoke test for stage2_tcs.py with a FAKE SAM 2, FAKE torch and FAKE embedders
on a tiny synthetic video. No GPU, no dataset, no model downloads.

It proves the plumbing (config -> memory -> TCS -> logs -> CSVs -> PNGs, gate on/off),
NOT accuracy. Real SAM 2 behaviour is only exercised by the real run.

Run (from baseline_v0/):   python tools/test_stage2_tcs_smoke.py
"""
import csv
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

H, W, N = 120, 160, 40
DIM = 32
rng = np.random.default_rng(3)


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


BASE, OTHER = unit(rng.normal(size=DIM)), unit(rng.normal(size=DIM))
TEXT = unit(rng.normal(size=DIM))
ORTHO = unit(rng.normal(size=DIM) - TEXT * (TEXT @ rng.normal(size=DIM)))
ORTHO = unit(ORTHO - TEXT * float(ORTHO @ TEXT))


def box_mask(x1, y1, x2, y2):
    m = np.zeros((H, W), bool)
    m[max(0, y1):y2, max(0, x1):x2] = True
    return m


def true_mask(t):
    """Where the real target is: one smooth path, 2 px/frame to the right."""
    x = 20 + 2 * t
    return box_mask(x, 40, x + 30, 70)


def scripted_mask(t):
    """What the fake tracker predicts: the true path, except frames 20-22 it jumps onto a distractor
    and frames 30-31 it reports the target absent."""
    if t in (30, 31):
        return None
    if t in (20, 21, 22):
        return box_mask(110, 5, 140, 35)
    return true_mask(t)


# ------------------------------------------------------------------ fakes
class FT:                                   # tiny stand-in for a torch tensor
    def __init__(self, a): self.a = a
    def squeeze(self): return self
    def __gt__(self, x): return FT(self.a > x)
    def cpu(self): return self
    def numpy(self): return self.a
    def float(self): return FT(self.a.astype(np.float32))


def install_fakes(tmp: Path):
    torch = types.ModuleType("torch")
    cuda = types.ModuleType("torch.cuda")
    cuda.is_available = lambda: False
    cuda.synchronize = lambda: None
    cuda.reset_peak_memory_stats = lambda: None
    cuda.max_memory_allocated = lambda: 0
    cuda.max_memory_reserved = lambda: 0
    cuda.empty_cache = lambda: None
    torch.cuda = cuda
    sys.modules["torch"], sys.modules["torch.cuda"] = torch, cuda

    class FakePredictor:
        def init_state(self, **kw):
            return {"output_dict_per_obj": {0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}}}

        def add_new_points_or_box(self, **kw):
            pass

        def propagate_in_video(self, state):
            od = state["output_dict_per_obj"][0]
            for t in range(N):
                m = scripted_mask(t)
                present = m is not None
                m = m if present else np.zeros((H, W), bool)
                out = {"object_score_logits": np.array([[6.0 if present else -5.0]])}
                (od["cond_frame_outputs"] if t == 0 else od["non_cond_frame_outputs"])[t] = out
                yield t, [1], [FT(np.where(m, 5.0, -5.0).astype(np.float32))]

        def reset_state(self, state):
            pass

    sam2 = types.ModuleType("sam2")
    build = types.ModuleType("sam2.build_sam")
    build.build_sam2_video_predictor = lambda *a, **k: FakePredictor()
    sam2.build_sam = build
    sys.modules["sam2"], sys.modules["sam2.build_sam"] = sam2, build

    class CropEmbedder:
        def __init__(self, **kw): pass
        def embed_masked(self, image, mask):
            cx = np.nonzero(mask)[1].mean()
            r = np.random.default_rng(int(cx * 10) + int(mask.sum()))      # deterministic per mask
            return unit((OTHER if cx > 100 else BASE) + 0.05 * r.normal(size=DIM))

    class ClipCropEmbedder:
        def __init__(self, **kw): pass
        def embed_masked(self, image, mask):
            return unit(0.30 * TEXT + np.sqrt(1 - 0.09) * ORTHO)

    emb = types.ModuleType("common.embed")
    emb.CropEmbedder, emb.ClipCropEmbedder = CropEmbedder, ClipCropEmbedder
    sys.modules["common.embed"] = emb


def build_dataset(tmp: Path, gate: bool) -> Path:
    root = tmp / "DAVIS"
    jd = root / "JPEGImages" / "480p" / "vid"
    jd.mkdir(parents=True, exist_ok=True)
    for t in range(N):
        Image.fromarray(np.full((H, W, 3), 128, np.uint8)).save(jd / f"{t:05d}.jpg")
    (tmp / "grounding").mkdir(exist_ok=True)
    (tmp / "grounding" / "vid.json").write_text(json.dumps(
        {"0": {"expression": "the blob", "obj_id": 1,
               "boxes": [[20, 40, 50, 70, 0.9], [60, 10, 80, 30, 0.4], [90, 60, 120, 90, 0.3]]}}))
    (tmp / "embeddings").mkdir(exist_ok=True)
    np.savez(tmp / "embeddings" / "vid.npz", **{"0__text": TEXT,
             "0__boxes": np.stack([BASE, OTHER, unit(rng.normal(size=DIM))])})
    cfg = {
        "dataset": {"root": str(root), "tier": "full", "annotator_set": 0},
        "propagation": {"model_cfg": "x.yaml", "checkpoint": "x.pt", "device": "cpu",
                        "offload_video_to_cpu": True, "offload_state_to_cpu": True},
        "evaluation": {"require_480p": True, "exclude_first_last": True, "bound_th_px": 4},
        "paths": {"grounding_out": str(tmp / "grounding"), "predictions_out": str(tmp / "pred"),
                  "results_out": str(tmp / "res")},
        "v1": {"use_sttm": True, "embeddings_out": str(tmp / "embeddings"),
               "params": {"K": 4, "delta": 3, "a_min": 200, "tau_sim": 0.3, "sim_mode": "anchor_proto"}},
        "v2": {"gate_writes": gate, "tcs": {"tau_low": 0.5, "tau_high": 0.7}},
    }
    p = tmp / f"cfg_{'gate' if gate else 'passive'}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def run_stage2(cfg_path: Path, extra=()):
    import importlib
    sys.argv = ["stage2_tcs.py", "--config", str(cfg_path), *extra]
    if "stage2_tcs" in sys.modules:
        del sys.modules["stage2_tcs"]
    mod = importlib.import_module("stage2_tcs")
    mod.main()


def read_csv(p):
    return list(csv.DictReader(open(p)))


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        install_fakes(tmp)

        # ------------------------------- passive run (gate OFF)
        cfg_p = build_dataset(tmp, gate=False)
        run_stage2(cfg_p)
        log = json.load(open(tmp / "res" / "tcs_log" / "vid.json"))["0"]
        frames = {f["f"]: f for f in log["frames"]}
        assert len(frames) == N - 1, len(frames)                     # frame 0 is the anchor
        assert log["gate_writes"] is False
        assert all(frames[t]["band"] == "HIGH" for t in range(5, 19)), \
            [(t, frames[t]["band"], frames[t]["c"]) for t in range(5, 19)]
        assert all(frames[t]["band"] != "HIGH" for t in (20, 21, 22)), \
            [(t, frames[t]["band"], frames[t]["c"]) for t in (20, 21, 22)]
        for t in (30, 31):
            assert frames[t]["visible"] is False and frames[t]["state"] == "absent" and frames[t]["c"] == 0.0
        assert frames[32]["state"] == "reappeared" and frames[32]["terms"]["motion"] is None
        assert len(list((tmp / "pred" / "vid" / "0").glob("*.png"))) == N
        st = read_csv(tmp / "res" / "sttm_per_expression.csv")[0]
        assert int(st["blocked_tcs_uncertain"]) == 0                  # passive never blocks
        passive_writes = int(st["writes"])
        tc = read_csv(tmp / "res" / "tcs_per_expression.csv")[0]
        assert int(tc["n_absent_frames"]) == 2 and int(tc["frames"]) == N - 1
        tm = read_csv(tmp / "res" / "timing_per_video.csv")[0]
        assert "tcs_overhead_s" in tm
        print("ok   passive run: bands, absence, reappearance, files")

        # ------------------------------- same config, forced passive via flag even when gate is on
        import shutil
        shutil.rmtree(tmp / "res"); shutil.rmtree(tmp / "pred")
        cfg_g = build_dataset(tmp, gate=True)
        run_stage2(cfg_g, ["--no-gate"])
        st = read_csv(tmp / "res" / "sttm_per_expression.csv")[0]
        assert int(st["blocked_tcs_uncertain"]) == 0 and int(st["writes"]) == passive_writes
        print("ok   --no-gate overrides config gate_writes: true")

        # ------------------------------- gated run
        shutil.rmtree(tmp / "res"); shutil.rmtree(tmp / "pred")
        run_stage2(cfg_g)
        log = json.load(open(tmp / "res" / "tcs_log" / "vid.json"))["0"]
        assert log["gate_writes"] is True
        for f in log["frames"]:
            if f["band"] != "HIGH" and f["visible"]:
                assert f["result"] in ("tcs_uncertain", "low_similarity", "truncated"), f
        st = read_csv(tmp / "res" / "sttm_per_expression.csv")[0]
        assert int(st["blocked_tcs_uncertain"]) >= 1                  # the teleport frames were blocked
        print("ok   gated run: uncertain frames blocked from memory:", st["blocked_tcs_uncertain"])

    print("\nsmoke test passed (wiring only; real SAM 2 not exercised)")


if __name__ == "__main__":
    main()
