import reid as R
import stage_reid as SR
def rk(*pairs): return [{"id": i, "score": s} for i, s in pairs]
def test_keep_when_current_is_close():
    assert R.margin_gate("d0", rk(("d0", 0.80), ("current", 0.75)), 0.10) == R.CURRENT
def test_restart_when_clearly_better():
    assert R.margin_gate("d0", rk(("d0", 0.90), ("current", 0.60)), 0.10) == "d0"
def test_restart_when_no_current_mask():
    assert R.margin_gate("d0", rk(("d0", 0.70)), 0.10) == "d0"
def test_none_and_current_pass_through():
    r = rk(("d0", 0.9), ("current", 0.5))
    assert R.margin_gate(R.NONE, r, 0.10) == R.NONE and R.margin_gate(R.CURRENT, r, 0.10) == R.CURRENT
def test_unknown_pick_is_none():
    assert R.margin_gate("zz", rk(("current", 0.5), ("d0", 0.9)), 0.10) == R.NONE
def test_margin_variant_vs_plain_memory():
    p = R.ReIDParams(); r = rk(("d0", 0.90), ("current", 0.85)); c = [{"id": "d0"}, {"id": "current"}]
    assert p.switch_margin == 0.10
    assert SR.decide_search("memory", c, r, None, p)["choice"] == "d0"
    assert SR.decide_search("memory_margin", c, r, None, p)["choice"] == R.CURRENT
if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); n += 1; print("ok  ", k)
    print(f"all {n} tests passed")
