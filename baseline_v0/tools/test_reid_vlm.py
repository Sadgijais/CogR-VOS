#!/usr/bin/env python3
"""CPU tests for the V4 re-ID question in vlm.py and for the thread-safe answer cache (mock VLM, no network)."""
import json
import sys
import tempfile
import threading
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vlm as V


def img(c):
    return Image.new("RGB", (16, 16), (c, c, c))


def client(script, td):
    return V.CachedVLM(V.MockVLM(script), Path(td) / "cache")


def test_prompt_names_the_expression_the_images_and_the_letters():
    p = V.reid_prompt("a man in red", "ABC")
    assert "a man in red" in p and "image 3 = candidate A" in p and "image 5 = candidate C" in p
    assert '"A" | "B" | "C" | "none"' in p
    assert "image 4" not in V.reid_prompt("x", "A")


def test_parser_accepts_fences_case_and_candidate_words():
    assert V.parse_reid('```json\n{"choice": "B", "confidence": 0.9, "reason": "same cap"}\n```', "ABC")["choice"] == "B"
    assert V.parse_reid('{"choice": "b", "confidence": 0.8}', "ABC")["choice"] == "B"
    assert V.parse_reid('{"choice": "Candidate C", "confidence": 0.8}', "ABC")["choice"] == "C"
    assert V.parse_reid('{"choice": "none", "confidence": 0.7}', "ABC")["choice"] == "none"


def test_parser_never_raises_and_defaults_to_none():
    for bad in ("", None, "I think it is A", "{not json", "[1, 2]", '{"choice": "D", "confidence": 0.9}',
                '{"choice": 3, "confidence": 0.9}', '{"choice": "AB", "confidence": 0.9}'):
        r = V.parse_reid(bad, "ABC")
        assert r["choice"] == "none", bad
    assert V.parse_reid('{"choice": "A", "confidence": 7}', "ABC")["confidence"] == 1.0
    assert V.parse_reid('{"choice": "A", "confidence": "high"}', "ABC")["confidence"] == 0.5


def test_ask_reid_picks_index_only_when_confident():
    with tempfile.TemporaryDirectory() as td:
        seen = {}

        def script(images, prompt):
            seen["n_images"] = len(images)
            return '{"choice": "B", "confidence": 0.9, "reason": "same"}'

        c = client(script, td)
        r = V.ask_reid(c, "a dog", img(1), img(2), [img(3), img(4), img(5)])
        assert r["index"] == 1 and r["letter"] == "B" and r["error"] == "" and seen["n_images"] == 5
        assert r["cached"] is False
        r2 = V.ask_reid(c, "a dog", img(1), img(2), [img(3), img(4), img(5)])
        assert r2["index"] == 1 and r2["cached"] is True                       # second identical question is free


def test_low_confidence_none_and_unparseable_all_mean_no_restart():
    with tempfile.TemporaryDirectory() as td:
        for k, text in enumerate(['{"choice": "A", "confidence": 0.59}', '{"choice": "none", "confidence": 0.99}',
                                  "no idea", '{"choice": "A", "confidence": 0.6}']):
            c = client(lambda i, p, t=text: t, Path(td) / str(k))
            r = V.ask_reid(c, "x", img(1), img(2), [img(3)])
            assert (r["index"] == 0) == (k == 3), (text, r)                       # exactly 0.6 is enough, 0.59 is not


def test_api_error_does_not_raise_and_is_reported():
    with tempfile.TemporaryDirectory() as td:
        def boom(images, prompt):
            raise RuntimeError("quota exceeded")
        r = V.ask_reid(client(boom, td), "x", img(1), img(2), [img(3), img(4)])
        assert r["index"] is None and "quota exceeded" in r["error"] and r["reason"] == "error"


def test_no_candidates_means_no_call():
    with tempfile.TemporaryDirectory() as td:
        calls = []
        r = V.ask_reid(client(lambda i, p: calls.append(1) or "{}", td), "x", img(1), img(2), [])
        assert r["index"] is None and calls == []


def test_cache_is_safe_when_many_threads_ask_the_same_question():
    import time
    with tempfile.TemporaryDirectory() as td:
        def slow(images, prompt):
            time.sleep(0.002)                                                  # widen the window in which threads overlap
            return '{"choice": "A", "confidence": 0.9, "reason": "r"}'

        c = client(slow, td)
        errs = []

        def work():
            for k in range(30):
                r = V.ask_reid(c, "x", img(1), img(2), [img(3 + k % 3)])      # 3 different questions, asked by all threads
                if r["error"] or r["index"] != 0:
                    errs.append(r["error"] or r["letter"])

        ts = [threading.Thread(target=work) for _ in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert not errs, errs[:3]
        files = list((Path(td) / "cache").iterdir())
        assert files and all(f.suffix == ".json" for f in files), files        # no leftover temp files
        for f in files:
            json.loads(f.read_text())                                           # every entry is complete


def test_damaged_cache_entry_is_asked_again_not_an_error():
    with tempfile.TemporaryDirectory() as td:
        c = client(lambda i, p: '{"choice": "A", "confidence": 0.9}', td)
        V.ask_reid(c, "x", img(1), img(2), [img(3)])
        f = next((Path(td) / "cache").iterdir())
        f.write_text('{"text": "abc", "latency_s": 0.1}{"text"')             # an interrupted double write
        r = V.ask_reid(c, "x", img(1), img(2), [img(3)])
        assert r["error"] == "" and r["index"] == 0 and r["cached"] is False


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok   {t.__name__}")
    print(f"all {len(tests)} tests passed")
