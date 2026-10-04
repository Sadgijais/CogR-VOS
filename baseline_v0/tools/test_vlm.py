"""Tests for vlm.py: answer parsing, request building for the three providers (no network), caching.
Run (from baseline_v0/):   python tools/test_vlm.py"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vlm as V

IMG = Image.fromarray(np.full((40, 60, 3), 90, np.uint8))


def test_parse_clean_json():
    d = V.parse_verdict('{"verdict": "no_match", "confidence": 0.8, "reason": "different person"}')
    assert d == {"verdict": "no_match", "confidence": 0.8, "reason": "different person"}


def test_parse_json_inside_prose_and_code_fence():
    d = V.parse_verdict('Sure!\n```json\n{"verdict": "match", "confidence": 1.4}\n```')
    assert d["verdict"] == "match" and d["confidence"] == 1.0          # confidence clamped to [0,1]


def test_parse_synonyms_and_garbage_never_raise():
    assert V.parse_verdict('{"verdict": "Same"}')["verdict"] == "match"
    assert V.parse_verdict('{"verdict": "different"}')["verdict"] == "no_match"
    assert V.parse_verdict('{"verdict": "not visible"}')["verdict"] == "not_visible"
    assert V.parse_verdict('{"verdict": "banana"}')["verdict"] == "unsure"
    assert V.parse_verdict("") == {"verdict": "unsure", "confidence": 0.0, "reason": ""}
    assert V.parse_verdict("I think this is no_match to be honest")["verdict"] == "no_match"
    assert V.parse_verdict("{broken json")["verdict"] == "unsure"


def test_prompts_carry_the_expression_and_ask_for_json():
    for p in (V.identity_prompt("the man in the red shirt"), V.presence_prompt("the man in the red shirt")):
        assert "the man in the red shirt" in p and "JSON" in p


def test_masked_crop_greys_the_background_and_resizes():
    img = Image.fromarray(np.full((300, 400, 3), 200, np.uint8))
    m = np.zeros((300, 400), bool)
    m[100:250, 100:350] = True
    c = V.masked_crop(img, m, pad=0, max_side=128)
    assert max(c.size) == 128
    assert V.masked_crop(img, np.zeros((300, 400), bool)) is None


def test_request_building_and_answer_extraction_for_all_providers():
    canned = {
        "gemini": {"candidates": [{"content": {"parts": [{"text": '{"verdict":"match"}'}]}}]},
        "openai": {"choices": [{"message": {"content": '{"verdict":"match"}'}}]},
        "anthropic": {"content": [{"type": "text", "text": '{"verdict":"match"}'}]},
    }
    for prov, resp in canned.items():
        c = V.RestVLM(prov)
        url, headers, body = c.build([IMG, IMG], "PROMPT", "KEY123")
        assert url.startswith("https://") and "KEY123" in json.dumps(headers)
        assert "KEY123" not in url and "KEY123" not in json.dumps(body)          # key only in headers
        assert "PROMPT" in json.dumps(body) and json.dumps(body).count("/9j/") == 2   # two JPEGs, base64
        assert c.extract(resp) == '{"verdict":"match"}'


def test_rest_client_needs_the_key_and_retries_then_fails_cleanly(monkeypatch=None):
    import os
    c = V.RestVLM("gemini", api_key_env="NO_SUCH_KEY_VAR_XYZ")
    try:
        c.ask([IMG], "p")
        raise AssertionError("should need a key")
    except RuntimeError as e:
        assert "NO_SUCH_KEY_VAR_XYZ" in str(e)
    os.environ["COGR_TEST_KEY"] = "k"
    calls = {"n": 0}
    orig_post, orig_sleep = V._post, V.time.sleep
    V.time.sleep = lambda s: None

    def flaky(url, headers, body, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise V.urllib.error.URLError("net down")
        return {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
    V._post = flaky
    try:
        assert V.RestVLM("gemini", api_key_env="COGR_TEST_KEY").ask([IMG], "p") == "ok" and calls["n"] == 3
        V._post = lambda *a: (_ for _ in ()).throw(V.urllib.error.URLError("down"))
        try:
            V.RestVLM("gemini", api_key_env="COGR_TEST_KEY", retries=2).ask([IMG], "p")
            raise AssertionError("should fail after retries")
        except RuntimeError as e:
            assert "failed after 2 tries" in str(e)
    finally:
        V._post, V.time.sleep = orig_post, orig_sleep


def test_cache_returns_same_answer_keeps_original_latency_and_skips_the_api():
    with tempfile.TemporaryDirectory() as td:
        inner = V.MockVLM(lambda imgs, p: '{"verdict": "match", "confidence": 0.9}')
        c = V.CachedVLM(inner, td)
        t1, lat1, cached1 = c.ask_timed([IMG], "prompt A")
        t2, lat2, cached2 = c.ask_timed([IMG], "prompt A")
        assert inner.calls == 1 and not cached1 and cached2 and t1 == t2 and lat2 == lat1
        c.ask_timed([IMG], "prompt B")                                   # different prompt = different key
        other = Image.fromarray(np.full((40, 60, 3), 10, np.uint8))
        c.ask_timed([other], "prompt A")                                 # different image = different key
        assert inner.calls == 3


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"\nall {len(tests)} tests passed")
