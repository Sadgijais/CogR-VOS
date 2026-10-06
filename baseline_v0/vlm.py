"""
VLM layer for V3: prompts, answer parsing, caching, and thin REST clients. No SDKs, no torch,
only the standard library + PIL, so it runs in the same environment as everything else.

The VLM never produces pixels. It answers ONE small question about crops that the tracker already
produced, and returns a verdict:

  identity  (target visible): image 1 = the target at the start, image 2 = what the tracker follows now
            match / no_match / unsure
  presence  (mask is empty):  one frame, "is the described object visible?"
            match (visible) / not_visible / unsure

Providers: gemini, openai, anthropic (plain HTTPS), mock (scripted, for tests).
The API key is read from an environment variable and never written anywhere.
Every answer is cached on disk by a hash of (model, prompt, image bytes), so a re-run costs nothing,
and the cache keeps the ORIGINAL latency so timing can still be reported honestly.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

VERDICTS = ("match", "no_match", "not_visible", "unsure")


# ------------------------------------------------------------------ images
def masked_crop(image: Image.Image, mask: np.ndarray, pad: int = 8, max_side: int = 256):
    """Tight crop around the mask, background greyed out, resized so the long side is <= max_side.
    None if the mask is empty."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    h, w = mask.shape
    y1, y2 = max(0, ys.min() - pad), min(h, ys.max() + 1 + pad)
    x1, x2 = max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)
    arr = np.array(image.convert("RGB"))
    arr = np.where(mask[..., None], arr, np.full_like(arr, 124))[y1:y2, x1:x2]
    im = Image.fromarray(arr)
    s = max(im.size)
    if s > max_side:
        im = im.resize((max(1, round(im.size[0] * max_side / s)), max(1, round(im.size[1] * max_side / s))))
    return im


def small_frame(image: Image.Image, max_side: int = 512):
    im = image.convert("RGB")
    s = max(im.size)
    if s > max_side:
        im = im.resize((round(im.size[0] * max_side / s), round(im.size[1] * max_side / s)))
    return im


def jpeg_bytes(im: Image.Image, quality: int = 90) -> bytes:
    b = io.BytesIO()
    im.convert("RGB").save(b, format="JPEG", quality=quality)
    return b.getvalue()


# ------------------------------------------------------------------ prompts
def identity_prompt(expression: str) -> str:
    return (
        "You check the output of an object tracker.\n"
        "Image 1 shows the TARGET object at the start of a video.\n"
        "Image 2 shows the object the tracker follows NOW (grey background removed).\n"
        f'The target is described as: "{expression}".\n'
        "Is image 2 the SAME object as image 1? Ignore changes of pose, lighting, scale and partial occlusion; "
        "answer no_match if it is a different object, even a similar-looking one (another person, animal or vehicle). "
        "If the crop is too small or unclear to tell, answer unsure.\n"
        'Reply with JSON only: {"verdict": "match" | "no_match" | "unsure", "confidence": <0..1>, '
        '"reason": "<at most 15 words>"}'
    )


def presence_prompt(expression: str) -> str:
    return (
        "You check the output of an object tracker. The tracker reports that the target is NOT in this frame.\n"
        f'The target is described as: "{expression}".\n'
        "Is an object matching that description visible anywhere in the image, even partly? "
        "Answer match if it is visible, not_visible if it is not, unsure if you cannot tell.\n"
        'Reply with JSON only: {"verdict": "match" | "not_visible" | "unsure", "confidence": <0..1>, '
        '"reason": "<at most 15 words>"}'
    )


def parse_verdict(text: str) -> dict:
    """Never raises. Anything unparseable becomes verdict 'unsure' with confidence 0."""
    out = {"verdict": "unsure", "confidence": 0.0, "reason": ""}
    if not text:
        return out
    m = re.search(r"\{.*?\}", text, flags=re.S)
    blob = m.group(0) if m else text
    try:
        d = json.loads(blob)
    except Exception:
        low = text.lower()
        for v in ("no_match", "not_visible", "match"):          # longest first
            if re.search(rf"\b{v}\b", low):
                out["verdict"] = v
                out["confidence"] = 0.5
                break
        return out
    v = str(d.get("verdict", "")).strip().lower().replace(" ", "_").replace("-", "_")
    if v in ("same", "yes", "visible", "present"):
        v = "match"
    elif v in ("different", "no", "other"):
        v = "no_match"
    elif v in ("absent", "invisible", "notvisible"):
        v = "not_visible"
    out["verdict"] = v if v in VERDICTS else "unsure"
    try:
        out["confidence"] = min(1.0, max(0.0, float(d.get("confidence", 0.5))))
    except Exception:
        out["confidence"] = 0.5
    out["reason"] = str(d.get("reason", ""))[:200]
    return out


# ------------------------------------------------------------------ clients
class MockVLM:
    """Scripted answers for tests: script(images, prompt) -> str (raw model text)."""
    model = "mock"

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def ask(self, images, prompt: str) -> str:
        self.calls += 1
        return self.script(images, prompt)


def _post(url: str, headers: dict, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class RestVLM:
    """gemini | openai | anthropic over plain HTTPS. Edit the model name to one your account has."""
    DEFAULTS = {
        "gemini": ("gemini-2.5-flash", "GEMINI_API_KEY"),
        "openai": ("gpt-4o-mini", "OPENAI_API_KEY"),
        "anthropic": ("claude-haiku-4-5-20251001", "ANTHROPIC_API_KEY"),
    }

    def __init__(self, provider: str, model: str | None = None, api_key_env: str | None = None,
                 timeout: float = 60.0, max_tokens: int = 200, retries: int = 3):
        if provider not in self.DEFAULTS:
            raise ValueError(f"provider must be one of {list(self.DEFAULTS)}")
        self.provider = provider
        self.model = model or self.DEFAULTS[provider][0]
        self.api_key_env = api_key_env or self.DEFAULTS[provider][1]
        self.timeout, self.max_tokens, self.retries = timeout, max_tokens, retries

    # ---- request building / answer extraction (unit-tested without network)
    def build(self, images, prompt: str, key: str):
        b64 = [base64.b64encode(jpeg_bytes(i)).decode() for i in images]
        if self.provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
            parts = [{"text": prompt}] + [{"inline_data": {"mime_type": "image/jpeg", "data": d}} for d in b64]
            body = {"contents": [{"parts": parts}],
                    "generationConfig": {"temperature": 0, "maxOutputTokens": self.max_tokens,
                                         "responseMimeType": "application/json"}}
        elif self.provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            content = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{d}"}} for d in b64]
            body = {"model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "user", "content": content}]}
        else:
            url = "https://api.anthropic.com/v1/messages"
            headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": d}}
                       for d in b64] + [{"type": "text", "text": prompt}]
            body = {"model": self.model, "max_tokens": self.max_tokens, "temperature": 0,
                    "messages": [{"role": "user", "content": content}]}
        return url, headers, body

    def extract(self, resp: dict) -> str:
        if self.provider == "gemini":
            return "".join(p.get("text", "") for p in resp["candidates"][0]["content"]["parts"])
        if self.provider == "openai":
            return resp["choices"][0]["message"]["content"]
        return "".join(b.get("text", "") for b in resp["content"] if b.get("type") == "text")

    def ask(self, images, prompt: str) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"environment variable {self.api_key_env} is not set")
        url, headers, body = self.build(images, prompt, key)
        last = None
        for attempt in range(self.retries):
            try:
                return self.extract(_post(url, headers, body, self.timeout))
            except urllib.error.HTTPError as e:
                last = e
                if e.code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"{self.provider} HTTP {e.code}: {e.read().decode()[:300]}") from None
            except (urllib.error.URLError, TimeoutError, KeyError) as e:
                last = e
            time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"{self.provider} failed after {self.retries} tries: {last}")


class CachedVLM:
    """Disk cache around any client. ask_timed() -> (text, latency_s, cached). Thread-safe enough
    for a thread pool (one file per key, written atomically)."""

    def __init__(self, inner, cache_dir: str | Path):
        self.inner = inner
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model = getattr(inner, "model", "unknown")

    def key(self, images, prompt: str) -> str:
        h = hashlib.sha256()
        h.update(f"{getattr(self.inner, 'provider', 'x')}|{self.model}|{prompt}".encode())
        for im in images:
            h.update(jpeg_bytes(im))
        return h.hexdigest()

    def ask_timed(self, images, prompt: str):
        k = self.key(images, prompt)
        p = self.dir / f"{k}.json"
        if p.exists():
            try:
                d = json.loads(p.read_text())
                return d["text"], float(d["latency_s"]), True
            except (ValueError, KeyError, OSError):
                pass                                    # damaged entry (an interrupted write): ask again
        t0 = time.perf_counter()
        text = self.inner.ask(images, prompt)
        lat = time.perf_counter() - t0
        tmp = p.with_suffix(f".{os.getpid()}.{os.urandom(4).hex()}.tmp")   # unique per write, so threads never share a file
        tmp.write_text(json.dumps({"text": text, "latency_s": lat, "model": self.model}))
        os.replace(tmp, p)
        return text, lat, False


def make_client(vcfg: dict):
    """Build a cached client from the config's v3.vlm block."""
    prov = vcfg.get("provider", "gemini")
    if prov == "mock":
        raise ValueError("mock clients are built in code (tests), not from a config")
    inner = RestVLM(prov, vcfg.get("model"), vcfg.get("api_key_env"),
                    timeout=float(vcfg.get("timeout_s", 60)), max_tokens=int(vcfg.get("max_tokens", 200)))
    return CachedVLM(inner, vcfg.get("cache_dir", "vlm_cache/"))


# ------------------------------------------------------------------ V4: the re-identification question
REID_LETTERS = "ABCDE"


def reid_prompt(expression: str, letters: str) -> str:
    """Image 1 = the target at the start. Image 2 = how the tracker saw the target last (from target memory).
    Images 3.. = the candidates on the current frame, labelled with `letters` (grey background removed)."""
    cand = ", ".join(f"image {3 + i} = candidate {c}" for i, c in enumerate(letters))
    return (
        "You help an object tracker find its target again after it lost it.\n"
        "Image 1 shows the TARGET object at the start of a video.\n"
        "Image 2 shows the same target as the tracker saw it most recently, before it was lost.\n"
        f"The next images show candidates on the current frame: {cand}.\n"
        f'The target is described as: "{expression}".\n'
        "Which candidate is the SAME object as the target? Ignore changes of pose, lighting, scale and partial "
        "occlusion. A similar-looking but different object (another person, animal or vehicle) is NOT the target. "
        f"If none of the candidates is the target, or you cannot tell, answer none.\n"
        f'Reply with JSON only: {{"choice": {" | ".join(chr(34) + c + chr(34) for c in letters)} | "none", '
        '"confidence": <0..1>, "reason": "<at most 15 words>"}'
    )


def parse_reid(text: str, letters: str) -> dict:
    """Never raises. Anything unparseable, unknown or out of range becomes choice 'none' with confidence 0."""
    out = {"choice": "none", "confidence": 0.0, "reason": ""}
    if not text:
        return out
    m = re.search(r"\{.*?\}", text, flags=re.S)
    try:
        d = json.loads(m.group(0) if m else text)
    except Exception:
        return out
    if not isinstance(d, dict):
        return out
    c = str(d.get("choice", "none")).strip().strip('"').upper().replace("CANDIDATE", "").strip()
    out["choice"] = c if (len(c) == 1 and c in letters.upper()) else "none"
    try:
        out["confidence"] = min(1.0, max(0.0, float(d.get("confidence", 0.5))))
    except Exception:
        out["confidence"] = 0.5
    out["reason"] = str(d.get("reason", ""))[:200]
    return out


def ask_reid(client, expression: str, start_img, memory_img, cand_imgs, conf_min: float = 0.6) -> dict:
    """One re-ID question. cand_imgs: PIL crops in shortlist order (at most len(REID_LETTERS)); they are labelled A, B, ...
    -> {"index": position in cand_imgs of the chosen candidate or None, "letter", "confidence", "reason",
        "latency_s", "cached", "error"}. A choice below conf_min, an unparseable answer or an API error is "none"
    (index None). Never raises: a failed call must not stop a run."""
    letters = REID_LETTERS[:len(cand_imgs)]
    res = {"index": None, "letter": "none", "confidence": 0.0, "reason": "", "latency_s": 0.0, "cached": False, "error": ""}
    if not letters:
        return res
    try:
        text, lat, cached = client.ask_timed([start_img, memory_img, *cand_imgs], reid_prompt(expression, letters))
    except Exception as e:                                           # network, quota, timeout ...
        res["error"], res["reason"] = f"{type(e).__name__}: {str(e)[:150]}", "error"
        return res
    p = parse_reid(text, letters)
    res.update(latency_s=lat, cached=cached, letter=p["choice"], confidence=p["confidence"], reason=p["reason"])
    if p["choice"] != "none" and p["confidence"] >= conf_min:
        res["index"] = letters.index(p["choice"])
    return res
