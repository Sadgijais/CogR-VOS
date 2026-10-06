#!/usr/bin/env python3
"""
One-off patch of vlm.py for V4 (run from baseline_v0/):   python tools/apply_v4_vlm_patch.py
 1. Makes the VLM answer cache safe when several threads ask the same question at once. Before, every thread of a process
    wrote the SAME temporary file name, so identical parallel requests could overwrite each other, and a half-written entry
    broke the next run (this was the rare failure of test_stage_events). Now each write has its own temporary file, and a
    damaged entry is treated as a miss.
 2. Appends the V4 re-identification question: reid_prompt(), parse_reid(), ask_reid() (text in tools/v4_vlm_block.txt).
Refuses to touch a vlm.py that is not the expected version, and does nothing if it was already applied.
"""
from pathlib import Path

p = Path("vlm.py")
s = p.read_text()
if "def ask_reid" in s:
    print("vlm.py already patched; nothing to do")
    raise SystemExit(0)
old1 = """        if p.exists():
            d = json.loads(p.read_text())
            return d["text"], float(d["latency_s"]), True
"""
new1 = """        if p.exists():
            try:
                d = json.loads(p.read_text())
                return d["text"], float(d["latency_s"]), True
            except (ValueError, KeyError, OSError):
                pass                                    # damaged entry (an interrupted write): ask again
"""
old2 = """        tmp = p.with_suffix(f".{os.getpid()}.tmp")"""
new2 = """        tmp = p.with_suffix(f".{os.getpid()}.{os.urandom(4).hex()}.tmp")   # unique per write, so threads never share a file"""
if s.count(old1) != 1 or s.count(old2) != 1:
    raise SystemExit("vlm.py is not the expected version; nothing was changed. Paste `git diff --stat` and `git status` to Claude.")
block = (Path(__file__).resolve().parent / "v4_vlm_block.txt").read_text()
p.write_text(s.replace(old1, new1).replace(old2, new2) + block)
print("patched vlm.py: thread-safe cache + re-ID question")
