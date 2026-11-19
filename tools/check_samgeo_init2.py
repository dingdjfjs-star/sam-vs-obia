# -*- coding: utf-8 -*-
"""探针3：SamGeo2.__init__ 完整源码 + patches 模块下载逻辑。"""
from __future__ import annotations

import inspect
import os

from samgeo import samgeo2

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "logs", "samgeo_init_probe.txt")
lines = []


def emit(s=""):
    lines.append(str(s))


init_src = inspect.getsource(samgeo2.SamGeo2.__init__)
body = init_src.splitlines()
emit(f"__init__ total lines: {len(body)}")
emit("\n".join(body[120:]))  # 前 120 行上一轮已看过（参数+docstring 头）

try:
    from samgeo import patches
    psrc = inspect.getsource(patches)
    emit("\n--- patches.py: download / hf / build 相关行 ---")
    for i, ln in enumerate(psrc.splitlines(), 1):
        low = ln.lower()
        if any(kw in low for kw in ("hf_hub", "snapshot", "download", "url", "checkpoint",
                                    "hiera")):
            emit(f"  L{i}: {ln.strip()}")
except Exception as e:
    emit(f"patches introspect failed: {e}")

emit("\n--- samgeo/common.py download 函数 ---")
from samgeo import common
csrc = inspect.getsource(common)
for i, ln in enumerate(csrc.splitlines(), 1):
    low = ln.lower()
    if "hf_hub_download" in low or "def download" in low or "resolve/main" in low:
        emit(f"  L{i}: {ln.strip()}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("written", OUT)
