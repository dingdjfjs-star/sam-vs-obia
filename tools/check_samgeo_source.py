# -*- coding: utf-8 -*-
"""补充探针：DEFAULT_MODEL_IDS、SamGeo2 权重下载源码、predict 坐标语义。"""
from __future__ import annotations

import inspect
import os

import samgeo
from samgeo import DEFAULT_MODEL_IDS

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "logs", "samgeo_source_probe.txt")
lines = []


def emit(s=""):
    lines.append(str(s))


emit("DEFAULT_MODEL_IDS:")
for k, v in DEFAULT_MODEL_IDS.items():
    emit(f"  {k} -> {v}")

from samgeo import samgeo2
src = inspect.getsource(samgeo2)
emit("\n--- samgeo2.py: download / hf / checkpoint 相关行 ---")
for i, ln in enumerate(src.splitlines(), 1):
    low = ln.lower()
    if any(kw in low for kw in ("hf_hub_download", "snapshot_download", "download_checkpoint",
                                "download_ckpt", "from_url", "url(", "huggingface",
                                "build_sam2", "load_state_dict", "checkpoint")):
        emit(f"  L{i}: {ln.strip()}")

emit("\n--- SamGeo2.__init__ 源码（前 120 行内的模型构建段） ---")
init_src = inspect.getsource(samgeo2.SamGeo2.__init__)
emit(init_src[:4000])

emit("\n--- predict 源码前 80 行（坐标语义） ---")
pred_src = inspect.getsource(samgeo2.SamGeo2.predict)
emit("\n".join(pred_src.splitlines()[:80]))

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("written", OUT)
