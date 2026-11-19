# -*- coding: utf-8 -*-
"""samgeo 安装后 API 探针：版本/类/关键方法签名/权重下载方式，写入 results/logs/samgeo_api.txt。

只做 introspection，不加载模型、不联网。
"""
from __future__ import annotations

import inspect
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "logs", "samgeo_api.txt")

lines = []


def emit(s=""):
    lines.append(str(s))


emit(f"python {sys.version}")
try:
    import samgeo
    emit(f"samgeo version: {getattr(samgeo, '__version__', 'N/A')}")
    emit(f"samgeo file: {samgeo.__file__}")
    emit(f"samgeo dir: {[n for n in dir(samgeo) if not n.startswith('_')]}")
except Exception as e:
    emit(f"IMPORT samgeo FAILED: {type(e).__name__}: {e}")
    raise SystemExit("\n".join(lines))

for cls_name in ("SamGeo2", "SamGeo"):
    cls = getattr(samgeo, cls_name, None)
    if cls is None:
        continue
    emit(f"\n=== class {cls_name} ===")
    try:
        emit(f"__init__{inspect.signature(cls.__init__)}")
    except Exception as e:
        emit(f"__init__ signature unavailable: {e}")
    for m in ("generate", "predict", "setImage", "set_image", "save_masks",
              "automatic_generate", "download_model_weights"):
        f = getattr(cls, m, None)
        if f is not None and callable(f):
            try:
                emit(f"{m}{inspect.signature(f)}")
            except Exception:
                emit(f"{m}: signature unavailable")
            doc = (inspect.getdoc(f) or "").strip().split("\n")
            if doc:
                emit(f"    doc: {doc[0]}")

# 源码级确认权重下载通道（HF or github release）
src_file = inspect.getsourcefile(samgeo)
emit(f"\nsource file: {src_file}")
try:
    import samgeo.samgeo2 as s2
    src = inspect.getsource(s2)
    for kw in ("hf_hub_download", "snapshot_download", "huggingface", "HUGGINGFACE",
               "github", "release", "torch.hub", "URL"):
        hits = [ln.strip() for ln in src.splitlines() if kw.lower() in ln.lower()][:6]
        if hits:
            emit(f"[samgeo2 source hits: {kw}]")
            for h in hits:
                emit(f"    {h}")
except Exception as e:
    emit(f"samgeo2 submodule introspect failed: {e}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
print(f"\n[probe] written {OUT}")
