# -*- coding: utf-8 -*-
"""chloechia/loveda 目录结构细察：按一级目录聚合统计 + 掩码文件实测下载 + 掩码像素值检查。"""
from __future__ import annotations

import collections
import io
import json
import urllib.request

UA = {"User-Agent": "sam-vs-obia-probe"}
REPO = "chloechia/loveda"


def get(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


items = json.loads(get(f"https://hf-mirror.com/api/datasets/{REPO}/tree/main?recursive=true"))
files = [it for it in items if it.get("type") == "file"]

# 一级/二级目录聚合
groups = collections.defaultdict(lambda: [0, 0])
for it in files:
    parts = it["path"].split("/")
    key = "/".join(parts[:2]) if len(parts) > 1 else parts[0]
    groups[key][0] += 1
    groups[key][1] += it.get("size", 0)

print(f"total files {len(files)}, {sum(s for _,s in groups.values())/1e6:.0f}MB")
for k in sorted(groups):
    n, s = groups[k]
    print(f"  {n:5d} files {s/1e6:8.1f}MB  {k}")

# 抽一个 mask 文件实测：下载 → 检查格式/像素值
mask_candidates = [it["path"] for it in files
                   if "masks" in it["path"].lower() or "label" in it["path"].lower() or "mask" in it["path"].lower()]
print(f"\nmask-like paths: {len(mask_candidates)}")
if mask_candidates:
    from urllib.parse import quote
    p = sorted(mask_candidates)[0]
    url = f"https://hf-mirror.com/datasets/{REPO}/resolve/main/{quote(p)}"
    data = get(url)
    print(f"sample mask: {p} ({len(data)/1024:.0f}KB)")
    from PIL import Image
    import numpy as np
    img = Image.open(io.BytesIO(data))
    arr = np.array(img)
    print(f"  mode={img.mode} size={img.size} dtype={arr.dtype} uniques={np.unique(arr).tolist()[:30]}")
    # 同名 images 文件
    img_p = p.replace("masks", "images")
    if img_p != p:
        data2 = get(f"https://hf-mirror.com/datasets/{REPO}/resolve/main/{quote(img_p)}")
        img2 = Image.open(io.BytesIO(data2))
        print(f"sample image: {img_p} mode={img2.mode} size={img2.size}")
