# -*- coding: utf-8 -*-
"""LoveDA 候选仓紧凑探针：一级目录聚合（修正分组）+ 掩码存在性 + id2label。"""
from __future__ import annotations

import collections
import json
import urllib.request

UA = {"User-Agent": "sam-vs-obia-probe"}
REPOS = [
    "chloechia/loveda",
    "Xiazhiqi2002/loveda_train",
    "licheng03/loveda",
    "ahsennazir/loveDA",
    "snchen1230/LoveDA",
    "mrmeepsle/LoveDa",
]


def get(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


for repo in REPOS:
    try:
        items = json.loads(get(f"https://hf-mirror.com/api/datasets/{repo}/tree/main?recursive=true"))
        files = [it for it in items if it.get("type") == "file"]
        groups = collections.defaultdict(lambda: [0, 0, 0])  # n, size, n_mask
        for it in files:
            top = it["path"].split("/")[0]
            g = groups[top]
            g[0] += 1
            g[1] += it.get("size", 0)
            low = it["path"].lower()
            if "mask" in low or "label" in low or "ann" in low or "gt" in low:
                g[2] += 1
        print(f"\n=== {repo}: {len(files)} files ===")
        for k in sorted(groups):
            n, s, nm = groups[k]
            print(f"  {n:5d} files {s/1e6:8.1f}MB masks~{nm:4d}  {k}")
        if len(files) <= 3:
            for it in files:
                print(f"    {it['path']}")
    except Exception as e:
        print(f"\n=== {repo}: ERR {type(e).__name__}: {str(e)[:120]}")

# id2label.json 内容
try:
    d = json.loads(get("https://hf-mirror.com/api/datasets/chloechia/loveda/resolve/main/id2label.json"))
    print("\nid2label(chloechia):", d)
except Exception as e:
    print("id2label fetch err:", e)
