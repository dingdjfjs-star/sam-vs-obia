# -*- coding: utf-8 -*-
"""候选数据集仓库的文件树 + 真实文件通道 Range 实测。"""
from __future__ import annotations

import json
import urllib.request

UA = {"User-Agent": "sam-vs-obia-probe"}

REPOS = [
    "chloechia/loveda",
    "aamante/potsdam",
    "JasonGao726/ISPRS_Potsdam_dataset_preprocessed",
    "snchen1230/GID-15",
    "tomas517/GID_benchmark",
]


def get(url, timeout=20):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def tree(repo):
    url = f"https://hf-mirror.com/api/datasets/{repo}/tree/main?recursive=true"
    try:
        items = json.loads(get(url))
        rows = []
        for it in items:
            if it.get("type") == "file":
                size = it.get("size", 0)
                rows.append((it["path"], size))
        rows.sort(key=lambda t: -t[1])
        return rows
    except Exception as e:
        return f"ERR:{type(e).__name__}:{str(e)[:150]}"


def ranged(repo, path, nbytes=131072):
    from urllib.parse import quote
    url = f"https://hf-mirror.com/datasets/{repo}/resolve/main/{quote(path)}"
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-%d" % (nbytes - 1)})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return f"HTTP {r.status}, got {len(r.read())} bytes, final={r.url[:110]}"
    except Exception as e:
        return f"ERR {type(e).__name__}: {str(e)[:150]}"


if __name__ == "__main__":
    for repo in REPOS:
        print(f"\n===== {repo} =====")
        rows = tree(repo)
        if isinstance(rows, str):
            print(rows)
            continue
        total = sum(s for _, s in rows)
        print(f"files={len(rows)} total={total/1e6:.0f}MB")
        for p, s in rows[:12]:
            print(f"  {s/1e6:9.1f}MB  {p}")
        if rows:
            # 实测最大文件（真正的大头）的下载通道
            p, s = rows[0]
            print(f"  [channel test] {p}: {ranged(repo, p)}")
