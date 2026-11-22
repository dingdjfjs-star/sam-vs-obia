# -*- coding: utf-8 -*-
"""v2 数据集探测：三个候选（LoveDA / GID15 / ISPRS Potsdam）在 hf-mirror 与 ModelScope 的可得性。

只做元数据搜索 + 小文件实测（Range GET 前几十 KB），不下载大数据。
用法：E:\\Ana\\envs\\samgeo\\python.exe tools\\probe_datasets.py
"""
from __future__ import annotations

import json
import urllib.request

UA = {"User-Agent": "sam-vs-obia-probe"}


def get(url, timeout=15):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def probe_range(url, want_bytes=65536, timeout=25):
    """实测文件通道：GET 前 want_bytes 字节。返回 (status, got_bytes) 或异常名。"""
    req = urllib.request.Request(url, headers={**UA, "Range": f"bytes=0-{want_bytes-1}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, len(r.read())
    except Exception as e:
        return "ERR", type(e).__name__ + ":" + str(e)[:120]


def hf_search(term):
    url = f"https://hf-mirror.com/api/datasets?search={term}&limit=20&full=false"
    try:
        _, body = get(url)
        items = json.loads(body)
        return [(d["id"], d.get("downloads", 0)) for d in items]
    except Exception as e:
        return f"ERR:{type(e).__name__}:{e}"


def ms_search(term):
    url = f"https://modelscope.cn/api/v1/datasets?PageSize=10&PageNumber=1&Query={term}"
    try:
        _, body = get(url)
        d = json.loads(body)
        rows = (d.get("Data") or {})
        items = rows.get("Datasets") or rows.get("Models") or []
        return [x.get("Namespace", "") + "/" + x.get("Name", "") for x in items][:10]
    except Exception as e:
        return f"ERR:{type(e).__name__}:{e}"


if __name__ == "__main__":
    for term in ("loveda", "LoveDA", "gaofen", "gid15", "GID", "potsdam", "Potsdam", "isprs"):
        print(f"\n=== hf-mirror search: {term} ===")
        print(hf_search(term))
    print("\n=== ModelScope ===")
    for term in ("loveda", "potsdam", "gaofen", "isprs", "segmentation remote"):
        print(f"{term}: {ms_search(term)}")
