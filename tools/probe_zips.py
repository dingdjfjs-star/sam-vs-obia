# -*- coding: utf-8 -*-
"""zip 通道实测 + mrmeepsle data/ 文件清单。"""
from __future__ import annotations

import json
import urllib.request

UA = {"User-Agent": "sam-vs-obia-probe"}


def head_resolve(repo, path):
    url = f"https://hf-mirror.com/datasets/{repo}/resolve/main/{urllib.request.quote(path)}"
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-131071"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            cr = r.headers.get("Content-Range", "?")
            cl = r.headers.get("Content-Length", "?")
            data = r.read()
            return f"HTTP {r.status} Content-Range={cr[:60]} got={len(data)}B head={data[:4].hex()}"
    except Exception as e:
        return f"ERR {type(e).__name__}: {str(e)[:140]}"


print("Xiazhiqi2002/loveda_train.zip:")
print("  ", head_resolve("Xiazhiqi2002/loveda_train", "loveda_train.zip"))
print("licheng03/loveda Val.zip:")
print("  ", head_resolve("licheng03/loveda", "Val.zip"))

# mrmeepsle data/ 清单
items = json.loads(urllib.request.urlopen(urllib.request.Request(
    "https://hf-mirror.com/api/datasets/mrmeepsle/LoveDa/tree/main/data?recursive=false",
    headers=UA), timeout=20).read())
print("\nmrmeepsle/LoveDa data/:")
for it in items:
    print(f"  {it.get('size',0)/1e6:9.1f}MB  {it['path']}")

# Xiazhiqi2002/LoveDA_union 也看一眼
items2 = json.loads(urllib.request.urlopen(urllib.request.Request(
    "https://hf-mirror.com/api/datasets/Xiazhiqi2002/LoveDA_union/tree/main?recursive=true",
    headers=UA), timeout=20).read())
print("\nXiazhiqi2002/LoveDA_union:")
for it in items2:
    if it.get("type") == "file":
        print(f"  {it.get('size',0)/1e6:9.1f}MB  {it['path']}")
