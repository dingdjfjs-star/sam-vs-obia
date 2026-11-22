# -*- coding: utf-8 -*-
"""检查 LoveDA parquet 分片 schema + 解码一个样本看掩码编码。"""
from __future__ import annotations

import io
import sys

import pyarrow.parquet as pq

path = sys.argv[1] if len(sys.argv) > 1 else r"E:\zcode\探索前沿\sam-vs-obia\data\loveda_raw\validation-00000.parquet"
pf = pq.ParquetFile(path)
print("num_rows:", pf.metadata.num_rows, "row_groups:", pf.metadata.num_row_groups)
print("schema:")
print(pf.schema_arrow)

t = next(pf.iter_batches(batch_size=4))
print("\nfirst batch columns:", t.column_names)
for i, name in enumerate(t.column_names):
    col = t.column(name)
    v = col[0].as_py()
    if isinstance(v, dict):
        print(f"  [{name}] dict keys: {list(v.keys())}, bytes len: "
              f"{len(v.get('bytes') or b'')}, path: {v.get('path')}")
    elif isinstance(v, (bytes, bytearray)):
        print(f"  [{name}] bytes len {len(v)}")
    else:
        s = str(v)
        print(f"  [{name}] {type(v).__name__}: {s[:120]}")

# 解码 image + mask（若有）
from PIL import Image
import numpy as np
for name in t.column_names:
    v = t.column(name)[0].as_py()
    if isinstance(v, dict) and v.get("bytes"):
        img = Image.open(io.BytesIO(v["bytes"]))
        arr = np.array(img)
        u = np.unique(arr)
        print(f"  decode [{name}]: mode={img.mode} size={img.size} dtype={arr.dtype} "
              f"uniques(head)={u.tolist()[:25]}{'...' if len(u)>25 else ''} n_uniq={len(u)}")
