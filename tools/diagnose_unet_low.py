# -*- coding: utf-8 -*-
"""UNet 低尾诊断：最低分芯片的 真值/预测 类分布 + train/val 各 shard 的类分布对比。"""
import json
import sys
from collections import Counter

import numpy as np
from PIL import Image

sys.path.insert(0, r"E:\zcode\探索前沿\sam-vs-obia")
import obiabridge  # noqa: F401
from scripts.common import DATA_DIR  # noqa: E402

base = r"E:\zcode\探索前沿\sam-vs-obia"
CLASS_NAMES = {0: "ignore", 1: "background", 2: "building", 3: "road", 4: "water",
               5: "barren", 6: "forest", 7: "agriculture"}


def hist(arr):
    c = Counter(arr.ravel().tolist())
    tot = sum(c.values())
    return {f"{k}:{CLASS_NAMES.get(k,k)}": round(v/tot, 2) for k, v in c.most_common(4)}


runs = {}
for line in open(base + r"\results\raw\v2_runs.jsonl", encoding="utf-8"):
    if line.strip():
        r = json.loads(line)
        runs[r["name"]] = r

chips = sorted(f[:-4] for f in __import__("os").listdir(
    base + r"\data\loveda\test512\img") if f.endswith(".png"))
scored = sorted((runs[f"v2_unet_{s}"]["map_oa"], s) for s in chips)
print("worst-5 chips (UNet raw OA):")
for oa, s in scored[:5]:
    t = np.asarray(Image.open(base + rf"\data\loveda\test512\mask\{s}.png"), np.int32)
    p = np.load(base + rf"\results\v2\preds\unet\{s}.npy")
    print(f"  {s} OA={oa:.3f}")
    print(f"    truth: {hist(t)}")
    print(f"    pred : {hist(p)}")

print("\nshard-level class distributions (per-shard 8 均值图):")
for split, sub in (("train512(all)", "train512\\mask"), ("val(shard0)", "val\\mask")):
    files = sorted(__import__("os").listdir(base + r"\data\loveda" + "\\" + sub))[:40]
    acc = np.zeros(8, np.int64)
    for f in files:                             # 抽前 40 张
        arr = np.asarray(Image.open(base + r"\data\loveda" + "\\" + sub + "\\" + f), np.int32)
        acc += np.bincount(arr.ravel(), minlength=8)
    tot = acc[1:].sum()
    print(f"  {split}: " + ", ".join(f"{CLASS_NAMES[k]}={acc[k]/tot:.2f}"
                                      for k in range(1, 8) if acc[k] / tot > 0.02))
