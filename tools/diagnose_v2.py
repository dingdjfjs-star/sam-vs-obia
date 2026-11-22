# -*- coding: utf-8 -*-
"""v2 数字体检：truth 的 0(ignore) 分布、单类芯片占比、逐芯片指标分布、OBIA 耗时。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from scripts.common import DATA_DIR, RESULTS_DIR  # noqa: E402

TEST = os.path.join(DATA_DIR, "loveda", "test512")

stats = []
for f in sorted(os.listdir(os.path.join(TEST, "mask"))):
    if not f.endswith(".png"):
        continue
    t = np.asarray(Image.open(os.path.join(TEST, "mask", f)), np.int32)
    cls = set(np.unique(t).tolist()) - {0}
    stats.append({"stem": os.path.splitext(f)[0], "zero_frac": float((t == 0).mean()),
                  "n_classes": len(cls), "classes": sorted(cls)})

zf = np.array([s["zero_frac"] for s in stats])
print(f"chips: {len(stats)}")
print(f"zero(ignore) fraction: mean={zf.mean():.3f} median={np.median(zf):.3f} "
      f"max={zf.max():.3f} chips>10% zero: {(zf>0.1).sum()}")
n1 = sum(1 for s in stats if s["n_classes"] < 2)
print(f"single-class chips: {n1}")
from collections import Counter
print("class-count histogram:", dict(Counter(s["n_classes"] for s in stats)))

# 逐芯片 UNet OA 分布
M = json.load(open(os.path.join(RESULTS_DIR, "metrics.json"), encoding="utf-8"))
runs = {}
for line in open(os.path.join(RESULTS_DIR, "raw", "v2_runs.jsonl"), encoding="utf-8"):
    if line.strip():
        r = json.loads(line)
        runs[r["name"]] = r
for m in ("obia", "sam_default", "unet"):
    oas = sorted(runs[f"v2_{m}_{s['stem']}"]["map_oa"] for s in stats)
    q = np.percentile(oas, [0, 25, 50, 75, 100])
    print(f"{m:12s} OA p0/25/50/75/100 = {np.round(q,3).tolist()}")

# OBIA 耗时与对象数
seg = [runs[f"v2_obia_{s['stem']}"]["seg_seconds"] for s in stats]
nobj = [runs[f"v2_obia_{s['stem']}"]["n_objects"] for s in stats]
print(f"OBIA seg seconds: mean={np.mean(seg):.0f} p50={np.median(seg):.0f} max={max(seg):.0f}")
print(f"OBIA n_objects: mean={np.mean(nobj):.0f} p50={np.median(nobj):.0f} min={min(nobj)}")

# 0 值对 bF1 的影响实验：单张芯片，把 truth 的 0 换成其邻域多数类后重算 UNet bF1
from obia.metrics import boundary_f
worst = max(stats, key=lambda s: s["zero_frac"])["stem"]
print(f"\nworst-zero chip: {worst} zero={max(s['zero_frac'] for s in stats):.2f}")
truth = np.asarray(Image.open(os.path.join(TEST, "mask", worst + ".png")), np.int32)
pred = np.load(os.path.join(RESULTS_DIR, "v2", "preds", "unet", worst + ".npy"))
print("  as-is bF1:", round(boundary_f(pred, truth, 2)["f1"], 4))
# 0 → 最近非零填充（诊断用，不改协议）
from scipy import ndimage
ind = ndimage.distance_transform_edt(truth == 0, return_distances=False, return_indices=True)
truth_filled = truth[tuple(ind)]
print("  zero-filled truth bF1:", round(boundary_f(pred, truth_filled, 2)["f1"], 4),
      "| zero frac was", round(float((truth == 0).mean()), 3))
