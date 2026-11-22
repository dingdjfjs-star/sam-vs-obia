# -*- coding: utf-8 -*-
"""按亮度抽 3 张 test 芯片实测 AMG 行为：判断个别现象 vs 系统性。"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from scripts.common import DATA_DIR  # noqa: E402

TEST = os.path.join(DATA_DIR, "loveda", "test512", "img")
files = sorted(f for f in os.listdir(TEST) if f.endswith(".png"))
brights = []
for f in files:
    a = np.asarray(Image.open(os.path.join(TEST, f)).convert("L"))
    brights.append((float(a.mean()), f))
brights.sort()
picks = [brights[0][1], brights[len(brights) // 2][1], brights[-1][1]]
print("brightness picks:", [(round(b, 1), f) for b, f in
                            (brights[0], brights[len(brights) // 2], brights[-1])])

from scripts.sam_adapter import _build_model
ckpt = os.path.join(DATA_DIR, "weights", "sam2_hiera_small.pt")
model, _ = _build_model(ckpt, "cuda")

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

for f in picks:
    stem = os.path.splitext(f)[0]
    img = np.asarray(Image.open(os.path.join(TEST, f)).convert("RGB"), np.uint8)
    out = []
    for name, cfg in (("def", dict(pred_iou_thresh=0.8, stability_score_thresh=0.95)),
                      ("perm", dict(pred_iou_thresh=0.6, stability_score_thresh=0.85))):
        amg = SAM2AutomaticMaskGenerator(model, points_per_side=32, points_per_batch=64,
                                         box_nms_thresh=0.7, crop_n_layers=0,
                                         min_mask_region_area=0, **cfg)
        masks = amg.generate(img)
        giant = sum(1 for m in masks if m["segmentation"].sum() > 512 * 512 * 0.9)
        # 涂写（剔除 >90% 全图退化 mask 后）
        lab = np.zeros((512, 512), np.int32)
        k = 0
        for m in sorted(masks, key=lambda x: x["predicted_iou"], reverse=True):
            if m["segmentation"].sum() > 512 * 512 * 0.9:
                continue
            free = m["segmentation"] & (lab == 0)
            if free.any():
                k += 1
                lab[free] = k
        out.append(f"{name}: n={len(masks)} giant={giant} painted={k} "
                   f"uncov={(lab==0).mean():.2f}")
    print(f"{stem}: {out}")
