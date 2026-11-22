# -*- coding: utf-8 -*-
"""调试：LoveDA 芯片上 SAM2 AMG 到底返回了什么。"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from scripts.common import DATA_DIR  # noqa: E402

chip_stem = sys.argv[1] if len(sys.argv) > 1 else "test_2541_00"
img_path = os.path.join(DATA_DIR, "loveda", "test512", "img", chip_stem + ".png")

img = np.asarray(Image.open(img_path).convert("RGB"), np.uint8)
print("chip:", img.shape, img.dtype, "range", img.min(), img.max(),
      "mean", img.mean(axis=(0, 1)).round(1))

from scripts.sam_adapter import _build_model
ckpt = os.path.join(DATA_DIR, "weights", "sam2_hiera_small.pt")
model, _ = _build_model(ckpt, "cuda")

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

for name, cfg in (("default", dict(pred_iou_thresh=0.8, stability_score_thresh=0.95)),
                  ("permissive", dict(pred_iou_thresh=0.6, stability_score_thresh=0.85))):
    amg = SAM2AutomaticMaskGenerator(model, points_per_side=32, points_per_batch=64,
                                     box_nms_thresh=0.7, crop_n_layers=0,
                                     min_mask_region_area=0, **cfg)
    masks = amg.generate(img)
    print(f"\n[{name}] n_masks={len(masks)}")
    if masks:
        ious = [m["predicted_iou"] for m in masks]
        areas = [int(m["segmentation"].sum()) for m in masks]
        order = np.argsort(ious)[::-1]
        print("  iou range:", round(min(ious), 3), "..", round(max(ious), 3))
        print("  top-5 by iou:")
        for k in order[:5]:
            m = masks[k]
            frac = areas[k] / (512 * 512)
            print(f"    iou={m['predicted_iou']:.3f} stab={m.get('stability_score', 0):.3f} "
                  f"area={areas[k]} ({frac*100:.1f}%) bbox={m.get('bbox')}")
        big = sum(1 for a in areas if a > 512 * 512 * 0.9)
        print(f"  masks covering >90% of chip: {big}")
        # 涂写模拟
        lab = np.zeros((512, 512), np.int32)
        k = 0
        for m in sorted(masks, key=lambda x: x["predicted_iou"], reverse=True):
            free = m["segmentation"] & (lab == 0)
            if free.any():
                k += 1
                lab[free] = k
        print(f"  painted objects: {k}, uncovered: {(lab==0).mean():.3f}")
