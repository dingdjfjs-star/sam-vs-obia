# -*- coding: utf-8 -*-
"""SAM2 整图自动分割（AMG）与尺度扫描：points_per_side ∈ {8,16,32,48,64}。

对每档 pps：
  1) SAM2 AutomaticMaskGenerator 整图推理计时（含编码器一次前向）
  2) mask 按分数降序涂写为非重叠 label 图（0=未覆盖，随后统一协议规整）
  3) label 图落盘 labelmaps/sam_pps{P}.npy
  4) eval_labelmap 统一协议评估 → results/raw/runs.jsonl

注意：AMG 原始 mask 数（n_masks_raw）与涂写后对象数（n_objects）都记录，
后者才是与 OBIA 可比的口径（非重叠地面实况对象）。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.common import (RESULTS_DIR, SEED, TOL_PX, append_raw,  # noqa: E402
                            ensure_dirs, eval_labelmap, load_scene,
                            save_classmap, save_labelmap)

PPS_SWEEP = [8, 16, 32, 48, 64]      # 32 为 SAM2 默认 AMG 密度


def masks_to_labelmap(masks):
    """AMG mask 列表 → 非重叠 label 图（predicted_iou 降序先到先得，0=未覆盖）。

    masks: list[dict]（含 'segmentation' bool (H,W) 与分数，AMG 已按分数排序）
    """
    h, w = np.asarray(masks[0]["segmentation"]).shape
    lab = np.zeros((h, w), np.int32)
    n_raw = len(masks)
    k = 0
    for m in sorted(masks, key=lambda x: x.get("predicted_iou", 0.0), reverse=True):
        seg = np.asarray(m["segmentation"], bool)
        free = seg & (lab == 0)
        if free.any():
            k += 1
            lab[free] = k
    return lab, {"n_masks_raw": n_raw, "n_painted": k}


def main():
    ensure_dirs()
    image, truth, parcels = load_scene()
    from scripts.sam_adapter import make_amg_fn
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    amg_fn = make_amg_fn(image_rgb_path=os.path.join(data_dir, "scene_rgb.tif"),
                         ckpt_path=os.path.join(data_dir, "weights", "sam2_hiera_small.pt"))

    for pps in PPS_SWEEP:
        name = f"sam_pps{pps}"
        t0 = time.time()
        masks, meta = amg_fn(pps)          # 返回 AMG mask 列表 + 模型元信息
        amg_seconds = time.time() - t0
        lab, paint = masks_to_labelmap(masks)
        print(f"[sam] {name}: raw_masks={paint['n_masks_raw']} "
              f"painted_objects={paint['n_painted']} "
              f"uncovered={(lab == 0).mean():.4f} {amg_seconds:.1f}s", flush=True)

        save_labelmap(name, lab)
        rec, obj_map, _ = eval_labelmap(image, truth, parcels, lab, seed=SEED,
                                        tol_px=TOL_PX, name=name,
                                        seg_seconds=amg_seconds, return_maps=True)
        rec.update(paint)
        rec["pps"] = pps
        rec["model"] = meta.get("model", "sam2")
        save_classmap(name, obj_map)
        append_raw(rec)
        print(f"[sam] {name}: objects={rec['n_objects']} map_oa={rec['map_oa']:.4f} "
              f"speckle={rec['speckle']:.4f} cls_bF1={rec['cls_boundary_f1']:.4f} "
              f"seg_bF1={rec['seg_boundary_f1']:.4f}", flush=True)

    print("[sam] auto sweep done", flush=True)

    # --- 阈值敏感性对照：放宽 pred_iou/stability（其余同 pps32 默认档）---
    amg_fn2 = make_amg_fn(image_rgb_path=os.path.join(data_dir, "scene_rgb.tif"),
                          ckpt_path=os.path.join(data_dir, "weights", "sam2_hiera_small.pt"),
                          pred_iou_thresh=0.6, stability_score_thresh=0.85)
    name = "sam_pps32_permissive"
    t0 = time.time()
    masks, meta = amg_fn2(32)
    amg_seconds = time.time() - t0
    lab, paint = masks_to_labelmap(masks)
    print(f"[sam] {name}: raw_masks={paint['n_masks_raw']} "
          f"painted={paint['n_painted']} uncovered={(lab == 0).mean():.4f} "
          f"{amg_seconds:.1f}s", flush=True)
    save_labelmap(name, lab)
    rec, obj_map, _ = eval_labelmap(image, truth, parcels, lab, seed=SEED,
                                    tol_px=TOL_PX, name=name,
                                    seg_seconds=amg_seconds, return_maps=True)
    rec.update(paint)
    rec["pps"] = 32
    rec["model"] = meta.get("model", "sam2")
    rec["note"] = "pred_iou=0.6 stability=0.85 (permissive)"
    save_classmap(name, obj_map)
    append_raw(rec)
    print(f"[sam] {name}: objects={rec['n_objects']} map_oa={rec['map_oa']:.4f} "
          f"speckle={rec['speckle']:.4f} cls_bF1={rec['cls_boundary_f1']:.4f} "
          f"seg_bF1={rec['seg_boundary_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
