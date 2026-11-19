# -*- coding: utf-8 -*-
"""OBIA 侧实跑：手写 MRS scale 扫描 + SLIC 对照，全部走统一下游协议。

对每个 scale ∈ {5,10,20,40,80,160}（10 为主参数）：
  1) multiresolution_segment(image, scale, shape=0.2, cpt=0.5) 计时
  2) label 图落盘 results/labelmaps/obia_scale{S}.npy + 分类图 classmaps/
  3) eval_labelmap 统一协议评估 → results/raw/runs.jsonl 追加

另跑 SLIC 对照（n_segments=440，与 obia README 6.3 同配置）同协议评估。
所有数字出自本次运行；runs.jsonl 可溯源。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.common import (OBIA_COMPACTNESS, OBIA_SHAPE, SEED, TOL_PX,  # noqa: E402
                            append_raw, ensure_dirs, eval_labelmap, load_scene,
                            save_classmap, save_labelmap)

import obiabridge  # noqa: E402,F401
from obia.segment import multiresolution_segment  # noqa: E402

SCALES = [5, 10, 20, 40, 80, 160]


def run_one(image, truth, parcels, labels, name, seg_seconds):
    """统一协议评估 + 双落盘（label 图 / 分类图）+ jsonl 追加。"""
    save_labelmap(name, labels)
    rec, obj_map, _ = eval_labelmap(image, truth, parcels, labels, seed=SEED,
                                    tol_px=TOL_PX, name=name,
                                    seg_seconds=seg_seconds, return_maps=True)
    save_classmap(name, obj_map)
    append_raw(rec)
    print(f"[eval] {name}: objects={rec['n_objects']} map_oa={rec['map_oa']:.4f} "
          f"speckle={rec['speckle']:.4f} cls_bF1={rec['cls_boundary_f1']:.4f} "
          f"seg_bF1={rec['seg_boundary_f1']:.4f} seg={seg_seconds:.1f}s", flush=True)
    return rec


def main():
    ensure_dirs()
    image, truth, parcels = load_scene()
    print(f"[obia] scene loaded: {image.shape}, parcels={int(parcels.max())}", flush=True)

    for s in SCALES:
        name = f"obia_scale{s}"
        t0 = time.time()
        labels, info = multiresolution_segment(image, scale=float(s),
                                               shape=OBIA_SHAPE,
                                               compactness=OBIA_COMPACTNESS,
                                               verbose=False)
        seg_seconds = time.time() - t0
        labels = np.asarray(labels).astype(np.int32)
        rec = run_one(image, truth, parcels, labels, name, seg_seconds)
        rec["mrs_n_merges"] = int(info.get("n_merges", -1))
        rec["mrs_internal_seconds"] = float(info.get("seconds", seg_seconds))
        append_raw(rec)   # 补写 mrs 运行信息（覆盖同名记录）
        print(f"[obia] {name} info: merges={rec['mrs_n_merges']} "
              f"internal={rec['mrs_internal_seconds']:.1f}s", flush=True)

    # --- SLIC 对照（与 obia README 6.3 同协议：n_segments≈440, compactness=10）---
    from skimage.segmentation import slic
    name = "slic_440"
    t0 = time.time()
    labels = slic(image, n_segments=440, compactness=10.0, channel_axis=-1,
                  start_label=1, enforce_connectivity=True).astype(np.int32)
    seg_seconds = time.time() - t0
    run_one(image, truth, parcels, labels, name, seg_seconds)

    print("[obia] done", flush=True)


if __name__ == "__main__":
    main()
