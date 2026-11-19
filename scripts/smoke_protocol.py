# -*- coding: utf-8 -*-
"""协议冒烟：用一幅小 SLIC label 图验证 obiabridge→特征→RF→指标 全链路。

不落盘 jsonl（避免污染正式 runs），只打印记录并断言关键字段。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.common import eval_labelmap, load_scene  # noqa: E402

import obiabridge  # noqa: E402,F401
from obia.metrics import boundary_f, salt_pepper_rate  # noqa: E402


def main():
    image, truth, parcels = load_scene()
    print(f"[smoke] scene {image.shape}, truth classes={np.unique(truth).tolist()}")

    from skimage.segmentation import slic
    labels = slic(image, n_segments=64, compactness=10.0, channel_axis=-1,
                  start_label=1, enforce_connectivity=True).astype(np.int32)
    print(f"[smoke] slic labels: {labels.min()}..{labels.max()}")

    rec, obj_map, lab = eval_labelmap(image, truth, parcels, labels,
                                      name="smoke", seg_seconds=0.0,
                                      return_maps=True)
    for k, v in rec.items():
        print(f"  {k}: {v}")

    # --- 断言 ---
    assert lab.min() == 1, "规整后 label 图必须从 1 开始"
    assert lab.max() == rec["n_objects"], "label 连续 1..K"
    assert set(np.unique(obj_map).tolist()) <= {1, 2, 3, 4, 5}, "分类图类别码 1..5"
    assert 0.0 <= rec["map_oa"] <= 1.0 and 0.0 <= rec["speckle"] <= 1.0
    assert 0.0 <= rec["cls_boundary_f1"] <= 1.0 and 0.0 <= rec["seg_boundary_f1"] <= 1.0

    # 指标函数与 obia 原实现同一来源的旁证：对同一分类图直接重算应完全一致
    sp = salt_pepper_rate(obj_map)
    bf = boundary_f(obj_map, truth, tol=2)["f1"]
    assert abs(sp - rec["speckle"]) < 1e-12, "椒盐率复算一致"
    assert abs(bf - rec["cls_boundary_f1"]) < 1e-12, "边界 F 复算一致"

    print("[smoke] PROTOCOL OK")


if __name__ == "__main__":
    main()
