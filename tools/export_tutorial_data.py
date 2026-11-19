# -*- coding: utf-8 -*-
"""导出交互教程所需的最小数据集 → tutorial/data.js（<script src> 引入，file:// 可用）。

内容：
* scene_rgb64 : 64² 真彩色缩略图（中心裁剪 256² 后 4×4 众数下采样）的 dataURL
* truth64 / parcels64 : 中心裁剪同区域同法下采样的真值/图斑 label
* labels64 : {obia_scale10, slic_440, sam_pps32, sam_pps32_permissive} 同区域 64² label
* masks_seq : SAM pps32 label 图前 6 个对象的二值 mask 序列（AMG 涂写演示用）
* main / sweeps / prompts / auto : summary.json 的关键数字
下采样用"块内众数"保 label 离散性。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE)))

from scripts.common import LABELMAP_DIR, RESULTS_DIR, load_scene  # noqa: E402
from scripts.make_scene import make_rgb  # noqa: E402

CROP = 256          # 中心裁剪窗
OUT_SIZE = 64       # 输出边长（4×4 众数块）
BLOCK = CROP // OUT_SIZE


def crop(arr):
    h, w = arr.shape[:2]
    r0 = (h - CROP) // 2
    c0 = (w - CROP) // 2
    return arr[r0:r0 + CROP, c0:c0 + CROP]


def block_mode(arr):
    """(CROP,CROP) int label → (OUT,OUT) 块内众数（0 优先级最低）。"""
    out = np.zeros((OUT_SIZE, OUT_SIZE), np.int32)
    a = crop(arr)
    for i in range(OUT_SIZE):
        for j in range(OUT_SIZE):
            blk = a[i * BLOCK:(i + 1) * BLOCK, j * BLOCK:(j + 1) * BLOCK]
            vals, cnts = np.unique(blk, return_counts=True)
            # 同票时偏向非零，保持边界可见
            order = np.argsort(-cnts - (vals == 0) * 0.5)
            out[i, j] = vals[order[0]]
    return out


def block_rgb(rgb):
    a = crop(rgb.astype(np.float32))
    return a.reshape(OUT_SIZE, BLOCK, OUT_SIZE, BLOCK, 3).mean((1, 3)).astype(np.uint8)


def main():
    import base64

    from PIL import Image

    image, truth, parcels = load_scene()
    rgb = make_rgb(image)

    labels = {}
    for name in ("obia_scale10", "slic_440", "sam_pps32", "sam_pps32_permissive"):
        lab = np.load(os.path.join(LABELMAP_DIR, f"{name}.npy"))
        labels[name] = block_mode(lab).tolist()

    # AMG 涂写演示：sam_pps32 label 图（整幅）前 6 个对象 → 整幅下采样 64²
    sam_full = np.load(os.path.join(LABELMAP_DIR, "sam_pps32.npy"))
    ids = [i for i in range(1, int(sam_full.max()) + 1)]
    sizes = [(i, int((sam_full == i).sum())) for i in ids]
    sizes.sort(key=lambda t: -t[1])
    seq_ids = [i for i, _ in sizes[:6]]
    masks_seq = [(sam_full == i).astype(np.uint8) for i in seq_ids]
    painted = np.zeros_like(sam_full, np.int32)
    seq64 = []
    k = 0
    for m in masks_seq:
        free = (m == 1) & (painted == 0)
        if free.any():
            k += 1
            painted[free] = k
            seq64.append(block_mode(painted).tolist())

    # 真彩缩略图 dataURL
    thumb = Image.fromarray(block_rgb(rgb))
    buf = thumb.resize((OUT_SIZE * 4, OUT_SIZE * 4), Image.NEAREST)
    import io
    b = io.BytesIO()
    buf.save(b, format="PNG")
    dataurl = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

    with open(os.path.join(RESULTS_DIR, "summary.json"), encoding="utf-8") as f:
        summ = json.load(f)

    data = {
        "crop": CROP, "out": OUT_SIZE,
        "scene_rgb64": dataurl,
        "truth64": block_mode(truth).tolist(),
        "parcels64": block_mode(parcels).tolist(),
        "labels64": labels,
        "masks_seq": seq64,
        "masks_note": f"sam_pps32 前 {len(seq64)} 个最大对象按 predicted_iou 降序涂写（整幅 512² 下采样 64²）",
        "main": summ["main_comparison"],
        "obia_sweep": [{"scale": r["scale"], "n": r["n_objects"],
                        "seg_bf1": r["seg_boundary_f1"], "oa": r["map_oa"]}
                       for r in summ["obia_scale_sweep"]],
        "sam_sweep": [{"pps": r["pps"], "n": r["n_objects"],
                       "seg_bf1": r["seg_boundary_f1"], "oa": r["map_oa"],
                       "unc": r["uncovered_frac"]}
                      for r in summ["sam_pps_sweep"]],
        "prompts": summ["prompts"]["summary"],
        "auto_containment": {"iou_mean": summ["auto_containment"]["iou_mean"]},
    }

    outdir = os.path.join(os.path.dirname(HERE), "tutorial")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "data.js")
    with open(path, "w", encoding="utf-8") as f:
        f.write("// 自动生成：tools/export_tutorial_data.py（2026-11-20 实跑数据快照，勿手改）\n")
        f.write("const TUTORIAL_DATA = ")
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        f.write(";\n")
    print(f"[export] {path} ({os.path.getsize(path)//1024} KB)")


if __name__ == "__main__":
    main()
