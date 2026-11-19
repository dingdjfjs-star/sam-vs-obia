# -*- coding: utf-8 -*-
"""生成合成场景并落盘：scene.npz（4 波段 float + 真值 + 图斑）+ scene_rgb.tif。

场景由 obia.synthetic.generate_scene(seed=20260829) 生成——与
obia-segmentation 完全同一份实现、同一个种子，保证两个项目看到的是
同一幅影像。scene_rgb.tif 是给 SAM 的"自然影像"输入（真彩色 8bit，
2% 线性拉伸），SAM 零样本时只能看到 RGB 三个波段（无 NIR）——这正是
零样本迁移的真实约束，写入报告讨论。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.common import DATA_DIR, PROJECT_ROOT, SEED, SCENE_SIZE, ensure_dirs  # noqa: E402

import obiabridge  # noqa: E402,F401
from obia.synthetic import generate_scene  # noqa: E402


def make_rgb(image, stretch_pct=2.0):
    """4 波段 [B,G,R,NIR] float → 真彩色 8bit RGB（R,G,B 通道，2% 线性拉伸）。"""
    rgb = image[:, :, [2, 1, 0]].astype(np.float32)      # R, G, B
    lo = np.percentile(rgb, stretch_pct, axis=(0, 1))
    hi = np.percentile(rgb, 100.0 - stretch_pct, axis=(0, 1))
    rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-9), 0.0, 1.0)
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def main():
    ensure_dirs()
    print(f"[scene] generating {SCENE_SIZE}x{SCENE_SIZE} seed={SEED} "
          f"(via obia.synthetic, same impl as obia-segmentation) ...", flush=True)
    image, truth, parcels = generate_scene(SCENE_SIZE, SCENE_SIZE, seed=SEED)
    print(f"[scene] image {image.shape} {image.dtype} "
          f"range [{image.min():.3f}, {image.max():.3f}]", flush=True)
    print(f"[scene] truth classes {np.unique(truth).tolist()}, "
          f"parcels {int(parcels.max())}", flush=True)

    npz = os.path.join(DATA_DIR, "scene.npz")
    np.savez_compressed(npz, image=image, truth=truth, parcels=parcels)
    print(f"[scene] saved {npz}", flush=True)

    # SAM 输入：真彩色 GeoTIFF（地理语义保留，segment-geospatial 可直接读）
    import rasterio
    from obia.vectorize import SCENE_TRANSFORM, SCENE_CRS
    rgb = make_rgb(image)
    tif = os.path.join(DATA_DIR, "scene_rgb.tif")
    with rasterio.open(
        tif, "w", driver="GTiff", height=rgb.shape[0], width=rgb.shape[1],
        count=3, dtype="uint8", transform=SCENE_TRANSFORM, crs=SCENE_CRS,
        compress="deflate",
    ) as dst:
        dst.write(rgb[:, :, 0], 1)
        dst.write(rgb[:, :, 1], 2)
        dst.write(rgb[:, :, 2], 3)
    print(f"[scene] saved {tif} (true-color uint8, 2% stretch)", flush=True)


if __name__ == "__main__":
    main()
