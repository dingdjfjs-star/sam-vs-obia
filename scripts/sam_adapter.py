# -*- coding: utf-8 -*-
"""SAM2 接入层：本项目所有 SAM 相关调用集中在此文件。

为什么直用 sam2 包 API 而不是 samgeo.SamGeo2？
------------------------------------------------
segment-geospatial 1.4.2 的 SamGeo2 内部就是
`SAM2AutomaticMaskGenerator.from_pretrained(model_id)` / `SAM2ImagePredictor
.from_pretrained(model_id)`（见其 samgeo2.py L152/L186），而 from_pretrained
最终走 `sam2.build_sam.build_sam2_hf` → `hf_hub_download` 拉
facebook/sam2-hiera-small 的 sam2_hiera_small.pt。实测（2026-11-20）：
huggingface.co 直连被阻，hf-mirror 的 API 元数据通、但其 Xet 文件代理对
本仓库 502（0 字节挂起）——SAM2 系仓库已迁移 Xet 存储，镜像文件通道失效。
故改为：权重经 **ModelScope 镜像 AI-ModelScope/sam2-hiera-small** 下载
（同文件同字节，sha256 见 README），构建走 sam2 原生 API——与 SamGeo2
内部完全同一类、同一参数，仅绕开坏掉的下载通道。

口径
----
* AMG 单帧耗时 = amg.generate(image) 全程（编码器前向 + 全部点批解码）；
* 提示单次耗时 = predictor.predict(...)（编码器在 set_image 时已完成）；
* 点提示 multimask_output=True 取最高分（单点歧义，SAM 官方建议）；
  框提示 multimask_output=False（无歧义）；
* AMG 参数：points_per_side 扫描，points_per_batch=64，pred_iou_thresh=0.8，
  stability_score_thresh=0.95，box_nms_thresh=0.7，crop_n_layers=0，
  min_mask_region_area=0，multimask_output=True（sam2 默认）。
"""
from __future__ import annotations

import os
import time

import numpy as np

SAM2_CONFIG = "configs/sam2/sam2_hiera_s.yaml"   # sam2 pip 包内置，对应 hiera-small
MODEL_TAG = "SAM2.0-hiera-small(PyPI sam2 / ModelScope 权重)"


def read_rgb(image_rgb_path):
    """3 波段 uint8 GeoTIFF → (H, W, 3) RGB uint8。"""
    import rasterio
    with rasterio.open(image_rgb_path) as src:
        arr = src.read([1, 2, 3])            # (3, H, W)，写入时按 R,G,B 顺序
    return np.transpose(arr, (1, 2, 0)).astype(np.uint8)


def _build_model(ckpt_path, device):
    import torch
    from sam2.build_sam import build_sam2
    assert os.path.exists(ckpt_path), f"SAM2 权重不存在: {ckpt_path}"
    t0 = time.time()
    model = build_sam2(SAM2_CONFIG, ckpt_path=ckpt_path, device=device)
    return model, time.time() - t0


def make_amg_fn(image_rgb_path, ckpt_path, device="cuda",
                pred_iou_thresh=0.8, stability_score_thresh=0.95):
    """工厂：amg_fn(pps) -> (masks, meta)。模型只构建一次，AMG 包装按 pps/阈值重建。

    pred_iou_thresh / stability_score_thresh 可放宽做阈值敏感性对照
    （默认 0.8/0.95 = SAM2 官方默认 AMG 配置）。
    """
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    image = read_rgb(image_rgb_path)
    model, build_seconds = _build_model(ckpt_path, device)

    def amg_fn(pps):
        amg = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=pps,
            points_per_batch=64,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            box_nms_thresh=0.7,
            crop_n_layers=0,
            min_mask_region_area=0,
        )
        t0 = time.time()
        masks = amg.generate(image)          # List[dict]: segmentation/area/predicted_iou/...
        seconds = time.time() - t0
        meta = {"model": MODEL_TAG, "device": device, "points_per_side": pps,
                "pred_iou_thresh": pred_iou_thresh,
                "stability_score_thresh": stability_score_thresh,
                "build_seconds": build_seconds, "generate_seconds": seconds}
        return masks, meta

    return amg_fn


def make_predict_fn(image_rgb_path, ckpt_path, device="cuda"):
    """工厂：predict_fn(kind, x, y, bbox) -> (H, W) bool mask（kind ∈ point/box）。"""
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    image = read_rgb(image_rgb_path)
    model, build_seconds = _build_model(ckpt_path, device)
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(image)               # 编码器一次前向，提示耗时不含此步

    def predict_fn(kind, x, y, bbox):
        if kind == "point":
            masks, scores, _ = predictor.predict(
                point_coords=np.array([[x, y]], np.float64),
                point_labels=np.array([1]),
                box=None, multimask_output=True)
            idx = int(np.argmax(scores))     # 单点歧义：取最高分掩码
            return masks[idx].astype(bool)
        if kind == "box":
            masks, _, _ = predictor.predict(
                point_coords=None, point_labels=None,
                box=np.array(bbox, np.float64), multimask_output=False)
            return masks[0].astype(bool)
        raise ValueError(f"unknown prompt kind: {kind}")

    predict_fn.model_tag = MODEL_TAG
    return predict_fn
