# -*- coding: utf-8 -*-
"""外科手术式重评 U-Net：修复"argmax 含 0 号 ignore 通道"的解码 bug，只重算 UNet 记录。

OBIA / SAM2 的 label 图与下游记录完全不动；UNet 权重不重训（同一 checkpoint，
仅解码改为 LoveDA 官方协议：logits[1:,:,:].argmax+1，0 号通道不参与）。
顺带统计 channel-0 曾赢得的像素占比（README 引用的诊断数字来源）。

写回：results/v2/preds/unet|unet_cmap/*.npy + results/raw/v2_runs.jsonl（替换 unet 记录）
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from scripts.common import (DATA_DIR, RESULTS_DIR, SEED, TOL_PX,  # noqa: E402
                            fill_uncovered)

import obiabridge  # noqa: E402,F401

DATA_LOVEDA = os.path.join(DATA_DIR, "loveda")
TEST_DIR = os.path.join(DATA_LOVEDA, "test512")
V2_DIR = os.path.join(RESULTS_DIR, "v2")
RUNS = os.path.join(RESULTS_DIR, "raw", "v2_runs.jsonl")


def main():
    import torch
    from scripts.run_v2_eval import (chip_list, eval_with_fallback,
                                     global_prior_class, load_chip, save_pred)
    from scripts.unet import UNet

    ckpt_path = os.path.join(V2_DIR, "unet_best.pt")
    ck = torch.load(ckpt_path, map_location="cuda", weights_only=True)
    unet = UNet(in_ch=3, n_classes=ck["n_classes"]).cuda().eval()
    unet.load_state_dict(ck["state_dict"])

    prior_c, _ = global_prior_class()
    chips = chip_list()
    print(f"[unet-fix] {len(chips)} chips, decode = argmax over channels 1..7", flush=True)

    # 逐芯片重算
    new_recs = {}
    zero_won_total = 0
    for i, (stem, ipath, mpath) in enumerate(chips, 1):
        img, truth = load_chip(ipath, mpath)
        t0 = time.time()
        with torch.no_grad():
            x = torch.from_numpy(img.transpose(2, 0, 1)[None]).cuda()
            logits = unet(x)[0]
            ch0 = int((logits.argmax(0) == 0).sum())     # 诊断：0 号通道曾赢的像素
            pred = logits[1:, :, :].argmax(0).cpu().numpy().astype(np.int32) + 1
        dt = time.time() - t0
        zero_won_total += ch0

        d = os.path.join(V2_DIR, "preds", "unet")
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, f"{stem}.npy"), pred.astype(np.int32))

        # 主指标 = 模型本体输出（raw argmax 直接对真值）——"训练类方案"的本来面目
        from obia.metrics import boundary_f, confusion_oa, salt_pepper_rate
        cmap = pred.astype(np.int32)
        _, oa_raw = confusion_oa(cmap, truth, 7)
        cls_bf = boundary_f(cmap, truth, tol=TOL_PX)
        # 参考：沿用 v1 下游协议（UNet 类别图当 label 图再过特征+RF）的精修数字
        rec_rf, cmap_rf, _ = eval_with_fallback(img, truth, pred, SEED, TOL_PX,
                                                f"v2_unet_{stem}", dt, 7, prior_c)
        rec = {
            "name": f"v2_unet_{stem}",
            "n_objects": int(len(np.unique(cmap))),
            "uncovered_frac": 0.0,
            "map_oa": float(oa_raw),
            "obj_oa": float(oa_raw),
            "speckle": salt_pepper_rate(cmap),
            "cls_boundary_p": cls_bf["precision"], "cls_boundary_r": cls_bf["recall"],
            "cls_boundary_f1": cls_bf["f1"],
            "seg_boundary_p": cls_bf["precision"], "seg_boundary_r": cls_bf["recall"],
            "seg_boundary_f1": cls_bf["f1"],
            "classify_seconds": 0.0, "seg_seconds": float(dt),
            "degenerate": False,
            "unet_pred_is_cmap": True,
            "decode": "argmax over channels 1..7 (ignore-channel excluded)",
            "口径": "主指标=模型本体输出；refined_*=沿用 v1 下游协议(特征+RF)的参考值",
            "refined_map_oa": rec_rf["map_oa"],
            "refined_cls_boundary_f1": rec_rf["cls_boundary_f1"],
        }
        dc = os.path.join(V2_DIR, "preds", "unet_cmap")
        os.makedirs(dc, exist_ok=True)
        np.save(os.path.join(dc, f"{stem}.npy"), cmap.astype(np.int32))
        new_recs[f"v2_unet_{stem}"] = rec
        if i % 16 == 0 or i == len(chips):
            print(f"  [{i}/{len(chips)}] latest OA={rec['map_oa']:.3f}", flush=True)

    # 重写 jsonl：替换 unet 记录，其余原样
    out_lines = []
    n_unet = 0
    for line in open(RUNS, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["name"].startswith("v2_unet_"):
            out_lines.append(json.dumps(new_recs[r["name"]], ensure_ascii=False) + "\n")
            n_unet += 1
        else:
            out_lines.append(line)
    with open(RUNS, "w", encoding="utf-8") as f:
        f.writelines(out_lines)
    print(f"[unet-fix] replaced {n_unet} unet records; "
          f"channel-0 won {zero_won_total} px total "
          f"({zero_won_total/ (len(chips)*512*512) * 100:.2f}% of all pixels)", flush=True)


if __name__ == "__main__":
    main()
