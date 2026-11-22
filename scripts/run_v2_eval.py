# -*- coding: utf-8 -*-
"""v2 三方案受控评估：OBIA / SAM2 零样本 / U-Net（训练）× 同一 64 个 512² test 芯片。

口径沿用 v1：eval_labelmap 统一下游协议（19 维特征→同种子 RF→回填→三指标）；
真实数据无图斑真值，seg 边界口径从 v1 的"对图斑"退化为"label 图边界 vs 地类真值
边界"，README 明示。四指标：OA / 椒盐率 / cls_bF1 / 速度。

OBIA 尺度标定：v1 结论"scale 安全上限与数据耦合，换数据须重新标定"——先在一张
【非 test】validation 图上扫 scale∈{5,10,20,40,80}，按"对象数 200~800 带内取最小
scale"规则选主 scale（规则先于数据确定，标定图不入 test），全 test 沿用。

三段式（OBIA 单芯片 CPU ~60-75s，64 片拆两进程并行）：
  --calibrate              尺度标定 → results/v2/obia_calibration.json
  --shard k --nshards n    逐芯片三方案 → results/raw/v2_runs.jsonl + results/v2/preds/
  --aggregate              汇总 → results/metrics.json（数字单一来源）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.common import (DATA_DIR, LABELMAP_DIR, OBIA_COMPACTNESS, OBIA_SHAPE,  # noqa: E402
                            RESULTS_DIR, SEED, TOL_PX, append_raw, eval_labelmap,
                            fill_uncovered, load_runs)

import obiabridge  # noqa: E402,F401
from obia.segment import multiresolution_segment  # noqa: E402

DATA_LOVEDA = os.path.join(DATA_DIR, "loveda")
TEST_DIR = os.path.join(DATA_LOVEDA, "test512")
VAL_DIR = os.path.join(DATA_LOVEDA, "val")
V2_DIR = os.path.join(RESULTS_DIR, "v2")
PREDS_DIR = os.path.join(V2_DIR, "preds")
CAL_SCALE_RULE = (200, 800)      # 对象数带（规则先于数据确定）
CAL_SCALES = [5, 10, 20, 40, 80]
GIANT_FRAC = 0.90                # >90% 芯片面积的 AMG mask 视为全图退化提案，剔除
N_PRIOR_MASKS = 200              # 全局类先验的抽样 mask 数（train 掩码按名排序取前 N）
CLASS_NAMES = {1: "background", 2: "building", 3: "road", 4: "water",
               5: "barren", 6: "forest", 7: "agriculture"}


def prior_class_name(c):
    return f"{c}:{CLASS_NAMES.get(c, '?')}"


def stretch_u8(rgb_u8):
    """逐波段 2% 线性拉伸（与 v1 make_rgb 的 SAM 输入口径一致）。"""
    a = rgb_u8.astype(np.float32)
    out = np.empty_like(a)
    for b in range(3):
        lo, hi = np.percentile(a[:, :, b], [2.0, 98.0])
        out[:, :, b] = np.clip((a[:, :, b] - lo) / max(hi - lo, 1e-9), 0, 1) * 255.0
    return (out + 0.5).astype(np.uint8)


def global_prior_class():
    """训练集全局类先验（前 N 张 train 掩码的像素直方图）→ 多数类。

    用于 SAM/OBIA 在某芯片退化（K<2 对象）时的"无信号常数预测器"——
    只用 train split，不碰 test；三方同规则，逐芯片记 degenerate 标记。
    """
    mask_dir = os.path.join(DATA_LOVEDA, "train512", "mask")
    files = sorted(f for f in os.listdir(mask_dir) if f.endswith(".png"))[:N_PRIOR_MASKS]
    hist = np.zeros(8, np.int64)
    for f in files:
        arr = np.asarray(Image.open(os.path.join(mask_dir, f)), np.int32)
        hist += np.bincount(arr.ravel(), minlength=8)
    c = int(np.argmax(hist[1:]) + 1)          # ignore(0) 不参与
    frac = float(hist[c] / max(hist[1:].sum(), 1))
    return c, frac


def eval_with_fallback(img, truth, lab, seed, tol_px, name, seg_seconds, n_classes, prior_c):
    """eval_labelmap + K<2 退化兜底：单对象时用全局先验常数图（诚实记 degenerate）。"""
    lab_filled = fill_uncovered(lab)
    k = int(lab_filled.max())
    if k >= 2:
        rec, cmap, _ = eval_labelmap(img, truth, truth, lab_filled, seed=seed,
                                     tol_px=tol_px, name=name, seg_seconds=seg_seconds,
                                     n_classes=n_classes, return_maps=True)
        rec["degenerate"] = False
        return rec, cmap, lab_filled
    # ---- 退化路径：无可用分割 → 常数预测器（全局先验类）----
    from obia.metrics import boundary_f, confusion_oa, salt_pepper_rate
    cmap = np.full(truth.shape, prior_c, np.int32)
    _, oa = confusion_oa(cmap, truth, n_classes)
    cls_bf = boundary_f(cmap, truth, tol=tol_px)
    seg_bf = boundary_f(lab_filled, truth, tol=tol_px)
    rec = {
        "name": name, "n_objects": max(k, 0),
        "uncovered_frac": float((np.asarray(lab) == 0).mean()),
        "map_oa": float(oa), "obj_oa": float(oa),
        "speckle": salt_pepper_rate(cmap),
        "cls_boundary_p": cls_bf["precision"], "cls_boundary_r": cls_bf["recall"],
        "cls_boundary_f1": cls_bf["f1"],
        "seg_boundary_p": seg_bf["precision"], "seg_boundary_r": seg_bf["recall"],
        "seg_boundary_f1": seg_bf["f1"],
        "classify_seconds": 0.0, "seg_seconds": float(seg_seconds),
        "degenerate": True,
    }
    return rec, cmap, lab_filled


def chip_list():
    imgs = sorted(os.listdir(os.path.join(TEST_DIR, "img")))
    out = []
    for f in imgs:
        stem = os.path.splitext(f)[0]
        mp = os.path.join(TEST_DIR, "mask", f)
        if os.path.exists(mp):
            out.append((stem, os.path.join(TEST_DIR, "img", f), mp))
    return out


def load_chip(img_path, mask_path):
    img = np.asarray(Image.open(img_path).convert("RGB"), np.float32) / 255.0
    truth = np.asarray(Image.open(mask_path), np.int32)
    return img, truth


def save_pred(method, stem, cmap):
    d = os.path.join(PREDS_DIR, method)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, f"{stem}.npy"), cmap.astype(np.int32))


def save_label(method, stem, lab):
    d = os.path.join(LABELMAP_DIR, "v2_" + method)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, f"{stem}.npy"), np.asarray(lab).astype(np.int32))


def calibrate_obia_scale(out_path=None):
    """在一张非 test 的 validation 图上标定主 scale（规则：带内对象数取最小 scale）。"""
    val_imgs = sorted(os.listdir(os.path.join(VAL_DIR, "img")))
    manifest = json.load(open(os.path.join(DATA_DIR, "loveda_split_manifest.json"),
                              encoding="utf-8"))
    test_names = set(manifest["test_img_names"])
    cal_name = next(f for f in val_imgs
                    if os.path.splitext(f)[0] not in test_names)   # 第一张非 test 图
    img, truth = load_chip(os.path.join(VAL_DIR, "img", cal_name),
                           os.path.join(VAL_DIR, "mask", cal_name))
    img = img[:512, :512]
    truth = truth[:512, :512]
    recs = []
    chosen = None
    for s in CAL_SCALES:
        t0 = time.time()
        labels, info = multiresolution_segment(img, scale=float(s), shape=OBIA_SHAPE,
                                               compactness=OBIA_COMPACTNESS, verbose=False)
        n_obj = int(np.asarray(labels).max())
        recs.append({"scale": s, "n_objects": n_obj, "seconds": time.time() - t0})
        print(f"[calib] scale={s}: objects={n_obj} ({recs[-1]['seconds']:.0f}s)", flush=True)
        if CAL_SCALE_RULE[0] <= n_obj <= CAL_SCALE_RULE[1] and chosen is None:
            chosen = s
    if chosen is None:
        chosen = min(CAL_SCALES, key=lambda s: abs(
            next(r for r in recs if r["scale"] == s)["n_objects"]
            - (CAL_SCALE_RULE[0] + CAL_SCALE_RULE[1]) / 2))
        print(f"[calib] 无带内档，退化为最接近带中心的 scale={chosen}", flush=True)
    print(f"[calib] calibration image = {cal_name}（非 test）→ 主 scale = {chosen}", flush=True)
    cal = {"main_scale": chosen, "calibration_image": cal_name,
           "calibration_rule": f"scale ∈ {CAL_SCALES} 扫描，对象数 "
                               f"{CAL_SCALE_RULE[0]}~{CAL_SCALE_RULE[1]} 带内取最小",
           "records": recs}
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cal, f, ensure_ascii=False, indent=1)
        print(f"[calib] wrote {out_path}", flush=True)
    return cal


def process_chips(chips, obia_scale, shard_tag=""):
    """逐芯片三方案评估（分片可并行）；结果逐条写 v2_runs.jsonl 与 preds/。"""
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    from scripts.run_sam_auto import masks_to_labelmap
    from scripts.sam_adapter import _build_model
    ckpt = os.path.join(DATA_DIR, "weights", "sam2_hiera_small.pt")
    sam_model, _ = _build_model(ckpt, "cuda")
    AMG_CFG = {"default": dict(pred_iou_thresh=0.8, stability_score_thresh=0.95),
               "permissive": dict(pred_iou_thresh=0.6, stability_score_thresh=0.85)}

    import torch
    from scripts.unet import UNet
    ckpt_path = os.path.join(V2_DIR, "unet_best.pt")
    ck = torch.load(ckpt_path, map_location="cuda", weights_only=True)
    unet = UNet(in_ch=3, n_classes=ck["n_classes"]).cuda().eval()
    unet.load_state_dict(ck["state_dict"])

    prior_c, prior_frac = global_prior_class()
    print(f"[eval{shard_tag}] global prior class = {prior_class_name(prior_c)} "
          f"(train pixel share {prior_frac:.2f}) → 退化芯片常数预测器", flush=True)

    # 断点续跑：jsonl 中已有四方法记录的芯片跳过（幂等，可安全重入）
    existing = set()
    for r in load_runs("v2_runs.jsonl"):
        existing.add(r["name"])
    todo = []
    for c in chips:
        stem = c[0]
        need = [f"v2_obia_{stem}", f"v2_sam_default_{stem}",
                f"v2_sam_permissive_{stem}", f"v2_unet_{stem}"]
        if not all(n in existing for n in need):
            todo.append(c)
    skipped = len(chips) - len(todo)
    if skipped:
        print(f"[eval{shard_tag}] resume: skip {skipped} done chips, {len(todo)} to go",
              flush=True)

    for i, (stem, ipath, mpath) in enumerate(todo, 1):
        img, truth = load_chip(ipath, mpath)

        # ---- 方案 1：OBIA ----
        t0 = time.time()
        lab_obia, info = multiresolution_segment(img, scale=float(obia_scale),
                                                 shape=OBIA_SHAPE,
                                                 compactness=OBIA_COMPACTNESS,
                                                 verbose=False)
        seg_s = time.time() - t0
        lab_obia = np.asarray(lab_obia).astype(np.int32)
        save_label("obia", stem, lab_obia)
        rec, cmap, _ = eval_with_fallback(img, truth, lab_obia, SEED, TOL_PX,
                                          f"v2_obia_{stem}", seg_s, 7, prior_c)
        rec["mrs_n_merges"] = int(info.get("n_merges", -1))
        save_pred("obia", stem, cmap)
        append_raw(rec, fname="v2_runs.jsonl")

        # ---- 方案 2：SAM2 两配置（输入=2% 拉伸，对齐 v1 口径）----
        rgb_u8 = stretch_u8((img * 255 + 0.5).astype(np.uint8))
        for cfg_name, cfg in AMG_CFG.items():
            amg = SAM2AutomaticMaskGenerator(
                sam_model, points_per_side=32, points_per_batch=64,
                box_nms_thresh=0.7, crop_n_layers=0, min_mask_region_area=0, **cfg)
            t0 = time.time()
            masks = amg.generate(rgb_u8)
            seg_s = time.time() - t0
            # 剔除全图退化提案（>90% 芯片面积，不携带分割信息；先到先得会堵死整幅）
            masks = [m for m in masks
                     if np.asarray(m["segmentation"], bool).sum()
                     <= GIANT_FRAC * img.shape[0] * img.shape[1]]
            if masks:
                lab, paint = masks_to_labelmap(masks)
            else:                     # AMG 零有效提案 → 全零 label，走退化兜底
                lab = np.zeros(img.shape[:2], np.int32)
                paint = {"n_masks_raw": 0, "n_painted": 0}
            save_label("sam_" + cfg_name, stem, lab)
            rec, cmap, _ = eval_with_fallback(img, truth, lab, SEED, TOL_PX,
                                              f"v2_sam_{cfg_name}_{stem}", seg_s,
                                              7, prior_c)
            rec.update(paint)
            rec["amg_cfg"] = cfg_name
            rec["n_masks_after_giant_filter"] = len(masks)
            save_pred("sam_" + cfg_name, stem, cmap)
            append_raw(rec, fname="v2_runs.jsonl")

        # ---- 方案 3：U-Net ----
        t0 = time.time()
        with torch.no_grad():
            x = torch.from_numpy(img.transpose(2, 0, 1)[None]).cuda()
            logits = unet(x)[0]                       # (8, H, W)
            pred = logits[1:, :, :].argmax(0).cpu().numpy().astype(np.int32) + 1
            # 类别通道 1..7 上取 argmax（0=ignore 通道不参与，LoveDA 官方协议）
        seg_s = time.time() - t0
        save_pred("unet", stem, pred)
        rec, cmap, _ = eval_with_fallback(img, truth, pred, SEED, TOL_PX,
                                          f"v2_unet_{stem}", seg_s, 7, prior_c)
        rec["unet_pred_is_cmap"] = True     # U-Net 的 label 图 = 分类图本身
        save_pred("unet_cmap", stem, cmap)
        append_raw(rec, fname="v2_runs.jsonl")

        print(f"[eval{shard_tag}] {i}/{len(todo)} {stem}: "
              f"OBIA oa={rec['map_oa']:.3f} OBIA_seg={rec['seg_seconds']:.0f}s "
              f"(SAM/UNet 见 jsonl)", flush=True)


def aggregate():
    """读 v2_runs.jsonl → results/metrics.json（数字单一来源）。"""
    runs_d = {}
    for r in load_runs("v2_runs.jsonl"):
        runs_d[r["name"]] = r          # 同名后写覆盖（断点重入会重写 OBIA 记录）
    chips = chip_list()
    n_expect = len(chips)
    methods = {"obia": "v2_obia_", "sam_default": "v2_sam_default_",
               "sam_permissive": "v2_sam_permissive_", "unet": "v2_unet_"}
    # 每芯片真值类数（单类芯片的 bF1 公式值为 0，属度量伪影，单独记账）
    chip_classes = {}
    for stem, ipath, mpath in chips:
        t = np.asarray(Image.open(mpath), np.int32)
        chip_classes[stem] = len(set(np.unique(t).tolist()) - {0})
    n_single = sum(1 for v in chip_classes.values() if v < 2)

    summary = {}
    for m, prefix in methods.items():
        sub = [runs_d[f"{prefix}{c[0]}"] for c in chips]
        assert len(sub) == n_expect, \
            f"{m}: 唯一记录 {len(sub)} ≠ 期望 {n_expect}（先跑完全部分片）"

        def ms(key):
            v = np.array([r[key] for r in sub], float)
            return {"mean": float(v.mean()), "std": float(v.std()), "n": len(v),
                    "median": float(np.median(v))}
        mc = [r for r in sub if chip_classes[r["name"][len(prefix):]] >= 2]
        summary[m] = {
            "n_chips": len(sub),
            "map_oa": ms("map_oa"),
            "speckle": ms("speckle"),
            "cls_boundary_f1": ms("cls_boundary_f1"),
            "cls_boundary_f1_multiclass": {
                "mean": float(np.mean([r["cls_boundary_f1"] for r in mc])),
                "std": float(np.std([r["cls_boundary_f1"] for r in mc])),
                "n": len(mc)},
            "seg_boundary_f1": ms("seg_boundary_f1"),
            "seg_seconds": ms("seg_seconds"),
            "n_objects": {"mean": float(np.mean([r["n_objects"] for r in sub])),
                          "std": float(np.std([r["n_objects"] for r in sub])),
                          "n": len(sub)},
            "n_degenerate_chips": int(sum(1 for r in sub if r.get("degenerate"))),
            "n_single_class_chips": n_single,
        }
        if all("refined_map_oa" in r for r in sub):
            summary[m]["refined_map_oa"] = ms("refined_map_oa")
            summary[m]["refined_cls_boundary_f1"] = ms("refined_cls_boundary_f1")

    cal = json.load(open(os.path.join(V2_DIR, "obia_calibration.json"), encoding="utf-8"))
    import torch as _t
    ckpt_path = os.path.join(V2_DIR, "unet_best.pt")
    ck = _t.load(ckpt_path, map_location="cpu", weights_only=True)
    manifest = json.load(open(os.path.join(DATA_DIR, "loveda_split_manifest.json"),
                              encoding="utf-8"))
    metrics = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "dataset": {"name": "LoveDA (mrmeepsle/LoveDa parquet via hf-mirror)",
                    "classes": {str(i): n for i, n in enumerate(
                        ["ignore", "background", "building", "road", "water",
                         "barren", "forest", "agriculture"])},
                    "img_size": 1024, "chip_size": 512,
                    "n_test_chips": len(chips),
                    "test_img_names": [c[0] for c in chips],
                    "n_train_chips": manifest["n_train_chips"],
                    "manifest": "data/loveda_split_manifest.json"},
        "seed": SEED, "tol_px": TOL_PX,
        "obia": {"main_scale": cal["main_scale"], "shape": OBIA_SHAPE,
                 "compactness": OBIA_COMPACTNESS,
                 "calibration_image": cal["calibration_image"],
                 "calibration_rule": cal["calibration_rule"],
                 "calibration_records": cal["records"]},
        "sam2": {"model": "facebook/sam2-hiera-small", "pps": 32,
                 "ckpt_sha256_16": "95949964d4e54840",
                 "configs": ["default", "permissive"]},
        "unet": {"arch": "handwritten UNet base32 (scripts/unet.py)",
                 "checkpoint": "results/v2/unet_best.pt",
                 "checkpoint_sha256_16": hashlib.sha256(
                     open(ckpt_path, "rb").read()).hexdigest()[:16],
                 "monitor_oa": ck["monitor_oa"], "epoch": ck["epoch"],
                 "train_log": "results/v2/train_log.json"},
        "torch_cuda": _t.cuda.get_device_name(0) if _t.cuda.is_available() else "cpu",
        "metrics": summary,
    }
    out = os.path.join(RESULTS_DIR, "metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    print(f"[eval] wrote {out}", flush=True)
    for m, s in summary.items():
        print(f"  {m:14s}: OA={s['map_oa']['mean']:.4f}±{s['map_oa']['std']:.4f} "
              f"speckle={s['speckle']['mean']:.4f} "
              f"cls_bF1={s['cls_boundary_f1']['mean']:.4f} "
              f"seg={s['seg_seconds']['mean']:.2f}s n={s['n_chips']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true", help="只跑 OBIA 尺度标定")
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--aggregate", action="store_true", help="只汇总 metrics.json")
    args = ap.parse_args()
    os.makedirs(V2_DIR, exist_ok=True)

    if args.calibrate:
        calibrate_obia_scale(os.path.join(V2_DIR, "obia_calibration.json"))
        return
    if args.aggregate:
        aggregate()
        return

    chips = chip_list()
    assert 0 < len(chips) <= 128, "test 芯片数异常"
    if args.nshards > 1:
        assert 0 <= args.shard < args.nshards
        chips = [c for i, c in enumerate(chips) if i % args.nshards == args.shard]
        tag = f"[shard{args.shard}/{args.nshards}]"
    else:
        tag = ""
    print(f"[eval{tag}] chips: {len(chips)}", flush=True)

    cal = json.load(open(os.path.join(V2_DIR, "obia_calibration.json"), encoding="utf-8"))
    assert os.path.exists(os.path.join(V2_DIR, "unet_best.pt")), \
        "results/v2/unet_best.pt 不存在（先训练）"
    process_chips(chips, cal["main_scale"], tag)
    print(f"[eval{tag}] shard done", flush=True)


if __name__ == "__main__":
    main()
