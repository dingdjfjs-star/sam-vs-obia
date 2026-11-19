# -*- coding: utf-8 -*-
"""SAM2 点/框提示实验：对采样图斑分别用点提示与框提示分割，量化提示价值。

协议
----
* 采样：seed 固定，按地类分层抽 20 个图斑（每类最多 4 个，面积 ≥150px）；
* 点提示 = 图斑质心像素 (x, y)；框提示 = 图斑外包框 [xmin, ymin, xmax, ymax]；
* 每次提示记录：mask 与图斑的 IoU / 覆盖率 / 越界率 / 窗口内边界 F（tol=2px）；
* 窗口 = 图斑 bbox 外扩 24px，边界 F 在窗口内计算（避免全图边界淹没）；
* 逐条记录 + 汇总写入 results/raw/prompts.jsonl（数字溯源）。

SAM 预测通过 scripts/sam_adapter.py 注入（samgeo 细节集中在那一个文件）。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.common import (RESULTS_DIR, SEED, append_raw, ensure_dirs,  # noqa: E402
                            load_scene)

import obiabridge  # noqa: E402,F401
from obia.metrics import boundary_f  # noqa: E402

N_PER_CLASS = 4
MIN_AREA = 150
WINDOW_MARGIN = 24
N_CLASSES = 5


def sample_parcels(parcels, truth, seed=SEED, n_per_class=N_PER_CLASS,
                   min_area=MIN_AREA):
    """分层抽样图斑：每类最多 n 个，面积≥min_area 且 ≥80% 像素属同一类。"""
    rng = np.random.default_rng(seed)
    ids = np.unique(parcels)
    ids = ids[ids > 0]
    picks = []
    for c in range(1, N_CLASSES + 1):
        pool = []
        for pid in ids:
            m = parcels == pid
            if m.sum() < min_area:
                continue
            cls = int(np.bincount(truth[m].ravel()).argmax())
            if cls == c and truth[m].mean() >= 0.8:
                pool.append((int(pid), int(m.sum())))
        if not pool:
            continue
        pool.sort(key=lambda t: -t[1])          # 大图斑优先进入候选池
        take = min(n_per_class, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        picks += [pool[i] for i in idx]
    return picks


def parcel_window(parcels, pid, margin=WINDOW_MARGIN):
    """图斑 bbox 外扩 margin 的窗口 (r0, r1, c0, c1)。"""
    ys, xs = np.where(parcels == pid)
    r0 = max(int(ys.min()) - margin, 0)
    r1 = min(int(ys.max()) + 1 + margin, parcels.shape[0])
    c0 = max(int(xs.min()) - margin, 0)
    c1 = min(int(xs.max()) + 1 + margin, parcels.shape[1])
    return r0, r1, c0, c1


def parcel_class(parcels, truth, pid):
    m = parcels == pid
    return int(np.bincount(truth[m].ravel()).argmax())


def prompt_metrics(mask, pmask, tol=2):
    """单个提示的质量指标：IoU / 覆盖率 / 越界率 / 窗口边界 F。"""
    inter = float((mask & pmask).sum())
    union = float((mask | pmask).sum())
    iou = inter / max(union, 1.0)
    coverage = inter / max(float(pmask.sum()), 1.0)
    spill = float((mask & ~pmask).sum()) / max(float(mask.sum()), 1.0)
    pred_map = mask.astype(np.int32)                # 窗口内预测 = mask 本身
    truth_map = pmask.astype(np.int32)
    bf = boundary_f(pred_map, truth_map, tol=tol)["f1"]
    return {"iou": iou, "coverage": coverage, "spill": spill, "window_bf1": bf}


def run_prompts(predict_fn, parcels, truth, seed=SEED, warmup=True, verbose=True):
    """predict_fn(kind, x, y, bbox) -> (H,W) bool mask；kind ∈ {point, box}。

    返回 (records, window_store)；window_store 为 {(pid, mode): (mask窗口, parcel窗口)}
    供 fig5 叠加可视化（保存 npz）。
    """
    picks = sample_parcels(parcels, truth, seed=seed)
    if verbose:
        print(f"[prompts] sampled {len(picks)} parcels", flush=True)

    if warmup and picks:
        pid0 = picks[0][0]
        cy, cx = ndimage.center_of_mass(parcels == pid0)
        predict_fn("point", int(round(cx)), int(round(cy)), None)  # 预热不计耗时

    records = []
    windows = {}
    for pid, area in picks:
        pmask = parcels == pid
        r0, r1, c0, c1 = parcel_window(parcels, pid)
        cy, cx = ndimage.center_of_mass(pmask)
        ys, xs = np.where(pmask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        cls = parcel_class(parcels, truth, pid)

        for kind, px, py, box in (("point", int(round(cx)), int(round(cy)), None),
                                  ("box", None, None, bbox)):
            t0 = time.time()
            mask = predict_fn(kind, px, py, box)
            dt = time.time() - t0
            m = np.asarray(mask, bool)[r0:r1, c0:c1]
            met = prompt_metrics(m, pmask[r0:r1, c0:c1])
            rec = {"parcel_id": int(pid), "class": cls, "area": int(area),
                   "mode": kind, "seconds": dt, **met,
                   "point_xy": [px, py] if px is not None else None,
                   "bbox": box,
                   "window": [int(r0), int(c0), int(r1), int(c1)]}
            records.append(rec)
            windows[(pid, kind)] = (m, pmask[r0:r1, c0:c1])
            if verbose:
                print(f"[prompts] pid={pid} cls={cls} {kind}: IoU={met['iou']:.3f} "
                      f"cov={met['coverage']:.3f} spill={met['spill']:.3f} "
                      f"bF1={met['window_bf1']:.3f} ({dt*1000:.0f}ms)", flush=True)
    return records, windows


def save_windows(windows, path):
    """窗口数据落盘 npz（fig5 数据源；key = pid{pid}_{mode}_{mask|parcel}）。"""
    arrays = {}
    for (pid, kind), (m, pm) in windows.items():
        arrays[f"pid{pid}_{kind}_mask"] = m
        arrays[f"pid{pid}_{kind}_parcel"] = pm
    np.savez_compressed(path, **arrays)


def summarize(records):
    out = {}
    for mode in ("point", "box"):
        sub = [r for r in records if r["mode"] == mode]
        if not sub:
            continue
        classes = sorted(set(r["class"] for r in sub))
        out[mode] = {
            "n": len(sub),
            "iou_mean": float(np.mean([r["iou"] for r in sub])),
            "iou_median": float(np.median([r["iou"] for r in sub])),
            "coverage_mean": float(np.mean([r["coverage"] for r in sub])),
            "spill_mean": float(np.mean([r["spill"] for r in sub])),
            "window_bf1_mean": float(np.mean([r["window_bf1"] for r in sub])),
            "seconds_mean": float(np.mean([r["seconds"] for r in sub])),
            "iou_per_class": {str(c): float(np.mean([r["iou"] for r in sub
                                                     if r["class"] == c]))
                              for c in classes},
        }
    return out


def main():
    ensure_dirs()
    image, truth, parcels = load_scene()
    from scripts.sam_adapter import make_predict_fn
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    predict_fn = make_predict_fn(
        image_rgb_path=os.path.join(data_dir, "scene_rgb.tif"),
        ckpt_path=os.path.join(data_dir, "weights", "sam2_hiera_small.pt"))
    records, windows = run_prompts(predict_fn, parcels, truth, seed=SEED)

    summary = summarize(records)
    for r in records:
        append_raw(r, fname="prompts.jsonl")
    append_raw({"name": "sam_prompts_summary", **summary}, fname="prompts.jsonl")
    save_windows(windows, os.path.join(RESULTS_DIR, "prompt_windows.npz"))
    print("[prompts] summary:", flush=True)
    for mode, s in summary.items():
        print(f"  {mode}: IoU mean={s['iou_mean']:.3f} median={s['iou_median']:.3f} "
              f"cov={s['coverage_mean']:.3f} spill={s['spill_mean']:.3f} "
              f"bF1={s['window_bf1_mean']:.3f} {s['seconds_mean']*1000:.0f}ms/prompt",
              flush=True)


if __name__ == "__main__":
    main()
