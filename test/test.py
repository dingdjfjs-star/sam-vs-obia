# -*- coding: utf-8 -*-
"""自动化自检（不跑模型推理，秒级完成）：

A1 obiabridge 桥接可用，obia 指标实现与原项目同一份
A2 场景确定性：同 seed 重新生成与 data/scene.npz 逐像素一致
A3 label 图健全性：连续 1..K、无 0 号残留（规整后）
A4 指标复现：summary.json 中的 map_oa / speckle / cls_bF1 / seg_bF1 可由
   落盘的 labelmap/classmap + obia 原指标函数复算到 1e-9
A5 JSON 往返：summary.json load→dump→load 逐字段相等；runs.jsonl 每行可解析
A6 提示记录健全性：prompts.jsonl 汇总存在、IoU∈[0,1]、两模式样本数一致

运行：E:\\Ana\\envs\\samgeo\\python.exe test\\test.py （或任意含 numpy/scipy 的环境）
全绿（all assertions pass, exit 0）即通过。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import obiabridge  # noqa: E402,F401
from obia.metrics import boundary_f, salt_pepper_rate  # noqa: E402
from obia.synthetic import generate_scene  # noqa: E402
from scripts.common import (DATA_DIR, RESULTS_DIR, SEED, SCENE_SIZE,  # noqa: E402
                            fill_uncovered, load_runs, runs_to_dict)

PASS = 0


def ok(msg):
    global PASS
    PASS += 1
    print(f"  PASS {PASS:02d} {msg}", flush=True)


def main():
    # --- A1 桥接 ---
    from obia.metrics import boundary_f as bf2
    from obia.metrics import salt_pepper_rate as sp2
    assert bf2 is boundary_f and sp2 is salt_pepper_rate, "指标实现非同一份"
    ok("A1 obiabridge 挂载成功，obia.metrics 为原项目实现")

    # --- A2 场景确定性 ---
    z = np.load(os.path.join(DATA_DIR, "scene.npz"))
    img0, truth0, parcels0 = z["image"], z["truth"], z["parcels"]
    img1, truth1, parcels1 = generate_scene(SCENE_SIZE, SCENE_SIZE, seed=SEED)
    assert np.array_equal(img0, img1), "image 与同 seed 重生成不一致"
    assert np.array_equal(truth0, truth1) and np.array_equal(parcels0, parcels1), \
        "truth/parcels 与同 seed 重生成不一致"
    ok(f"A2 场景 bit-exact 可复现（seed={SEED}, {SCENE_SIZE}², "
       f"classes={np.unique(truth0).tolist()}, parcels={int(parcels0.max())}）")

    # --- A3/A4 数据齐备 + 指标复现 ---
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    assert os.path.exists(summary_path), "results/summary.json 不存在（先跑实验与汇总）"
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    ok("A3 summary.json 存在")

    runs = runs_to_dict(load_runs("runs.jsonl"))
    for name in ("obia_scale10", "slic_440", "sam_pps32"):
        assert name in runs, f"runs.jsonl 缺 {name}"
        lab = np.load(os.path.join(RESULTS_DIR, "labelmaps", f"{name}.npy"))
        cmap = np.load(os.path.join(RESULTS_DIR, "classmaps", f"{name}.npy"))
        lab_f = fill_uncovered(lab)
        k = int(lab_f.max())
        assert lab_f.min() == 1 and np.array_equal(np.unique(lab_f),
                                                   np.arange(1, k + 1)), \
            f"{name} label 非连续 1..K"
        # 指标复算（obia 原实现，与汇总管线同源）
        speckle = salt_pepper_rate(cmap)
        cls_f1 = boundary_f(cmap, truth0, tol=2)["f1"]
        seg_f1 = boundary_f(lab_f, parcels0, tol=2)["f1"]
        rec = runs[name]
        for got, key, want in ((speckle, "speckle", rec["speckle"]),
                               (cls_f1, "cls_boundary_f1", rec["cls_boundary_f1"]),
                               (seg_f1, "seg_boundary_f1", rec["seg_boundary_f1"])):
            assert abs(got - want) < 1e-9, \
                f"{name}.{key} 复算 {got} ≠ 记录 {want}"
        # OA 复算（逐像素）
        valid = (truth0 >= 1) & (truth0 <= 5)
        oa = float((cmap[valid] == truth0[valid]).mean())
        assert abs(oa - rec["map_oa"]) < 1e-9, \
            f"{name}.map_oa 复算 {oa} ≠ 记录 {rec['map_oa']}"
        ok(f"A4 {name}: label 连续 1..{k}，map_oa/speckle/cls_bF1/seg_bF1 复算一致")

    # --- A5 JSON 往返 ---
    text = json.dumps(summary, ensure_ascii=False, sort_keys=True)
    rt = json.loads(text)
    assert json.dumps(rt, ensure_ascii=False, sort_keys=True) == text, "JSON 往返不等"
    for line in open(os.path.join(RESULTS_DIR, "raw", "runs.jsonl"), encoding="utf-8"):
        if line.strip():
            json.loads(line)
    ok("A5 summary.json 往返一致，runs.jsonl 全部行可解析")

    # --- A6 提示记录 ---
    precs = load_runs("prompts.jsonl")
    psum = [r for r in precs if r.get("name") == "sam_prompts_summary"]
    assert psum, "prompts.jsonl 缺汇总记录"
    s = psum[-1]
    for mode in ("point", "box"):
        assert mode in s and s[mode]["n"] > 0, f"提示模式 {mode} 无样本"
        assert 0.0 <= s[mode]["iou_mean"] <= 1.0
    assert s["point"]["n"] == s["box"]["n"], "两模式样本数不一致"
    ok(f"A6 提示实验：point n={s['point']['n']} IoU={s['point']['iou_mean']:.3f} | "
       f"box n={s['box']['n']} IoU={s['box']['iou_mean']:.3f}")

    print(f"\nALL {PASS} ASSERTIONS PASSED", flush=True)


if __name__ == "__main__":
    main()
