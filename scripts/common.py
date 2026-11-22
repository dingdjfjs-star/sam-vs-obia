# -*- coding: utf-8 -*-
"""公共工具：路径、场景读取、label 图规整、下游评估协议（公平性核心）。

下游评估协议 eval_labelmap() 是本项目的公平性基石——它把 obia-segmentation
baselines.py 中 SLIC 对照所走的"分割 label 图 → 19 维特征 → 同种子 RF →
回填分类图 → 三指标"协议参数化为任意 label 图，使 OBIA / SLIC / SAM 的
产物在同一协议下评估。除 label 图本身外，其余步骤逐行沿用 obia 原实现。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import obiabridge  # noqa: F401  （挂载 ../obia-segmentation 到 sys.path）

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LABELMAP_DIR = os.path.join(RESULTS_DIR, "labelmaps")
CLASSMAP_DIR = os.path.join(RESULTS_DIR, "classmaps")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

SEED = 20260829            # 与 obia-segmentation 完全一致的种子
SCENE_SIZE = 512           # 512×512 主场景
N_CLASSES = 5
TOL_PX = 2                 # 边界贴合 F 的容差（同 obia README 口径）
OBIA_MAIN_SCALE = 10       # obia 主参数（scale=10 / shape=0.2 / compactness=0.5）
OBIA_SHAPE = 0.2
OBIA_COMPACTNESS = 0.5


def ensure_dirs():
    for d in (DATA_DIR, RESULTS_DIR, LABELMAP_DIR, CLASSMAP_DIR, RAW_DIR, LOGS_DIR):
        os.makedirs(d, exist_ok=True)


def load_scene():
    """读取场景 npz（image/truth/parcels，均来自 obia.synthetic 同 seed 生成）。"""
    path = os.path.join(DATA_DIR, "scene.npz")
    z = np.load(path)
    return z["image"], z["truth"], z["parcels"]


def relabel_continuous(labels):
    """任意 label 图 → 连续 1..K（顺序按首次出现，稳定可复现）。"""
    lab = np.asarray(labels)
    uniq = np.unique(lab)
    uniq = uniq[uniq > 0]                      # 0 号（未覆盖像素）后面单独处理
    remap = np.zeros(int(lab.max()) + 1 if lab.max() >= 0 else 1, np.int32)
    remap[uniq] = np.arange(1, len(uniq) + 1, dtype=np.int32)
    out = remap[lab]
    out[lab == 0] = 0
    return out.astype(np.int32)


def fill_uncovered(labels, coverage_bg=0):
    """把 label 图中的 0（AMG 未覆盖像素）并入相邻对象，返回连续 1..K。

    用最近非零 label 的距离变换填充（AMG 缝隙通常是孤立小缝，并入最近
    对象不改变分割语义）。若整图无 0，原样规整返回。
    """
    from scipy import ndimage
    lab = np.asarray(labels)
    if (lab == 0).sum() == 0:
        return relabel_continuous(lab)
    ind = ndimage.distance_transform_edt(lab == 0, return_distances=False,
                                         return_indices=True)
    filled = lab[tuple(ind)]
    return relabel_continuous(filled)


def eval_labelmap(image, truth, parcels, labels, seed=SEED, tol_px=TOL_PX,
                  name="", seg_seconds=None, return_maps=False, n_classes=None):
    """通用下游评估：任意分割 label 图走与 SLIC 对照完全相同的协议。

    步骤（与 obia.baselines.slic_baseline 逐行同源）：
      1) label 图规整为连续 1..K（0 号缝隙并入最近对象）
      2) 19 维图斑特征（obia.features.extract_object_features 原实现）
      3) 多数票真值标定 + 同种子 RandomForest（obia.classify 原实现）
      4) 预测回填栅格 → 三个全图指标（obia.metrics 原实现）
      5) 另测分割质量：label 图边界 vs 真值图斑边界（README 6.1 口径）

    Returns
    -------
    dict : n_objects / map_oa / obj_oa / speckle / cls_boundary_f(3 项) /
           seg_boundary_f(3 项) / uncovered_frac / classify_seconds / seg_seconds
    return_maps=True 时返回 (dict, obj_map, lab)，便于落盘分类图与出图。
    """
    from obia.features import extract_object_features, object_majority_truth
    from obia.classify import classify_objects
    from obia.vectorize import rasterize_object_pred
    from obia.metrics import confusion_oa, salt_pepper_rate, boundary_f

    img = np.asarray(image, np.float32)
    labels0 = np.asarray(labels)
    uncovered = float((labels0 == 0).mean())
    lab = fill_uncovered(labels0)

    feats = extract_object_features(img, lab)
    nc = int(n_classes) if n_classes is not None else int(truth.max())
    obj_truth = object_majority_truth(lab, truth, n_classes=nc)
    res = classify_objects(feats, obj_truth, seed=seed)
    obj_map = rasterize_object_pred(lab, res["pred"])
    cm, map_oa = confusion_oa(obj_map, truth, int(truth.max()))

    cls_bf = boundary_f(obj_map, truth, tol=tol_px)     # 分类图 vs 地类边界
    seg_bf = boundary_f(lab, parcels, tol=tol_px)       # label 图 vs 图斑边界
    out = {
        "name": name,
        "n_objects": int(lab.max()),
        "uncovered_frac": uncovered,
        "map_oa": float(map_oa),
        "obj_oa": float(res["oa"]),
        "speckle": salt_pepper_rate(obj_map),
        "cls_boundary_p": cls_bf["precision"],
        "cls_boundary_r": cls_bf["recall"],
        "cls_boundary_f1": cls_bf["f1"],
        "seg_boundary_p": seg_bf["precision"],
        "seg_boundary_r": seg_bf["recall"],
        "seg_boundary_f1": seg_bf["f1"],
        "classify_seconds": float(res["seconds"]),
    }
    if seg_seconds is not None:
        out["seg_seconds"] = float(seg_seconds)
    if return_maps:
        return out, obj_map.astype(np.int32), lab
    return out


def save_labelmap(name, labels):
    path = os.path.join(LABELMAP_DIR, f"{name}.npy")
    np.save(path, np.asarray(labels).astype(np.int32))
    return path


def save_classmap(name, obj_map):
    path = os.path.join(CLASSMAP_DIR, f"{name}.npy")
    np.save(path, np.asarray(obj_map).astype(np.int32))
    return path


def append_raw(record, fname="runs.jsonl"):
    """原始记录逐条追加 jsonl（所有数字的第一落盘点，可溯源）。"""
    path = os.path.join(RAW_DIR, fname)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def load_runs(fname="runs.jsonl"):
    path = os.path.join(RAW_DIR, fname)
    records = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    return records


def runs_to_dict(records):
    """jsonl 记录按 name 归并成 {name: record}（同名后写覆盖先写）。"""
    d = {}
    for r in records:
        d[r["name"]] = r
    return d
