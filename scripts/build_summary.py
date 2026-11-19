# -*- coding: utf-8 -*-
"""汇总：runs.jsonl + prompts.jsonl → results/summary.json（报告唯一数据源）。

额外计算 auto-containment：对提示实验采样的每个图斑，取 AMG(pps32) label 图中
覆盖其质心的分割段，算它与图斑的 IoU——这是"零提示自动模式下同一图斑能拿到的
分割质量"，与点/框提示 IoU 同基可比，直接量化提示的增益。
"""
from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.common import (RESULTS_DIR, SEED, SCENE_SIZE, TOL_PX,  # noqa: E402
                            fill_uncovered, load_runs, load_scene, runs_to_dict)

MAIN_NAMES = ["obia_scale10", "slic_440", "sam_pps32", "sam_pps32_permissive"]
MODEL_ID = "facebook/sam2-hiera-small"


def auto_containment(sam_lab, parcels, truth, records, seed=SEED):
    """AMG(pps32) 质心分割段 vs 图斑 IoU（提示实验同批图斑）。"""
    from scripts.run_sam_prompts import sample_parcels
    picks = sample_parcels(parcels, truth, seed=seed)
    out = []
    for pid, area in picks:
        pmask = parcels == pid
        cy, cx = ndimage.center_of_mass(pmask)
        seg_id = sam_lab[int(round(cy)), int(round(cx))]
        if seg_id <= 0:
            out.append({"parcel_id": int(pid), "iou": 0.0, "area": int(area)})
            continue
        seg = sam_lab == seg_id
        inter = float((seg & pmask).sum())
        union = float((seg | pmask).sum())
        out.append({"parcel_id": int(pid), "area": int(area),
                    "iou": inter / max(union, 1.0)})
    ious = [r["iou"] for r in out]
    return {"n": len(out), "iou_mean": float(np.mean(ious)),
            "iou_median": float(np.median(ious)), "per_parcel": out}


def main():
    image, truth, parcels = load_scene()
    runs = runs_to_dict(load_runs("runs.jsonl"))
    precs = load_runs("prompts.jsonl")
    psum = [r for r in precs if r.get("name") == "sam_prompts_summary"][-1]
    precords = [r for r in precs if r.get("name") != "sam_prompts_summary"]

    sam_lab_raw = np.load(os.path.join(RESULTS_DIR, "labelmaps", "sam_pps32.npy"))
    sam_lab = fill_uncovered(sam_lab_raw)

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED, "scene_size": SCENE_SIZE, "n_parcels_truth": int(parcels.max()),
        "n_classes": int(truth.max()), "tol_px": TOL_PX,
        "sam_model_id": MODEL_ID,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": None,
    }
    try:
        import torch
        meta["device"] = ("cuda:" + torch.cuda.get_device_name(0)
                          if torch.cuda.is_available() else "cpu")
        meta["torch"] = torch.__version__
    except Exception:
        meta["torch"] = None
    try:
        import samgeo
        meta["samgeo"] = getattr(samgeo, "__version__", "unknown")
    except Exception:
        meta["samgeo"] = None
    import numpy, scipy, sklearn, skimage
    meta["numpy"] = numpy.__version__
    meta["scipy"] = scipy.__version__
    meta["sklearn"] = sklearn.__version__
    meta["skimage"] = skimage.__version__

    obia_sweep = []
    for s in (5, 10, 20, 40, 80, 160):
        rec = dict(runs[f"obia_scale{s}"])
        rec["scale"] = s
        obia_sweep.append(rec)
    sam_sweep = [runs[f"sam_pps{p}"] for p in (8, 16, 32, 48, 64)]

    summary = {
        "meta": meta,
        "main_comparison": {k: runs[k] for k in MAIN_NAMES},
        "obia_scale_sweep": obia_sweep,
        "sam_pps_sweep": sam_sweep,
        "prompts": {"summary": psum, "records": precords},
        "auto_containment": auto_containment(sam_lab, parcels, truth, precords),
    }

    out = os.path.join(RESULTS_DIR, "summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"[summary] wrote {out}", flush=True)
    mc = summary["main_comparison"]
    for k, v in mc.items():
        print(f"  {k}: oa={v['map_oa']:.4f} speckle={v['speckle']:.4f} "
              f"cls_bF1={v['cls_boundary_f1']:.4f} seg_bF1={v['seg_boundary_f1']:.4f} "
              f"n={v['n_objects']} seg={v.get('seg_seconds', -1):.1f}s", flush=True)
    ac = summary["auto_containment"]
    print(f"  auto_containment IoU mean={ac['iou_mean']:.3f} "
          f"(point {psum['point']['iou_mean']:.3f} / box {psum['box']['iou_mean']:.3f})",
          flush=True)


if __name__ == "__main__":
    main()
