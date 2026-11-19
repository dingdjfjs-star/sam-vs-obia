# -*- coding: utf-8 -*-
"""出图：5 张结果图（results/figs/），数据源仅限 results/ 下本次实跑产物。

fig1_segmentation_compare.png  六宫格：RGB/地类真值/图斑真值 + 三方分割边界叠加
fig2_classification_compare.png 六宫格：真值 + 三方分类图 + 两张误差图
fig3_scale_response.png        对象数-尺度响应（OBIA scale 扫描 vs SAM pps 扫描）
fig4_metrics_bar.png           主对比指标柱状（OA/椒盐率/边界F/分割边界F/耗时）
fig5_prompts.png               点/框提示样例叠加 + IoU 对比柱状（含自动模式基线）
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.common import (DATA_DIR, LABELMAP_DIR, RESULTS_DIR, fill_uncovered,  # noqa: E402
                            load_scene)
from scripts.make_scene import make_rgb  # noqa: E402

CLASS_NAMES = ["water", "forest", "farmland", "soil", "builtup"]
CLASS_COLORS = ["#2f6fd0", "#2e7d32", "#c0ca33", "#d7a86e", "#757575"]
PALETTE = np.array([[47, 111, 208], [46, 125, 50], [192, 202, 51],
                    [215, 168, 110], [117, 117, 117]], np.uint8)
FIG_DIR = os.path.join(RESULTS_DIR, "figs")


def truth_rgb(truth):
    out = np.zeros(truth.shape + (3,), np.uint8)
    for c in range(1, 6):
        out[truth == c] = PALETTE[c - 1]
    return out


def boundary_overlay(ax, rgb, labels, color="yellow", dilate=1):
    """label 图边界叠加在 RGB 上。"""
    lab = np.asarray(labels)
    b = np.zeros(lab.shape, bool)
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    if dilate > 0:
        b = ndimage.binary_dilation(b, iterations=dilate)
    ov = np.zeros(lab.shape + (4,), np.float32)
    ov[b] = matplotlib.colors.to_rgba(color)
    ax.imshow(rgb, interpolation="nearest")
    ax.imshow(ov, interpolation="nearest")


def method_rgb(labels):
    """label 图 → 随机上色（图斑可视化）。"""
    lab = fill_uncovered(labels)
    k = int(lab.max())
    rng = np.random.default_rng(0)
    cols = rng.integers(40, 215, (k + 1, 3), np.uint8)
    cols[0] = 0
    return cols[lab]


def error_overlay(rgb, cmap, truth):
    err = (cmap != truth) & (truth >= 1)
    ov = np.zeros(rgb.shape[:2] + (4,), np.float32)   # RGBA 叠加层
    ov[err] = [1.0, 0.15, 0.15, 0.85]
    return rgb, ov


def panel(ax, img, title):
    ax.imshow(img, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def load_summary():
    with open(os.path.join(RESULTS_DIR, "summary.json"), encoding="utf-8") as f:
        return json.load(f)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    image, truth, parcels = load_scene()
    rgb = make_rgb(image)
    summary = load_summary()
    mc = summary["main_comparison"]

    # ---------- fig1 分割对比 ----------
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9))
    panel(axes[0, 0], rgb, "RGB true color (SAM input)")
    panel(axes[0, 1], truth_rgb(truth), "Pixel truth (5 classes)")
    boundary_overlay(axes[0, 2], rgb, parcels, color="white")
    axes[0, 2].set_title(f"Parcel truth ({int(parcels.max())} fields)")
    for ax, name, tag in ((axes[1, 0], "obia_scale10", "OBIA MRS (scale=10)"),
                          (axes[1, 1], "slic_440", "SLIC (n=440)"),
                          (axes[1, 2], "sam_pps32", "SAM2 AMG (pps=32)")):
        lab = np.load(os.path.join(LABELMAP_DIR, f"{name}.npy"))
        rec = mc[name]
        boundary_overlay(ax, rgb, lab, color="yellow")
        ax.set_title(f"{tag}  |  {rec['n_objects']} objects")
    fig.suptitle("Fig1 Segmentation: handwritten OBIA vs zero-shot SAM2 "
                 "(512² synthetic scene, seed=20260829)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_segmentation_compare.png"), dpi=130)
    plt.close(fig)

    # ---------- fig2 分类对比 ----------
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9))
    panel(axes[0, 0], truth_rgb(truth), "Truth")
    slots = ((axes[0, 1], "obia_scale10"), (axes[0, 2], "slic_440"),
             (axes[1, 0], "sam_pps32"))
    for ax, name in slots:
        cmap = np.load(os.path.join(RESULTS_DIR, "classmaps", f"{name}.npy"))
        rec = mc[name]
        panel(ax, truth_rgb(cmap.astype(np.int32)),
              f"{rec['name']} classification\nOA={rec['map_oa']:.3f} "
              f"speckle={rec['speckle']:.4f} bF1={rec['cls_boundary_f1']:.3f}")
    for ax, name in ((axes[1, 1], "obia_scale10"), (axes[1, 2], "sam_pps32")):
        cmap = np.load(os.path.join(RESULTS_DIR, "classmaps", f"{name}.npy")).astype(np.int32)
        rgb2, ov = error_overlay(rgb, cmap, truth)
        ax.imshow(rgb2, interpolation="nearest")
        ax.imshow(ov, interpolation="nearest")
        wrong = float((cmap != truth).mean())
        ax.set_title(f"Error map: {name} ({wrong*100:.2f}% pixels)")
    fig.suptitle("Fig2 Classification maps via the SAME downstream protocol "
                 "(19-dim features + seed-fixed RF)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_classification_compare.png"), dpi=130)
    plt.close(fig)

    # ---------- fig3 尺度响应 ----------
    obia_sweep = summary["obia_scale_sweep"]
    sam_sweep = summary["sam_pps_sweep"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, sweep, xkey, xlabel, title in (
        (axes[0], obia_sweep, "mrs", "OBIA scale parameter",
         "OBIA MRS: scale sweep (shape=0.2, cpt=0.5)"),
        (axes[1], sam_sweep, "pps", "SAM2 points_per_side",
         "SAM2 AMG: points_per_side sweep"),
    ):
        xs = [r["scale"] if xkey == "mrs" else r["pps"] for r in sweep]
        ax.plot(xs, [r["n_objects"] for r in sweep], "o-", color="#d32f2f",
                label="n objects")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("n objects", color="#d32f2f")
        ax.tick_params(axis="y", labelcolor="#d32f2f")
        ax2 = ax.twinx()
        ax2.plot(xs, [r["seg_boundary_f1"] for r in sweep], "s--", color="#1976d2",
                 label="seg boundary F1")
        ax2.set_ylabel("seg boundary F1 (vs parcels)", color="#1976d2")
        ax2.tick_params(axis="y", labelcolor="#1976d2")
        ax.set_title(title)
    ax = axes[2]
    ax.plot([r["n_objects"] for r in obia_sweep],
            [r["seg_boundary_f1"] for r in obia_sweep], "o-", color="#d32f2f",
            label="OBIA (scale 5→160)")
    ax.plot([r["n_objects"] for r in sam_sweep],
            [r["seg_boundary_f1"] for r in sam_sweep], "s-", color="#1976d2",
            label="SAM2 AMG (pps 8→64)")
    ax.set_xscale("log")
    ax.set_xlabel("n objects (log)")
    ax.set_ylabel("seg boundary F1 (vs parcels)")
    ax.set_title("Boundary quality vs granularity")
    ax.legend()
    fig.suptitle("Fig3 Object-count–scale response", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_scale_response.png"), dpi=130)
    plt.close(fig)

    # ---------- fig4 指标柱状 ----------
    metrics = [("map_oa", "Overall accuracy", 1.0),
               ("speckle", "Salt-pepper rate", 1.0),
               ("cls_boundary_f1", "Boundary F1 vs class truth", 1.0),
               ("seg_boundary_f1", "Boundary F1 vs parcel truth", 1.0),
               ("seg_seconds", "Segmentation seconds (log)", 1.0)]
    tags = [("obia_scale10", "OBIA\n(scale10)"), ("slic_440", "SLIC\n(n440)"),
            ("sam_pps32", "SAM2\n(pps32)")]
    colors = ["#d32f2f", "#f9a825", "#1976d2"]
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, (key, label, _) in zip(axes, metrics):
        vals = [mc[t[0]][key] for t in tags]
        bars = ax.bar([t[1] for t in tags], vals, color=colors)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if key == "seg_seconds":
            ax.set_yscale("log")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}" if key != "seg_seconds"
                    else f"{v:.1f}s", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Fig4 Main comparison (same downstream protocol)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_metrics_bar.png"), dpi=130)
    plt.close(fig)

    # ---------- fig5 提示实验 ----------
    psum = summary["prompts"]["summary"]
    ac = summary["auto_containment"]
    precords = summary["prompts"]["records"]
    wins = np.load(os.path.join(RESULTS_DIR, "prompt_windows.npz"))

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9))
    for row, mode in enumerate(("point", "box")):
        sub = [r for r in precords if r["mode"] == mode]
        sub.sort(key=lambda r: -r["iou"])
        for col, pick in enumerate((sub[0], sub[-1])):
            ax = axes[row, col]
            r0, c0, r1, c1 = pick["window"]
            ax.imshow(rgb[r0:r1, c0:c1], interpolation="nearest")
            pm = wins[f"pid{pick['parcel_id']}_{mode}_parcel"]
            mk = wins[f"pid{pick['parcel_id']}_{mode}_mask"]
            # 叠加：mask 边界黄色、图斑边界红色
            for m, color in ((mk, "yellow"), (pm, "red")):
                b = np.zeros(m.shape, bool)
                b[:, :-1] |= m[:, :-1] != m[:, 1:]
                b[:-1, :] |= m[:-1, :] != m[1:, :]
                ov = np.zeros(m.shape + (4,), np.float32)
                ov[b] = matplotlib.colors.to_rgba(color)
                ax.imshow(ov, interpolation="nearest")
            if mode == "point" and pick.get("point_xy"):
                px, py = pick["point_xy"]
                ax.plot(px - c0, py - r0, "w+", markersize=14, markeredgewidth=2)
            if mode == "box" and pick.get("bbox"):
                bx0, by0, bx1, by1 = pick["bbox"]
                ax.add_patch(mpatches.Rectangle((bx0 - c0, by0 - r0),
                                                bx1 - bx0, by1 - by0,
                                                fill=False, edgecolor="white",
                                                linewidth=1.6))
            tag = "best" if col == 0 else "worst"
            ax.set_title(f"{mode} {tag}: pid={pick['parcel_id']} "
                         f"cls={CLASS_NAMES[pick['class']-1]}\n"
                         f"IoU={pick['iou']:.3f} cov={pick['coverage']:.3f} "
                         f"spill={pick['spill']:.3f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        ax = axes[row, 2]
        x = np.arange(5)
        ious = [psum[mode]["iou_per_class"].get(str(c), 0.0) for c in range(1, 6)]
        ax.bar(x, ious, color="#1976d2" if mode == "point" else "#f9a825")
        ax.axhline(psum[mode]["iou_mean"], color="k", ls="--", lw=1,
                   label=f"mean {psum[mode]['iou_mean']:.3f}")
        ax.set_xticks(x, CLASS_NAMES, rotation=20, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"{mode} IoU per class (n={psum[mode]['n']})", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Fig5 Prompt engineering value: point vs box "
                 f"(auto-mode baseline IoU={ac['iou_mean']:.3f} at same parcels)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_prompts.png"), dpi=130)
    plt.close(fig)

    print(f"[figs] 5 figures written to {FIG_DIR}", flush=True)


if __name__ == "__main__":
    main()
