# -*- coding: utf-8 -*-
"""v2 出图（5 张，results/figs/v2_*.png），数据源仅限 metrics.json / train_log.json / 落盘 preds。

figv2_1_methods.png      2 个展示芯片 × 5 面板（RGB/真值/三方案分类图）
figv2_2_metrics.png      四方法 × 四指标柱状（mean±std）
figv2_3_train_curve.png  U-Net 训练曲线（loss + 监控 OA/mIoU）
figv2_4_v1_v2.png        v1 合成 vs v2 真实 结论对照（共同三方法）
figv2_5_gallery.png      4 芯片 × 4 方法 分类图墙
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.common import DATA_DIR, LABELMAP_DIR, RESULTS_DIR  # noqa: E402

V2_DIR = os.path.join(RESULTS_DIR, "v2")
PREDS = os.path.join(V2_DIR, "preds")
FIGS = os.path.join(RESULTS_DIR, "figs")
TEST512 = os.path.join(DATA_DIR, "loveda", "test512")

# LoveDA 0..7 调色（0=ignore 黑）
PALETTE = np.array([
    [0, 0, 0],        # 0 ignore
    [158, 158, 158],  # 1 background
    [211, 47, 47],    # 2 building
    [249, 168, 37],   # 3 road
    [25, 118, 210],   # 4 water
    [215, 168, 110],  # 5 barren
    [46, 125, 50],    # 6 forest
    [174, 213, 129],  # 7 agriculture
], np.uint8)
METHODS = [("obia", "OBIA MRS"), ("sam_default", "SAM2 default"),
           ("sam_permissive", "SAM2 permissive"), ("unet", "U-Net (trained)")]
SHOW_CHIPS = None   # main() 里按类多样性选


def cmap_rgb(cmap):
    out = np.zeros(cmap.shape + (3,), np.uint8)
    vals = np.unique(cmap)
    for v in vals:
        out[cmap == v] = PALETTE[min(int(v), 7)]
    return out


def load_metrics():
    with open(os.path.join(RESULTS_DIR, "metrics.json"), encoding="utf-8") as f:
        return json.load(f)


def pick_show_chips(manifest, k=2):
    """按类别丰富度挑展示芯片（确定性：类多样性降序 + 名称序稳定排序）。

    manifest.test_img_names 已是完整芯片 stem（test_XXXX_rr）。"""
    names = manifest["test_img_names"]
    scored = []
    for stem in names:
        mp = os.path.join(TEST512, "mask", stem + ".png")
        if not os.path.exists(mp):
            continue
        arr = np.asarray(Image.open(mp))
        diversity = len(np.unique(arr))
        scored.append((-diversity, stem))
    scored.sort()
    return [s for _, s in scored[:k]]


def main():
    os.makedirs(FIGS, exist_ok=True)
    M = load_metrics()
    manifest = M["dataset"]
    chips = manifest["test_img_names"]

    # ---------- figv2_1 方法并排（三方案主对比）----------
    show = pick_show_chips(manifest, k=2)
    main_methods = [("obia", "OBIA MRS (calibrated)"),
                    ("sam_default", "SAM2 AMG default (pps=32)"),
                    ("unet", "U-Net (trained)")]
    fig, axes = plt.subplots(len(show), 5, figsize=(19, 4.1 * len(show)))
    axes = np.atleast_2d(axes)
    for row, stem in enumerate(show):
        img = np.asarray(Image.open(os.path.join(TEST512, "img", stem + ".png")))
        truth = np.asarray(Image.open(os.path.join(TEST512, "mask", stem + ".png")), np.int32)
        panels = [("RGB (SAM/U-Net input)", img, None),
                  ("Truth (LoveDA 7 classes)", cmap_rgb(truth), None)]
        for m, tag in main_methods:
            pred = np.load(os.path.join(PREDS, m, stem + ".npy"))
            panels.append((tag, cmap_rgb(pred), M["metrics"][m]["map_oa"]["mean"]))
        for col, (title, p, oa_mean) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(p, interpolation="nearest")
            if oa_mean is not None:
                ax.set_xlabel(f"mean OA={oa_mean:.3f}", fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Fig v2-1  Three methods on the same LoveDA test chips (512²)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "figv2_1_methods.png"), dpi=130)
    plt.close(fig)

    # ---------- figv2_2 指标柱状 ----------
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    metric_keys = [("map_oa", "Overall accuracy", False),
                   ("speckle", "Salt-pepper rate", False),
                   ("cls_boundary_f1", "Boundary F1 vs class truth", False),
                   ("seg_seconds", "Segment seconds (log)", True)]
    colors = ["#d32f2f", "#1976d2", "#64b5f6", "#2e7d32"]
    for ax, (key, label, logy) in zip(axes, metric_keys):
        names = [m[1] for m in METHODS]
        means = [M["metrics"][m[0]][key]["mean"] for m in METHODS]
        stds = [M["metrics"][m[0]][key]["std"] for m in METHODS]
        bars = ax.bar(names, means, yerr=stds, capsize=4, color=colors)
        if logy:
            ax.set_yscale("log")
        ax.set_title(label, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=8, rotation=12)
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}" if v < 100 else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle("Fig v2-2  Four metrics, same 64 test chips (mean±std)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "figv2_2_metrics.png"), dpi=130)
    plt.close(fig)

    # ---------- figv2_3 训练曲线 ----------
    with open(os.path.join(V2_DIR, "train_log.json"), encoding="utf-8") as f:
        log = json.load(f)
    eps = [e["epoch"] for e in log["epochs"]]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    ax[0].plot(eps, [e["loss_mean"] for e in log["epochs"]], "o-", color="#d32f2f")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("train loss (CE)")
    ax[0].set_title("U-Net training loss")
    ax[0].grid(alpha=0.3)
    ax[1].plot(eps, [e["monitor_oa"] for e in log["epochs"]], "o-", color="#1976d2",
               label="monitor OA")
    ax[1].plot(eps, [e["monitor_miou"] for e in log["epochs"]], "s--", color="#2e7d32",
               label="monitor mIoU")
    ax[1].set_xlabel("epoch")
    ax[1].set_title("Held-out 5% of train chips")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.suptitle(f"Fig v2-3  U-Net training curves (seed={log['seed']}, "
                 f"{len(log['epochs'])} epochs, {log['device']})", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "figv2_3_train_curve.png"), dpi=130)
    plt.close(fig)

    # ---------- figv2_4 v1 vs v2 对照 ----------
    with open(os.path.join(RESULTS_DIR, "summary.json"), encoding="utf-8") as f:
        v1 = json.load(f)
    common_map = [("obia", "obia_scale10", "OBIA MRS"),
                  ("sam_default", "sam_pps32", "SAM2 default"),
                  ("sam_permissive", "sam_pps32_permissive", "SAM2 permissive")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    x = np.arange(3)
    w = 0.36
    for ax, key, v1key, label in (
        (axes[0], "map_oa", "map_oa", "Overall accuracy"),
        (axes[1], "cls_boundary_f1", "cls_boundary_f1", "Boundary F1 vs class truth"),
    ):
        v1v = [v1["main_comparison"][vk][v1key] for _, vk, _ in common_map]
        v2v = [M["metrics"][mk][key]["mean"] for mk, _, _ in common_map]
        ax.bar(x - w / 2, v1v, w, label="v1 synthetic (512², seed=20260829)", color="#f9a825")
        ax.bar(x + w / 2, v2v, w, label="v2 real LoveDA (64 chips)", color="#1976d2")
        ax.set_xticks(x, [t for _, _, t in common_map], fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        for xi, (a, b) in enumerate(zip(v1v, v2v)):
            ax.text(xi - w / 2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
            ax.text(xi + w / 2, b, f"{b:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].legend(fontsize=8)
    fig.suptitle("Fig v2-4  Does the synthetic-scene conclusion hold on real data?", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "figv2_4_v1_v2.png"), dpi=130)
    plt.close(fig)

    # ---------- figv2_5 芯片墙 ----------
    gallery = chips[:4]              # test_img_names 已是完整芯片 stem
    fig, axes = plt.subplots(4, 6, figsize=(17.5, 12.4))
    for row, stem in enumerate(gallery):
        img = np.asarray(Image.open(os.path.join(TEST512, "img", stem + ".png")))
        truth = np.asarray(Image.open(os.path.join(TEST512, "mask", stem + ".png")), np.int32)
        panels = [("RGB", img), ("Truth", cmap_rgb(truth))]
        for m, tag in METHODS:
            pred = np.load(os.path.join(PREDS, m, stem + ".npy"))
            oa = M["metrics"][m]["map_oa"]["mean"]
            panels.append((f"{tag}", cmap_rgb(pred), oa))
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel[1], interpolation="nearest")
            if col == 0:
                ax.set_ylabel(stem, fontsize=8)
            if col >= 2:
                ax.set_xlabel(f"mean OA={panel[2]:.3f}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(panel[0], fontsize=9)
    fig.suptitle("Fig v2-5  Gallery: 4 chips × 4 methods (classification maps)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "figv2_5_gallery.png"), dpi=120)
    plt.close(fig)

    print(f"[figs-v2] 5 figures → {FIGS}", flush=True)


if __name__ == "__main__":
    main()
