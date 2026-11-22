# -*- coding: utf-8 -*-
"""v2 第三方案：U-Net 在 LoveDA train split 上从零训练（GPU），训练曲线全程落盘。

纪律：种子固定（20260829，与 v1 一脉）；cudnn 确定性；训练只用 data/loveda/train，
validation 整体保留给三方案 test（另从 train 尾部划 5% 做监控曲线，不参与三方案评估）。

产出：
* results/v2/unet_best.pt       最优 state_dict（按监控 OA）
* results/v2/train_log.json     每 epoch 的 loss / 监控 OA / mIoU / lr / 秒
* results/v2/train_log.png      训练曲线（loss + OA/mIoU）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.common import RESULTS_DIR  # noqa: E402

SEED = 20260829


def set_deterministic(seed=SEED):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def list_pairs(root):
    """data/loveda/train/{img,mask}/*.png → 排序配对列表（prepare_v2_data 产物）。"""
    img_dir = os.path.join(root, "img")
    mask_dir = os.path.join(root, "mask")
    imgs = sorted(os.listdir(img_dir))
    pairs = []
    for f in imgs:
        m = os.path.join(mask_dir, f)
        if os.path.exists(m):
            pairs.append((os.path.join(img_dir, f), m))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(RESULTS_DIR), "data", "loveda", "train512"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-chips", type=int, default=2000, help="训练芯片上限（种子抽样）")
    ap.add_argument("--n-classes", type=int, default=8, help="LoveDA 掩码值域 0..7 → 类数（0=ignore 由 prepare 脚本确认）")
    ap.add_argument("--ignore-index", type=int, default=0)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    set_deterministic()
    rng = np.random.default_rng(SEED)

    pairs = list_pairs(args.data)
    if not pairs:
        raise SystemExit(f"[train] 没有配对样本：{args.data}（先跑 scripts/prepare_v2_data.py）")
    rng.shuffle(pairs)
    val_frac = 0.05
    n_val = max(8, int(len(pairs) * val_frac))
    val_pairs = pairs[:n_val]
    tr_pairs = pairs[n_val:][:args.max_chips]
    print(f"[train] chips: train={len(tr_pairs)} monitor-val={n_val} (总 {len(pairs)})", flush=True)

    class ChipDS(Dataset):
        def __init__(self, pairs):
            self.pairs = pairs

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, i):
            ip, mp = self.pairs[i]
            img = np.asarray(Image.open(ip).convert("RGB"), np.float32) / 255.0
            mask = np.asarray(Image.open(mp), np.int64)
            x = torch.from_numpy(img.transpose(2, 0, 1))
            y = torch.from_numpy(mask)
            return x, y

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from scripts.unet import UNet
    model = UNet(in_ch=3, n_classes=args.n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_index)

    dl_tr = DataLoader(ChipDS(tr_pairs), batch_size=args.batch, shuffle=True,
                       num_workers=0, drop_last=True)
    dl_va = DataLoader(ChipDS(val_pairs), batch_size=args.batch, shuffle=False, num_workers=0)

    log = {"seed": SEED, "device": device, "args": vars(args), "epochs": []}
    best_oa = -1.0
    out_dir = os.path.join(RESULTS_DIR, "v2")
    os.makedirs(out_dir, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        sched.step()

        # 监控集评估（OA + mIoU）
        model.eval()
        inter = np.zeros(args.n_classes, np.int64)
        union = np.zeros(args.n_classes, np.int64)
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                pred = pred.cpu().numpy()
                gt = y.cpu().numpy()
                valid = gt != args.ignore_index
                correct += int((pred[valid] == gt[valid]).sum())
                total += int(valid.sum())
                for c in range(1, args.n_classes):
                    p = pred == c
                    g = gt == c
                    inter[c] += int((p & g).sum())
                    union[c] += int((p | g).sum())
        oa = correct / max(total, 1)
        miou = float(np.mean([inter[c] / union[c] for c in range(1, args.n_classes) if union[c] > 0]))
        dt = time.time() - t0
        log["epochs"].append({"epoch": ep, "loss_mean": float(np.mean(losses)),
                              "monitor_oa": oa, "monitor_miou": miou,
                              "lr": sched.get_last_lr()[0], "seconds": dt})
        print(f"[train] ep{ep:02d}/{args.epochs} loss={np.mean(losses):.4f} "
              f"monOA={oa:.4f} mIoU={miou:.4f} {dt:.0f}s", flush=True)
        with open(os.path.join(out_dir, "train_log.json"), "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=1)
        if oa > best_oa:
            best_oa = oa
            torch.save({"state_dict": model.state_dict(), "n_classes": args.n_classes,
                        "epoch": ep, "monitor_oa": oa}, os.path.join(out_dir, "unet_best.pt"))

    print(f"[train] done, best monitor OA={best_oa:.4f} → {out_dir}/unet_best.pt", flush=True)


if __name__ == "__main__":
    main()
