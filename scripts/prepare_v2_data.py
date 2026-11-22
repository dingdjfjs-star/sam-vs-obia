# -*- coding: utf-8 -*-
"""v2 数据管线：LoveDA parquet 分片 → PNG 抽取 → 512² 切片 → 三方案共用数据集。

划分纪律（种子 20260829，与 v1 同源）：
* train = 3 片 train parquet 全部图 → 1024²→4×512² 非重叠切片（U-Net 训练专用）；
* test  = validation 分片池中固定种子抽 16 张图 → 各切 4×512² = 64 个 test 芯片
  （OBIA/SAM/U-Net 三方吃同一份切片；validation 其余图绝不入训练）；
* 掩码值域断言 ⊆ {1..7}（LoveDA 官方编码：1 background 2 building 3 road
  4 water 5 barren 6 forest 7 agriculture，无 0/ignore）。

数据全部落 data/loveda/（.gitignore，不入仓库）。
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.common import DATA_DIR, SEED, ensure_dirs  # noqa: E402

RAW_DIR = os.path.join(DATA_DIR, "loveda_raw")
OUT_DIR = os.path.join(DATA_DIR, "loveda")
CHIP = 512
N_TEST_IMGS = 16
TRAIN_SHARDS = ["train-00000.parquet", "train-00001.parquet", "train-00002.parquet"]
VAL_SHARDS = ["validation-00000.parquet"]

CLASS_NAMES = ["", "background", "building", "road", "water",
               "barren", "forest", "agriculture"]  # index=类值 1..7


def extract_shard(path, split, img_dir, mask_dir):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    names = []
    for batch in pf.iter_batches(batch_size=8):
        imgs = batch.column("image").to_pylist()
        labels = batch.column("label").to_pylist()
        for rec_i, (im, lb) in enumerate(zip(imgs, labels)):
            name = os.path.splitext(os.path.basename(im["path"]))[0]
            ipath = os.path.join(img_dir, f"{name}.png")
            mpath = os.path.join(mask_dir, f"{name}.png")
            if not os.path.exists(ipath):          # 幂等：重跑跳过已抽取
                with open(ipath, "wb") as f:
                    f.write(im["bytes"])
            if not os.path.exists(mpath):
                with open(mpath, "wb") as f:
                    f.write(lb["bytes"])
            # 掩码值域断言（抽检每 batch 首个；LoveDA 官方编码 0=ignore + 1..7）
            if rec_i == 0:
                arr = np.asarray(Image.open(mpath))
                bad = set(np.unique(arr).tolist()) - set(range(0, 8))
                if bad:
                    raise ValueError(f"掩码值域越界 {bad} @ {mpath}")
                ignore_frac = float((arr == 0).mean())
                if ignore_frac > 0.5:
                    print(f"[prepare] note: {mpath} ignore 占比 {ignore_frac:.2f}", flush=True)
            names.append(name)
    return names


def cut_chips(img_path, mask_path, chip_prefix, out_img, out_mask):
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path)
    w, h = img.size
    assert (w, h) == (1024, 1024), f"LoveDA 应为 1024²，得到 {(w, h)}"
    n = 0
    for r in range(2):
        for c in range(2):
            box = (c * CHIP, r * CHIP, (c + 1) * CHIP, (r + 1) * CHIP)
            img.crop(box).save(os.path.join(out_img, f"{chip_prefix}_{r}{c}.png"),
                               compress_level=3)
            mask.crop(box).save(os.path.join(out_mask, f"{chip_prefix}_{r}{c}.png"),
                                compress_level=3)
            n += 1
    return n


def main():
    ensure_dirs()
    import pyarrow  # noqa: F401  确认依赖在位

    # --- 1) 抽取 ---
    for shards, split in ((TRAIN_SHARDS, "train"), (VAL_SHARDS, "val")):
        img_dir = os.path.join(OUT_DIR, split, "img")
        mask_dir = os.path.join(OUT_DIR, split, "mask")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        all_names = []
        for s in shards:
            p = os.path.join(RAW_DIR, s)
            if not os.path.exists(p):
                raise SystemExit(f"[prepare] 分片缺失：{p}（先完成下载）")
            names = extract_shard(p, split, img_dir, mask_dir)
            all_names += names
            print(f"[prepare] {s}: {len(names)} imgs (累计 {len(all_names)})", flush=True)
        print(f"[prepare] split={split}: {len(all_names)} 张 1024² 原图", flush=True)

    # --- 2) 训练切片（train 全量 4×512²）---
    t_img = os.path.join(OUT_DIR, "train512", "img")
    t_mask = os.path.join(OUT_DIR, "train512", "mask")
    os.makedirs(t_img, exist_ok=True)
    os.makedirs(t_mask, exist_ok=True)
    src_imgs = sorted(os.listdir(os.path.join(OUT_DIR, "train", "img")))
    n_chips = 0
    for f in src_imgs:
        stem = os.path.splitext(f)[0]
        n_chips += cut_chips(os.path.join(OUT_DIR, "train", "img", f),
                             os.path.join(OUT_DIR, "train", "mask", f),
                             f"train_{stem}", t_img, t_mask)
    print(f"[prepare] train512 chips: {n_chips}", flush=True)

    # --- 3) test 集（validation 固定种子抽 16 图 → 64 芯片）---
    rng = np.random.default_rng(SEED)
    val_imgs = sorted(os.listdir(os.path.join(OUT_DIR, "val", "img")))
    pick = sorted(rng.choice(len(val_imgs), size=min(N_TEST_IMGS, len(val_imgs)),
                             replace=False).tolist())
    s_img = os.path.join(OUT_DIR, "test512", "img")
    s_mask = os.path.join(OUT_DIR, "test512", "mask")
    os.makedirs(s_img, exist_ok=True)
    os.makedirs(s_mask, exist_ok=True)
    n_test = 0
    picked_names = []
    for i in pick:
        f = val_imgs[i]
        stem = os.path.splitext(f)[0]
        picked_names.append(stem)
        n_test += cut_chips(os.path.join(OUT_DIR, "val", "img", f),
                            os.path.join(OUT_DIR, "val", "mask", f),
                            f"test_{stem}", s_img, s_mask)
    print(f"[prepare] test imgs={len(picked_names)} chips={n_test} "
          f"(seed={SEED}, names={picked_names})", flush=True)

    # 记录清单（溯源）
    import json
    with open(os.path.join(DATA_DIR, "loveda_split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "chip": CHIP,
                   "train_shards": TRAIN_SHARDS, "val_shards": VAL_SHARDS,
                   "n_train_imgs": len(src_imgs), "n_train_chips": n_chips,
                   "n_test_imgs": len(picked_names), "n_test_chips": n_test,
                   "test_img_names": picked_names,
                   "classes": {str(i): CLASS_NAMES[i] for i in range(1, 8)}},
                  f, ensure_ascii=False, indent=1)
    print("[prepare] manifest → data/loveda_split_manifest.json", flush=True)


if __name__ == "__main__":
    main()
