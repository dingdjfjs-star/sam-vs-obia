# -*- coding: utf-8 -*-
"""诊断 UNet B3 失配：重算 vs 记录。"""
import json
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, r"E:\zcode\探索前沿\sam-vs-obia")
import obiabridge  # noqa: F401
from obia.metrics import confusion_oa

base = r"E:\zcode\探索前沿\sam-vs-obia"
runs = {}
for line in open(base + r"\results\raw\v2_runs.jsonl", encoding="utf-8"):
    if line.strip():
        r = json.loads(line)
        runs[r["name"]] = r

for stem in ("test_2541_00", "test_2541_01"):
    truth = np.asarray(Image.open(base + r"\data\loveda\test512\mask" + "\\" + stem + ".png"), np.int32)
    pred = np.load(base + r"\results\v2\preds\unet" + "\\" + stem + ".npy")
    cm, oa = confusion_oa(pred, truth, 7)
    rec = runs["v2_unet_" + stem]
    print(f"{stem}: recomputed={oa:.6f} record={rec['map_oa']:.6f} "
          f"pred_uniq={np.unique(pred).tolist()} truth_uniq={np.unique(truth).tolist()} "
          f"n_obj_rec={rec['n_objects']}")
