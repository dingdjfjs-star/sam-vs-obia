# -*- coding: utf-8 -*-
"""提取 v2 报告所需的精确数字（唯一来源 metrics.json + runs.jsonl）。"""
import json

M = json.load(open(r"E:\zcode\探索前沿\sam-vs-obia\results\metrics.json", encoding="utf-8"))
for m in ("obia", "sam_default", "sam_permissive", "unet"):
    s = M["metrics"][m]
    print(f"[{m}]")
    for k in ("map_oa", "speckle", "cls_boundary_f1", "cls_boundary_f1_multiclass",
              "seg_boundary_f1", "seg_seconds", "n_objects"):
        blk = s[k]
        line = f"  {k}: mean={blk['mean']:.4f} std={blk['std']:.4f} median={blk.get('median', float('nan')):.4f} n={blk['n']}"
        print(line)
    print(f"  n_degenerate={s['n_degenerate_chips']} n_single_class={s['n_single_class_chips']}")
    if "refined_map_oa" in s:
        print(f"  refined_map_oa: mean={s['refined_map_oa']['mean']:.4f} "
              f"median={s['refined_map_oa']['median']:.4f}")
print("\nobia calibration:", M["obia"]["main_scale"], M["obia"]["calibration_image"])
print("unet:", M["unet"]["checkpoint_sha256_16"], "epoch", M["unet"]["epoch"],
      "monitor_oa", round(M["unet"]["monitor_oa"], 4))
print("generated_at:", M["generated_at"])
