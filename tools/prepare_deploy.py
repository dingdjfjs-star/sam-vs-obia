# -*- coding: utf-8 -*-
"""生成 GitHub Pages 部署拷贝 → ../sam-vsobia-site/（与 obia-segmentation 的 prepare_deploy 同惯例）。

收录：落地页/README/教程（文档+交互网页）/结果图/源码/测试/溯源账本(summary.json+jsonl)/质检报告/SAM输入GeoTIFF。
排除：大二进制（scene.npz 4MB、SAM 权重 176MB、labelmaps/classmaps ~12MB、prompt_windows.npz）、
日志、__pycache__、AGENTS.md（工作区内部档案，不入库——同 obia 主项目惯例）。
更新线上：改完东西重跑本脚本再走 github-pages-deploy 技能同仓名重发。
"""
from __future__ import annotations

import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEST = os.path.normpath(os.path.join(ROOT, "..", "sam-vsobia-site"))

COPY_FILES = [
    "index.html", ".nojekyll", ".gitignore", "README.md", "TUTORIAL.md",
    os.path.join("data", "scene_rgb.tif"),
    os.path.join("results", "summary.json"),
]
COPY_DIRS = [
    "tutorial", "scripts", "obiabridge", "test", "tools",
    os.path.join("results", "figs"),
    os.path.join("results", "raw"),
    "report",
]
EXCLUDE_IN_DIRS = {"__pycache__", "logs"}
EXCLUDE_EXT = {".pyc", ".npz", ".npy"}
EXCLUDE_NAMES = {"AGENTS.md", "setup_env.bat"}


def main():
    if os.path.isdir(DEST):
        shutil.rmtree(DEST)
    os.makedirs(DEST)

    for rel in COPY_FILES:
        src = os.path.join(ROOT, rel)
        dst = os.path.join(DEST, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    n = 0
    for d in COPY_DIRS:
        src_dir = os.path.join(ROOT, d)
        for dirpath, dirnames, filenames in os.walk(src_dir):
            dirnames[:] = [x for x in dirnames if x not in EXCLUDE_IN_DIRS]
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in EXCLUDE_EXT or fn in EXCLUDE_NAMES:
                    continue
                rel_in = os.path.relpath(os.path.join(dirpath, fn), ROOT)
                dst = os.path.join(DEST, rel_in)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(os.path.join(dirpath, fn), dst)
                n += 1

    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(DEST) for f in fs)
    print(f"[deploy-copy] {DEST}")
    print(f"[deploy-copy] files: {n} (+{len(COPY_FILES)} roots), total {total/1024/1024:.1f} MB")
    # 敏感文件自查
    for dp, _, fs in os.walk(DEST):
        for f in fs:
            if f.lower() in (".env",) or f.endswith((".key", ".pem")):
                print("[deploy-copy] !!! SENSITIVE:", os.path.join(dp, f))


if __name__ == "__main__":
    main()
