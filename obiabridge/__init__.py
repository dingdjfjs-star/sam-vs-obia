# -*- coding: utf-8 -*-
"""obiabridge —— 把姊妹项目 obia-segmentation 的代码原地接入本项目（零拷贝）。

设计动机
--------
对比实验最大的公平性风险是"口径漂移"：如果本项目复制一份 metrics.py，
两边的指标实现会随时间各自演化，数字失去可比性。因此本包不做任何复制，
只在 sys.path 上挂载 ../obia-segmentation，让 `obia` 原包直接可用：

    import obiabridge                      # 挂载 sys.path（import 即生效）
    from obia.metrics import boundary_f    # 与 obia-segmentation 完全同一份实现

要求 obia-segmentation 与本项目同级目录（E:\\zcode\\探索前沿\\obia-segmentation）。
"""
from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OBIA_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "obia-segmentation"))

if not os.path.isdir(_OBIA_ROOT):
    raise ImportError(
        f"obia-segmentation 不在预期位置: {_OBIA_ROOT}\n"
        "obiabridge 要求两项目同级（探索前沿/obia-segmentation 与 探索前沿/sam-vs-obia）。"
    )

if _OBIA_ROOT not in sys.path:
    sys.path.insert(0, _OBIA_ROOT)

__all__ = ["OBIA_ROOT"]
OBIA_ROOT = _OBIA_ROOT
