# 当 Meta 的 SAM 遇上手写 OBIA：遥感分割的零样本正面对比

> **同一幅 512²×4 波段合成场景（seed=20260829）、同一套下游分类协议、同一套指标代码**——
> 让 segment-geospatial（Meta SAM2，零样本、只用真彩色 RGB）与
> [obia-segmentation](https://github.com/dingdjfjs-star/obia-segmentation)（手写 Baatz & Schäpe 多尺度区域生长，4 波段全谱+19 维特征）正面对比。
> 本 README 所有数字出自 2026-11-20 本次实跑（`results/summary.json` / `results/raw/*.jsonl` 可溯源，`test/test.py` 8 项断言全绿复算一致）。

**一句话结论**：SAM2 零样本分割**赢在边界与速度**（边界贴合 F1 最高 0.95 vs OBIA 0.48，GPU 3.2 s vs CPU 78 s），**输在"开箱即用的分类精度"与尺度可控性**（OA 0.924 vs OBIA 0.959，且默认阈值会把 23% 的场景切成"无人区"）；提示工程能把同一图斑的 IoU 从自动模式的 0.376 拉到点提示 0.602、框提示 0.870——**遥感零样本工作流里，一次点击不够，一个框刚好**。

---

## 1. 实验设计

| 要素 | 设置 |
|---|---|
| 场景 | obia-segmentation 自带合成场景：512²×4 波段（B/G/R/NIR），5 类地物 × 70 个 Voronoi 图斑，自带像素级真值与图斑真值（`obiabridge` 原地复用 `obia.synthetic`，同 seed 逐字节一致，测试 A2 验证） |
| SAM 侧输入 | **只有真彩色 RGB GeoTIFF**（2% 线性拉伸）——零样本的真实约束：没有 NIR、没有热红外 |
| OBIA 侧输入 | 4 波段 float + 19 维光谱/纹理/形状特征 |
| SAM 模型 | SAM2.0 hiera-small（`facebook/sam2-hiera-small`，sha256 `95949964…` 与 HF 逐字节一致，经 ModelScope 镜像下载，RTX 4060 Laptop 8GB / torch 2.6.0+cu124） |
| 公平协议 | SAM 的 AMG mask 涂写为非重叠 label 图后，走与 SLIC 对照**完全相同**的下游：19 维图斑特征 → 同种子 RandomForest → 回填分类图 → 三指标（`scripts/common.py::eval_labelmap`，逐行沿用 obia 的 `baselines.py` 协议） |
| 指标 | 全图 OA；椒盐率（3×3 多数类不一致像素占比）；边界贴合 F1（tol=2px 双向）分两口径：**分类图 vs 地类真值**（cls_bF1）与 **分割 label 图 vs 图斑真值**（seg_bF1）——口径区分是本系列项目的核心纪律 |
| 对照 | 手写 MRS（scale=10/shape=0.2/cpt=0.5）、skimage SLIC（n=440）、SAM2 AMG 默认阈值、SAM2 AMG 放宽阈值 |

## 2. 主结果：四方同台

| 指标 | **手写 OBIA**（scale=10） | SLIC（n=440） | **SAM2 默认 AMG**（pps=32） | **SAM2 放宽阈值**（iou≥0.6/stab≥0.85） |
|---|---|---|---|---|
| 对象数 | 440 | 441 | 28 | 47 |
| AMG 未覆盖区占比 | —（全图分割） | — | **23.2%** | **1.4%** |
| 全图 OA | **0.9588** | 0.9452 | 0.9240 | 0.8230 |
| 椒盐率（越低越净） | 0.0018 | 0.0002 | 0.0005 | 0.0005 |
| 边界 F1 vs 地类（cls_bF1） | 0.4812 | 0.2851 | 0.6241 | **0.9505** |
| 边界 F1 vs 图斑（seg_bF1） | 0.2508 | 0.2021 | 0.3531 | **0.4853** |
| 分割耗时 | 77.9 s（CPU） | 0.2 s（CPU） | **3.2 s（GPU）** | 3.6 s（GPU） |
| 产物形态 | 矢量图斑+栅格，特征可解释 | 栅格 | mask 栅格 | mask 栅格 |

![fig4](results/figs/fig4_metrics_bar.png)
![fig1](results/figs/fig1_segmentation_compare.png)
![fig2](results/figs/fig2_classification_compare.png)

### 2.1 SAM 赢在哪

1. **边界感知碾压**：seg_bF1 0.353 vs OBIA 0.251；阈值放宽后 cls_bF1 直达 **0.9505**（OBIA 的 2 倍）。SAM 在自然图像上学到的边缘先验对"图斑交界 1px 压暗壕沟"极其敏感——它切出来的段**边缘干净地贴住田块线**（fig1 右下）。
2. **速度**：GPU 上 3.2 s vs 手写 MRS 77.9 s（24×）。AMG 是一次编码器前向 + 批量解码；OBIA 的合并循环是 26 万次贪心合并。
3. **椒盐率**：0.0005 < OBIA 0.0018——大 mask 天然无碎斑（但 SLIC 0.0002 仍是最低，超像素的本职）。

### 2.2 SAM 输在哪

1. **开箱 OA**：0.9240 < OBIA 0.9588。两个原因：① AMG 的 mask 是"视觉同质区"，**跨地类合并**常见（同类相邻田块被并成一段，一段错分即损失大片 OA，fig2 下排误差图的红块 vs OBIA 的红边）；② 只有 28 个对象，RF 训练集（60% 划分）仅 17 个样本，统计上先天不足。阈值放宽后对象变多反而 OA 掉到 0.823——**大段跨界吞并加剧**（47 段覆盖全图，段均 5600px，错一段伤一片）。
2. **默认阈值留下 23.2% 无人区**：SAM2 官方默认 AMG（pred_iou≥0.8, stability≥0.95）把"模型不确定"的区域全部过滤——对自然照片合理，对遥感是致命的覆盖缺口（fig1 右下角的灰色空洞）。补上这些像素只能靠最近邻填充（本实验协议）或后处理，**零样本直接用会有大片漏分**。
3. **尺度不可控**：pps 从 8 扫到 64，对象数只从 10 涨到 58（OBIA scale 5→160 是 925→21）——**AMG 没有"对象≈目标图斑"的尺度语义**，pps 是提示密度不是异质性预算。要 440 个图斑级别的粒度，SAM 给不了。
4. **无光谱语义**：看不到 NIR， NDVI/NDWI/纹理这些 OBIA 的看家特征全部缺位；水域的 NIR≈0 在 RGB 里只是"比较暗"。

## 3. 对象数量-尺度响应

![fig3](results/figs/fig3_scale_response.png)

- **OBIA**：对象数随 scale 单调锐减（925→21），seg_bF1 在 scale=10 峰值 0.2508 后缓慢下降——与 obia-segmentation README 的敏感性结论一致（本次复跑数字逐档吻合）。
- **SAM2**：对象数随 pps 近线性缓增（10→58），seg_bF1 在 pps=16~32 达峰（0.33~0.35）；pps 增大主要**加密度不加边界质量**。
- **右图（边界质量 vs 粒度）是本实验最锋利的一张图**：SAM 曲线整体压在 OBIA 上方——同样甚至更少的对象数下，SAM 的段边界质量高 30%~50%。**SAM 用 1/15 的对象数做到了 1.4 倍的边界 F**。代价是粒度不可调、且未覆盖区要靠阈值牺牲来换。

## 4. 提示工程的价值（18 个分层抽样图斑）

| 模式 | IoU 均值 | IoU 中位数 | 覆盖率 | 越界率(spill) | 窗口边界 F | 单次耗时 |
|---|---|---|---|---|---|---|
| 自动模式（AMG 质心段） | 0.376 | — | — | — | — | — |
| **点提示**（质心 1 点） | 0.602 | 0.512 | 0.921 | **0.324** | 0.564 | 16 ms |
| **框提示**（图斑 bbox） | **0.870** | **0.942** | 0.954 | 0.089 | 0.744 | 14 ms |

![fig5](results/figs/fig5_prompts.png)

- **三级阶梯**：自动 0.376 → 点 +0.23 → 框 +0.49。提示不是锦上添花，是**从"不可用"到"可用"的开关**。
- **点的失败模式一眼可见**：覆盖率 0.921 高但越界率 0.324——点告诉 SAM"从哪开始"，没告诉它"到哪停"，mask 顺着同类光谱一路漫溢到邻斑（fig5 上排右：白点在图斑内，mask 边界冲出红色真值框一大圈）。框提示把溢出压回 0.089。
- **地类差异**：点提示下 water/farmland 0.86/0.96 很好，soil 0.47、**builtup 仅 0.26**（8px 块状纹理诱导 SAM 碎切+外溢）；框提示全面抬升（builtup 0.26→0.68），forest 0.83。**纹理越乱、越需要框**。
- 成本：14~16 ms/次（编码器一次前向后只跑解码器）。逐图斑框提示全图 70 斑 < 1 s——**"人工框一秒一个"换 +0.49 IoU**，是遥感零样本流程里性价比最高的一步。

## 5. 结论：什么条件下选谁

| 场景 | 推荐 | 依据 |
|---|---|---|
| 要**矢量图斑 + 可解释特征 + 高 OA 自动分类**（国土调查/变化检测入库） | **OBIA** | OA 0.959、NDVI/纹理/形状可解释、产物直接入库；SAM 给不了特征语义 |
| 要**贴边界的快速勾绘**（制图前勾边、大区块提取、样本预标注） | **SAM2** | 边界 F 高 1.4~2 倍、快 24 倍、零样本免调参 |
| **小样本/无标注**的新区域冷启动 | SAM2 + 框提示 | 框提示 IoU 0.87 无需任何训练；OBIA 的 RF 仍需标注样本 |
| **对象粒度可控**（"对象≈图斑"的尺度语义） | **OBIA** | scale 参数就是异质性预算；AMG 的 pps 不是尺度 |
| **CPU-only 环境** | OBIA/SLIC | SAM2 需 GPU；OBIA 纯 CPU 78 s 也可接受 |
| 需要全图 100% 覆盖的硬约束 | OBIA | SAM 默认阈值漏 23%，放宽阈值要重新标定且掉 OA |

**遥感场景下提示工程的价值一句话**：SAM2 的预训练先验在遥感上**边界极准、语义不全**——框提示是把"边界准"变现成"分割可用"的最小代价动作；而要把分割变成"分类可用"，还得靠 OBIA 那套光谱-纹理-形状特征接力。

## 6. 复现

```bat
:: 环境（一次性）—— 见 AGENTS.md「复现命令」节；权重经 ModelScope 镜像
::   data/weights/sam2_hiera_small.pt  sha256=95949964d4e548409021d47b22712d5f1abf2564cc0c3c765ba599a24ac7dce3
::   （与 HuggingFace facebook/sam2-hiera-small 的 LFS blob 逐字节一致；hf-mirror 对 Xet 仓库文件通道 502，故走 ModelScope）
E:\Ana\python.exe scripts\make_scene.py            :: 场景（obia.synthetic 同 seed 生成）
E:\Ana\python.exe scripts\run_obia.py              :: OBIA 6 档 scale 扫描 + SLIC（~9 min CPU）
E:\Ana\envs\samgeo\python.exe scripts\run_sam_auto.py    :: SAM AMG 5 档 pps + 放宽阈值档（~3 min GPU）
E:\Ana\envs\samgeo\python.exe scripts\run_sam_prompts.py :: 点/框提示实验（18 图斑，~1 min）
E:\Ana\envs\samgeo\python.exe scripts\build_summary.py   :: 汇总 summary.json
E:\Ana\envs\samgeo\python.exe scripts\make_figures.py    :: 5 张图
E:\Ana\envs\samgeo\python.exe test\test.py               :: 8 项断言自检（秒级）
```

测试内容：场景 bit-exact 复现、四方主结果的 map_oa/椒盐率/两口径边界 F 从落盘 label 图**复算到 1e-9**、summary.json 往返一致、runs.jsonl 全行可解析、提示记录健全。

## 7. 诚实声明（协议与局限）

- **SAM mask 涂写**：AMG mask 按 predicted_iou 降序"先到先得"涂写，重叠区归高分者；未覆盖像素用最近邻距离变换并入相邻段（默认档 23.2% 像素经历此过程，是协议的一部分，已按 run 记录 `uncovered_frac`）。
- **SAM 只见 RGB**：这是零样本真实约束而非缺陷修复——给 NIR 通道（伪彩色合成）是否显著提升是后续实验。
- **单场景单种子**：结论绑定 seed=20260829 的 512² 场景；跨场景泛化需扩充场景库（合成场景生成器已参数化）。
- **RF 小样本效应**：SAM 默认档仅 28 对象，其 OA 的方差大；对比结论（边界赢/精度输）方向稳健，具体数值不宜外推。
- **耗时口径**：SAM 计 generate() 全程含编码器；OBIA 计合并循环全程（`multiresolution_segment` 内部计时）；提示计单次 predict（编码器已在 set_image 完成）。硬件 RTX 4060 Laptop 8GB。
- **fig2 下排 SAM 分类图的条纹**是 AMG mask 行状涂写的真实产物，非渲染 bug——恰可视化解说了"mask 并非为制图而生"。

## 8. 文件与结构

```
obiabridge/            sys.path 桥接 ../obia-segmentation（指标/特征/分类零拷贝复用）
scripts/common.py      统一下游评估协议 eval_labelmap（公平性核心）
scripts/sam_adapter.py SAM2 接入层（ModelScope 权重 + sam2 原生 API，含原因说明）
scripts/run_obia.py · run_sam_auto.py · run_sam_prompts.py · build_summary.py · make_figures.py
test/test.py           8 项断言自检
results/summary.json   本次实跑全部数字（唯一数据源）
results/raw/*.jsonl    逐 run 溯源账本
results/figs/          fig1~fig5
AGENTS.md              项目详档（环境/命令/踩坑）
```

姊妹项目：[obia-segmentation](https://github.com/dingdjfjs-star/obia-segmentation)（手写 OBIA 全流程）· 同系列对比口径。评测工具链：segment-geospatial 1.4.2 / sam2 (PyPI) / torch 2.6.0+cu124 / rasterio / scikit-image / scikit-learn。
