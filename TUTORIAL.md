# TUTORIAL —— sam-vs-obia 从设计逻辑到逐段精讲

> 读者画像：三个月后的你自己 + 复试/面试官。所有数字可回查 `results/summary.json`（2026-11-20 实跑快照），所有代码摘录自真实源文件并标注 `文件:行号`。交互版教程（可玩的指标计算器/AMG 涂写演示）见 `tutorial/index.html`，file:// 双击直开。

## 0. 这个项目是什么

**一句话**：把 Meta 的 SAM2（零样本，只见真彩色 RGB）和自己的手写 OBIA（4 波段全谱 + 19 维特征 + Baatz&Schäpe 区域生长）放进**同一幅合成场景、同一套下游分类器、同一套指标代码**里打一架，回答三个问题：SAM 赢在哪、输在哪、提示值多少钱。

**成品三看**：① `results/figs/fig3_scale_response.png` 右图（SAM 边界质量曲线整体压在 OBIA 上方，最锋利的一张）；② README §4 的提示三级阶梯表（auto 0.376 → 点 0.602 → 框 0.870）；③ `tutorial/index.html` 的边界 F 计算器（拖滑块亲眼看容差如何改写 P/R/F）。

**30 秒电梯陈述**：遥感分割有两条路线——传统 OBIA 先分后分（异质性准则切图斑、再提特征分类）和大模型 SAM 直接零样本出 mask。但文献里很少有人在**受控条件下**正面对比过。我造了一个自带像素级真值的合成场景，用 sys.path 桥接让两个项目共享同一份指标实现（杜绝口径漂移），SAM 的 mask 涂写成 label 图后走与 SLIC 基线完全相同的"特征+RF"下游。结论：SAM 边界 F 最高 0.95（OBIA 0.48）、快 24 倍，但开箱 OA 低 3.5 个点、默认阈值漏掉 23% 场景、尺度不可控；框提示一秒一个能把图斑 IoU 从 0.38 拉到 0.87。

## 1. 设计逻辑

### 1.1 模块划分

| 模块 | 职责 | 为什么独立存在 |
|---|---|---|
| `obiabridge/` | 把 `../obia-segmentation` 挂上 sys.path | **指标零拷贝**——复制一份 metrics.py 迟早各自演化，数字失去可比性 |
| `scripts/common.py` | 场景读取、label 规整、**统一下游协议 eval_labelmap** | OBIA/SLIC/SAM 三方产物都要过同一道协议门，公平性集中在一处 |
| `scripts/make_scene.py` | 生成场景 npz + SAM 用的真彩色 GeoTIFF | SAM 只许看 RGB——零样本约束要显式做成"输入差异" |
| `scripts/sam_adapter.py` | SAM2 全部调用隔离层 | 第三方 API 会变， adapters 换库只改一个文件 |
| `scripts/run_obia.py` | MRS scale 扫描 + SLIC 对照 | OBIA 侧数据生产者 |
| `scripts/run_sam_auto.py` | AMG pps 扫描 + 阈值对照档 | SAM 侧数据生产者 |
| `scripts/run_sam_prompts.py` | 点/框提示实验 | 独立问题独立脚本（提示价值） |
| `scripts/build_summary.py` | 汇总 → summary.json | 报告唯一数据源，jsonl 可溯源 |
| `scripts/make_figures.py` | 5 张图 | 图只读 summary/labelmaps，不藏数字 |
| `test/test.py` | 8 断言自检 | 不跑模型秒级复算，防止"数字从此没人管" |

```
obia.synthetic(seed=20260829) ──► scene.npz ──┬─► run_obia.py ──► labelmaps + runs.jsonl
                                              └─► scene_rgb.tif ──► run_sam_auto.py ──┐
                                                    sam2(AMG) ───────────────────────┤
                                                    sam2(predict) ─► run_sam_prompts.py ─► prompts.jsonl
                                                                                      ▼
                                     labelmaps ─► eval_labelmap（19 维特征→同种子RF→回填→三指标）─► runs.jsonl
                                                                                      ▼
                                                  build_summary.py ─► summary.json ─► make_figures.py / README / tutorial
```

### 1.2 五个关键决策点（为什么选 A 不选 B）

1. **为什么桥接不拷贝**：对比实验最大的暗坑是"两边指标实现不一样"。`obiabridge/__init__.py` 只做 sys.path.insert，测试 A1 断言 `boundary_f is 同一函数对象`。不这样做的后果：某天一边改了容差或膨胀方向，所有对比数字作废且无人察觉。
2. **为什么 SAM 也要过 RF 下游**（而不是只比分割 IoU）：SAM 不输出类别。若给 SAM 用"多数票真值"当分类（oracle），OA 会虚高不可比。选择：mask→label 图→**同一套 19 维特征 + 同种子 RF**——这是 obia 主项目给 SLIC 定下的对照协议，三方同门同卷。代价：SAM 档的对象数少（28/47），RF 训练集小、方差大——这个代价本身写进了结论（§7 局限）。
3. **为什么直用 sam2 API 不硬依赖 samgeo.SamGeo2**：samgeo 1.4.2 的 `SamGeo2` 内部就是 `SAM2AutomaticMaskGenerator.from_pretrained` → `hf_hub_download`；实测 hf-mirror 对 Xet 仓库文件通道 502（权重 0 字节挂起）。改从 ModelScope 下载**同一文件**（sha256 与 HF LFS blob 一致），构建走 sam2 原生 API——与 samgeo 内部完全同类同参，只换掉坏掉的下载通道（`scripts/sam_adapter.py` 模块注释记录了完整决策链）。
4. **为什么加"放宽阈值档"**：默认 AMG（pred_iou≥0.8/stab≥0.95）在本场景留下 23.2% 无人区——只报这一个数会把"SAM 漏分"和"SAM 阈值保守"混为一谈。加一档 0.6/0.85 对照，才能拆出"阈值决定覆盖 vs 覆盖换质量"的机理（放宽后未覆盖 1.4%、cls_bF1 0.95、但 OA 掉到 0.82）。
5. **为什么全链路落盘 jsonl**：用户纪律"所有数字只出自本次实际运行输出"——runs.jsonl 逐 run 追加、同名后写覆盖（重跑幂等），summary.json 由它聚合，README/教程只许引用 summary。数字有了账本，才能被 test.py 复算到 1e-9。

## 2. 代码骨架

```
sam-vs-obia/
├── obiabridge/__init__.py   sys.path 桥接（import 即生效，带位置校验）
├── scripts/
│   ├── common.py            路径常量/场景/label 规整/eval_labelmap 统一协议/jsonl 账本
│   ├── make_scene.py        场景 npz + 真彩色 GeoTIFF（2% 拉伸）
│   ├── sam_adapter.py       HF_ENDPOINT 说明 + sam2 构建 + AMG/predict 两工厂
│   ├── run_obia.py          MRS 6 档扫描 + SLIC，逐档 eval+落盘
│   ├── run_sam_auto.py      AMG 5 档 pps + 放宽阈值档
│   ├── run_sam_prompts.py   18 图斑分层抽样 × {点,框}，IoU/覆盖/越界/窗口边界F
│   ├── build_summary.py     runs.jsonl+prompts.jsonl → summary.json（含 auto-containment）
│   ├── make_figures.py      fig1~fig5（matplotlib Agg，只读落盘数据）
│   └── smoke_protocol.py    64 超像素协议冒烟
├── tools/                   probe_net.py / check_samgeo_*.py / export_tutorial_data.py
├── data/                    scene.npz、scene_rgb.tif、weights/sam2_hiera_small.pt
├── results/                 summary.json / raw\*.jsonl / labelmaps / classmaps / figs / logs
├── test/test.py             8 断言（A1~A6）
└── report/质检报告.md       12 处踩坑 + 跨项目对账
```

核心数据结构——`eval_labelmap` 返回的记录（每方一条，进 jsonl）：

```python
{
  "name": "sam_pps32",            # 方法名（同名重跑覆盖）
  "n_objects": 28,                # 涂写后非重叠对象数（与 OBIA 可比的口径）
  "uncovered_frac": 0.2319,       # AMG 未覆盖像素占比（OBIA/SLIC 恒 0）
  "map_oa": 0.9240,               # 全图 OA（分类图 vs 像素真值）
  "obj_oa": ...,                  # 对象级测试划分 OA（RF 内部）
  "speckle": 0.0005,              # 椒盐率（3×3 多数类不一致占比）
  "cls_boundary_f1": 0.6241,      # 分类图边界 vs 地类真值边界（tol=2px）
  "seg_boundary_f1": 0.3531,      # label 图边界 vs 图斑真值边界（tol=2px）
  "classify_seconds": ...,        # 下游 RF 耗时
  "seg_seconds": 3.2,             # 分割本身耗时（SAM=generate 全程）
}
```

## 3. 核心代码精讲（4 段）

### 3.1 eval_labelmap —— 公平性的一道门

**问题**：三方产物形态不同（MRS 连续 label 图 / SLIC label 图 / AMG 带 0 缝隙的重叠 mask 集），如何保证"19 维特征→RF→三指标"逐行同源？
**思路**：把协议做成函数，输入只认"任意 label 图"：规整 → 缝隙填充 → 特征 → 同种子 RF → 回填 → 指标。谁想进对比表，谁先变成 label 图。

伪代码：
```
eval(label):
    lab = 规整为连续 1..K
    lab[0] = 最近邻填充            # AMG 缝隙
    feats = extract_object_features(img, lab)     # ← obia 原实现
    y = object_majority_truth(lab, truth)
    pred = RandomForest(seed=20260829).fit(60%y).predict(feats)
    obj_map = 按 lab 回填 pred
    return {OA, 椒盐率, cls_bF1(obj_map, truth), seg_bF1(lab, parcels), ...}
```

真实代码（`scripts/common.py:81`，`scripts/common.py:106`）：

```python
def eval_labelmap(image, truth, parcels, labels, seed=SEED, tol_px=TOL_PX,
                  name="", seg_seconds=None, return_maps=False):
    ...
    img = np.asarray(image, np.float32)
    labels0 = np.asarray(labels)
    uncovered = float((labels0 == 0).mean())
    lab = fill_uncovered(labels0)

    feats = extract_object_features(img, lab)          # obia.features 原实现
    obj_truth = object_majority_truth(lab, truth)
    res = classify_objects(feats, obj_truth, seed=seed)  # obia.classify 原实现，同种子
    obj_map = rasterize_object_pred(lab, res["pred"])
    cm, map_oa = confusion_oa(obj_map, truth, int(truth.max()))

    cls_bf = boundary_f(obj_map, truth, tol=tol_px)     # 分类图 vs 地类边界
    seg_bf = boundary_f(lab, parcels, tol=tol_px)       # label 图 vs 图斑边界
```

**坑**：`extract_object_features` 要求 label 连续 1..K（内部 `bincount[1:]` 按编号索引），SAM 涂写图必须先过 `fill_uncovered`+`relabel_continuous`，否则特征表错位——错位不报错、只污染数字，最阴险。对应踩坑 #8。

### 3.2 masks_to_labelmap —— 把 AMG 变成"分割图"

**问题**：AMG 返回数百个**互相重叠**的 mask（每个网格点一个候选），下游协议要的是**非重叠** label 图。
**思路**：按 predicted_iou 降序"先到先得"涂写：每个像素归第一个（最高分）盖住它的 mask；没被任何 mask 盖住 → 0（未覆盖，如实记录占比）。

真实代码（`scripts/run_sam_auto.py:29`）：

```python
def masks_to_labelmap(masks):
    h, w = np.asarray(masks[0]["segmentation"]).shape
    lab = np.zeros((h, w), np.int32)
    n_raw = len(masks)
    k = 0
    for m in sorted(masks, key=lambda x: x.get("predicted_iou", 0.0), reverse=True):
        seg = np.asarray(m["segmentation"], bool)
        free = seg & (lab == 0)
        if free.any():
            k += 1
            lab[free] = k
    return lab, {"n_masks_raw": n_raw, "n_painted": k}
```

**坑**：① sam2 AMG 不保证按分数返回（踩坑 #6），不排序则结果依赖库内实现；② `n_masks_raw=75 → n_painted=28`——一半 mask 在涂写时已无空地，报告里**只有 n_objects 与 OBIA 可比**，两个数都入账。③ 涂写顺序造成 fig2 下排的行状条纹——不是 bug，是 AMG"逐点提案"本性的可视化（README §7 特意声明）。

### 3.3 提示实验的三件套（抽样 / 窗口 / 指标）

**问题**：提示价值要可量化：同一批图斑上，自动模式、点、框各拿多少 IoU？
**思路**：分层抽样（每类最多 4 个、面积 ≥150px、纯度 ≥80%）固定 seed；点=质心像素、框=图斑 bbox；指标除 IoU 外，专门拆出 **coverage**（盖住图斑多少）和 **spill**（溢出到图斑外多少）——点提示的失败模式（"知道从哪开始、不知道到哪停"）就藏在这两个数的反差里。

真实代码（`scripts/run_sam_prompts.py:76`）：

```python
def prompt_metrics(mask, pmask, tol=2):
    inter = float((mask & pmask).sum())
    union = float((mask | pmask).sum())
    iou = inter / max(union, 1.0)
    coverage = inter / max(float(pmask.sum()), 1.0)
    spill = float((mask & ~pmask).sum()) / max(float(mask.sum()), 1.0)
    pred_map = mask.astype(np.int32)
    truth_map = pmask.astype(np.int32)
    bf = boundary_f(pred_map, truth_map, tol=tol)["f1"]
    return {"iou": iou, "coverage": coverage, "spill": spill, "window_bf1": bf}
```

**坑**：边界 F 若在全图算，40 万像素的真值边界会把窗口内 200px 的提示边界淹没——必须先裁到"图斑 bbox 外扩 24px"窗口（`parcel_window`，`scripts/run_sam_prompts.py:61`）再算。实测数据：点 coverage=0.921 但 spill=0.324，框把 spill 压到 0.089。

### 3.4 auto_containment —— 给提示实验装"反事实基线"

**问题**：0.870 的框提示 IoU 算好还是不好？没有"不开提示"的基线就没有"价值"可言。
**思路**：对**同一批抽样图斑**，回到已落盘的 AMG(pps32) label 图，取覆盖图斑质心的那个分割段算 IoU——零成本拿到"如果不开提示，SAM 会把这个图斑切什么样"。

真实代码（`scripts/build_summary.py:27`）：

```python
def auto_containment(sam_lab, parcels, truth, records, seed=SEED):
    from scripts.run_sam_prompts import sample_parcels
    picks = sample_parcels(parcels, truth, seed=seed)   # 与提示实验同批图斑
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
```

结果：auto 0.376 → 点 0.602（+0.23）→ 框 0.870（+0.49）。**坑**：质心可能落在未覆盖区（seg_id=0，默认档 23% 概率）——记 0.0 而不是跳过，否则基线被静默抬高。

## 4. 从零复刻路线

| 里程碑 | 做什么 | 怎么验证 | 产出 |
|---|---|---|---|
| M1 环境 | conda env `samgeo`（py311 + torch cu124 + `segment-geospatial[samgeo2]` + geo 栈）；ModelScope 下权重并核 sha256 | `torch.cuda.is_available()` 为 True；sha256 = `95949964…` | 可用 GPU 环境 |
| M2 场景+桥接 | make_scene.py；写 obiabridge | smoke_protocol.py 输出 PROTOCOL OK；`parcels.max()==70` | scene.npz / scene_rgb.tif |
| M3 OBIA 侧 | run_obia.py（~9 min） | scale10=440 对象、SLIC OA=0.9452，与 obia README 对账 | labelmaps + runs.jsonl |
| M4 SAM 侧 | run_sam_auto.py + run_sam_prompts.py（GPU ~4 min） | 两次运行 pps 各档数字完全一致（确定性）；uncovered 有记录 | labelmaps + prompts.jsonl |
| M5 汇总出图测试 | build_summary → make_figures → test | test.py 打印 ALL 8 ASSERTIONS PASSED | summary.json + 5 图 + 绿测试 |

## 5. 踩坑实录（编号与 report/质检报告.md 一致）

- **#1 假成功**：工作区 Bash 工具非真 bash，复合命令被肢解误执行，"环境装好"是回显碎片。→ 教训：每条调用一条简单命令；"成功"必须独立命令实证。
- **#2 中文路径 cmd 静默失败**：`cmd //c` + 含中文的 .bat 只弹提示符不执行、exit 0。→ bat 内容纯 ASCII + `%~dp0`；或干脆不用 bat。
- **#3 依赖藏在 extra**：`segment-geospatial` 不含 `sam2`，装完照样 `ModuleNotFoundError`。→ `pip install "segment-geospatial[samgeo2]"`；装完必须 `import sam2` 实测。
- **#4 HF 镜像的半程可用**：hf-mirror API 元数据通、Xet 文件通道 502（0 字节挂起）。→ ModelScope 同文件下载 + sha256 对账 HF blob 指纹。
- **#5 返回二元组**：`multiresolution_segment` 返回 `(labels, info)`，只看签名想当然 → inhomogeneous shape。→ 读 Returns；info 里的官方计时顺手入账。
- **#6 mask 顺序不确定**：AMG 返回顺序不保证按分数。→ 涂写前显式按 predicted_iou 降序。
- **#7 口径漂移**：复制指标代码是对比实验的慢性病。→ sys.path 桥接 + 测试断言同一函数对象。
- **#8 label 必须连续 1..K**：特征管线按编号索引，0 号缝隙/断号不报错只污染数字。→ 统一 `fill_uncovered`+`relabel_continuous`。
- **#9~#12 出图层键位错**：fig2 索引越界 / RGBA 通道数 / summary 嵌套层级 / OBIA 记录缺 scale 键。→ 出图脚本只能引用 summary 的已知 schema；改 schema 必须同步改图。

## 6. 扩展练习

1. **入门**：把 AMG 阈值再放宽一档（`pred_iou_thresh=0.4, stability_score_thresh=0.7`，改 `scripts/run_sam_auto.py` 的 permissive 调用），预测未覆盖率和 OA 怎么变，跑完对照你的预测。（验证：`runs.jsonl` 新增记录、`test.py` 仍全绿）
2. **进阶**：给 SAM 喂伪彩色（NIR,R,G）而非真彩色——改 `make_scene.py` 加 false-color GeoTIFF、`sam_adapter.py` 换输入路径，看 forest/farmland 的 seg_bF1 是否提升（NIR 让植被边界更锐）。（验证：新 label 图 + 对比表更新）
3. **挑战**：把 SLIC 从"固定 n_segments=440"改成"与 SAM pps 档对象数对齐"的扫描（32/47/441 三档），画进 fig3 第三面板——回答"超像素在同等粒度下输 SAM 多少"。（验证：fig3 重生成、三点落在两条曲线之间还是之外）

## 7. 面试/复试话术

**30 秒版**：我做了 SAM2 零样本分割和手写 OBIA（eCognition 多分辨率分割复现）的受控对比：同一合成场景、同一套指标代码（sys.path 桥接防口径漂移）、SAM 的 mask 走和 SLIC 基线完全相同的特征+随机森林下游。发现 SAM 边界 F 最高 0.95（OBIA 0.48）且快 24 倍，但开箱 OA 低 3.5 个点、默认阈值漏 23% 场景、尺度不可控；框提示能把图斑 IoU 从 0.38 提到 0.87，一次 14 毫秒。

**3 分钟版骨架**：① 动机：两条路线缺受控对比 → ② 设计：合成场景自带两级真值 + 桥接零拷贝 + 统一下游协议 → ③ 主结果表（四方数字，背 scale10 列和 pps32 列）→ ④ 两个机理发现：AMG 阈值决定"覆盖 vs 质量"（23%→1.4%、0.62→0.95、OA 反降）、点提示的 spill 失败模式（cov 0.92 / spill 0.32）→ ⑤ 结论表（什么条件选谁）→ ⑥ 局限（单场景、SAM 无 NIR、RF 小样本）。

**追问 QA**：

| 追问 | 答法（与正文数字一致） |
|---|---|
| 为什么用合成场景不用真实影像？ | 真实影像没有像素级真值，"对比"会退化成"不可验证的观感"。合成场景（seed=20260829）自带像素级+图斑级两级真值，椒盐率/边界 F 才能严格算；代价是外推性，我在局限里声明了。 |
| SAM 的 OA 低是不是 RF 训练样本太少？ | 是主因之一但不是全部：28 对象档 OA 0.924，放宽阈值后 47 对象反而掉到 0.823——主因是 AMG 大 mask 跨地类合并（误差图上红块成片 vs OBIA 红边贴线）。样本量效应我在局限里单列。 |
| 23% 未覆盖为什么会出现？ | AMG 用 pred_iou≥0.8 + stability≥0.95 过滤"模型不确定"的提案；本场景类内纹理噪声大（σ=0.028+纹理项），不少区域提案置信度不足。对自然照片合理、对遥感是覆盖缺口——这就是我加放宽档对照的原因。 |
| 边界 F 两口径为什么必须分开？ | 分类图边界只能表达"地类变化"，同类相邻田块的边界在地类真值里不存在；label 图边界对图斑真值才衡量分割本身。混用会把宽容口径的数字安到严格问题上（obia 主项目质检报告的深坑之一）。 |
| 提示实验怎么保证公平？ | 同一批 18 个图斑（分层抽样 seed 固定）、同一套窗口与指标函数；另算 auto-containment 做"反事实基线"——0.376 是不开提示时同一图斑能拿到的水平，+0.23/+0.49 才是提示的净贡献。 |
| 下一步做什么？ | 三个方向：伪彩色输入（NIR 进 SAM）、LoRA 微调（零样本→少样本光谱适应）、扩充场景库做跨场景方差。 |
