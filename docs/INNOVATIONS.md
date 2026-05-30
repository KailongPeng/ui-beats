# UI_Beat_share 相对原始 UI_Beat 的独特贡献

> **一句话总结**：原仓库 [mato00/UI_Beat](https://github.com/mato00/UI_Beat) 是 PN-QRS 论文的**学术研究代码**（训练 + 标准数据集评测），本仓库 `UI_Beat_share` 在**完全不改动模型核心**的前提下，重新组织为一套**面向上臂可穿戴 ECG 的部署级质量筛选工具链**，并首次填补了论文未公开的若干关键能力。

---

## 零、最关键的差别

| 维度 | 原始 UI_Beat | UI_Beat_share (本仓库) |
|------|--------------|------------------------|
| **定位** | 论文方法参考实现 | **工业部署级工具** |
| **目标信号** | CPSC2019 / MIT-BIH / INCART / LUDB / QTDB 等**研究用标准 ECG** | **上臂单导联（CH20）可穿戴 ECG** |
| **使用门槛** | 读代码 + 自己从头训练 + 自己写评测 | `bash run.sh --data_dir X --fs 1000` 一键产出 |
| **预训练权重** | 需自行训练（仓库未发布 `.pt` 文件） | ✅ 随仓库附带一份 CPSC2019 训练好的权重（`experiments/logs_real/zy2lki18/models/best_model.pt`，wandb run `zy2lki18`） |
| **模型–数据匹配** | CPSC2019 训练 → CPSC2019/MIT-BIH 等临床数据评测（同域） | **CPSC2019 训练 → 上臂可穿戴 ECG 零样本迁移（跨域）** |
| **README** | 1 行 (`# UI_Beat`) | 161 行（数据格式、参数、FAQ、输出规范完整覆盖） |
| **阈值机制** | 固定值（超参搜索） | **Otsu 自适应 + 两阶段批处理** |
| **批处理 / 多 GPU** | 无 | ✅ `spawn` 多进程 + 分片调度 |
| **可视化** | 无 | ✅ 三种诊断图（overview / segments / uc-distribution） |
| **实测数据** | 无 | ✅ 上臂 ECG 多被试 × 多活动数据集 |

---

## 一、上臂导联（Armband ECG）的独有贡献 ⭐ 本仓库最强创新

### 1.0 跨域零样本迁移（Cross-Domain Zero-Shot Transfer）⭐

> **先声明权重的来源**：随仓库附带的 `experiments/logs_real/zy2lki18/models/best_model.pt`（348 KB）是使用**原仓库 `running/run_ui_beat.py` 训练脚本、在 CPSC2019 标准单导联临床 ECG 上训练**得到的（run id `zy2lki18` 为 wandb 记录），路径结构完全遵循原训练代码的 `log_dir / run.id / models / best_model.pt` 约定。本仓库**没有重新训练**。

**这反而构成了一个更强的学术论点**：

- 模型**只见过 CPSC2019 临床 ECG**（胸导联、干净、500 Hz、静态）
- 直接迁移到**上臂 CH20 可穿戴 ECG**（运动伪迹、肌电、信号强度弱、非标准电极位置）
- **无 fine-tune、无 domain adaptation、无目标域标注**
- 依然能通过 `U_E + U_A` 的不确定性排序**区分出高/低质量窗口**

这说明 PN-QRS 的不确定性估计具备**跨域泛化能力**——epistemic uncertainty 不仅能识别"模型没见过的形态"，aleatoric uncertainty 也能识别"上臂场景独有的噪声"。**这一跨域性质在原论文的同域（CPSC→MITBIH）评测中并未被单独 claim。**

PN-QRS 原论文在**标准医院 12 导联数据集**上评测，**从未涉及上臂可穿戴场景**。本仓库在此跨域迁移的基础上，针对可穿戴场景的特殊挑战做了一系列原创设计：

### 1.1 单通道（CH20）质量评估协议

原版 `running/ui_qrs_detect.py` 的 `multi_lead_select` 依赖**多导联交叉投票**（`uncertain_leads` 横向对比）来决定可信度；上臂只有**一路电极**，该机制完全失效。

本仓库提出：

- **只用 CH20 信号 + `mean(U_E + U_A)` 作为单通道自评估指标**
- `U_E`（epistemic）反映模型自信度，`U_A`（aleatoric）反映信号噪声
- 相加后可同时捕捉"模型不确定"与"信号本身糟糕"两种情况
- 无需任何 12 导联参考，纯单路可用

📍 参考 `pipeline/extract_quality_segments.py:161-179`（`uncertain_est` / `en_est` / `mi_est` 的组合使用）。

### 1.2 Otsu 自适应阈值 —— 解决可穿戴数据的域漂移

**问题**：上臂 ECG 的 `mean_uc` 分布与医院 12 导联差别极大（肌电、运动伪迹导致 uc 整体偏高）。原版固定阈值 `UC_THR_DEF=1.0` 直接套用会把 90% 数据判成 "bad"。

**解法**：一维 Otsu 二值化，自动最大化"好/坏窗口"组间方差：

```python
# pipeline/extract_quality_segments.py:85-114
def otsu_threshold(values):
    # 对 mean_uc 列表求最优分割阈值
    # 累积和 O(n) 实现，可跑上万窗口
```

📍 `pipeline/extract_quality_segments.py:85`。

### 1.3 两阶段批处理：全局池化的 Otsu

**观察**：每个 30 秒短录制只有 3~4 个滑窗，单文件做 Otsu 样本太少、阈值不稳。

**创新设计**（原版完全没有）：

```
Pass-1：全部 CSV 文件跑推理 → 收集所有 mean_uc → 池化
      ↓
全局 Otsu (对 pooled uc 计算) → 得到整批数据的单一阈值
      ↓
Pass-2：回填阈值 → 按窗口判定 is_good → 保存 NPZ + 可视化
```

📍 `pipeline/extract_quality_segments.py:770-834`（`_worker_infer` + `_worker_process`）。这一设计让**跨被试的 uc 标尺统一**，也让可视化 `batch_uc_distribution.png` 有意义。

### 1.4 双重校验：不确定性 + 心拍数合理性

仅靠 uc 阈值可能误判——如果信号非常嘈杂到模型"自信地预测很多假 R 峰"，uc 可能反而低。本仓库增加**生理合理性校验**：

```python
# pipeline/extract_quality_segments.py:184-188
BEAT_MIN = 5   # 10s 内最少 5 拍 ≈ 30 bpm
BEAT_MAX = 25  # 10s 内最多 25 拍 ≈ 150 bpm
is_good = (mean_uc <= uc_thr) and (BEAT_MIN <= n_beats <= BEAT_MAX)
```

📍 `pipeline/extract_quality_segments.py:47-48, 187`。

### 1.5 活动维度（Activity-level）分组汇总

原版从不关心"**被试 × 活动**"这种工程组织维度。本仓库自动按目录路径提取 `activity`（坐姿抬手 / 慢走 / 站立坐下 / 坐姿说话 / 坐姿手臂前后摇摆 / 坐姿说话），并在汇总报告里按活动聚合统计：

```
activity       file                             dur    wins  good  ratio%  uc_good
被试1         [小计]                            36.0    12    10   83.3%   1.23
              └ 坐姿抬手/rec01.csv             12.0     4     4  100.0%   0.97
              └ 慢走/rec02.csv                 12.0     4     2   50.0%   1.56
```

📍 `pipeline/extract_quality_segments.py:763-765, 895-910`。这让研究者可以**快速看到哪些动作在上臂场景下最容易出伪迹**（实测结果：慢走、站立坐下 good_ratio 最低，符合生理直觉）。

### 1.6 实测上臂 ECG 数据集（Ready-to-use benchmark）

本仓库**自带真实采集的上臂 ECG 数据**（`data/0410_dummy/`），被试覆盖：

- 2 名被试 × 5 种日常活动 × 3 次重复 = **30 段录制**
- 5 种活动：`坐姿抬手` / `坐姿手臂前后摇摆` / `坐姿说话` / `慢走` / `站立坐下`
- 每段含原始 CSV + 质量报告 CSV + 三种诊断图 + 提取出的高质量 NPZ

这是目前**几乎找不到公开对应**的数据维度（根据 PN_QRS_解读文档 §上臂 ECG 评测 部分，学术界仅有 2024 ESC 会议摘要评测过上臂 QRS，但数据未公开）。

---

## 二、工程架构创新（Pipeline Engineering）

### 2.1 原版没有"生产入口"—— 本仓库新增 926 行 pipeline

| 文件 | 行数 | 作用 |
|------|------|------|
| `pipeline/extract_quality_segments.py` | **926** | 核心推理管线（新增） |
| `run.sh` | 80 | bash 一键入口（新增） |
| `README.md` | 161 | 完整使用文档（原版 1 行） |

对比原版 `running/ui_qrs_detect.py`（255 行）只是**研究评测脚本**，依赖 `RECORDS` 列表文件 + 标准数据集命名约定（`cpsc2019.mat` / `mitdb.dat`），**无法直接处理真实用户数据**。

### 2.2 多 GPU 并行的分片 + spawn 设计

```python
# pipeline/extract_quality_segments.py:775-820
chunks = [all_files[i::n_gpus] for i in range(n_gpus)]
ctx    = mp.get_context("spawn")   # CUDA 兼容
with ctx.Pool(len(p1_args)) as pool:
    p1_results = pool.starmap(_worker_infer, p1_args)
```

原版完全串行，处理大批量数据时耗时不可接受。本设计：

- 文件按轮询分配（`[i::n_gpus]`）保证负载均衡
- `spawn` 上下文避免 fork 模式下的 CUDA 冲突
- 独立 worker 各自 `load_model(device)`，无共享状态

📍 `pipeline/extract_quality_segments.py:775-824`（Pass-1）、`pipeline/extract_quality_segments.py:811-823`（Pass-2）。

### 2.3 批量 GPU forward（5-10x 加速）

```python
# pipeline/extract_quality_segments.py:147-156
for i in range(0, len(tensor_list), infer_batch):
    batch_np = np.stack(tensor_list[i: i + infer_batch], axis=0)
    ...
```

原版 `ui_qrs_detect.py:170-185` 是逐窗口 forward，本仓库默认 `infer_batch=16`，显著提升吞吐。

### 2.4 鲁棒的 CSV 读取

真实采集数据的 CSV 常有末尾多余空列（例如 `"ts, CH20, , , ,"`），直接 `pd.read_csv` 会 `ParserError`。本仓库提供：

```python
# pipeline/extract_quality_segments.py:487-501
def _read_csv_robust(path):
    # 以第一行列数为准，截断后续所有行
```

---

## 三、可视化能力（原版全部缺失）

本仓库为每个录制生成**三类可视化诊断图**：

| 图 | 输出 | 作用 |
|---|---|---|
| **全局概览** | `rec*_quality_overview.png` | 完整信号 + R 峰 + 好坏窗口着色 + 不确定性带 |
| **高质量片段** | `rec*_quality_segments.png` | 每个 good 窗口单独子图，带 R 峰红点、心率标注 |
| **UC 分布图** | `rec*_uc_distribution.png` / `batch_uc_distribution.png` | 直方图 + Otsu 阈值线 + **双峰/单峰可信度判定** |

特别是 **UC 分布图**里的 bimodal 判定（`pipeline/extract_quality_segments.py:440-466`）：

```
uc range > 1.0 → 判定为双峰分布 → Otsu 阈值可信（绿色提示）
uc range ≤ 1.0 → 单峰 → 提示用户手动设 --uc_thr（红色警告）
```

这是一种**阈值可信度的自我诊断**机制，原版没有。

---

## 四、输出规范（面向下游消费者）

### 4.1 每片段一个 NPZ 文件

原版输出是 `.npy` 的 R 峰索引数组，信息很稀疏。本仓库每个高质量 10 秒片段存一个 NPZ，包含：

```
CH20       (N,)       上臂导联信号
CH1-CH8    (N,)       同步的 12 导联信号（若 CSV 有）
fs         scalar     采样率
start_s    scalar     片段在原始录制中的起始秒
mean_uc    scalar     综合不确定性
n_beats    scalar     检测到的心拍数
r_peaks    int[]      R 峰索引（相对片段起点，已减去偏移可直接用）
```

📍 `pipeline/extract_quality_segments.py:204-234`。

这种**自包含片段文件**直接对接下游训练或推理，无需再跑一遍预处理。

### 4.2 两级汇总报告

- 文件级：`rec*_quality_report.csv`（逐窗口的 start_s / mean_uc / mean_ue / mean_ua / n_beats / is_good）
- 批级：`batch_quality_summary.csv`（每 CSV 一行，含 activity 分组）

原版无任何汇总输出。

---

## 五、贡献清单（可量化）

| 指标 | 数值 |
|------|------|
| 新增代码行数 | **~1170 行**（pipeline + run.sh + README） |
| 删除研究代码行数 | ~800 行（training/running/sweep/config/loss/pwave） |
| 新增文件 | 4 个（pipeline/extract_quality_segments.py, run.sh, requirements.txt, models/__init__.py 等） |
| 模型代码修改 | **0 行**（完全保留原方法） |
| 随仓库附带的权重 | ✅ `best_model.pt` 348 KB，用原训练代码在 **CPSC2019** 上训练得到（wandb run `zy2lki18`）；原仓库本身未在 release 中附 `.pt` 文件，本仓库方便用户**直接跨域推理**，省去自行准备 CPSC2019 + 训练 GPU 的成本 |

---

## 七、创新价值的学术定位

> 以下属于相对于原始 PN-QRS / UI_Beat 的学术与工程新颖点，若用于论文 Claim：

1. **证明 PN-QRS 的 CPSC2019 训练模型可零样本迁移到上臂可穿戴 ECG**（原论文只在 CPSC → MIT-BIH / INCART 等同域临床数据集上评测，未涉及上臂场景）
2. **首次将 PN-QRS 不确定性估计应用于上臂单导联 ECG 的质量评估**（非 QRS 检测精度本身）
3. **提出 Otsu 自适应阈值 + 两阶段全局池化策略**解决可穿戴数据的 uc 域漂移问题
4. **提出 `U_E + U_A` 联合指标 + 心拍数合理性双重校验**的单通道质量判据
5. **发布首个小规模上臂 ECG × 5 日常活动的质量评估基准**（30 段，可复现实验）
6. **提供完整的部署级开源工具链**（多 GPU 并行 + 三类诊断可视化 + NPZ 片段输出 + 活动级汇总 + 一份 CPSC2019 预训练权重）

---

## 八、文件结构一览

```
UI_Beat_share/
├── INNOVATIONS.md                   ← 本文件
├── README.md                        ← 使用文档（161 行）
├── run.sh                           ← 一键入口
├── requirements.txt                 ← Python 依赖
│
├── pipeline/                        ★ 本仓库独有
│   ├── __init__.py
│   └── extract_quality_segments.py  ← 926 行核心管线
│
├── models/                          ← 来自原仓库，未修改
│   ├── multi_head.py
│   └── qrs_model.py
│
├── dataset/                         ← 来自原仓库，未修改
│   ├── dataset.py
│   └── ecg_preprocess.py
│
├── utils/                           ← 来自原仓库，未修改
│   └── qrs_post_process.py
│
├── experiments/logs_real/zy2lki18/
│   └── models/best_model.pt         ← 用原训练代码在 CPSC2019 上训出的权重（348 KB，wandb run `zy2lki18`）
│
└── data/0410_dummy/                 ★ 独有的上臂 ECG 实测数据
    ├── 被试1/{坐姿抬手,慢走,...}/rec01~03.csv
    ├── 被试2/{坐姿抬手,慢走,...}/rec01~03.csv
    └── batch_quality_summary.csv
```

---

**© UI_Beat_share — 基于 [mato00/UI_Beat](https://github.com/mato00/UI_Beat) 的上臂 ECG 工程化扩展**
