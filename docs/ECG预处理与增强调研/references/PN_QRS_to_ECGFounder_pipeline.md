---
title: PN-QRS → ECGFounder 数据准备管道
tags: [PN-QRS, ECGFounder, pipeline, data-preparation, SelfMIS]
date: 2026-04-08
type: guide
up: "[[index]]"
related: "[[PN_QRS_on_custom_ECG]]"
---

# PN-QRS → ECGFounder 数据准备管道

> ← [[index|返回索引]] | 相关代码：`PN-QRS/apply_pnqrs.py` · `PN-QRS/extract_quality_segments.py` · `ECGFounder/selfmis_pretrain.py`

---

## 核心理解

**PN-QRS 在整个管道中的角色是质量过滤器，而不是分割器。**

ECGFounder（SelfMIS）的输入是**连续的 10 秒原始信号**，不是单个心拍。
模型直接在整段信号上做对比学习，无需对齐到心拍边界。

PN-QRS 解决的问题是：**切出来的 10 秒窗口里，有没有真实的心电信号？**

---

## 整体管道

```
自采 Excel 文件（几分钟连续记录）
         │
         ▼
┌─────────────────────┐
│  Step 1: apply_pnqrs │  → 输出 *_CH1-8_rpeaks.csv / *_CH20_rpeaks.csv
│  检测 R-peak 位置    │
└─────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Step 2: 滑动窗口切片     │  → 10 秒窗口，2 秒 overlap
│  + 用 R-peak 密度过滤     │
└──────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Step 3: 送入 ECGFounder SelfMIS    │
│  输入对: (CH20, CH1-8) 10 秒窗口    │
└─────────────────────────────────────┘
```

---

## Step 1：R-peak 检测

```bash
cd /home/kailong/ECG/ECG/ECGFounder/PN-QRS
python apply_pnqrs.py --data_dir /path/to/data --fs 1000
```

输出 `*_CH1-8_rpeaks.csv` 和 `*_CH20_rpeaks.csv`，包含每个心拍的 `sample_index`。

---

## Step 2：滑动窗口切片 + 质量过滤

切片粒度是 **10 秒整段**，不是单个心拍。脚本 `extract_quality_segments.py` 完整实现此步骤。

```bash
cd /home/kailong/ECG/ECG/ECGFounder/PN-QRS
# 自动阈值（推荐可穿戴数据）
python extract_quality_segments.py --csv /path/to/data.csv --fs 1000 --uc_thr auto

# 批量处理（按行为子目录分组）
python extract_quality_segments.py --batch --data_dir /path/to/data_dir --fs 1000 --uc_thr auto
```

### 窗口示意

```
整段记录（例如 5 分钟）
│
├── 0s ──────── 10s   mean_uc=0.18  beats=8  → ✓ 绿色
├── 8s ──────── 18s   mean_uc=0.22  beats=9  → ✓ 绿色
├── 16s ─────── 26s   mean_uc=9.87  beats=0  → ✗ 红色（电极脱落）
├── 24s ─────── 34s   mean_uc=0.19  beats=2  → ✗ 红色（漏检过多）
└── ...（步长 8s，overlap 2s）
```

### 双重过滤条件（同时满足才算高质量）

| 条件 | 判断 | 说明 |
|------|------|------|
| `mean_uc <= uc_thr` | ✓ 保留 | 噪声帧占比低；默认阈值 `uc_thr=1.0` |
| 心拍数 5–25 个 / 10s | ✓ 保留 | 对应 30–150 bpm 正常范围 |
| 其中任一不满足 | ✗ 丢弃 | |

### 质量分数的含义

**`mean_uc` 的计算（`uncertain_est()` 内部）：**

```python
au  = en_est(logits)     # 逐帧预测熵（偶然不确定性）[T]
eu  = mi_est(logits)     # 逐帧 KL 散度（认知不确定性）[T]
eu[eu > 0.12] = 10       # 噪声帧二值化：超过 0.12 的帧直接打成 10
mean_uc = mean(eu + au)  # 窗口均值 → 质量分数
```

`mean_uc` 实质上是**噪声帧占比的加权计数**：只要窗口里有少数噪声帧（eu > 0.12），它们就以 10 的权重拉高均值，使干净窗口（~0.1–0.5）和噪声窗口（~10）之间差异悬殊，阈值容易区分。

**与论文原始定义的关系：**

论文中 U_E 和 U_A 是分开使用的：

| 论文中的用途 | 粒度 | 阈值 |
|------------|------|------|
| U_E：判断导联是否有效（非 ECG 则丢弃） | 整段信号标量 | α = 0.1 |
| U_E：多导联融合，每帧选 U_E 最小的导联 | 逐帧 | — |
| U_A：标记可疑心拍（R-peak 前后 7 帧均值） | 逐拍 | β = 0.12 |

`uncertain_est()` 把两者合并为逐帧分数再取窗口均值，是代码层面的工程适应，不是论文的直接定义。实用效果等价：噪声信号被可靠识别，干净信号被保留。

**此方法只需要 CH20 单路信号**，不需要 12 导联参考。

### 输出文件

| 文件 | 说明 |
|------|------|
| `*_quality_overview.png` | 全局概览：完整信号 + 绿/红背景色 + 不确定性柱状图 |
| `*_quality_segments.png` | 高质量片段网格：每片段一个子图，带 R-peak 红点 |
| `*_quality_report.csv` | 每个窗口的 `start_s, end_s, n_beats, mean_uc, is_good` |
| `quality_segments/*.npz` | 每个高质量片段的 NumPy 存档（含信号、R-peak、元数据） |

### 调参建议

```bash
# 先用默认阈值跑一遍，看概览图里好窗口的比例
python extract_quality_segments.py --csv data.csv --fs 1000

# 觉得过滤太宽松（质量报告里还有很多 uc~0.8 的窗口），收紧阈值
python extract_quality_segments.py --csv data.csv --fs 1000 --uc_thr 0.5

# 觉得过滤太严格（好片段太少），放宽阈值
python extract_quality_segments.py --csv data.csv --fs 1000 --uc_thr 2.0
```

---

## Step 3：送入 ECGFounder

### 数据映射关系

你的穿戴设备与 SelfMIS 训练范式天然对应：

| SelfMIS 期望 | 你的设备 | 形状（500 Hz 重采样后）|
|-------------|---------|----------------------|
| single-lead（Lead I） | CH20（上臂导联） | `(1, 5000)` |
| multi-lead（12 导联） | CH1–8（穿戴多导联） | `(8, 5000)`* |

> *ECGFounder 标准输入是 12 导联，可将 CH1–8 zero-pad 到 `(12, 5000)`（后 4 通道填 0），不需要修改模型结构。或写自定义 Dataset，修改 encoder 输入通道数为 8。

### 预处理步骤

```
原始信号（1000 Hz）
    → 重采样至 500 Hz（线性插值）
    → Z-score 归一化（per-channel）
    → 输出 (1, 5000) + (8, 5000)
```

与 ECGFounder 对 MIMIC-IV-ECG 的预处理对齐（MIMIC 额外有带通滤波，可选加上）。

---

## 为什么不按心拍切片？

ECGFounder 是**记录级（recording-level）**模型，而非**心拍级（beat-level）**模型：

- **训练数据**：MIMIC-IV-ECG（800K 条 10 秒记录）、PTB-XL（10 秒记录）
- **模型输入**：`(leads, 5000)` 的完整 10 秒张量
- **学习目标**：理解整段 ECG 的节律、形态、导联关系

按心拍切分（~0.8 秒/拍）是另一类任务（心拍分类器），不是 ECGFounder 的用法。

---

## PN-QRS 的附加价值

除了质量过滤，R-peak 结果还可以用于：

| 用途 | 说明 |
|------|------|
| 心率元数据 | `mean_hr_bpm` 可作为弱监督标签或辅助特征 |
| 导联质量评分 | `lead_usage_pct ≈ 0%` 的导联说明脱落，该段可降权 |
| CH20 vs CH1-8 对齐验证 | 用 [[PN_QRS_on_custom_ECG#评估上臂导联质量（CH20 vs CH1-8）\|evaluate_upper_arm.py]] 量化两路信号是否同步 |
| 未来 beat-level 任务 | 若下游任务需要心拍分类，R-peak 直接用作分割点 |

---

## 相关文档

- [[PN_QRS_on_custom_ECG]] — apply_pnqrs.py 使用方法和输出格式
- `ECGFounder/selfmis_pretrain.py` — SelfMIS 预训练入口
- `ECGFounder/selfmis_dataset.py` — `MIMICECGPretrainDataset`：参考实现，仿照写自定义 Dataset
