---
title: PN-QRS 应用于自采 ECG 数据
tags: [PN-QRS, ECG, QRS-detection, guide]
date: 2026-04-13
type: guide
up: "[[index]]"
related: "[[PN_QRS_解读]]"
---

# PN-QRS 应用于自采 ECG 数据

> ← [[index|返回索引]] | 相关：[[PN_QRS_解读]]

---

## 概览

本 pipeline 把 PN-QRS 模型应用于**自采可穿戴 ECG**（上臂导联 CH20 + 可选 12 导联 CH1–CH8），完成从原始 CSV 到信号质量评估的完整链路。

```
原始 CSV
  │
  ▼ Step 1  apply_pnqrs.py              → R-peak 检测（*_rpeaks.csv）
  │
  ▼ Step 2  visualize_rpeaks.py         → 可视化验证 + 幅度-质量分析
  │
  ▼ Step 3  extract_quality_segments.py → 基于不确定性筛选高质量片段
  │
  ▼ Step 4  evaluate_upper_arm.py       → CH20 vs CH1-8 精度评估
  │
  ▼ Step 5  wave_salience_calculator.py → P/Q/S/T 波形显著性 SQI
```

以上 5 步通过 `bash pipeline/run_pipeline.sh` 一键运行（`--skip` 可跳过指定步骤）。

```
（可选，独立运行）
  finetune/run_finetune.sh  → 在自采数据上微调模型（LOSO）
                              注意：与 run_pipeline.sh 无关，需单独执行
```

Step 1–5 的脚本在 `pipeline/` 下；微调脚本在 `finetune/` 下。均从仓库根目录运行。

---

## 数据格式

| 格式 | 列 | 说明 |
|------|----|------|
| 旧格式 | `timestamp` + `CH20` | 单导联，上臂 |
| 新格式 | `timestamp` + `CH1~CH8` + `CH20` | 多导联 + 上臂 |

脚本自动识别列结构，两种格式无需分别处理。信号单位（ADC 计数或 mV）不影响结果，内部做 z-score 归一化。

---

## Step 1：R-peak 检测

```bash
cd /home/kailong/ECG/ECG/ECGFounder/PN-QRS
conda run -n ECGFounder \
  python pipeline/apply_pnqrs.py --data_dir /path/to/data --fs 1000 --gpu 0
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `--data_dir` | 数据根目录（递归扫描所有 CSV/Excel）| 必填 |
| `--fs` | 采样率 Hz | 必填 |
| `--gpu` | GPU 编号 | `0` |

**输出（每个 CSV 旁生成）：**

| 文件 | 内容 |
|------|------|
| `*_CH1-8_rpeaks.csv` | 多导联融合 R-peak（`sample_index`, `time_seconds`）|
| `*_CH20_rpeaks.csv` | 上臂导联 R-peak |
| `rpeaks_summary.json` | 所有文件汇总（beats 数、心率、各导联使用率）|

`lead_usage_pct` 表示每条导联被选为"最优导联"的帧比例，接近 0% 说明该路电极接触不良。

**原理**：CH1–8 逐帧选 U_E（认知不确定性）最小的导联（论文 Algorithm 1）；CH20 直接单导联推理。详见 [[PN_QRS_解读]]。

---

## Step 2：可视化验证

### 2a. 基本可视化

```bash
# 单文件，显示全部通道前 30 秒
python pipeline/visualize_rpeaks.py --csv data/rec01.csv --fs 1000

# 只看 CH20，从第 60 秒看 20 秒
python pipeline/visualize_rpeaks.py \
  --csv data/rec01.csv --fs 1000 \
  --channels CH20 --start 60 --duration 20
```

生成 `*_rpeaks_vis_0-30s.png`，红点为检测到的 R-peak。

### 2b. 幅度-质量分析（`--low_amp`）

硬件信号幅度不稳定时，跨文件统计 CH20 幅度分布，并对比低/中/高幅度区段的检测质量：

```bash
python pipeline/visualize_rpeaks.py \
  --batch --data_dir data/0410_real/ --fs 1000 \
  --low_amp --top_n 9
```

生成三张图：

| 文件 | 内容 |
|------|------|
| `global_CH20_amp_distribution.png` | 所有窗口的 ptp 分布直方图（含 p25/p50）|
| `global_CH20_amp_comparison.png` | Low / Mid / High 各 N 个窗口对比，含 R-peak 标注 |
| `global_CH20_amp_vs_cvrr.png` | ptp vs CV_RR 散点图 + 线性回归 |

**子图标题解读：**
```
[Low#1] rec01.csv
0.0-10.0s  ptp=0.0821  cv_rr=0.312  8beats
```

- **ptp**（peak-to-peak）= `max - min`，衡量该窗口信号幅度大小
- **cv_rr**（RR 变异系数）= `std(RR间隔) / mean(RR间隔)`，衡量检测节律是否规律
  - `< 0.10`：节律规律，检测可信
  - `> 0.20`：忽长忽短，通常是低信噪比导致的漏检/误报

**散点图怎么读：**
- 斜率 < 0，p < 0.05 → 幅度越大检测质量越好
- 斜率 > 0，p < 0.05 → 幅度越大反而越乱（常见于运动伪迹：大幅度 = EMG 噪声而非心跳）
- 注意：ptp 对基线漂移敏感，高 ptp 不一定等于 QRS 幅度大

| 参数 | 说明 | 默认 |
|------|------|------|
| `--win_sec` | 窗口时长（秒）| 10.0 |
| `--top_n` | 每组显示窗口数 | 6 |

---

## Step 3：高质量片段提取

基于模型不确定性，筛选出噪声帧占比低、心拍数正常的 10 秒片段。

**高质量判定：同时满足：**
1. `mean_uc ≤ uc_thr`（噪声帧少）
2. `5 ≤ n_beats ≤ 25`（心率 30–150 bpm，排除脱落或误检爆炸）

```bash
# 单文件，自动阈值（推荐）
python pipeline/extract_quality_segments.py \
  --csv data/rec01.csv --fs 1000 --uc_thr auto

# 批量 + 自动阈值
conda run -n ECGFounder \
  python pipeline/extract_quality_segments.py \
  --batch --data_dir data/0410_real/ --fs 1000 --uc_thr auto --gpu 0

# 可穿戴数据建议起点（阈值宽松）
python pipeline/extract_quality_segments.py \
  --batch --data_dir data/0410_real/ --fs 1000 --uc_thr 2.0
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `--uc_thr` | 不确定性阈值；`auto` 用 Otsu 自动决定 | `1.0` |
| `--step` | 滑动步长（秒）；嘈杂数据设 1–2s | `8` |
| `--gpu` | GPU 编号 | `0` |

**输出文件：**

| 文件 | 内容 |
|------|------|
| `*_quality_overview.png` | 全段信号 + 绿/红背景 + 不确定性折线 |
| `*_quality_segments.png` | 高质量片段网格，含 R-peak + uc 值 |
| `*_uc_distribution.png` | mean_uc 分布直方图 + 阈值线（绿="Bimodal"可信，红="Unimodal"建议手动设阈）|
| `*_quality_report.csv` | 每窗口明细：`start_s, end_s, n_beats, mean_uc, is_good` |
| `quality_segments/*.npz` | 每个高质量片段的信号，可直接送 ECGFounder |
| `batch_quality_summary.csv` | 批量汇总，按 activity 分组 |

**`mean_uc` 原理：**
```python
eu[eu > 0.12] = 10        # 噪声帧打成 10（硬判决）
mean_uc = mean(eu + au)   # 本质上是噪声帧占比的加权计数
```
干净信号 ~0.1–0.5；电极脱落 ~10。
默认阈值 `uc_thr=1.0` 为临床 ECG 设计，可穿戴导联建议从 `2.0` 或 `auto` 开始。

---

## Step 4：上臂导联精度评估

以 CH1-8 多导联融合结果为伪标准答案，量化 CH20 的检测精度：

```bash
python pipeline/evaluate_upper_arm.py --data_dir data/0410_real/ --fs 1000
```

**输出示例：**
```
rec01.csv
  ref(CH1-8)=450  pred(CH20)=448
  TP=441  FP=7  FN=9
  Se=97.78%  P+=98.44%  F1=98.11%

汇总（27 个文件）
  Se=96.89%  P+=96.14%  F1=96.51%
```

| 指标 | 含义 |
|------|------|
| Se（敏感度）| CH20 检出了多少 CH1-8 检到的心拍（低 → 漏检多）|
| P+（精确率）| CH20 的检出中有多少是真实心拍（低 → 误报多）|

> CH1-8 不是绝对 ground truth，只是比单导联更可靠的参考。

---

## Step 5：波形显著性 SQI

评估 P/Q/S/T 各波相对于 R 波的可见程度，补充 `mean_uc` 只能回答"QRS 找不找得到"的不足。

有两个版本可选：

| 版本 | 脚本 | 后端 | 适用场景 |
|------|------|------|---------|
| 独立版 | `wave_salience_calculator.py` | NeuroKit2（内置）| 无外部依赖 |
| Domain 库版 | `wave_salience_calculator_call.py` | `_wave_salience_calculator`（需安装）| 接入 domain SQI 框架 |

### 独立版（推荐）

```bash
# 单文件
python pipeline/wave_salience_calculator.py --csv data/rec01.csv --fs 1000

# 批量
python pipeline/wave_salience_calculator.py \
  --batch --data_dir data/0410_real/ --fs 1000
```

**输出示例：**
```
rec01.csv  fs=1000Hz  duration=36.0s  segments=12
  P: salience=0.127  detection=73.2%  (104/142)
  Q: salience=0.085  detection=91.5%  (130/142)
  S: salience=0.203  detection=95.8%  (136/142)
  T: salience=0.312  detection=80.3%  (114/142)
  composite=0.207
```

| 指标 | 计算 | 含义 |
|------|------|------|
| salience | `median(|wave_amp| / |R_amp|)` | 该波幅度相对 R 波的比值 |
| detection_rate | `n_detected / n_R_peaks` | 该波被检出的心拍占比 |
| composite | `Σ(salience × detection_rate) / Σ(detection_rate)` | 检出率加权综合 |

**CH20 上臂导联参考范围：**

| 波段 | 预期范围 | 说明 |
|------|---------|------|
| P | salience 0.05–0.15 | 上臂 P 波很弱，detection < 70% 正常 |
| Q | salience 0.03–0.10 | 幅度小 |
| S | salience 0.10–0.25 | 相对稳定 |
| T | salience 0.15–0.35 | detection 应 > 80%，否则 ST 分析不可靠 |

### Domain 库版

需要先安装 `_wave_salience_calculator`，且**必须先跑 Step 1** 生成 `*_rpeaks.csv`。

```bash
python pipeline/wave_salience_calculator_call.py \
  --batch --data_dir data/0410_real/ --fs 1000 --detail
```

输出 `wave_sqi_summary_.csv`（每文件一行）和 `wave_sqi_detail_.csv`（每波一行）。

---

## 完整流程一键脚本

```bash
cd /home/kailong/ECG/ECG/ECGFounder/PN-QRS

# 一键跑完全部 5 步
bash pipeline/run_pipeline.sh --data_dir data/0413_real/ --fs 1000

# 常用参数
bash pipeline/run_pipeline.sh \
  --data_dir data/0413_real/ \
  --fs       1000 \
  --gpu      0 \
  --conda_env ECGFounder \
  --top_n    9 \
  --uc_thr   auto

# 跳过已完成的步骤（逗号分隔，如只重跑 Step 3）
bash pipeline/run_pipeline.sh \
  --data_dir data/0413_real/ --fs 1000 --skip 1,2,4,5
```

每步完成后打印耗时，任意一步报错自动停止。

### 手动逐步运行

```bash
cd /home/kailong/ECG/ECG/ECGFounder/PN-QRS
CONDA="conda run -n ECGFounder"
DATA="data/0410_real"

# Step 1: R-peak 检测（需 GPU）
$CONDA python pipeline/apply_pnqrs.py --data_dir $DATA --fs 1000 --gpu 0

# Step 2: 幅度-质量分析（不需要 GPU）
python pipeline/visualize_rpeaks.py \
  --batch --data_dir $DATA --fs 1000 --low_amp --top_n 9

# Step 3: 高质量片段提取（需 GPU）
$CONDA python pipeline/extract_quality_segments.py \
  --batch --data_dir $DATA --fs 1000 --uc_thr auto --gpu 0

# Step 4: 上臂导联评估
python pipeline/evaluate_upper_arm.py --data_dir $DATA --fs 1000

# Step 5: 波形显著性 SQI
python pipeline/wave_salience_calculator.py --batch --data_dir $DATA --fs 1000
```

---

## 实测结果（data/0410_real，27 个文件）

数据：被试 1+2，各行为 3 段录制，CSV 约 36 秒/文件，1000 Hz，新格式（CH1–CH8 + CH20）。

**Step 1 结果：**

| 活动 | HR 典型范围 |
|------|-----------|
| 坐姿系列 | 66–88 bpm |
| 慢走 | 84–107 bpm |
| 站立坐下 | 69–96 bpm |

**Step 3 结果（auto 阈值）：**

| 被试 | 总窗口 | 好窗口 | 比例 |
|------|--------|--------|------|
| 被试 1 | 60 | 35 | 58.3% |
| 被试 2 | 75 | 48 | 64.0% |
| **合计** | **135** | **83** | **61.5%** |

坐姿抬手好窗口 80–100%；慢走仅 20–60%（运动伪迹拉高 mean_uc）。

**Step 4 结果（CH20 vs CH1-8）：**

| TP | FP | FN | Se | P+ | F1 |
|----|----|----|----|----|-----|
| 1246 | 50 | 40 | 96.89% | 96.14% | 96.51% |

**Step 5 结果（wave_salience_calculator）：**

| 波段 | 检出率 | 备注 |
|------|--------|------|
| P 波 | ~100% | salience 0.07–0.60，活动依赖 |
| T 波 | ~77% | 运动伪迹下下降符合预期 |
| composite SQI | 0.31 | 在上臂导联参考范围内 |

---

## 可选：在自采数据上微调

> **前置条件**：Step 1 的 `*_CH1-8_rpeaks.csv` 和 Step 3 的 `*_quality_report.csv` 已生成。
>
> **注意**：微调脚本（`finetune/`）与 `run_pipeline.sh` 完全独立，需单独执行，不能通过 `--skip` 控制。

模型默认权重来自 CPSC2019（标准 12 导联）。如果 CH20 检测质量不理想（F1 < 90%），可用 Step 1 产生的 12 导联伪标签对模型做 AEU 微调，让它专门适配上臂导联。

### 一键 LOSO 微调 + 评估

```bash
cd /home/kailong/ECG/ECG/UI_Beat
bash finetune/run_finetune.sh --data_dir data/0410_real --fs 1000 --gpu 0
```

- 对 `data_dir` 下每个被试目录分别做一次 LOSO fold（用其他被试训练，当前被试验证）
- 每 fold 自动跑完训练 + 评估，结果写入 `experiments/logs_armband/`

### 手动单 fold

```bash
# 训练
conda run -n ECGFounder python finetune/train_armband.py \
    --data_dir data/0410_real --fs 1000 \
    --test_subject subject01 --gpu 0

# 评估（baseline vs fine-tuned）
conda run -n ECGFounder python finetune/eval_armband.py \
    --data_dir data/0410_real --fs 1000 \
    --test_subject subject01 \
    --baseline_ckpt  experiments/logs_real/zy2lki18/models/best_model.pt \
    --finetuned_ckpt experiments/logs_armband/<timestamp>_subject01/models/best_model.pt \
    --gpu 0
```

### 输出

```
experiments/logs_armband/<timestamp>_<subject>/
├── models/
│   ├── best_model.pt       ← 微调后权重（可替换 best_model.pt 用于 pipeline）
│   └── last_model.pt
├── history.csv             ← 每 epoch 的 α/θ loss + val_bce
├── args.json
└── eval/
    ├── eval_results.csv    ← 每文件 Se/P+/F1（baseline + fine-tuned）
    └── eval_summary.md     ← 按活动分组的 F1 对比表（含 Δ）
```

### 关键参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--epochs` | 30 | 最大训练轮数（early stop 默认 patience=10）|
| `--alpha_lr` | 5e-5 | encoder+decoder 学习率（保守，避免覆盖预训练特征）|
| `--theta_lr` | 5e-5 | projection head 学习率 |
| `--early_stop` | 10 | val_bce 不下降多少 epoch 后停止 |

> **注意**：评估所用的参考标签是 PN-QRS 自己产出的 12 导联伪标签，不是人工标注。  
> F1 提升 = 模型更好地在 CH20 上模仿 12 导联教师，不等同于绝对精度提升。  
> 详见 `finetune/README.md`。

---

## 常见问题

**CH20 漏检多**：上臂导联 QRS 可能倒置，尝试在 `load_excel_ecg` 后对信号取反再检测，对比数量。

**采样率推断错误**：加 `--fs 250`（或设备实际值）手动指定。

**检测数量极少**：信号单位是 ADC 计数（幅度 >>10）时，z-score 预处理有时不生效。检查 CSV 里信号的实际数值范围，确认 `std > 1` 触发了归一化。

**某导联 `lead_usage_pct` 始终 0%**：该路电极脱落或极性错误，U_E 机制已自动排除，不影响其他导联结果。

**`mean_uc` 可穿戴数据普遍偏高**：默认阈值 1.0 为临床 ECG 设计。可穿戴建议先跑 `--uc_thr auto` 看 Otsu 阈值落在哪，再决定是否手动调。

**幅度 vs CV_RR 斜率显著 > 0**：高幅度窗口 CV_RR 反而高，通常是运动伪迹（EMG 幅度大于 QRS）或基线漂移（ptp 大但不是 QRS 幅度大）。翻出 High 组窗口图直观确认。
