---
title: PN-QRS 应用于自采 Excel ECG 数据
tags: [PN-QRS, ECG, QRS-detection, guide]
date: 2026-04-08
type: guide
up: "[[index]]"
related: "[[PN_QRS_解读]]"
---

# PN-QRS 应用于自采 Excel ECG 数据

> ← [[index|返回索引]] | 实现代码：`PN-QRS/apply_pnqrs.py` · `PN-QRS/visualize_rpeaks.py` · `PN-QRS/evaluate_upper_arm.py` · `PN-QRS/extract_quality_segments.py`

---

## 数据格式

| 格式 | 列 | 说明 |
|------|----|------|
| 旧格式 | `timestamp` + `CH20` | 单导联，上臂 |
| 新格式 | `timestamp` + `CH1~CH8` + `CH20` | 多导联 + 上臂 |

脚本自动识别列结构，两种格式无需分别处理。信号幅度（ADC 计数）不影响结果，`preprocess_ecg` 内部做 z-score 归一化。

---

## 快速开始

```bash
cd /home/kailong/ECG/ECG/ECGFounder/PN-QRS

# 基本用法（自动推断采样率）
python apply_pnqrs.py --data_dir /path/to/xlsx

# 手动指定采样率（推断不准时用）
python apply_pnqrs.py --data_dir /path/to/xlsx --fs 500

# 指定 GPU
python apply_pnqrs.py --data_dir /path/to/xlsx --gpu 1
```

---

## 输出

每个 Excel 文件产生：

| 文件 | 内容 |
|------|------|
| `*_CH1-8_rpeaks.csv` | 多导联融合结果（`sample_index`, `time_seconds`） |
| `*_CH20_rpeaks.csv` | 上臂导联结果 |
| `rpeaks_summary.json` | 所有文件的汇总（beats 数、心率、导联使用率） |

`lead_usage_pct` 表示每条导联被选为"最优导联"的帧比例，可用来判断哪路电极接触不良（使用率长期接近 0%）。

---

## 可视化

检测完成后用 `visualize_rpeaks.py` 画图验证，红点即为检测到的 R-peak：

```bash
cd /home/kailong/ECG/ECG/ECGFounder/PN-QRS

# 显示全部通道，前 30 秒（默认）
python visualize_rpeaks.py \
  --csv "data/recording.csv" \
  --fs  1000

# 只看 CH20，从第 60 秒开始看 20 秒
python visualize_rpeaks.py \
  --csv      "data/recording.csv" \
  --fs       1000 \
  --channels CH20 \
  --start    60 \
  --duration 20
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `--csv` | 原始 ECG CSV 路径 | 必填 |
| `--fs` | 采样率 Hz | 必填 |
| `--start` | 起始时间（秒） | 0 |
| `--duration` | 显示时长（秒） | 30 |
| `--channels` | 显示的通道，逗号分隔，或 `all` | all |
| `--out` | 输出图片路径 | CSV 同目录自动命名 |

图片自动保存为 `*_rpeaks_vis_0-30s.png`，右上角标注该段的 beats 数和 bpm。

---

## 常见问题

**CH20 漏检多**：上臂导联 QRS 可能倒置，在 `load_excel_ecg` 调用后尝试 `rec.ch_upper_arm = -rec.ch_upper_arm`，再对比检测数量。

**采样率推断错误**：加 `--fs 250`（或设备实际采样率）手动指定。

**检测数量极少（如 46 个可见 spike 只检出 4 个）**：信号单位是 ADC 计数（幅度 >>10）时，`preprocess_ecg` 内部的 `pp()` 函数会把整段信号抹平成常数，导致模型看不到任何波形。`apply_pnqrs.py` 已在进入 `preprocess_ecg` 前自动做 z-score 预处理规避此问题（`_run_window` 中 `std > 1` 时触发）。

**某导联 lead_usage_pct 始终为 0%**：U_E 机制已自动排除该导联，结果不受影响，但说明该导联可能脱落或极性错误。

---

## 基于不确定性提取高质量片段（仅需 CH20）

不需要 12 导联，只用上臂导联信号即可判断每个 10 秒窗口的信号质量，并提取高质量片段：

```bash
cd /home/kailong/ECG/ECG/ECGFounder/PN-QRS

# 基本用法（阈值默认 1.0）
python extract_quality_segments.py --csv /path/to/data.csv --fs 1000

# 调严阈值（只保留最干净的片段）
python extract_quality_segments.py --csv /path/to/data.csv --fs 1000 --uc_thr 0.5
```

**输出：**

| 文件 | 内容 |
|------|------|
| `*_quality_overview.png` | 全段信号，绿色=高质量窗口，红色=低质量，下方柱状图显示每窗口的 mean(U_E+U_A) |
| `*_quality_segments.png` | 所有高质量片段网格图，每格含 R-peak 红点 + 心率 + 不确定性值 |
| `*_quality_report.csv` | 每窗口的详细数值：`start_s, end_s, n_beats, mean_uc, is_good` |
| `quality_segments/*.npz` | 每个高质量片段的 NumPy 存档，可直接加载送进 ECGFounder |

**原理**：PN-QRS 在推理每个窗口时，内部同时计算 U_E（认知不确定性）和 U_A（偶然不确定性）。`mean(U_E + U_A)` 低 → 模型自信 + 信号干净；高 → 信号嘈杂或电极脱落。详见 [[PN_QRS_to_ECGFounder_pipeline#两种不确定性的含义]]。

---

## 评估上臂导联质量（CH20 vs CH1-8）

以 12 导联融合结果为伪标准答案，量化上臂导联 CH20 的检测质量：

```bash
cd /home/kailong/ECG/ECG/ECGFounder/PN-QRS

# 先运行 apply_pnqrs.py 生成 *_rpeaks.csv，再运行评估
python evaluate_upper_arm.py --data_dir /path/to/data --fs 1000
```

输出示例：

```
recording
  ref(12导联)=450  pred(CH20)=448
  TP=441  FP=7  FN=9
  Se=97.78%  P+=98.44%  F1=98.11%

汇总（3 个文件）
  Se=97.80%  P+=98.20%  F1=98.00%
```

| 指标 | 含义 |
|------|------|
| Se（敏感度） | CH20 检出了多少 12 导联检到的心拍（低 → 漏检多） |
| P+（精确率） | CH20 的检出中有多少是真实心拍（低 → 误报多） |
| F1 | 综合指标 |

> **注意**：12 导联结果本身不是绝对 ground truth，只是比单导联更可靠的参考。

---

## 原理

- **CH1–8（多导联）**：每条导联独立推理，逐帧选 U_E 最小的导联（论文 Algorithm 1）
- **CH20（单导联）**：直接推理，无导联融合
- 详见 [[PN_QRS_解读]] Section 5（U_E/U_A）和 Section 21（多导联融合）
