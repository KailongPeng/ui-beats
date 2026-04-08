---
title: PN-QRS 应用于自采 Excel ECG 数据
tags: [PN-QRS, ECG, QRS-detection, guide]
date: 2026-04-08
type: guide
up: "[[index]]"
related: "[[PN_QRS_解读]]"
---

# PN-QRS 应用于自采 Excel ECG 数据

> ← [[index|返回索引]] | 实现代码：`PN-QRS/apply_pnqrs.py`

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

## 常见问题

**CH20 漏检多**：上臂导联 QRS 可能倒置，在 `load_excel_ecg` 调用后尝试 `rec.ch_upper_arm = -rec.ch_upper_arm`，再对比检测数量。

**采样率推断错误**：加 `--fs 250`（或设备实际采样率）手动指定。

**某导联 lead_usage_pct 始终为 0%**：U_E 机制已自动排除该导联，结果不受影响，但说明该导联可能脱落或极性错误。

---

## 原理

- **CH1–8（多导联）**：每条导联独立推理，逐帧选 U_E 最小的导联（论文 Algorithm 1）
- **CH20（单导联）**：直接推理，无导联融合
- 详见 [[PN_QRS_解读]] Section 5（U_E/U_A）和 Section 21（多导联融合）
