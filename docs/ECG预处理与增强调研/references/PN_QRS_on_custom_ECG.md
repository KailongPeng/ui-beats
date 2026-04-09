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

不需要 12 导联，只用上臂导联信号即可判断每个 10 秒窗口的信号质量，并提取高质量片段。

### 高质量片段的判定标准

每个 10 秒窗口必须**同时满足两个条件**才被标记为高质量：

| 条件 | 判断 | 说明 |
|------|------|------|
| `mean(U_E + U_A) <= uc_thr` | 不确定性足够低 | 模型对该段信号的预测自信；电极脱落时 mean_uc~10，干净信号~0.1–0.3 |
| `5 <= n_beats <= 25` | 心拍数正常 | 对应 30–150 bpm；排除完全脱落（0拍）或误检爆炸 |

> **注意**：默认阈值 `uc_thr=1.0` 是为临床 ECG 设计的。**可穿戴/上臂导联**信噪比更低，mean_uc 普遍在 1–3 之间，建议先用 `--uc_thr auto` 自动决定，或从 `--uc_thr 2.0` 开始调试。

### 单文件模式

```bash
cd /home/kailong/ECG/ECG/ECGFounder/PN-QRS

# 自动阈值（推荐，Otsu 方法自动分割好/坏窗口）
python extract_quality_segments.py --csv /path/to/data.csv --fs 1000 --uc_thr auto

# 基本用法（不确定性阈值默认 1.0，临床 ECG 适用）
python extract_quality_segments.py --csv /path/to/data.csv --fs 1000

# 调严阈值：只保留最干净的片段（阈值越低越严格）
python extract_quality_segments.py --csv /path/to/data.csv --fs 1000 --uc_thr 0.5

# 调宽阈值：尽量多保留片段（可穿戴数据推荐起点）
python extract_quality_segments.py --csv /path/to/data.csv --fs 1000 --uc_thr 2.0

# 嘈杂数据：缩小滑动步长（默认 8s → 1s），密集扫描发现夹在噪声里的干净片段
python extract_quality_segments.py --csv /path/to/data.csv --fs 1000 --uc_thr auto --step 1
```

### 批量模式（`--batch`）

数据按行为分子文件夹存放时，开启 `--batch` 递归处理所有文件，并**按行为分组汇总**：

```
data_dir/
  slow_walk/    rec1.csv  rec2.csv
  raise_hand/   rec1.csv
  sitting/      rec1.csv  rec2.csv
```

```bash
cd /home/kailong/ECG/ECG/ECGFounder/PN-QRS

# 递归扫描，按行为子目录自动分组
python extract_quality_segments.py --batch --data_dir /path/to/data_dir --fs 1000

# 批量 + 自动阈值（两遍：先全量推理，汇总 Otsu 阈值，再统一过滤）
python extract_quality_segments.py --batch --data_dir /path/to/data_dir --fs 1000 --uc_thr auto

# 批量 + 手动阈值
python extract_quality_segments.py --batch --data_dir /path/to/data_dir --fs 1000 --uc_thr 2.0

# 嘈杂数据批量：密集滑动窗口（step=2s）+ 自动阈值
python extract_quality_segments.py --batch --data_dir /path/to/data_dir --fs 1000 --uc_thr auto --step 2
```

**终端输出示例（按行为分组，小计加粗）：**

```
─────────────────────────────────────────────────────────────────────
activity      file              dur(s)  windows   good  ratio%  uc_good
─────────────────────────────────────────────────────────────────────
raise_hand                       280.0       35     12   34.3%    0.201
              └ rec1.csv         280.0       35     12   34.3%    0.201
sitting                          870.0      109     98   89.9%    0.179
              └ rec1.csv         290.0       36     33   91.7%    0.175
              └ rec2.csv         300.0       37     34   91.9%    0.181
slow_walk                        592.0       74     41   55.4%    0.193
              └ rec1.csv         312.0       39     22   56.4%    0.190
              └ rec2.csv         280.0       35     19   54.3%    0.196
─────────────────────────────────────────────────────────────────────
TOTAL                           1742.0      218    151   69.3%
```

批量模式说明：
- `activity` = CSV 相对于 `data_dir` 的**第一级子目录名**（即行为标签）
- 自动跳过脚本自身生成的 `*_quality_report.csv`、`*_rpeaks.csv`，不重复处理
- 每个文件的 PNG / NPZ / report 写在**各自所在目录**下，不混淆
- 根目录下生成 `batch_quality_summary.csv`，含所有文件的汇总指标

### 参数一览

| 参数 | 说明 | 默认 |
|------|------|------|
| `--csv` | 单文件模式：目标 CSV/Excel 路径 | 必填（单文件）|
| `--data_dir` | 批量模式：根目录路径 | 必填（批量）|
| `--batch` | 开启批量模式 flag | 关闭 |
| `--fs` | 采样率 Hz | 必填 |
| `--uc_thr` | 不确定性阈值，高于此值丢弃；填 `auto` 自动用 Otsu 方法决定 | `1.0` |
| `--step` | 滑动窗口步长（秒）；默认 8s（2s overlap）；嘈杂数据设 1–2s 可密集扫描 | `8` |
| `--infer_batch` | 每次 GPU forward 的窗口数，越大越快 | `16` |
| `--out_dir` | NPZ 保存目录（不指定则各自放在 CSV 旁） | 自动 |
| `--gpu` | 使用的 GPU 编号 | `0` |

### 输出文件

| 文件 | 位置 | 内容 |
|------|------|------|
| `*_quality_overview.png` | 各 CSV 同目录 | 全段信号 + 绿/红背景 + 不确定性柱状图 |
| `*_quality_segments.png` | 各 CSV 同目录 | 高质量片段网格，含 R-peak 红点 + 心率 + uc 值 |
| `*_quality_report.csv` | 各 CSV 同目录 | 每窗口明细：`start_s, end_s, n_beats, mean_uc, is_good` |
| `quality_segments/*.npz` | 各 CSV 同目录 | 每个高质量片段的 NumPy 存档，可直接送进 ECGFounder |
| `batch_quality_summary.csv` | `data_dir` 根目录 | 所有文件汇总：`activity, rel_path, duration_s, n_windows, n_good, good_ratio_pct, mean_uc_good, mean_uc_all, mean_beats_good` |

**原理**：PN-QRS 推理时同时计算 U_E（认知不确定性）和 U_A（偶然不确定性）。`mean(U_E + U_A)` 低 → 信号干净；高 → 噪声或电极脱落。详见 [[PN_QRS_to_ECGFounder_pipeline#两种不确定性的含义]]。

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
