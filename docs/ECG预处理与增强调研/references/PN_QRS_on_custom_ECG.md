---
title: PN-QRS 应用于自采 Excel ECG 数据
tags: [PN-QRS, ECG, QRS-detection, guide]
date: 2026-04-08
type: guide
up: "[[index]]"
related: "[[PN_QRS_解读]]"
---

# PN-QRS 应用于自采 Excel ECG 数据

> ← [[index|返回索引]] | 实现代码：`PN-QRS/apply_pnqrs.py` · `PN-QRS/visualize_rpeaks.py` · `PN-QRS/evaluate_upper_arm.py` · `PN-QRS/extract_quality_segments.py` · `PN-QRS/wave_salience_calculator.py`

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
| `mean_uc <= uc_thr` | 噪声帧占比足够低 | 见下方原理说明；电极脱落时 mean_uc~10，干净信号~0.1–0.5 |
| `5 <= n_beats <= 25` | 心拍数正常 | 对应 30–150 bpm；排除完全脱落（0拍）或误检爆炸 |

**`mean_uc` 的实际计算（`uncertain_est()` 内部）：**

```python
au  = en_est(logits)     # 逐帧预测熵（偶然不确定性，连续值）[T]
eu  = mi_est(logits)     # 逐帧 KL 散度（认知不确定性，连续值）[T]
eu[eu > 0.12] = 10       # 关键：把噪声帧二值化 → 10（硬判决）
mean_uc = mean(eu + au)  # 对所有帧取均值
```

`eu > 0.12` 的帧被直接打成 10，所以 `mean_uc` 实质上是**噪声帧占比的加权计数**，而不是论文里连续定义的 U_E + U_A 均值。只要窗口里有少数几帧被判定为噪声，mean_uc 就会被显著拉高。

> **注意**：论文里 U_E 和 U_A 是分开使用的（U_E 判断导联有效性，U_A 标记可疑心拍），`uncertain_est()` 的合并写法是代码层面的工程适应，并非论文定义。

> **阈值建议**：默认 `uc_thr=1.0` 是为临床 ECG 设计的。**可穿戴/上臂导联**信噪比更低，mean_uc 普遍在 1–3 之间，建议先用 `--uc_thr auto` 自动决定，或从 `--uc_thr 2.0` 开始调试。

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
| `*_quality_overview.png` | 各 CSV 同目录 | 全段信号 + 绿/红背景 + 不确定性折线/柱状图 |
| `*_quality_segments.png` | 各 CSV 同目录 | 高质量片段网格，含 R-peak 红点 + 心率 + uc 值 |
| `*_uc_distribution.png` | 各 CSV 同目录 | mean_uc 直方图 + 阈值线；绿字"Bimodal"= 阈值可信，红字"Unimodal"= 建议手动设置 |
| `*_quality_report.csv` | 各 CSV 同目录 | 每窗口明细：`start_s, end_s, n_beats, mean_uc, is_good` |
| `quality_segments/*.npz` | 各 CSV 同目录 | 每个高质量片段的 NumPy 存档，可直接送进 ECGFounder |
| `batch_quality_summary.csv` | `data_dir` 根目录 | 所有文件汇总：`activity, rel_path, duration_s, n_windows, n_good, good_ratio_pct, mean_uc_good, mean_uc_all, mean_beats_good` |
| `batch_uc_distribution.png` | `data_dir` 根目录 | 批量 auto 模式专有：所有文件的 mean_uc pooled 分布 + Otsu 阈值线 |

**原理**：`uncertain_est()` 逐帧计算认知不确定性（eu）和偶然不确定性（au），将 `eu > 0.12` 的帧二值化为 10，然后对 `eu + au` 取窗口均值得到 `mean_uc`。该值本质上反映了窗口内**噪声帧的占比**：干净信号 ~0.1–0.5，电极脱落 ~10。详见 [[PN_QRS_to_ECGFounder_pipeline#质量分数的含义]]。

---

## 波形显著性 SQI（P/Q/S/T 波质量评估）

`mean_uc` 只回答"QRS 找不找得到"，不回答"P 波 / T 波 / ST 段可不可见"。对于设备验证和需要完整 PQRST 形态的下游任务（如心梗检测），需要额外的形态学质量评估。

`wave_salience_calculator.py` 基于 NeuroKit2 波形检测，计算每个波段（P/Q/S/T）相对于 R 波的**显著性分数**（salience）和**检出率**（detection rate），并以检出率为权重计算综合评分。

### 快速开始

```bash
cd /home/kailong/ECG/ECG/ECGFounder/PN-QRS

# 单文件分析
python wave_salience_calculator.py --csv /path/to/data.csv --fs 1000

# 输出逐段明细
python wave_salience_calculator.py --csv /path/to/data.csv --fs 1000 --detail

# 批量模式（按行为子目录分组）
python wave_salience_calculator.py --batch --data_dir /path/to/data_dir --fs 1000
```

### 输出示例

```
>> data.csv
   fs=1000Hz  duration=120.0s  segments=12
   P: salience=0.127  detection=73.2%  (104/142)
   Q: salience=0.085  detection=91.5%  (130/142)
   S: salience=0.203  detection=95.8%  (136/142)
   T: salience=0.312  detection=80.3%  (114/142)
   composite=0.207
```

### 参数一览

| 参数 | 说明 | 默认 |
|------|------|------|
| `--csv` | 单文件模式：CSV/Excel 路径 | 必填（单文件）|
| `--data_dir` | 批量模式：根目录路径 | 必填（批量）|
| `--batch` | 开启批量模式 | 关闭 |
| `--fs` | 采样率 Hz | 必填 |
| `--segment_sec` | 分析片段长度（秒）| `10` |
| `--detail` | 输出逐段明细 CSV | 关闭 |
| `--out_dir` | 输出目录（不指定则放 CSV 旁）| 自动 |

### 输出文件

| 文件 | 内容 |
|------|------|
| `*_wave_sqi.csv` | 汇总：各波 salience、detection_rate、composite |
| `*_wave_sqi_detail.csv` | 逐段明细（`--detail` 时生成）|
| `batch_wave_sqi_summary.csv` | 批量汇总（`--batch` 时生成）|

### 各波评分含义

| 指标 | 计算 | 含义 |
|------|------|------|
| salience | `median(|wave_amp| / |R_amp|)` | 该波相对 R 波的幅度比值（0-1）|
| detection_rate | `n_detected / n_R_peaks` | 该波被检出的心拍占比 |
| composite | `Σ(salience × detection_rate) / Σ(detection_rate)` | 加权综合 |

### 上臂导联 CH20 参考范围

| 波段 | 临床 ECG 典型值 | 上臂 CH20 预期范围 | 说明 |
|------|---------------|------------------|------|
| P | 0.10–0.25 | **0.05–0.15** | 上臂 P 波很弱，检出率 < 70% 正常 |
| Q | 0.05–0.15 | 0.03–0.10 | Q 波幅度小 |
| S | 0.10–0.30 | 0.10–0.25 | S 波相对稳定 |
| T | 0.20–0.40 | **0.15–0.35** | T 波检出率应 > 80%，否则 ST 分析不可靠 |

### 类结构

```
SQICalculatorRole (ABC)
  └── _BaseSalienceCalculator
        ├── _get_wave_array()       ← 提取波峰位置
        ├── _get_amplitudes()       ← 提取振幅
        ├── _filter_intervals()     ← 过滤不合理间期
        ├── _compute_salience_score()← 振幅→分数
        ├── _segment_by_gaps()      ← 按 R-peak 间隔分段
        ├── PWaveSalienceCalculator  (PR: 80-300ms)
        ├── QWaveSalienceCalculator  (QR: 10-80ms)
        ├── SWaveSalienceCalculator  (SR: 10-80ms)
        ├── TWaveSalienceCalculator  (TR: 100-500ms)
        └── WaveSalienceCalculator   综合评分（检出率加权）
```

---

## 波形显著性 SQI（domain 库版，wave_salience_calculator_call.py）

`wave_salience_calculator.py` 基于 NeuroKit2 自行实现波形检测。如果项目中已有 `_wave_salience_calculator.py`（domain SQI 库），可改用 `wave_salience_calculator_call.py` 作为调用层，它封装了相同的 P/Q/S/T/综合计算器接口，输出格式更结构化。

### 前置条件

**必须先运行 `apply_pnqrs.py`**，生成 `*_CH20_rpeaks.csv`（或 `*_CH1-8_rpeaks.csv`）。`wave_salience_calculator_call.py` 直接读取这些文件，不重复做 R-peak 检测。

### 快速开始

```bash
cd /home/kailong/ECG/ECG/ECGFounder/PN-QRS

# 单文件
python wave_salience_calculator_call.py --csv /path/to/data.csv --fs 1000

# 单文件 + 分波明细 CSV
python wave_salience_calculator_call.py --csv /path/to/data.csv --fs 1000 --detail

# 批量（递归扫描子目录）
python wave_salience_calculator_call.py --batch --data_dir /path/to/data_dir --fs 1000

# 批量 + 明细 + 指定输出目录
python wave_salience_calculator_call.py --batch --data_dir /path/to/data_dir --fs 1000 --detail --out_dir /path/to/out
```

### 输出示例（终端）

```
[1/33] 被试1/坐姿抬手/rec01.csv
  fs=1000Hz  duration=36.0s  r_peaks=49
    P: value=0.082  confidence=100.0%  score=0.328
    Q: value=0.051  confidence=94.2%   score=0.204
    S: value=0.187  confidence=97.1%   score=0.748
    T: value=0.241  confidence=81.3%   score=0.603
  → composite  value=0.157  score=0.521  conf=93.2%

──────────────────────────────────────────────────────────────────────────
file                                       dur   P_val  P_conf   T_val  T_conf    comp
──────────────────────────────────────────────────────────────────────────
rec01.csv                                 36.0   0.082  100.0%   0.241   81.3%   0.521
```

### 参数一览

| 参数 | 说明 | 默认 |
|------|------|------|
| `--csv` | 单文件模式：CSV/Excel 路径 | 必填（单文件）|
| `--data_dir` | 批量模式：根目录路径 | 必填（批量）|
| `--batch` | 开启批量模式 | 关闭 |
| `--fs` | 采样率 Hz | 必填 |
| `--detail` | 同时输出每个波的明细 CSV | 关闭 |
| `--out_dir` | 输出目录（默认与 `--data_dir` 或 `--csv` 同目录）| 自动 |

### 输出文件

| 文件 | 内容 |
|------|------|
| `wave_sqi_summary_.csv` | 每个文件一行：`composite_value/score/conf` + `P/Q/S/T_value/confidence/score` |
| `wave_sqi_detail_.csv` | 每波一行，含 `description` 字段（`--detail` 时生成）|

### 与 wave_salience_calculator.py 的区别

| 维度 | wave_salience_calculator.py | wave_salience_calculator_call.py |
|------|----------------------------|---------------------------------|
| 波形检测后端 | NeuroKit2（内置） | `_wave_salience_calculator.py`（domain 库）|
| R-peak 来源 | 脚本内部重新检测 | 读取 `apply_pnqrs.py` 生成的 rpeaks CSV |
| 输出字段 | salience / detection_rate / composite | value / confidence / score / description |
| 适用场景 | 独立使用，无外部依赖 | 已跑过 apply_pnqrs，需要接入 domain SQI 框架 |

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

---

## 完整流程示例（data/0410_real）

### 数据结构

```
data/0410_real/
  被试1/
    坐姿抬手/      rec01.csv  rec02.csv  rec03.csv
    坐姿手臂前后摇摆/ rec01.csv  rec02.csv  rec03.csv
    慢走/          rec01.csv  rec02.csv  rec03.csv
    站立坐下/      rec01.csv  rec02.csv  rec03.csv
  被试2/
    坐姿抬手/      rec01.csv  rec02.csv  rec03.csv
    坐姿手臂前后摇摆/ rec01.csv  rec02.csv  rec03.csv
    坐姿说话/      rec01.csv  rec02.csv  rec03.csv
    慢走/          rec01.csv  rec02.csv  rec03.csv
    站立坐下/      rec01.csv  rec02.csv  rec03.csv
```

每个 CSV：`timestamp` + `CH1~CH8` + `CH20`，35992 行，采样率 1000 Hz（约 36 秒）。  
数据来源：4 段 MIMIC-IV 12 导联记录拼接后升采样至 1000 Hz，CH20 模拟上臂导联（lead II × 0.4 + 活动相关运动噪声）。

> **3 级目录注意**：`--batch` 模式的 `activity` 列取的是**相对于 `data_dir` 的第一级子目录名**，即这里显示 `被试1 / 被试2`，而非具体活动名。如需按活动分组，可将 `data_dir` 指向单个被试目录（如 `data/0410_real/被试1`）。

### 环境说明

本机 base 环境的 PyTorch 编译 CUDA 版本与驱动不匹配，须通过 `conda run -n ECGFounder` 切换到兼容环境（RTX 5090，CUDA 12.8）：

```bash
cd /home/kailong/ECG/ECG/ECGFounder/PN-QRS
CONDA="conda run -n ECGFounder"
DATA="data/0410_real"
```

### 步骤 1：R-peak 检测

```bash
$CONDA python apply_pnqrs.py --data_dir $DATA --fs 1000 --gpu 0
```

输出：每个 CSV 旁生成 `*_CH1-8_rpeaks.csv` 和 `*_CH20_rpeaks.csv`，根目录生成 `rpeaks_summary.json`。

### 步骤 2：R-peak 可视化

```bash
# 对单个文件可视化（指定时间段）
$CONDA python visualize_rpeaks.py \
  --csv $DATA/被试1/坐姿抬手/rec01.csv \
  --fs 1000 --duration 20 \
  --out $DATA/被试1/坐姿抬手/rec01_rpeaks_vis.png
```

### 步骤 3：不确定性质量分析

```bash
$CONDA python extract_quality_segments.py \
  --batch --data_dir $DATA --fs 1000 --uc_thr auto --gpu 0
```

### 步骤 4：上臂导联评估

```bash
$CONDA python evaluate_upper_arm.py --data_dir $DATA --fs 1000
```

### 步骤 5：波形显著性 SQI

```bash
$CONDA python wave_salience_calculator.py --batch --data_dir $DATA --fs 1000
```

### 结果汇总

**R-peak 检测（27 个文件，HR 范围 64–96 bpm）**

| 活动 | HR 典型范围 | 说明 |
|------|-----------|------|
| 坐姿系列 | 66–88 bpm | 信号相对干净 |
| 慢走 | 84–107 bpm | 运动伪迹明显 |
| 站立坐下 | 69–96 bpm | 过渡动作有瞬间噪声 |

**质量分析（extract_quality_segments，auto 阈值）**

| 主体 | 总窗口 | 好窗口 | 好窗口比例 |
|------|--------|--------|-----------|
| 被试1 | 60 | 35 | 58.3% |
| 被试2 | 75 | 48 | 64.0% |
| **汇总** | **135** | **83** | **61.5%** |

活动间差异：坐姿抬手 80–100% 好窗口；慢走仅 20–60%（运动伪迹导致 mean_uc 升高）。

**上臂导联评估（evaluate_upper_arm，以 CH1-8 为参考）**

| 指标 | 结果 |
|------|------|
| TP / FP / FN | 1246 / 50 / 40 |
| Se（敏感度） | **96.89%** |
| P+（精确率） | **96.14%** |
| F1 | **96.51%** |

**波形显著性 SQI（wave_salience_calculator）**

| 波段 | 检出率 | 说明 |
|------|--------|------|
| P 波 | ~100% | 显著性 0.07–0.60（活动依赖） |
| T 波 | ~77% | 运动伪迹下 T 波检出率下降符合预期 |
| composite SQI | 0.31 | 上臂导联参考范围内 |
