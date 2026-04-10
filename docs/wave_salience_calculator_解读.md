# wave_salience_calculator.py 通俗解读

> 文件路径：`PN-QRS/wave_salience_calculator.py`

---

## 一句话总结

**这是一个"心电图各波形清晰度打分"工具。**

它回答一个问题：**这段心电信号里，除了最明显的 R 波（心跳尖峰）之外，P波、Q波、S波、T波还能不能看清楚？**

---

## 背景：为什么需要这个工具？

标准 12 导联心电图（医院里做的那种）信号很干净，P/Q/S/T 波都清晰可见。

但我们用的是**上臂单导联 ECG（CH20 列）**——上臂离心脏远，信号经过肌肉、骨骼衰减，结果是：

```
标准心电图：   P波清晰 → QRS尖峰 → T波清晰
上臂心电图：   P波很弱 → QRS尖峰 → T波偏弱
```

所以需要一个工具来**量化**每种波"还有多明显"，用于判断信号质量。

---

## 核心思路：salience（显著性）分数

**Salience 分数 = 某个波的幅度 ÷ R波的幅度**

```
salience(P) = |P波高度| / |R波高度|
salience(T) = |T波高度| / |R波高度|
salience(Q) = |Q波深度| / |R波高度|
salience(S) = |S波深度| / |R波高度|
```

- 分数范围：**0 ~ 1**（超过1被截断为1）
- 分数越高 → 该波相对R波越明显
- 用**中位数**汇总多个心拍，避免个别异常拍干扰

---

## 流程图

```
输入：一段 ECG 信号（CH20 列）
         │
         ▼
   1. NeuroKit2 清洗信号
         │
         ▼
   2. 检测所有 R 峰（每次心跳的最高点）
         │
         ▼
   3. 在每个 R 峰附近，分别找 P/Q/S/T 峰
         │
         ▼
   4. 生理约束过滤（不合理的间期标记为"未检出"）
      P 波：必须在 R 峰前 80-300ms
      Q 波：必须在 R 峰前后 10-80ms
      S 波：必须在 R 峰后 10-80ms
      T 波：必须在 R 峰后 100-500ms
         │
         ▼
   5. 计算每个波的 salience 分数（幅度比值）
         │
         ▼
   6. 加权综合 → composite_salience（综合质量分）
         │
         ▼
输出：P/Q/S/T 各自的 salience + 检出率 + 综合分
```

---

## 加权综合分怎么算？

```python
composite = Σ(score_i × detection_rate_i) / Σ(detection_rate_i)
```

**直觉理解**：如果 P 波只检出了 30%（经常看不到），那它对综合分的贡献权重就低；T 波检出了 90%，权重就高。

这样做的好处：**不会因为某个波检测失败就拉低整体分数**。

---

## 各波的典型值（上臂导联参考）

| 波形 | 典型 salience | 典型检出率 | 说明 |
|------|-------------|-----------|------|
| **Q 波** | 0.05–0.15 | >80% | QRS内的小向下偏转，近R波所以好找 |
| **S 波** | 0.05–0.20 | >80% | R波后的向下偏转，也近R波 |
| **T 波** | 0.15–0.35 | >80% | 心室复极波，幅度仅次于R波 |
| **P 波** | 0.05–0.15 | 50–80% | 心房收缩波，上臂导联最弱，低检出率正常 |

---

## 代码结构（类图）

```
SQICalculatorRole（抽象基类）
    └── _BaseSalienceCalculator（通用流程：检波→过滤→打分）
            ├── PWaveSalienceCalculator   P波，PR间期 80-300ms
            ├── QWaveSalienceCalculator   Q波，QR间期 10-80ms
            ├── SWaveSalienceCalculator   S波，RS间期 10-80ms
            └── TWaveSalienceCalculator   T波，RT间期 100-500ms

WaveSalienceCalculator（综合计算器）
    ├── 内含上面4个计算器
    ├── 只做一次波形检测（节省计算）
    └── 加权平均 → composite_salience
```

---

## 输入/输出

### 输入
- CSV 文件，必须有 `CH20` 列（上臂心电信号）
- 采样率 `fs`（如 1000 Hz）

### 输出文件

| 文件 | 内容 |
|------|------|
| `xxx_wave_sqi.csv` | 汇总：P/Q/S/T 各波 salience + 检出率 + 综合分 |
| `xxx_wave_sqi_detail.csv` | 逐片段（每10秒）明细（`--detail` 开启） |
| `batch_wave_sqi_summary.csv` | 批量模式下所有文件的汇总表 |

### 输出示例
```
>> subject01_sitting.csv
   fs=1000Hz  duration=60.0s  segments=6
   P: salience=0.082  detection=68.3%  (41/60)
   Q: salience=0.134  detection=95.0%  (57/60)
   S: salience=0.178  detection=96.7%  (58/60)
   T: salience=0.241  detection=88.3%  (53/60)
   composite=0.189
```

---

## 用法

```bash
# 单文件分析
python wave_salience_calculator.py --csv /path/to/data.csv --fs 1000

# 单文件 + 逐段明细
python wave_salience_calculator.py --csv /path/to/data.csv --fs 1000 --detail

# 批量分析整个目录
python wave_salience_calculator.py --batch --data_dir /path/to/data_dir --fs 1000
```

---

## 与 PN-QRS 的关系

这个工具是 PN-QRS 项目中**信号质量评估（SQI）模块**的一部分。

| 工具 | 作用 |
|------|------|
| `qrs_post_process.py` | 用**不确定性**（U_E/U_A）判断"这段信号是不是 ECG" |
| `wave_salience_calculator.py` | 用**波形清晰度**判断"这段 ECG 里各波形态好不好" |

两者互补：前者过滤掉噪声段，后者评估保留段的波形质量。

---

## 关键设计细节

### 为什么用中位数而不是均值？
心电信号里偶尔有异常拍（早搏、伪差），中位数对这些异常点不敏感，均值会被拉偏。

### 为什么要分段处理（每10秒一段）？
长程信号中间可能有电极松动导致的信号空洞。分成小段后，只统计有效段，最后取中位数汇总，避免空洞污染整体评分。

### 什么是"生理约束过滤"？
NeuroKit2 有时会把噪声误检为 P 波或 T 波。加上时间约束（P 波必须在 R 峰前 80-300ms），可以过滤掉位置明显不对的假阳性检测结果。
