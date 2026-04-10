---
title: 新设备信号质量验证指南
tags: [device-validation, SQA, morphology, PN-QRS, wearable]
date: 2026-04-09
type: guide
up: "[[index]]"
related: "[[PN_QRS_on_custom_ECG]]", "[[PN_QRS_uncertainty_analysis]]"
---

# 新设备信号质量验证指南

> ← [[index|返回索引]] | 相关代码：`PN-QRS/evaluate_upper_arm.py` · `PN-QRS/extract_quality_segments.py`

---

## 核心问题：数据筛选 vs 设备验证

这是两个根本不同的目标，需要的分析深度完全不同。

| 目标 | 核心问题 | 需要的分析 | PN-QRS 够吗 |
|------|---------|-----------|------------|
| ECGFounder 预训练数据筛选 | 这段信号**能不能用** | 有没有 ECG、心拍数正常吗 | ✅ 够 |
| **设备验证** | 设备采到的信号**质量如何** | 信号有没有失真、形态是否正常 | ⚠️ 不够，需要补充 |

设备验证要回答的问题更细：

- 信号有没有系统性失真（放大器非线性、滤波截频偏移）？
- 导联极性对不对（QRS 是否正向）？
- 基线漂移在不同动作下有多严重？
- CH20（上臂导联）的 QRS 形态与标准导联 CH1-8 一致程度如何？
- 哪种动作/体位下信号质量最差？
- 设备在长时间佩戴后信号质量是否稳定下降（电极老化）？

---

## 为什么 PN-QRS 不够

PN-QRS 的 `mean_uc` 只回答"**QRS 找不找得到**"，不回答"**QRS 形态对不对**"：

```
mean_uc 低（质量好）≠ 形态正常

例子：
  基线漂移严重但 QRS 仍可辨认  → mean_uc 低，但信号形态差
  LBBB（宽 QRS 异常形态）      → mean_uc 低，但形态与正常心跳完全不同
  高频肌电干扰叠在 QRS 上      → mean_uc 偏低，但 SNR 实际很差
```

PN-QRS 能发现的问题：电极脱落、纯噪声、完全无法辨认的信号。

PN-QRS 发现不了的问题：基线漂移程度、QRS 形态失真、信噪比具体数值、P 波/T 波完整性。

---

## 推荐的设备验证流程

```
采集一段标准动作数据（每种动作 3–5 分钟）
静坐 / 慢走 / 快走 / 抬手 / 弯腰
           │
           ▼
【第一层：PN-QRS 基础筛查】
  extract_quality_segments.py --batch --uc_thr auto
  → 有没有信号？心拍数正常吗？各动作好片段比例是多少？
           │
           ▼
【第二层：R-peak 一致性验证】
  evaluate_upper_arm.py
  → CH20 vs CH1-8 的 Se / P+ / F1
  → 间接反映 QRS 形态是否足够被检测器识别
           │
           ▼
【第三层：形态学分析】（本文档重点）
  → QRS 模板相关系数
  → SNR 估计
  → 基线漂移幅度
           │
           ▼
按动作分组汇总，找出最差场景
```

---

## 第三层：形态学分析方法

### 3.1 QRS 模板相关系数（最直接）

**原理**：提取所有心拍（R-peak ± 固定窗口），计算每拍与平均模板的相关系数。相关系数越低，说明该拍形态越偏离典型波形（可能是伪差、形态失真、或真实异常心拍）。

```python
import numpy as np

def qrs_template_correlation(signal, rpeaks, fs, half_win_ms=200):
    """
    计算每个心拍与平均模板的相关系数。

    参数：
        signal   : 1D 信号数组（采样点）
        rpeaks   : R-peak 采样点索引数组
        fs       : 采样率（Hz）
        half_win_ms : R-peak 两侧各取多少毫秒（默认 ±200ms）

    返回：
        corrs    : 每个有效心拍的相关系数（-1 到 1，越接近 1 越好）
        template : 平均模板
    """
    half_win = int(half_win_ms / 1000 * fs)
    beats = []
    valid_idx = []

    for i, r in enumerate(rpeaks):
        s, e = r - half_win, r + half_win
        if s >= 0 and e < len(signal):
            beat = signal[s:e].astype(float)
            # Z-score 归一化（消除幅度差异，只看形态）
            std = beat.std()
            if std > 1e-6:
                beat = (beat - beat.mean()) / std
                beats.append(beat)
                valid_idx.append(i)

    if len(beats) < 2:
        return np.array([]), np.array([])

    beats = np.stack(beats)          # (N, 2*half_win)
    template = beats.mean(axis=0)    # 平均模板

    corrs = np.array([
        np.corrcoef(b, template)[0, 1] for b in beats
    ])
    return corrs, template


# 使用示例
corrs, template = qrs_template_correlation(signal, rpeaks, fs=1000)

print(f"平均相关系数: {corrs.mean():.3f}")
print(f"相关系数 < 0.8 的心拍占比: {(corrs < 0.8).mean()*100:.1f}%")
# 经验参考：
#   > 0.95  → 优质信号，形态一致
#   0.85–0.95 → 可接受，有轻微形态变异
#   0.70–0.85 → 较差，基线漂移或伪差较多
#   < 0.70  → 信号质量很差
```

**局限**：相关系数对整体形态敏感，对幅度不敏感（已归一化）。如果设备导联极性反了（QRS 倒置），模板本身也是倒置的，相关系数仍然接近 1——所以需要额外检查 QRS 极性。

### 3.2 QRS 极性检查

```python
def check_qrs_polarity(signal, rpeaks, fs, half_win_ms=100):
    """检查 QRS 主波方向：正向（R 波向上）还是反向（QS 波型）"""
    half_win = int(half_win_ms / 1000 * fs)
    peak_vals = []
    for r in rpeaks:
        s, e = r - half_win, r + half_win
        if s >= 0 and e < len(signal):
            seg = signal[s:e]
            peak_vals.append(seg.max() - seg.mean())  # 最大偏移

    peak_vals = np.array(peak_vals)
    pos_ratio = (peak_vals > 0).mean()
    print(f"正向 QRS 占比: {pos_ratio*100:.1f}%")
    if pos_ratio < 0.5:
        print("⚠️  QRS 主波方向可能反转，建议对信号取反后重跑")
    return pos_ratio
```

### 3.3 SNR 估计

**原理**：把信号分成"QRS 段"和"非 QRS 段"，用非 QRS 段的方差估计噪声功率，QRS 段的峰值功率估计信号功率。

```python
def estimate_snr(signal, rpeaks, fs, qrs_half_ms=60, noise_margin_ms=200):
    """
    估计 ECG 信号的 SNR（dB）。

    qrs_half_ms    : QRS 复合波半宽（±60ms 覆盖 QRS，约 120ms 总宽）
    noise_margin_ms: QRS 两侧各留多少毫秒作为"纯噪声段"的起始间距
    """
    qrs_half = int(qrs_half_ms / 1000 * fs)
    margin   = int(noise_margin_ms / 1000 * fs)

    # 建立 QRS mask
    qrs_mask = np.zeros(len(signal), dtype=bool)
    for r in rpeaks:
        s = max(0, r - qrs_half)
        e = min(len(signal), r + qrs_half)
        qrs_mask[s:e] = True

    # 噪声段：距所有 R-peak 足够远的区域
    noise_mask = np.zeros(len(signal), dtype=bool)
    for r in rpeaks:
        s = max(0, r - qrs_half - margin)
        e = min(len(signal), r + qrs_half + margin)
        noise_mask[s:e] = True
    noise_mask = ~noise_mask  # 取反：远离 QRS 的区域

    if noise_mask.sum() < fs * 0.5:  # 噪声段太少
        return np.nan

    signal_power = np.var(signal[qrs_mask])
    noise_power  = np.var(signal[noise_mask])

    if noise_power < 1e-12:
        return np.inf

    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db

# 经验参考（临床 ECG）：
#   > 20 dB  → 优质信号
#   10–20 dB → 可接受
#   < 10 dB  → 噪声严重
# 注意：可穿戴上臂导联 SNR 普遍低 5–10 dB，参考值需相应下调
```

### 3.4 基线漂移幅度

```python
def baseline_wander_amplitude(signal, fs, cutoff_hz=0.5):
    """
    估计基线漂移幅度（峰峰值，单位与信号相同）。
    基线 = 低通滤波后的信号（< 0.5Hz）。
    """
    from scipy.signal import butter, filtfilt

    # 低通滤波提取基线
    b, a = butter(2, cutoff_hz / (fs / 2), btype='low')
    baseline = filtfilt(b, a, signal)

    amplitude = baseline.max() - baseline.min()
    # 相对幅度（相对于信号整体峰峰值）
    relative = amplitude / (signal.max() - signal.min())
    return amplitude, relative

# 经验参考：
#   relative < 0.1  → 基线漂移轻微
#   0.1–0.3         → 中等，可接受
#   > 0.3           → 严重，影响 ST 段分析
```

---

## 汇总报告格式

建议按动作分组输出以下表格：

| 动作 | 好片段% | 平均相关系数 | 相关系数<0.8占比 | SNR(dB) | 基线漂移% | Se vs CH1-8 |
|------|---------|------------|----------------|---------|----------|------------|
| 静坐 | 89.9% | 0.972 | 2.1% | 18.3 | 8.2% | 99.1% |
| 慢走 | 55.4% | 0.921 | 11.3% | 12.7 | 19.4% | 97.8% |
| 快走 | 34.3% | 0.873 | 28.6% | 9.2 | 31.1% | 95.3% |
| 抬手 | 41.2% | 0.845 | 35.2% | 8.8 | 28.7% | 94.6% |

通过这张表可以直接看出：
- 哪种动作下设备最不稳定
- 质量下降的主要原因（SNR 低 → 高频噪声；基线漂移高 → 低频干扰；相关系数低 → 形态失真）
- 是否需要对某种动作做特殊处理（如加强滤波、调整佩戴位置）

---

## 各指标的分工

```
PN-QRS mean_uc（已有）
  └── 判断：有没有可识别的 ECG 信号
        → 过滤电极脱落、纯噪声

evaluate_upper_arm.py Se/P+（已有）
  └── 判断：CH20 能不能代替多导联做 QRS 检测
        → 间接反映形态是否足够清晰

QRS 模板相关系数（待实现）
  └── 判断：心拍形态是否一致、有无系统性失真
        → 直接量化形态质量

SNR 估计（待实现）
  └── 判断：高频噪声有多严重
        → 量化噪声水平

基线漂移幅度（待实现）
  └── 判断：低频漂移有多严重
        → 影响 ST 段、T 波分析的先决条件
```

---

## 已有代码 vs 待实现

| 功能 | 状态 | 脚本 |
|------|------|------|
| PN-QRS 推理 + 好片段筛选 | ✅ 完成 | `extract_quality_segments.py` |
| CH20 vs CH1-8 一致性评估 | ✅ 完成 | `evaluate_upper_arm.py` |
| QRS 模板相关系数 | ⚠️ 代码片段在 §3.1 | — |
| SNR 估计 | ⚠️ 代码片段在 §3.3 | — |
| 基线漂移幅度 | ⚠️ 代码片段在 §3.4 | — |
| P/Q/S/T 波显著性 SQI | ✅ 完成 | `wave_salience_calculator.py` |
| 按动作分组汇总报告 | ✅ 完成 | `wave_salience_calculator.py --batch` + `extract_quality_segments.py --batch` |

---

## 文献调研：可用的形态学 SQA 算法全景

> 调研时间：2026-04-09，覆盖学术文献 + 开源工具箱

### 4.1 经典统计类指标（无需训练，可直接用）

| 指标 | 全称 | 衡量什么 | 核心计算 | 单导联可用 |
|------|------|---------|---------|-----------|
| **kSQI** | Kurtosis SQI | 信号峰度——QRS 有尖锐峰，噪声峰度低 | `kurtosis(signal)`；> 5 为干净，< 5 为噪声 | ✅ |
| **pSQI** | Power SQI | QRS 频段（5–20 Hz）占总功率比例 | `P(5-20Hz) / P(0-62.5Hz)` | ✅ |
| **basSQI** | Baseline SQI | 基线漂移功率占比（< 1Hz）| `1 - P(0-1Hz) / P(0-40Hz)` | ✅ |
| **sSQI** | Skewness SQI | 信号偏斜度（噪声分布对称，ECG 不对称）| `skewness(signal)` | ✅ |
| **SNRsqi** | SNR-based SQI | 信噪比（dB）| QRS 段功率 / 非 QRS 段功率 | ✅ |

**经验阈值（临床 ECG）：**
- kSQI > 5 且 pSQI > 0.5 → 可接受
- SNR > 18 dB → 适合全波形分析；10–18 dB → 仅适合 QRS 检测；< 10 dB → 噪声严重
- 注：可穿戴上臂导联 SNR 普遍低 5–10 dB，阈值应相应下调

**实现：** NeuroKit2 内置，`pip install neurokit2`
```python
import neurokit2 as nk
quality = nk.ecg_quality(ecg_signal, sampling_rate=fs, method="zhao2018")
# 返回 "Excellent" / "Barely acceptable" / "Unacceptable"
```

---

### 4.2 基于 R-peak 一致性的方法

| 方法 | 思路 | 适合场景 | 局限 |
|------|------|---------|------|
| **bSQI** | 两种 QRS 检测器结果吻合率 | 通用 SQA | 两个检测器都骗过则失效 |
| **GbSQI** | 4 种检测器的泛化版（推荐用于可穿戴）| 可穿戴 ECG | 计算量较大 |
| **QRS 模板相关系数** | 每拍 vs 平均模板的 Pearson 相关 | 直接量化形态一致性 | 不检测系统性漂移 |
| **Beat Dissimilarity Index** | 每拍与模板的形态差异量 | 伪差检测 | 同上 |

GbSQI 文献建议：使用 U3 / UNSW / DOM / OKB 四种检测器，投票决策。

---

### 4.3 深度学习方法

| 方法 | 输入 | 特点 | 实现状态 |
|------|------|------|---------|
| **dECG-CNN** | 导数 ECG（一阶差分）| 轻量，适合嵌入式 | 研究代码，无统一开源包 |
| **cGAN + CNN** | 原始 ECG | 用生成对抗网络做数据增强，提升噪声类别覆盖 | 研究代码 |
| **LSTM/RNN** | 原始 ECG 序列 | 捕捉跨心拍的时序一致性 | 研究代码 |
| **PN-QRS U_E**（已有）| 原始 ECG | 轻量，单次推理，OoD 检测 | ✅ 已集成 |

深度学习方法的共同问题：需要针对特定设备 / 导联位置的标注数据微调，否则泛化性差。对于上臂导联，目前没有现成的预训练质量模型。

---

### 4.4 可用工具箱汇总

| 工具箱 | 语言 | 安装 | 包含的 SQA 方法 | 适合我们吗 |
|--------|------|------|----------------|-----------|
| **NeuroKit2** | Python | `pip install neurokit2` | kSQI、pSQI、basSQI、模板相关、Zhao2018 质量分类 | ✅ 首选 |
| **ECGAssess** | Python | GitHub 安装 | 平稳性、心率、SNR 三项联合判断；94% 准确率 | ✅ 可参考 |
| **WFDB** | Python/Matlab/C | `pip install wfdb` | QRS 检测、波形标注、特征提取基础库 | ✅ 已在用 |
| **BioSPPy** | Python | `pip install biosppy` | 预处理、FIR 滤波（0.67–45 Hz） | ✅ NeuroKit2 底层 |
| **EcgScorer** | Matlab | GitHub | 通用质量评分 | ⚠️ Matlab 环境 |

---

### 4.5 针对可穿戴上臂导联的建议优先级

基于文献调研结论（上臂导联的主要问题：运动伪差 >> 基线漂移 >> 高频噪声）：

**第一优先级（立即可用，无需训练）：**
1. **kSQI**：最快速的噪声检测，NeuroKit2 一行调用
2. **pSQI / basSQI**：分别量化高频和低频干扰的比例
3. **QRS 模板相关系数**：直接量化形态一致性（代码已在本文档 §3.1 给出）

**第二优先级（增加鲁棒性）：**
4. **GbSQI**：文献推荐专门用于可穿戴场景，四检测器投票
5. **SNR 估计**：量化绝对噪声水平（代码已在本文档 §3.3 给出）

**暂不推荐（当前阶段）：**
- 深度学习 SQA：需要针对上臂导联的标注数据，目前没有
- P 波 / T 波专项质量：信息量小，实现复杂，对设备验证帮助有限

---

### 4.6 关键文献

| 论文 | 贡献 | 链接 |
|------|------|------|
| Clifford et al. (2012) | 定义 bSQI / kSQI / pSQI 等经典指标体系 | IEEE EMBC |
| Liu et al. (2018) | GbSQI：推广到 4 检测器，可穿戴推荐 | IEEE Access |
| Zhao et al. (2018) | 融合多指标的质量分类器（NeuroKit2 内置）| IEEE TBME |
| Smital et al. (2020) | 可穿戴长程 ECG 实时质量评估 | IEEE TBME |
| Krasteva et al. (2022) | ECGAssess 工具箱，94% 准确率 | PMC9120362 |
| CNN+dECG (2025) | 导数 ECG + 轻量 CNN，适合嵌入式设备 | ScienceDirect |

---

## 开源形态学 SQI 代码库调研（2026-04-09）

> 针对"波形显著性 + 区间变异性"两类形态学评分的开源实现调研

### 5.1 用户代码片段的类名来源

用户展示的类结构（`_BaseSalienceCalculator`、`PWaveSalienceCalculator` 等）**在公开 GitHub 仓库中未找到完全匹配的实现**。可能来自：
- 私有/内部代码库
- 尚未公开的研究代码
- 基于某个通用框架的自定义扩展

以下是功能最接近的开源库。

---

### 5.2 开源库对比

| 库 | P/Q/S/T 波显著性 | 区间变异性 SQI | 综合评分 | 单导联 | 安装 |
|----|----------------|--------------|---------|--------|------|
| **vital_sqi** | ❌ | ✅（DTW、qrs_energy）| ✅（Ruleset，74 项指标）| ✅ | `pip install vital-sqi` |
| **ekg_tda** | ✅（拓扑方法）| ✅（PR/QT/QRS/P宽/T宽）| ❌ | ✅ | GitHub clone |
| **NeuroKit2** | ⚠️ 给位置，不给显著性分 | ✅（zhao2018：pSQI/kSQI/basSQI）| ✅ | ✅ | `pip install neurokit2` |
| **PyECG** | ⚠️ 仅 Q onset + T offset | ✅（QT/QTc/QTVI）| ❌ | ✅ | GitHub clone |
| **ecg_qc** | ❌ | ❌ | ✅（ML 分类，4 个模型）| ✅ | `pip install ecg-qc` |
| **ECGAssess** | ❌ | ❌ | ✅（平稳性 + 心率 + SNR）| ✅ | GitHub clone |

---

### 5.3 各库详解

#### vital_sqi（最完整的 SQI 框架）

- **GitHub**：https://github.com/Oucru-Innovations/vital-sqi
- **论文**：Frontiers in Physiology, 2022
- **特点**：74 项 SQI 指标 + Rule/Ruleset 框架，支持自定义组合规则
- **适合用途**：综合质量流水线
- **局限**：没有显式的 P/Q/S/T 波相对 R 波的显著性打分

```python
import vital_sqi as sq
# 计算 kSQI、DTW 模板相关等
sqis = sq.compute_sqi(ecg_segment, fs=1000)
```

#### ekg_tda（最接近"波段显著性"需求）

- **GitHub**：https://github.com/hdlugas/ekg_tda
- **特点**：用**持续同调**（拓扑方法）检测 P/Q/S/T 波，直接输出各波的 prominence 和时间位置
- **输出**：PR 间期、QT 间期、QRS 宽度、P 波宽度、T 波宽度
- **适合用途**：区间变异性 SQI + 波形检出质量

```python
# 输出各波检出位置和 prominence
from ekg_tda import detect_waves
waves = detect_waves(ecg_signal, fs=500)
# waves['P_prominence'], waves['T_prominence'], waves['PR_interval'], ...
```

#### NeuroKit2（最易用，推荐入门）

- **GitHub**：https://github.com/neuropsychology/NeuroKit
- **特点**：一行调用 zhao2018 综合质量分类（Excellent / Barely acceptable / Unacceptable）
- **波形检测**：给出 P/T/QRS 的峰值位置，但不直接给显著性分数
- **适合用途**：kSQI + pSQI + basSQI 快速评估

```python
import neurokit2 as nk

# 方法1：综合质量分类
quality = nk.ecg_quality(ecg, sampling_rate=fs, method="zhao2018")
# 返回 "Excellent" / "Barely acceptable" / "Unacceptable"

# 方法2：模板相关系数（平均 QRS 法）
quality_score = nk.ecg_quality(ecg, sampling_rate=fs, method="averageQRS")
# 返回 0–1 的连续分数

# 方法3：波形检测（获取 P/T 波位置）
signals, info = nk.ecg_process(ecg, sampling_rate=fs)
# signals["ECG_P_Peaks"], signals["ECG_T_Peaks"] 等
```

---

### 5.4 如何自己实现"波段显著性"打分

如果需要类似用户展示的 `PWaveSalienceCalculator` 框架，可基于 NeuroKit2 的波形位置检测，自行计算各波相对 R 波的显著性：

```python
import neurokit2 as nk
import numpy as np

def wave_salience(ecg, fs):
    """
    计算 P/Q/S/T 各波相对于 R 波幅度的显著性（0–1）。
    需要 NeuroKit2 先检测各波位置。
    """
    signals, info = nk.ecg_process(ecg, sampling_rate=fs)

    r_peaks   = info["ECG_R_Peaks"]
    p_peaks   = np.where(signals["ECG_P_Peaks"] == 1)[0]
    t_peaks   = np.where(signals["ECG_T_Peaks"] == 1)[0]
    q_peaks   = np.where(signals["ECG_Q_Peaks"] == 1)[0]
    s_peaks   = np.where(signals["ECG_S_Peaks"] == 1)[0]

    r_amp = np.median(np.abs(ecg[r_peaks]))  # R 波幅度基准

    def salience(peaks, ref_amp):
        if len(peaks) == 0:
            return 0.0, 0.0   # 分数, 检出率
        amps = np.abs(ecg[peaks])
        score = float(np.median(amps) / (ref_amp + 1e-9))
        detection_rate = len(peaks) / max(len(r_peaks), 1)
        return min(score, 1.0), min(detection_rate, 1.0)

    p_score, p_rate = salience(p_peaks, r_amp)
    q_score, q_rate = salience(q_peaks, r_amp)
    s_score, s_rate = salience(s_peaks, r_amp)
    t_score, t_rate = salience(t_peaks, r_amp)

    # 加权综合（以检出率为权重）
    scores  = [p_score, q_score, s_score, t_score]
    weights = [p_rate,  q_rate,  s_rate,  t_rate]
    total_w = sum(weights) + 1e-9
    composite = sum(s * w for s, w in zip(scores, weights)) / total_w

    return {
        "P_salience": p_score,  "P_detection_rate": p_rate,
        "Q_salience": q_score,  "Q_detection_rate": q_rate,
        "S_salience": s_score,  "S_detection_rate": s_rate,
        "T_salience": t_score,  "T_detection_rate": t_rate,
        "composite_salience": composite,
    }
```

---

### 5.5 区间变异性 SQI（PR/QT/QTc/波宽）

```python
import neurokit2 as nk
import numpy as np

def interval_variability_sqi(ecg, fs):
    """
    计算各心拍间区间的变异性，变异性越低 → 节律越稳定 → SQI 越高。
    返回 0–1 的 SQI 分数（1=最稳定）。
    """
    signals, info = nk.ecg_process(ecg, sampling_rate=fs)

    def cv_to_sqi(values):
        """变异系数（CV = std/mean）转 SQI：CV 越低越好"""
        if len(values) < 2:
            return np.nan
        cv = np.std(values) / (np.mean(values) + 1e-9)
        return float(np.exp(-5 * cv))   # CV=0 → SQI=1.0；CV=0.2 → SQI≈0.37

    # RR 间期（ms）
    rr = np.diff(info["ECG_R_Peaks"]) / fs * 1000
    rr_sqi = cv_to_sqi(rr)

    # QRS 宽度（ms）—— Q onset 到 S offset
    qrs_on  = np.where(signals["ECG_Q_Peaks"] == 1)[0]
    qrs_off = np.where(signals["ECG_S_Peaks"] == 1)[0]
    n = min(len(qrs_on), len(qrs_off))
    qrs_width = (qrs_off[:n] - qrs_on[:n]) / fs * 1000 if n > 1 else np.array([])
    qrs_sqi = cv_to_sqi(qrs_width)

    return {
        "RR_variability_sqi":  rr_sqi,
        "QRS_width_sqi":       qrs_sqi,
        "composite_interval_sqi": np.nanmean([rr_sqi, qrs_sqi]),
    }
```

> **注意**：PR 间期、QT 间期、QTc 需要 P 波和 T 波的精确 onset/offset 检测，NeuroKit2 的 `dwt` 方法在干净信号上可检测，但上臂导联 P 波幅度小，检出率可能不稳定。建议先验证 P 波检出率再启用 PR 间期计算。

---

### 5.6 推荐组合方案

针对上臂可穿戴 ECG 设备验证：

```
Step 1：NeuroKit2 zhao2018    → 整体质量分类（Excellent/Acceptable/Unacceptable）
Step 2：basSQI                → 基线漂移评估（ST 段是否可信）
Step 3：wave_salience()       → P/T 波检出率和显著性（上臂导联最关键）
Step 4：interval_variability_sqi() → RR / QRS 宽度变异性
Step 5：汇总报告              → 按动作分组，输出各动作的平均分
```

---

## 相关文档

- [[PN_QRS_on_custom_ECG]] — apply_pnqrs.py 使用方法、evaluate_upper_arm.py 说明
- [[PN_QRS_uncertainty_analysis]] — U_E/U_A 的精确定义与代码实现对比
- [[PN_QRS_to_ECGFounder_pipeline]] — 数据筛选（预训练用途，不需要形态学分析）
