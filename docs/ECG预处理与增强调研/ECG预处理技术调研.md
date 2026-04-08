# ECG 预处理技术综述

**调研日期**: 2026-03-27
**调研背景**: MIMIC-IV-ECG 800K 预训练 + PTB-XL 9类MI亚型分类微调项目
**当前设置**: Lead I 单导联, 500Hz, 10秒 (5000采样点), Mamba (SSM) 架构, 2-Loss 预训练 (SigLIP 对比 + 重建)
**当前性能**: Macro AUROC 0.8244
**未来场景**: 上臂非标准导联可穿戴心电设备

---

## 目录

1. [经典 ECG 预处理流程](#1-经典-ecg-预处理流程)
   - 1.1 基线漂移去除
   - 1.2 工频干扰去除
   - 1.3 肌电噪声滤波
   - 1.4 R 波检测与心拍分割
   - 1.5 信号质量评估 (SQI)
   - 1.6 重采样策略
2. [深度学习时代的预处理](#2-深度学习时代的预处理)
   - 2.1 端到端学习 vs 手工预处理
   - 2.2 可学习滤波器
   - 2.3 预训练模型的预处理策略差异
   - 2.4 归一化策略
3. [单导联特殊考虑](#3-单导联特殊考虑)
   - 3.1 单导联 vs 多导联预处理差异
   - 3.2 上臂非标准导联特殊噪声
   - 3.3 运动伪差处理
4. [我们项目当前的预处理流程分析](#4-我们项目当前的预处理流程分析)
5. [创新机会分析](#5-创新机会分析)
6. [参考文献与工具](#6-参考文献与工具)

---

## 1. 经典 ECG 预处理流程

ECG 信号的主要噪声源包括：基线漂移 (Baseline Wander, BW)、工频干扰 (Power Line Interference, PLI)、肌电噪声 (Electromyographic Noise, EMG)、运动伪差 (Motion Artifact, MA) 以及电极接触噪声。经典预处理流程通常按照以下顺序进行。

### 1.1 基线漂移去除

**问题描述**: 基线漂移主要由呼吸运动和电极阻抗变化引起，频率通常在 0.05-0.5 Hz 范围内，表现为 ECG 信号的低频缓慢波动，会严重影响 ST 段分析和波形特征提取。

#### 1.1.1 高通滤波

- **原理**: 使用高通滤波器 (HPF) 去除低于截止频率的成分
- **常用参数**: 截止频率 0.5-0.67 Hz，Butterworth 4 阶滤波器
- **代表方法**:
  - IIR Butterworth 高通滤波器 (最常用)
  - FIR 高通滤波器 (线性相位，但阶数较高)
- **优点**: 实现简单，计算效率高，实时性好
- **缺点**: 可能损失 ST 段低频信息；截止频率选择过高会扭曲 T 波和 P 波形态
- **AHA/IEC 标准建议**: 诊断级 ECG 的高通截止频率不应超过 0.05 Hz；监护级可放宽至 0.67 Hz
- **与我们项目的相关性**: **高度相关**。我们当前使用 0.67 Hz 作为带通滤波器下限，属于监护级设置，对 MI 分类（依赖 ST 段）可能存在信息损失风险

#### 1.1.2 多项式拟合

- **原理**: 对信号拟合低阶多项式（通常 3-6 阶），估计基线趋势后减去
- **代表方法**: 三次样条插值 (Cubic Spline)，利用等电位点（如 PQ 段）作为节点
- **优点**: 保留更多低频生理信息
- **缺点**: 需要先检测 R 波或其他基准点，对异常心律不适用；计算复杂度较高
- **与我们项目的相关性**: 中等。因为需要先做 R 波检测，增加了预处理复杂度

#### 1.1.3 中值滤波

- **原理**: 使用两级中值滤波器估计基线漂移。第一级窗口宽度约 200ms（覆盖 QRS 波群），第二级窗口宽度约 600ms（覆盖 T 波），从原信号中减去估计的基线
- **代表方法**: 双级中值滤波法 (Two-stage Median Filter)
- **优点**: 非线性方法，能保留 QRS 波群的尖锐特征
- **缺点**: 窗口大小依赖采样率，对非常慢的基线漂移效果有限
- **与我们项目的相关性**: **高度相关**。我们当前代码在带通滤波后使用了 0.4s 窗口的中值滤波做基线去除（见 `util.py` 第 47-53 行），存在重复处理问题——带通滤波已经去除了低频成分，之后再做中值滤波可能引入不必要的信号失真

#### 1.1.4 小波方法

- **原理**: 利用离散小波变换 (DWT) 将信号分解到多个频率尺度，将低频近似系数置零或阈值化后重构，达到去除基线漂移的目的
- **代表小波基**: db4, db6, sym8, coif5
- **代表论文/工具**:
  - Empirical Wavelet Transform (EWT) 方法（Elouaham et al., 2024, J. Electrical & Computer Engineering）
  - DWT-ADTF 自适应双阈值滤波方法
- **优点**: 多分辨率分析，可同时处理多种噪声；能保留 QRS 波群的时域尖锐特征
- **缺点**: 小波基和分解层数的选择缺乏统一标准；可能在信号和噪声频谱重叠时引入失真；计算复杂度高于简单滤波
- **与我们项目的相关性**: 中等。对 800K 大规模预训练来说，小波方法的计算开销可能是瓶颈；但对上臂可穿戴设备的噪声处理可能有价值

### 1.2 工频干扰去除

**问题描述**: 电力线干扰是 50/60 Hz 的窄带噪声（及其谐波 100/120 Hz 等），幅度可达数 mV，严重遮蔽 ECG 形态特征。

#### 1.2.1 陷波滤波 (Notch Filter)

- **原理**: 使用 IIR 陷波滤波器在 50 Hz (或 60 Hz) 处进行窄带抑制
- **关键参数**: 中心频率 50/60 Hz，品质因子 Q（Q 越高，陷波带宽越窄）
- **代表方法**: `scipy.signal.iirnotch(50, Q=30, fs)` + `filtfilt` (零相位滤波)
- **优点**: 实现简单，仅去除特定频率，对其他成分影响极小
- **缺点**: 只能去除基频，对谐波需要额外的陷波器；Q 值过高会导致滤波器不稳定
- **与我们项目的相关性**: **高度相关**。我们当前使用 Q=30 的 50Hz 陷波滤波器。需注意：(1) MIMIC-IV-ECG 来自美国，应该使用 60Hz 而非 50Hz；(2) 后续的 40Hz 低通已经会去除工频干扰，陷波可能是多余的

#### 1.2.2 自适应滤波

- **原理**: 利用参考噪声信号（如电源线上的信号）和 LMS/RLS 等自适应算法实时估计和消除工频干扰
- **优点**: 能跟踪频率漂移，适应性强
- **缺点**: 需要额外的参考通道；在单导联场景下难以获得参考信号
- **与我们项目的相关性**: 低。单导联没有参考通道，且数据库信号通常已在采集端做了工频滤波

### 1.3 肌电噪声滤波

**问题描述**: 肌电噪声 (EMG) 来自骨骼肌电活动，频率范围 5-500 Hz，与 ECG 信号频谱严重重叠，是最难去除的噪声类型之一。在上臂可穿戴设备中，EMG 是最主要的干扰源。

#### 1.3.1 低通/带通滤波

- **原理**: ECG 诊断信息主要集中在 0.5-40 Hz（监护级）或 0.05-150 Hz（诊断级），通过带通滤波限制信号带宽
- **常用参数**:
  - 监护级: 0.67-40 Hz（用于心律监测）
  - 诊断级: 0.05-100/150 Hz（用于形态分析）
  - 深度学习常用: 0.5-40/45 Hz
- **代表方法**: Butterworth 带通滤波器 (4 阶)
- **优点**: 实现简单，能有效抑制高频 EMG 成分
- **缺点**: 40 Hz 低通会丢失 QRS 波群的高频细节；对与 ECG 频谱重叠的低频 EMG 无效
- **AHA 建议**: 诊断级 ECG 应保留至少 150 Hz 的带宽
- **与我们项目的相关性**: **高度相关**。我们当前使用 0.67-40 Hz 带通。40Hz 上限对 MI 分类（主要看 ST 段和 Q 波）可能足够，但会丢失 QRS 高频成分。对于上臂设备，EMG 干扰更严重，可能需要更激进的滤波或深度学习方法

#### 1.3.2 小波阈值去噪

- **原理**: 对信号做 DWT 分解，对高频小波系数做阈值化处理（硬阈值或软阈值），保留低频和显著的高频成分
- **常用阈值**: VisuShrink、SureShrink、BayesShrink
- **优点**: 能在去噪的同时更好地保留 QRS 等尖锐特征
- **缺点**: 阈值选择依赖于噪声估计的准确性；可能引入 Gibbs 现象

### 1.4 R 波检测与心拍分割

**问题描述**: R 波（QRS 波群中的最高峰）检测是 ECG 分析的基础步骤，用于心率计算、HRV 分析和心拍级分类。

#### 1.4.1 经典方法: Pan-Tompkins 算法

- **原理**: (1) 5-18 Hz 带通滤波突出 QRS；(2) 求导强调斜率；(3) 平方运算增强 R 波；(4) 移动窗口积分平滑；(5) 自适应双阈值检测
- **性能**: 在高质量 ECG 上准确率 >99%，但在低质量（可穿戴）ECG 上降至约 74.5%
- **改进版本**: Pan-Tompkins++ (2024) 在 4 个数据集上平均减少 2.8% 误报，1.8% 漏报，F-score 提升 2.2%，计算时间减少 33%
- **与我们项目的相关性**: 中等。我们当前不做心拍分割，直接处理 10 秒片段；但如果未来做心拍级分类或 HRV 分析则需要

#### 1.4.2 深度学习方法

- **代表方法**:
  - U-Net + BLSTM 的 R 波检测（2025，精确率 95.68%，召回率 97.95%）
  - Residual U-Net for R-peak detection in noisy ECG (2025, Scientific Reports)
  - CNN + BLSTM 混合模型
- **优点**: 对噪声鲁棒性远优于经典方法；能同时检测 P, QRS, T 各波
- **缺点**: 需要大量标注训练数据；推理延迟较高
- **工具**: NeuroKit2 (`ecg_peaks()`)、BioSPPy (`ecg.hamilton_segmenter()`)
- **与我们项目的相关性**: 低。我们当前不做心拍分割

#### 1.4.3 心拍分割策略

- **固定窗口**: 以 R 波为中心，取前 250ms + 后 400ms
- **自适应窗口**: 根据 RR 间期动态调整窗口长度
- **重叠窗口**: 滑动窗口方式，不依赖 R 波检测
- **与我们项目的相关性**: 低。我们使用 10 秒固定长度片段

### 1.5 信号质量评估 (SQI)

**问题描述**: 数据集中不可避免地存在质量极差的记录（电极脱落、严重运动伪差等），这些坏段如果参与训练会引入噪声标签，降低模型性能。

#### 1.5.1 基于规则的 SQI

- **常用指标**:
  - **bSQI (Beat SQI)**: 比较两种 R 波检测算法的一致性
  - **kSQI (Kurtosis SQI)**: 信号峰度，正常 ECG 通常 >5
  - **sSQI (Skewness SQI)**: 信号偏度
  - **pSQI (Power SQI)**: 5-15 Hz 功率占总功率的比例
  - **basSQI (Baseline SQI)**: 基线漂移程度
- **代表论文**: "SQI Quality Evaluation Mechanism of Single-Lead ECG" (PMC, 2018)
- **代表工具**: `vital_sqi` Python 包 (GitHub: meta00/vital_sqi)
- **与我们项目的相关性**: **高度相关**。在 800K MIMIC-IV-ECG 数据中，坏段剔除可能对预训练质量有显著影响。我们当前仅做了 NaN 检查和跳过，缺乏系统的 SQI 评估

#### 1.5.2 基于深度学习的 SQI

- **代表方法**:
  - CNN + LSTM 将 ECG 转为时频谱图后分类（2024，Springer）
  - cGAN 数据增强 + 迁移学习的 SQI 模型（PMC, 2021）
  - 模板匹配 + 生理可行性检查（Scientific Reports, 2025, 针对纺织电极可穿戴设备）
  - 任务特定 SQI 框架：先合成噪声数据标注 "Clean/Noisy"，再用深度学习分类（CinC 2025）
- **与我们项目的相关性**: **高度相关**。特别是针对上臂可穿戴设备，实时 SQI 评估是必须的功能

### 1.6 重采样策略

**问题描述**: 不同设备采集的 ECG 采样率不同（常见 100, 250, 300, 500, 1000 Hz），需要统一到模型输入要求的采样率。

#### 1.6.1 各采样率的特点

| 采样率 | 特点 | 适用场景 |
|--------|------|----------|
| 100 Hz | 数据量最小，计算效率最高；丢失 QRS 高频细节 | 心律分类、HRV 分析 |
| 250 Hz | 平衡性能与效率；保留大部分形态特征 | 通用 ECG 分类 |
| 500 Hz | 临床标准；保留完整形态信息 | 诊断级 ECG 分析 |
| 1000 Hz | 极少额外信息增益；计算开销大 | 起搏器检测等特殊场景 |

#### 1.6.2 关键研究发现

- **Perez-Valero et al. (2023, arXiv:2311.04229)** 在 3 个多标签 ECG 数据集和 3 个分类器上的系统实验发现：
  - 50 Hz 采样率即可达到与 500 Hz 相当的分类性能
  - Min-max 归一化整体上略有害
  - 带通滤波对分类性能几乎无可测量的改善
- **PMC 2025 多标签分类研究**: 激进下采样（低至 50-100 Hz）反而在部分任务上优于 500 Hz
- **QRS 检测 CNN 研究 (arXiv:2007.02052)**: 100 Hz 时 DFL 和 DenseNet 达到最高准确率；100→250 Hz 提升不超过 0.6%

#### 1.6.3 重采样方法

- **线性插值**: 简单快速，可能引入混叠
- **sinc 插值 (scipy.signal.resample)**: 频域方法，保留频谱特性
- **多相滤波 (scipy.signal.resample_poly)**: 高效的整数比率重采样
- **简单抽取 (::2)**: 仅适用于整数倍下采样，需先做抗混叠滤波

**与我们项目的相关性**: **高度相关**。我们当前使用线性插值重采样到 5000 点（即 500Hz * 10s）。根据最新研究，降至 250Hz 甚至 100Hz 可能不损失性能但大幅减少计算量——这对 Mamba 模型尤为重要，因为序列长度直接影响 SSM 的状态空间大小和内存占用。将 5000 点降至 2500 或 1000 点可能在不损失 AUROC 的前提下显著减少训练时间和 OOM 风险。

---

## 2. 深度学习时代的预处理

### 2.1 端到端学习 vs 手工预处理的取舍

这是 ECG 深度学习中最根本的争论之一。两种主流观点：

#### 2.1.1 观点一：手工预处理仍然重要

- **理据**: 去除已知噪声源可以简化模型学习任务，使模型专注于学习诊断特征而非降噪
- **代表**: ECGFounder (Li et al., 2024) 使用完整的预处理流水线：高通 0.5 Hz → 50 Hz Butterworth 低通 → 50/60 Hz 陷波 → 10s 窗口截取
- **适用场景**: 数据量有限、标注稀缺、模型容量小

#### 2.1.2 观点二：端到端学习可以替代预处理

- **理据**: 深度学习模型（特别是 CNN 的第一层）可以学习到等效的滤波操作；手工预处理可能引入信息损失
- **代表**: KED 基金会模型直接使用 100 Hz 原始信号，不做任何信号预处理
- **关键证据**: Perez-Valero et al. (2023) 发现带通滤波对深度学习分类器的性能几乎无可测量的改善
- **适用场景**: 大规模预训练、数据多样性高

#### 2.1.3 实际最佳实践

多数最新论文采取 **"轻量预处理 + 端到端学习"** 的折中路线：
- 做基本的 NaN/Inf 清洗和重采样
- 做 z-score 归一化
- 不做或做最小限度的滤波
- 让模型学习从（轻度噪声的）信号中提取特征

**与我们项目的相关性**: **核心相关**。我们的 MIMIC 预训练数据做了完整的 3 步滤波 (陷波→带通→中值滤波)，但 PTB-XL 微调数据只做了 z-score 归一化，**预训练和微调的预处理不一致**。这种不一致可能导致域偏移，影响迁移效果。

### 2.2 可学习滤波器 (Learnable Filters)

#### 2.2.1 DeepFilter

- **论文**: "DeepFilter: An ECG baseline wander removal filter using deep learning techniques" (Romero & Pinol, 2021, arXiv:2101.03423)
- **架构**: FCN-DAE (Fully Convolutional Denoising Autoencoder)，使用卷积层提取特征 + 转置卷积层重构信号
- **训练**: 在标准 ECG 上叠加合成噪声（基线漂移、肌电、工频）构造训练对
- **优点**: 端到端学习降噪映射，无需手工选择滤波器参数
- **缺点**: 需要 "干净" 参考信号做训练对
- **与我们项目的相关性**: 中等。可以作为预处理模块集成到 pipeline 中

#### 2.2.2 ECGD-Net

- **论文**: "ECGD-Net: Deep Learning-based ECG Signal Denoising with MIEMD Filtering" (2025, IETE J. Research)
- **方法**: 两阶段——先用 MIEMD (Multivariate Intrinsic EMD) 做初步滤波，再用 CNN 做深度去噪
- **优点**: 结合传统信号分解和深度学习的优势
- **与我们项目的相关性**: 中等

#### 2.2.3 CNN 第一层作为可学习滤波器

- **现象**: 研究发现 CNN 在 ECG 分类任务中，第一层卷积核会自动学习到类似带通滤波器的功能
- **含义**: 如果模型第一层卷积核足够大（覆盖至少 1 个心跳周期），模型可以自动学习降噪
- **代表**: 1D-CNN 的第一层通常使用较大的卷积核 (kernel_size=7~15)
- **与我们项目的相关性**: **高度相关**。Mamba 的输入嵌入层 (1D Conv patch embedding) 实际上扮演了类似角色。如果 patch 大小足够大，模型可能已经在学习等效滤波

### 2.3 预训练模型的预处理策略差异

不同 ECG 基金会模型的预处理策略差异巨大，以下是主要模型的对比：

| 模型 | 年份 | 预处理策略 | 采样率 | 备注 |
|------|------|-----------|--------|------|
| **ECGFounder** | 2024 | 高通 0.5Hz + 低通 50Hz Butterworth + 50/60Hz 陷波 | 500 Hz | 完整预处理 |
| **ECG-FM** | 2024 | 多源统一预处理 pipeline (fairseq_signals 框架) | 多种 | 开源，基于 wav2vec 2.0 |
| **KED** | 2024 | 无预处理，直接用原始信号 | 100 Hz | 极简路线 |
| **HuBERT-ECG** | 2024 | 标准化 + 重采样 | 500 Hz | 9.1M ECG 预训练 |
| **ST-MEM** | 2024 (ICLR) | z-score + 重采样 | 多种 | 时空掩码建模 |
| **我们 (SelfMIS)** | 当前 | MIMIC: 陷波+带通+中值滤波; PTB-XL: 仅 z-score | 500 Hz | **预训练/微调不一致** |

**关键观察**:
- 大规模预训练模型倾向于使用更少的手工预处理
- 预处理一致性（预训练与微调使用相同策略）比预处理方法本身更重要
- 开源基金会模型 (ECG-FM) 提供标准化 pipeline 是趋势

### 2.4 归一化策略

#### 2.4.1 Per-sample Z-score 归一化

- **公式**: `x_norm = (x - mean(x)) / (std(x) + eps)`，其中 mean/std 在整个样本上计算
- **优点**: 消除幅度差异（不同设备增益不同）、简单高效
- **缺点**: 极端异常值会压缩正常范围；全样本计算会混淆不同导联的幅度关系
- **我们的实现**: 当前代码使用全样本 z-score（`np.mean(signal)` / `np.std(signal)`），对于单导联是合理的；但对 12 导联，这种全局 z-score 会破坏导联间的相对幅度关系

#### 2.4.2 Per-channel Z-score 归一化

- **公式**: 对每个导联独立计算 mean 和 std
- **优点**: 保留导联间的独立性
- **缺点**: 破坏导联间的相对幅度关系（例如 V1-V6 的 R 波递增规律）
- **适用场景**: 多导联输入但不需要导联间幅度比较的任务

#### 2.4.3 Global Statistics 归一化

- **公式**: 使用训练集的全局 mean/std 归一化
- **优点**: 保留绝对幅度信息
- **缺点**: 对分布外 (OOD) 数据敏感；不同设备的增益差异无法消除

#### 2.4.4 Min-Max 归一化

- **公式**: `x_norm = (x - min) / (max - min)`
- **发现**: Perez-Valero et al. (2023) 的实验表明 min-max 归一化在 ECG 分类中整体略有害
- **原因推测**: 极端异常值会导致正常信号被压缩到很小的范围

#### 2.4.5 建议

- **单导联**: per-sample z-score（我们当前的做法）是合理的
- **多导联**: per-channel z-score 或 per-sample z-score（取决于是否需要保留导联间关系）
- **跨设备泛化**: per-sample z-score 是最鲁棒的选择

**与我们项目的相关性**: **核心相关**。当前的 per-sample z-score 对单导联 Lead I 是合适的。但需要注意：在 SigLIP 对比学习中，如果 Lead I 和 12 导联分别做 z-score，它们的尺度可能不一致，影响对比学习的效果。

---

## 3. 单导联特殊考虑

### 3.1 单导联 vs 多导联预处理差异

| 方面 | 单导联 | 多导联 (12-lead) |
|------|--------|-----------------|
| **噪声识别** | 只能通过时域/频域特征判断 | 可利用导联间相关性识别噪声 |
| **基线漂移** | 无法利用参考导联 | 可用独立导联验证 |
| **ICA 降噪** | 不可用（需多通道） | 可用 ICA/BSS 分离噪声源 |
| **SQI 评估** | 只能用单通道指标 | 可用导联间一致性 (bSQI) |
| **信息冗余** | 无冗余，信息损失不可逆 | 导联间有冗余，可交叉验证 |

**关键挑战**: 单导联预处理的最大困难在于无法区分 "噪声" 和 "异常 ECG 形态"。例如，一个被噪声污染的正常 QRS 波群和一个形态异常的 QRS 波群在单导联上可能看起来一样。

### 3.2 上臂非标准导联的特殊噪声模式

上臂心电信号与标准 12 导联 ECG 有本质区别：

#### 3.2.1 信号特点

- **幅度显著降低**: 上臂 ECG 幅度通常只有标准导联的 1/5 到 1/10
- **信噪比极低**: 信号微弱而肌电噪声强
- **形态差异**: 缺乏 Wilson 中心电端 (WCT)，波形与标准导联形态不匹配
- **导联位置不确定**: 用户佩戴位置的微小变化会显著改变信号形态

#### 3.2.2 主要噪声源

1. **肌电噪声 (EMG)**: 上臂肌肉群丰富，EMG 是**最主要的干扰源**
2. **运动伪差**: 手臂运动直接影响电极接触，产生大幅度低频伪差
3. **电极阻抗变化**: 干电极（常见于可穿戴设备）与皮肤接触不稳定
4. **汗液/水分变化**: 长期佩戴时汗液可导致电极阻抗和信号漂移

#### 3.2.3 相关研究

- **Signal Quality Analysis of Single-Arm ECG (PMC, 2023)**: 系统分析了单臂 ECG 的信号质量特征，发现 EMG 是最主要的干扰
- **Dynamic Cardiac Event Detection from Single-Arm Wearable ECG (PubMed, 2025)**: 提出基于对比多任务框架的动态心脏事件检测方法
- **Wearable Armband ECG (PMC, 2025)**: 浮动可移动探索电极指尖触摸式多导联 ECG 采集方案

**与我们项目的相关性**: **核心相关**。如果未来产品是上臂设备，预处理流程需要根本性重新设计：
1. 需要更激进的 EMG 滤波（低通可能需要降至 25-30 Hz）
2. 需要实时 SQI 评估以标识不可用段
3. 预训练模型需要在含噪声的信号上训练，或者使用数据增强模拟上臂噪声

### 3.3 运动伪差 (Motion Artifact) 处理

#### 3.3.1 问题特性

- 频谱范围 0.01-10 Hz，与 ECG 低频成分（P 波、T 波、ST 段）严重重叠
- 非平稳、非线性，传统线性滤波效果有限
- 幅度可达 ECG 信号幅度的数倍

#### 3.3.2 处理方法

| 方法 | 原理 | 优势 | 局限 |
|------|------|------|------|
| 自适应滤波 (LMS/RLS) | 利用加速度计参考信号 | 实时性好 | 需要额外传感器 |
| EMD/EEMD | 分解为本征模态函数 | 自适应，不需预设参数 | 模态混叠问题 |
| 小波阈值 | 多尺度分解+阈值 | 保留尖峰特征 | 阈值选择困难 |
| 深度学习 (DAE) | 学习噪声→干净映射 | 非线性建模 | 需要配对训练数据 |
| 冗余去噪 ICA | 多通道盲源分离 | 无需参考 | 仅适用多导联 |

- **加速度计辅助方法**: 利用 IMU 传感器记录运动信息作为参考，用自适应滤波器消除运动伪差。这是可穿戴设备中最实用的方法
- **阻抗体积描记法 (IPG)**: 利用呼吸引起的胸部阻抗变化作为参考信号（PMC, 2022）

**与我们项目的相关性**: **高度相关**。上臂设备面临严重的运动伪差问题。建议：
1. 设备端集成 IMU (加速度计+陀螺仪)
2. 利用 IMU 信号作为自适应滤波的参考
3. 在深度学习端，可以将 IMU 信号作为辅助输入通道

---

## 4. 我们项目当前的预处理流程分析

### 4.1 当前实现

基于对代码库的分析 (`util.py`, `dataset.py`, `selfmis_dataset.py`)，当前预处理流程如下：

#### MIMIC-IV-ECG 预训练数据

```
原始信号 → NaN→0 → 导联重排序 → 50Hz陷波(Q=30) → 0.67-40Hz带通(Butterworth 4阶) → 0.4s中值滤波基线去除 → z-score归一化 → 重采样到5000点
```

#### PTB-XL 微调数据

```
原始信号 → NaN检查 → z-score归一化 → 线性插值重采样到5000点
```

### 4.2 发现的问题

#### 问题 1: 预训练/微调预处理不一致

**严重程度: 高**

MIMIC 数据经过了 3 步滤波 (陷波+带通+中值)，而 PTB-XL 数据仅做了 z-score 归一化。这意味着预训练模型学到的特征分布与微调数据的分布存在系统性差异（域偏移）。预训练模型可能学会了依赖 0.67-40 Hz 带限信号的特征，但微调时接收到的是全带宽信号。

**建议**: 统一两个阶段的预处理，要么都做滤波，要么都不做。

#### 问题 2: 50Hz 陷波可能不适用于 MIMIC-IV-ECG

**严重程度: 中**

MIMIC-IV-ECG 来自 Beth Israel Deaconess Medical Center（美国），使用 60 Hz 电力系统。当前代码使用 50 Hz 陷波，应该是 60 Hz。不过，由于后续的 40 Hz 低通已经会去除 50/60 Hz 成分，这个问题的实际影响可能较小。

#### 问题 3: 带通滤波后的中值滤波可能是多余的

**严重程度: 低-中**

0.67 Hz 高通已经去除了基线漂移的主要成分，之后再做中值滤波基线去除是冗余的。中值滤波是非线性操作，可能引入额外的信号失真。

#### 问题 4: 全样本 z-score 在多导联场景下可能不合理

**严重程度: 中**

当前的 z-score 计算是在所有导联和所有时间点上做全局 mean/std。对于 12 导联预训练数据，这意味着 V1（通常 R 波较小）和 V5（通常 R 波较大）的幅度差异被抹平。不过，由于我们微调只用 Lead I，这个问题的实际影响可能有限。

#### 问题 5: 缺乏信号质量评估 (SQI)

**严重程度: 高**

800K MIMIC-IV-ECG 数据中必然存在大量低质量记录。当前仅做了 NaN 检查和跳过，缺乏系统的 SQI 筛选。低质量数据会引入噪声标签，降低对比学习的效果。

### 4.3 优化建议优先级

| 优先级 | 建议 | 预期影响 | 实施难度 |
|--------|------|----------|----------|
| P0 | 统一预训练/微调预处理流程 | AUROC +0.5-2% | 低 |
| P1 | 添加 SQI 筛选，剔除最差 5-10% 数据 | AUROC +0.3-1% | 中 |
| P2 | 尝试降采样至 250Hz (2500点) | 训练加速 2x，OOM 风险降低 | 低 |
| P3 | 修正 50Hz→60Hz 陷波 | 微小或无影响 | 极低 |
| P4 | 移除冗余的中值滤波步骤 | 训练提速，信号失真减少 | 低 |

---

## 5. 创新机会分析

### 5.1 当前痛点与未解决问题

#### 5.1.1 预处理-模型联合优化的空白

**痛点**: 当前几乎所有 ECG 深度学习论文都将预处理视为固定的前处理步骤，与模型训练完全分离。预处理参数（滤波器截止频率、阈值等）通过人工经验或简单网格搜索确定，没有与下游任务联合优化。

**创新方向**:
- **可微分预处理层**: 将滤波操作参数化为可微分模块，作为模型第一层，端到端训练
- **发表潜力**: 高。"Task-aware adaptive ECG preprocessing" 目前文献中几乎空白
- **与 Mamba 结合**: Mamba 的 SSM 本身就是一个可学习的线性时不变系统，天然适合建模滤波操作。可以设计 "预处理 SSM 层" 作为 Mamba 网络的第一层

#### 5.1.2 单导联信号质量的自动评估与自适应处理

**痛点**: 现有 SQI 方法主要针对标准 12 导联 ECG，对可穿戴单导联（尤其是非标准导联位置）的适用性差。

**创新方向**:
- **自监督 SQI**: 利用对比学习预训练 SQI 模型——同一患者不同时间的干净信号是正对，噪声信号是负对
- **噪声感知预训练**: 在预训练阶段同时学习信号质量表示和诊断表示
- **发表潜力**: 高。特别是结合上臂可穿戴设备场景

#### 5.1.3 跨设备/跨导联位置的域适应

**痛点**: 在标准 12 导联 ECG 上预训练的模型，迁移到非标准导联位置（如上臂）时性能急剧下降。

**创新方向**:
- **导联位置不变表示学习**: 通过数据增强或对抗训练学习与导联位置无关的心脏电活动表示
- **发表潜力**: 高。这是可穿戴 ECG 领域的核心问题

### 5.2 与 Mamba + 2-Loss 框架的创新结合

#### 5.2.1 SSM 作为可学习滤波器

**想法**: Mamba 的核心是选择性状态空间模型 `y = SSM(A, B, C, D)(x)`。一个 SSM 在数学上等价于一个线性时不变系统（即 IIR 滤波器）。可以显式地初始化 Mamba 的第一层为已知的 ECG 滤波器（如带通滤波器），然后允许其在训练中适应。

**创新点**:
- "Mamba as a Learnable ECG Filter: Unifying Preprocessing and Feature Extraction in State Space Models"
- 与现有的 CNN 可学习滤波器相比，SSM 提供了更物理可解释的滤波行为（状态空间描述）

#### 5.2.2 噪声感知对比学习

**想法**: 在 SigLIP 对比学习中，利用信号质量信息调整对比学习的损失权重。干净信号对的对比损失权重大，噪声信号对的权重小或直接丢弃。

**创新点**:
- "Quality-Aware Contrastive Learning for Robust ECG Representation"
- 不需要额外的 SQI 模型，可以通过简单的统计指标（kurtosis, SNR 估计）实时计算

#### 5.2.3 多尺度预处理 + Mamba

**想法**: 将 ECG 信号在多个预处理级别（原始、轻度滤波、强滤波）分别输入 Mamba，利用 cross-attention 或 gating 机制融合不同预处理级别的表示。

**创新点**:
- "Multi-Resolution Preprocessing Fusion for Robust ECG Analysis"
- 避免了选择单一预处理策略的困境

#### 5.2.4 预处理增强的数据增强

**想法**: 将预处理参数（滤波器截止频率、归一化方式）作为数据增强的一部分，在训练时随机选择不同的预处理配置。这使模型对预处理策略具有鲁棒性，有利于跨设备泛化。

**创新点**:
- "Preprocessing Augmentation: Training ECG Models Robust to Signal Processing Variations"
- 实现简单，不需要修改模型架构

### 5.3 实验建议（按优先级排序）

#### 快速验证 (1-2 天)

1. **统一预处理实验**: 用统一的 "仅 z-score" 预处理重新跑 3-seed finetune，对比当前的不一致预处理
2. **降采样实验**: 在 250Hz (2500 点) 下重新训练，验证性能是否保持

#### 中期实验 (1-2 周)

3. **SQI 筛选实验**: 用 kSQI + pSQI 对 MIMIC 数据做质量排序，剔除最差 5%/10% 后重新预训练
4. **预处理增强实验**: 训练时随机选择 {无滤波, 带通, 带通+中值} 中的一种

#### 长期研究方向 (1-3 月)

5. **SSM 可学习滤波器**: 设计并实验 "预处理 SSM 层"
6. **噪声感知对比学习**: 实现 quality-weighted SigLIP loss

---

## 6. 参考文献与工具

### 6.1 综述与系统性评估

1. **Preprocessing and Denoising Techniques for ECG and MCG: A Review** (PMC, 2024) — 近十年 ECG/MCG 去噪方法的全面综述
   https://pmc.ncbi.nlm.nih.gov/articles/PMC11591354/

2. **Exploring Best Practices for ECG Pre-Processing in Machine Learning** (arXiv:2311.04229, 2023) — 系统评估下采样、归一化、滤波对 ECG 分类性能的影响
   https://arxiv.org/abs/2311.04229

3. **Advances in Machine and Deep Learning for ECG Beat Classification: A Systematic Review** (Frontiers, 2025)
   https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1649923/full

4. **A Survey of Transformers and LLMs for ECG Diagnosis** (Springer AI Review, 2025)
   https://link.springer.com/article/10.1007/s10462-025-11259-x

### 6.2 ECG 基金会模型

5. **ECGFounder: An ECG Foundation Model Built on over 10 Million Recordings** (PMC, 2024)
   https://pmc.ncbi.nlm.nih.gov/articles/PMC12327759/

6. **ECG-FM: An Open Electrocardiogram Foundation Model** (arXiv:2408.05178, 2024) — 开源，基于 wav2vec 2.0，90.9M 参数
   https://arxiv.org/abs/2408.05178
   GitHub: https://github.com/bowang-lab/ECG-FM

7. **HuBERT-ECG: A Self-Supervised Foundation Model** (medRxiv, 2024) — 9.1M ECG 预训练
   https://www.medrxiv.org/content/10.1101/2024.11.14.24317328v2.full

8. **ST-MEM: Spatio-Temporal Masked ECG Modeling** (ICLR 2024) — 时空掩码建模自监督学习
   https://proceedings.iclr.cc/paper_files/paper/2024/file/412fb8623bf8b6d56fb6285ea295447e-Paper-Conference.pdf

### 6.3 深度学习降噪

9. **DeepFilter: ECG Baseline Wander Removal Using Deep Learning** (arXiv:2101.03423, 2021)
   https://arxiv.org/abs/2101.03423

10. **ECGD-Net: Deep Learning-based ECG Signal Denoising** (IETE J. Research, 2025)
    https://www.tandfonline.com/doi/full/10.1080/03772063.2025.2470375

11. **A Proposed Deep Learning Model for Multichannel ECG Noise Reduction** (Springer, 2025) — FCN + Jacobian 正则化
    https://link.springer.com/article/10.1007/s44163-025-00292-y

### 6.4 信号质量评估

12. **Introduction to ECG Signal Quality Assessment for Textile Electrodes** (Scientific Reports, 2025)
    https://www.nature.com/articles/s41598-025-25365-x

13. **Assessment of ECG SQI Algorithms Using Synthetic Data** (CinC 2024)
    https://www.cinc.org/archives/2024/pdf/CinC2024-270.pdf

14. **ECG Quality Assessment Using Deep Learning** (Springer, 2024)
    https://link.springer.com/chapter/10.1007/978-3-031-52382-3_21

### 6.5 可穿戴 ECG 与运动伪差

15. **Motion Artifacts in Capacitive ECG: A Review** (Springer MBEC, 2024) — 运动伪差建模与去除方法综述
    https://link.springer.com/article/10.1007/s11517-024-03165-1

16. **Signal Quality Analysis of Single-Arm ECG** (PMC, 2023)
    https://pmc.ncbi.nlm.nih.gov/articles/PMC10346735/

17. **Dynamic Cardiac Event Detection from Single-Arm Wearable ECG** (PubMed, 2025) — 对比多任务框架
    https://pubmed.ncbi.nlm.nih.gov/41525618/

18. **Next-Generation Wearable ECG Systems: Soft Materials, AI, and Personalized Healthcare** (ScienceDirect, 2025)
    https://www.sciencedirect.com/science/article/pii/S1385894725109601

19. **Opportunities and Challenges of Noise Suppression for Dynamic ECG in Wearable Devices** (ScienceDirect, 2025)
    https://www.sciencedirect.com/science/article/abs/pii/S0263224125004269

### 6.6 R 波检测

20. **Pan-Tompkins++: A Robust Approach to Detect R-peaks** (arXiv:2211.03171, 2024)
    https://arxiv.org/abs/2211.03171

21. **Robust R-peak Detection in Noisy ECG Using Deep Residual U-Net** (PMC, 2025)
    https://pmc.ncbi.nlm.nih.gov/articles/PMC12499854/

### 6.7 Mamba/SSM 与 ECG

22. **S2M2ECG: Spatio-temporal Bi-directional SSM Multi-branch Mamba for ECG** (arXiv, 2025) — Mamba 在 ECG 中的最新应用
    https://arxiv.org/abs/2509.03066

### 6.8 实时预处理评估

23. **Evaluation of Real-Time Preprocessing Methods in AI-Based ECG Signal Analysis** (IEEE AIIoT, 2025)
    https://arxiv.org/abs/2510.12541

### 6.9 开源工具

| 工具 | 语言 | 功能 | 链接 |
|------|------|------|------|
| **NeuroKit2** | Python | ECG 清洗、R 波检测、HRV 分析（支持多种算法） | https://github.com/neuropsychology/NeuroKit |
| **BioSPPy** | Python | ECG 滤波 (FIR 0.67-45Hz)、R 波检测 | https://github.com/PIA-Group/BioSPPy |
| **vital_sqi** | Python | 信号质量评估、预处理 | https://github.com/meta00/vital_sqi |
| **wfdb** | Python | PhysioNet 数据读取（我们已在使用） | https://github.com/MIT-LCP/wfdb-python |
| **ECG-FM** | Python | 端到端预处理 + 基金会模型 (fairseq_signals) | https://github.com/bowang-lab/ECG-FM |
| **ecg-selfsupervised** | Python | 12 导联自监督预训练 | https://github.com/tmehari/ecg-selfsupervised |

---

## 总结

### 核心发现

1. **预处理的影响可能被高估了**: 最新研究（2023-2025）表明，对于深度学习模型，带通滤波对分类性能的提升几乎不可测量，采样率降至 50-100 Hz 也可保持相当性能。

2. **预处理一致性比方法选择更重要**: 预训练和微调阶段使用不同的预处理流程（我们当前的情况）会引入域偏移，这可能比选择哪种滤波器更重要。

3. **SQI 筛选是被忽视的关键步骤**: 在 800K 大规模数据中，质量控制可能比精细的信号处理带来更大的性能提升。

4. **上臂可穿戴是全新的信号域**: 非标准导联位置、EMG 主导的噪声、运动伪差——这些问题需要全新的解决方案，不能简单沿用标准 12 导联的预处理流程。

5. **Mamba (SSM) 与信号处理的天然契合**: SSM 在数学上等价于线性时不变系统，这为设计可学习的预处理模块提供了独特的理论基础。

### 对我们项目的最高优先级行动项

1. **立即**: 统一预训练和微调的预处理流程（建议统一为 "仅 z-score + 重采样"）
2. **短期**: 实验 250Hz 降采样，可能解决 OOM 问题同时不损失性能
3. **中期**: 添加 SQI 筛选到 MIMIC 数据 pipeline
4. **长期**: 探索 "SSM 作为可学习滤波器" 的研究方向
