# ECG 预处理与数据增强交叉领域：前沿创新方向分析

**调研日期**: 2026-03-27
**调研范围**: 2024-2026 年最新文献，涵盖 NeurIPS/ICML/ICLR/AAAI 顶会、Nature Medicine/npj Digital Medicine/IEEE TBME 等期刊、arXiv 预印本
**项目背景**: Mamba + 2-Loss 预训练（SigLIP 对比学习 + 重建损失），MIMIC-IV-ECG 800K 预训练 + PTB-XL 9 类 MI 亚型分类微调

---

## 目录

1. [执行摘要：优先做什么](#1-执行摘要)
2. [预处理 + 预训练的协同创新](#2-预处理--预训练的协同创新)
3. [增强 + 对比学习的创新](#3-增强--对比学习的创新)
4. [Mamba/SSM 特有的机会](#4-mambassm-特有的机会)
5. [上臂心电设备的创新需求](#5-上臂心电设备的创新需求)
6. [可发论文的创新方向排序](#6-可发论文的创新方向排序)
7. [实施路线图](#7-实施路线图)
8. [参考文献](#8-参考文献)

---

## 1. 执行摘要

### 当前项目现状

| 指标 | 数值 |
|------|------|
| 当前最佳 AUROC | 0.8244 (33M Uni-Mamba, 1-seed) |
| 对标目标 | SelfMIS 0.8255 (LP, ResNet1D 30.81M) |
| 临床目标 | sens/spec > 0.8 (对应 AUROC 约 0.88) |
| 差距 | 约 0.055 AUROC 单位 |

### 核心结论：可以做什么、优先做什么

**立即可做（1-2 周内可产出结果）：**

1. **去噪预训练 pretext task** -- 将噪声注入作为第 3 个预训练损失，与现有 SigLIP + 重建损失组合。预期 AUROC 提升 1-3pp。
2. **信号质量感知的 Curriculum Learning** -- 按 MIMIC-IV 信号质量排序训练样本，先学干净后学噪声。几乎零额外代码成本。
3. **ECG 专用对比学习 view generation** -- 设计保持 MI 诊断语义不变的增强策略（如保持 ST 段幅度的增强），替换通用 random augmentation。

**中期可做（1-2 个月）：**

4. **可学习预处理层** -- 用可训练的 Conv1d 滤波器替代硬编码 bandpass filter，端到端优化。
5. **12 导联 Teacher 到单导联 Student 的知识蒸馏 + 增强联合优化** -- 在现有 SigLIP 框架上扩展。
6. **Mamba 专用的 Selective Masking 预训练** -- 利用 SSM 选择性机制设计与 Transformer [MASK] 不同的 masking 策略。

**长期布局（3-6 个月）：**

7. **非标准导联 Domain Adaptation** -- 标准 Lead I 数据到上臂心电的迁移。
8. **个人基线建模的增强策略** -- 长期监测场景的时序变化检测训练数据生成。

---

## 2. 预处理 + 预训练的协同创新

### 2.1 去噪作为预训练 Pretext Task（Denoising Pretext Task）

**核心思路**：当前 2-Loss 预训练包含 (1) SigLIP 对比损失 和 (2) 重建损失。可添加第 3 个损失：向输入信号注入已知噪声（高频肌电噪声、基线漂移、50Hz 工频干扰），要求模型在重建分支中同时完成去噪。

**文献支撑**：
- CREMA (2024, arXiv:2407.07110) 提出 Contrastive Regularized Masked Autoencoder，结合生成学习与对比正则化，在跨临床域的 ECG 诊断中显著优于监督基线。其核心发现是：**纯对比学习的 augmentation 可能扭曲诊断信息，而结合生成重建可以缓解这一问题**——这与我们 2-Loss 设计的哲学一致。
- Fully-Gated Denoising AutoEncoder (FGDAE, 2025, Sensors 25(3):801) 证明去噪自编码器在 ECG 伪差消除中的有效性。
- "Ditch the Denoiser" (2025, arXiv:2505.12191) 发现通过 noise-aware data curriculum 可以让 SSL backbone 涌现去噪能力，无需单独的去噪器——**这直接支持将噪声注入整合到预训练中**。

**与我们项目的结合点**：
- 现有 `ECGReconDecoder` 已在做重建，只需在输入端添加噪声注入模块
- 重建损失自然变为 "去噪重建损失"，不改变架构
- MIMIC-IV 数据已通过 `filter_bandpass()` 做了预处理，但原始信号仍可获取

**具体实现方案**：
```
原始 ECG x → 添加随机噪声 → x_noisy
x_noisy → Mamba Encoder → features
features → Decoder → x_recon
Loss_denoise = MSE(x_recon, x_clean)  # 注意：目标是干净信号
```

噪声类型及参数：
| 噪声类型 | 参数范围 | 物理含义 |
|---------|---------|---------|
| 高斯白噪声 | SNR 10-30dB | 热噪声/设备噪声 |
| 基线漂移 | 0.1-0.5Hz 正弦 | 呼吸运动 |
| 50Hz 工频 | 幅度 0.01-0.1 | 电磁干扰 |
| 肌电噪声 | 带通 20-500Hz | 肌肉活动 |

### 2.2 可学习预处理层（Learnable Preprocessing）

**核心思路**：当前预处理管道（`selfmis_dataset.py`）使用硬编码的 50Hz notch + 0.67-40Hz bandpass + 中值滤波基线移除。这些参数基于通用 ECG 工程经验设计。**可学习预处理层**将这些固定滤波器替换为端到端可训练的卷积层，让模型自动学习最优的频率响应。

**文献支撑**：
- 在语音领域，SincNet (Ravanelli & Bengio, 2018) 用参数化正弦滤波器替换固定特征提取，在说话人识别中大幅优于手工特征。ECG 信号与语音的相似之处在于：两者都是准周期的一维信号，且最优滤波器因任务而异。
- 最新的端到端 ECG 平台 ExChanGeAI (JMIR, 2026) 强调了预处理步骤对下游性能的关键影响。
- Preprocessing and Denoising Techniques review (PMC, 2024) 指出传统滤波器的局限性：**固定截止频率无法适应个体差异和不同噪声环境**。

**具体设计**：
```python
class LearnablePreprocessor(nn.Module):
    """端到端可训练的 ECG 预处理层"""
    def __init__(self, n_filters=32, kernel_size=65):
        super().__init__()
        # 多组可学习滤波器，初始化为 bandpass 响应
        self.filters = nn.Conv1d(1, n_filters, kernel_size,
                                  padding=kernel_size//2, bias=False)
        # 初始化权重为 Butterworth bandpass 0.67-40Hz
        self._init_bandpass(0.67, 40, 500)
        # 选择性融合（学习哪些滤波器的输出组合最优）
        self.gate = nn.Conv1d(n_filters, 1, 1)

    def forward(self, x):
        filtered = self.filters(x)          # (B, n_filters, 5000)
        weights = torch.sigmoid(self.gate(filtered))
        return (filtered * weights).sum(dim=1, keepdim=True)
```

**关键创新点**：
- 滤波器初始化为传统 bandpass 的系数（warm start），避免从零学习
- 门控机制让模型自适应选择滤波器组合
- 可在预训练和微调阶段分别学习不同的预处理策略

### 2.3 噪声注入作为预训练正则化

**核心思路**：不同于 2.1 的去噪 pretext task，这里的噪声注入纯粹作为**正则化手段**，类似于 Dropout 的思想——在训练时随机扰动输入信号，迫使模型学习对噪声鲁棒的表征。

**文献支撑**：
- "Ditch the Denoiser" (arXiv:2505.12191, 2025) 证明在 ImageNet 上，noise-aware data curriculum + teacher-guided regularization 可使 DINOv2 的 linear probing 准确率提升 4.8%，**无需单独的去噪模块**。
- Parametric Noise Injection (Neelakantan et al.) 在神经网络权重中注入可训练噪声作为正则化，在多个任务上优于 Dropout。

**实现方式**：
- 在 `MIMICECGPretrainDataset.__getitem__()` 中以 50% 概率注入随机噪声
- 噪声强度随训练进度递减（curriculum: epoch 0-10 高噪声，epoch 10-30 低噪声）
- 对比损失和重建损失的目标不变（干净的 12 导联嵌入和干净的单导联信号）

### 2.4 信号质量感知的 Curriculum Learning

**核心思路**：MIMIC-IV-ECG 800K 样本的信号质量参差不齐。Curriculum Learning 策略是**先让模型学习高质量（低噪声）的样本，建立稳定的特征表征，再逐步引入低质量样本**，使模型学会在噪声中提取有用信号。

**文献支撑**：
- 在语音识别领域，accordion annealing curriculum (Amodei et al.) 使用从低 SNR 到高 SNR 的多阶段训练，显著提升噪声环境下的识别率。
- Signal Quality Auditing (arXiv:2402.00803, 2024) 提出了时间序列信号质量的标准化评估框架。
- MTL-NET (IEEE TBME, 2024) 结合信号质量评估和去噪，使用 Bi-LSTM + attention 的多任务框架。

**实现方案**：
1. 预计算每条 MIMIC-IV 记录的信号质量指数 (SQI)：
   - `SQI = kurtosis * (1 - flat_ratio) * (1 - nan_ratio) * snr_estimate`
2. 按 SQI 排序训练数据
3. Epoch 1-10: 只使用 SQI top-50% 的样本
4. Epoch 11-20: 扩展到 SQI top-80%
5. Epoch 21-30: 使用全部数据

**成本评估**：几乎零额外代码成本，只需修改 DataLoader 的 sampler 逻辑。

---

## 3. 增强 + 对比学习的创新

### 3.1 ECG 专用的 View Generation 策略

**核心问题**：在对比学习中，正样本对应通过 augmentation 生成同一信号的两个 "view"。关键问题是：**哪些变换保持 ECG 的诊断语义不变？**

**当前问题**：我们的 SigLIP 对比学习使用 12 导联 Teacher 和单导联 Student 作为正对，不涉及 augmentation。但在 Student 端引入语义保持的增强可以提升表征质量。

**文献支撑**：
- Multi-stage Temporal and Cross-view Contrastive Learning (ScienceDirect, 2025) 提出多阶段预测 + 跨视图对比学习，使用多样化增强配合 SE-ResNet18 骨干，在 PTB-XL、Chapman、CPSC-2018 上超越 SOTA SSL 方法。
- CLOCS (2021) 将时间相邻的 ECG 段作为正对，利用心功能短期稳定性。
- ECG-FM (2025) 使用混合自监督（masked reconstruction + contrastive learning + ECG 专用增强）在 150 万条 12 导联 ECG 上预训练。

**ECG 语义保持增强的分类**：

| 变换类型 | 保持 MI 语义？ | 理由 | 推荐 |
|---------|-------------|------|------|
| 时间缩放 (0.9-1.1x) | 是 | 心率变异不改变形态学 | 强推荐 |
| 幅度缩放 (0.8-1.2x) | 是 | 模拟增益差异 | 强推荐 |
| 随机裁剪+填充 | 部分 | 需保留完整 QRS-ST-T 周期 | 推荐（需保留至少 3 个完整心搏） |
| 基线漂移注入 | 是 | 模拟呼吸运动 | 推荐 |
| 高斯噪声注入 | 是 | 模拟设备噪声 | 推荐（SNR > 10dB） |
| 频率 masking | 部分 | 可能破坏 ST 段频率成分 | 谨慎 |
| 时间 masking | 部分 | 若 mask 到 ST 段则破坏语义 | 谨慎 |
| ST 段幅度改变 | **否** | 直接改变 MI 诊断标准 | 禁止 |
| QRS 形态变换 | **否** | 改变传导异常特征 | 禁止 |

**创新方向**：**心电生理感知的增强** -- 利用心电学先验知识，设计只在诊断无关区间（如 TP 段、PR 段）施加变换的增强策略。

### 3.2 医学先验引导的增强（Physiology-Informed Augmentation）

**核心思路**：传统增强（时间拉伸、噪声注入等）对 ECG 信号盲目施加变换。**医学先验引导的增强**利用心电学知识，确保增强后的信号仍然满足生理学约束。

**文献支撑**：
- GeoECG (PMLR 2022, arXiv:2208.01220) 使用 Wasserstein 测地线扰动进行 ECG 增强，设计了基于生理特征的 ground metric，使增强沿着生理学合理的方向进行。
- Specialized ECG augmentation (Biomedical Engineering Letters, 2025) 利用胸前导联位置变异性进行增强，模拟电极放置的自然变异。
- iAAFT Surrogates (arXiv:2504.03761, 2025) 提出保留 QRS 峰的增强方法，通过 changepoint detection 分割信号，确保基本心电形态在增强过程中不变。

**我们的创新方向**：

**PhysioAug: 心电生理约束下的对比学习增强**

```
步骤 1: R 峰检测 → 心搏分割
步骤 2: 对每个心搏标记功能区间：
         P 波 | PR 段 | QRS 波群 | ST 段 | T 波 | TP 段
步骤 3: 只在 "安全区间"（TP 段、PR 段）施加强增强
         在 "关键区间"（QRS、ST-T）只做微弱扰动
步骤 4: 增强后信号 + 原始信号作为正对
```

### 3.3 ECG 的 Hard Negative 生成

**核心问题**：在对比学习中，hard negative（与 anchor 相似但属于不同类别的样本）对表征学习至关重要。ECG 中的 hard negative 天然存在：**不同 MI 亚型之间的信号差异极其细微**。

**文献支撑**：
- Semantic-Aware Hard Negative mining (SAHN, ACM MM 2025) 在医学视觉-语言对比预训练中，使用语义感知的 hard negative 挖掘来区分真正的负样本和"假负样本"（语义相似但标签不同的样本）。
- Cross-Modality Cardiac Insight Transfer (MICCAI 2024) 使用自适应 hard negative 加权的跨模态对比目标。

**我们的创新方向**：

**MI-Aware Hard Negative Mining**：
- 在 SigLIP 预训练中，当前所有其他样本的 12 导联嵌入都是负样本
- 改进：按临床相似度加权负样本，使模型更关注难以区分的 MI 亚型对：
  - ALMI vs LMI（都是侧壁 MI）
  - IMI vs IPLMI（都是下壁 MI）
  - AMI vs ASMI（都是前壁 MI）

这需要在微调阶段（有标签时）实现，预训练阶段（无标签）可通过 embedding 距离近似。

### 3.4 多导联 <-> 单导联的跨模态对比

**当前状态**：我们的 SigLIP 已在做 12 导联 <-> 单导联的对比学习。可进一步增强的方向：

**文献支撑**：
- Contrastive Random Lead Coding (arXiv:2410.19842, 2024) 提出随机导联 masking 作为 ECG 专用增强——在预训练时随机遮蔽部分导联，使模型学习导联不变的表征。
- Nature Communications (2023) 展示了基于大规模数据集的自监督预训练可以实现可穿戴 12 导联 ECG 的智能诊断。

**创新方向**：**渐进式导联退化训练**
```
阶段 1: 12 导联 Teacher → Lead I Student (当前方案)
阶段 2: 随机 6 导联 Teacher → Lead I Student（增加 Teacher 难度）
阶段 3: 随机 3 导联 Teacher → Lead I Student
阶段 4: Lead II Teacher → Lead I Student（单导联→单导联蒸馏）
```
这迫使 Student 学习更鲁棒的表征，因为 Teacher 也在退化。

---

## 4. Mamba/SSM 特有的机会

### 4.1 SSM 对长序列的优势在预处理/增强中的应用

**核心洞察**：Mamba 的选择性状态空间机制天然适合处理长序列，其 O(N) 时间复杂度和线性内存使得在更长的 ECG 序列上训练成为可能。

**创新方向**：**超长上下文预训练**
- 当前输入：10 秒，5000 采样点
- 扩展到：30 秒或 60 秒（15000 或 30000 采样点）
- Mamba 的 O(N) 复杂度使之可行（Transformer 的 O(N^2) 在此长度上不可行）
- 更长上下文可以捕获心率变异、ST 段演变等诊断信息

**文献支撑**：
- MSECG (arXiv:2412.04861, 2024) 使用 Mamba 进行 ECG 超分辨率，利用 Mamba 的循环结构捕获局部和全局依赖性，在干净和噪声环境下都表现稳健。
- Adaptive long-range modeling (Scientific Reports, 2025) 结合 Mamba 和动态图学习进行 EEG/ECG 的自适应长程建模。

### 4.2 选择性状态空间模型对噪声的鲁棒性分析

**核心洞察**：Mamba 的"选择性"机制（input-dependent B, C, delta 参数）理论上可以学习自动忽略噪声段、聚焦于信号段。这种**自适应滤波**能力是传统 CNN 和 Transformer 不具备的。

**文献支撑**：
- ECG-Mamba (PMC, 2025) 发现模型对噪声敏感性较高，但通过 Non-Uniform-Mix augmentation（跨 epoch 选择性 MixUp）可以有效缓解。
- MSEMG (2024) 证明 Mamba 在潜在表征空间中处理序列可以有效抑制噪声干扰。

**创新方向**：**Noise-Adaptive Selective Scanning**
- 在 Mamba 块中添加噪声感知模块：根据输入段的信号质量自适应调整 delta（时间步长），在噪声段使用较小的 delta（快速跳过），在干净段使用较大的 delta（仔细处理）
- 这是 Mamba 独有的创新点，Transformer 和 CNN 无法实现

### 4.3 Mamba 特有的 Masking 策略

**核心问题**：Transformer 的 masked pretraining 使用 [MASK] token 替换部分输入。Mamba 的因果结构使得 [MASK] token 的语义不同——**SSM 的状态是连续递推的，"空洞"会导致状态中断**。

**文献支撑**：
- EEGM2 (arXiv:2502.17873, 2025) 使用 Mamba-2 进行 EEG 自监督预训练，设计了长序列的高效 masking 策略。
- MambaTS (NeurIPS 2024) 提出 Variable Scan along Time (VST) 替代 Mamba 的因果卷积，认为时间序列中因果卷积是不必要的。

**创新方向**：**SSM-Native Masking for ECG**

不同于 Transformer 的 random token masking，为 SSM 设计的 masking 策略：

| 策略 | 描述 | 优势 |
|------|------|------|
| **State Reset Masking** | mask 位置不注入 [MASK] token，而是重置 SSM 状态 | 避免虚假状态传播 |
| **Continuous Segment Masking** | 连续 mask 整个心搏周期 | 迫使模型从上下文心搏预测被 mask 的心搏 |
| **Selective Dropout** | 在 Mamba 的选择性参数 (B, C) 上做 dropout | 正则化选择性机制 |
| **Frequency-Band Masking** | mask 特定频段（如 ST 段频率），要求从其他频段预测 | 学习频域鲁棒表征 |

**推荐**：**Continuous Segment Masking + State Reset** 最适合 ECG 的周期性结构。一个被 mask 的完整心搏可以从前后心搏的形态学中推断，这迫使模型学习心搏间的一致性——正是 MI 诊断所需的能力。

---

## 5. 上臂心电设备的创新需求

### 5.1 非标准导联的信号特点与特殊预处理

**技术背景**：上臂导联（类 aVL，约 -30deg 到 0deg）与标准 Lead I（0deg）存在角度差异，导致信号幅度和形态学差异。具体表现为：
- 信号幅度可能更小（电极间距短）
- 频率成分分布不同（肌电噪声更多）
- 不同 MI 亚型的可检测性变化

**文献支撑**：
- Wearable armband ECG (PMC, 2025) 展示了上臂石墨烯-纺织品电极的可穿戴 ECG 系统，食指探索电极可测量多导联 ECG。
- CNNAED (Biomedical Engineering Letters, 2025) 专门针对 armband ECG 的运动伪差和 EMG 噪声设计了深度学习 R 峰检测框架。
- Domain Adaptation of ECG Signals (Applied Sciences, 2025) 提出模糊能量-频率频谱网络 (FEFSN) 进行 ECG 域适应。

**创新方向**：**导联转换预训练**
```
训练阶段: Lead I 信号 → Mamba Encoder → 预测 aVL 信号
部署阶段: 上臂信号（类 aVL）→ 同一 Encoder → MI 分类
```
利用 PTB-XL 的 12 导联数据，在 Lead I 和 aVL 之间建立映射关系，使模型学习导联间的变换不变性。

### 5.2 运动伪差的智能检测和剔除

**文献支撑**：
- 两阶段框架 (medRxiv, 2025)：第一阶段使用小波变换 + Savitzky-Golay 滤波，第二阶段使用深度 CNN 分类信号为"可用"或"伪差污染"，达到 98.76% 分类准确率。
- MAICR 框架 (ScienceDirect, 2024)：利用加速度时域信息和频域信息评估运动伪差强度。

**创新方向**：**运动伪差感知的自适应推理**
- 在推理时自动检测信号质量
- 低质量段：跳过或降低置信度权重
- 高质量段：正常推理
- 时序聚合：多次检测取高质量段结果的加权平均

### 5.3 个人基线建模的增强策略

**核心问题**：长期监测场景需要检测个体的 ECG 变化（相对于自身基线），而非人群级分类。

**创新方向**：**Personalized Deviation Detection via Augmented Contrastive Learning**

```
训练数据生成：
  - 对同一个人的多条正常 ECG 做对比学习正对
  - 通过模拟 MI 变化（ST 抬高/压低）生成负对
  - 增强：对正常 ECG 施加生理性变异（心率变化、体位变化）
    但保持正常范围，作为 "hard positive"

推理时：
  - 编码新 ECG 与个人基线 ECG 的嵌入距离
  - 距离超过阈值 → 报警
```

### 5.4 Domain Adaptation: 标准导联 -> 非标准导联

**文献支撑**：
- FEFSN (Applied Sciences, 2025) 在频谱域进行模糊规则生成实现 ECG 域适应。
- 12-lead reconstruction from EASI leads (Bioengineering, 2024) 使用深度学习从减少的导联组重建 12 导联 ECG。

**创新方向**：**Adversarial Domain Adaptation for Lead Transfer**
```
Source Domain: PTB-XL Lead I (大量有标签数据)
Target Domain: 上臂心电 (少量无标签数据)

方法：
  1. 共享 Mamba Encoder
  2. 域判别器：区分 Lead I 和 上臂信号
  3. 对抗训练：Encoder 学习域不变的心电表征
  4. 分类器：在 Lead I 标签上训练
  5. 部署时直接用于上臂信号
```

---

## 6. 可发论文的创新方向排序

### 方向 1: Denoising-Augmented Contrastive Pretraining (DACP)

**标题候选**: "Denoising as a Third Loss: Noise-Aware Self-Supervised Pretraining for Robust Single-Lead ECG Analysis"

| 维度 | 评估 |
|------|------|
| **新颖性** | 中高。去噪作为预训练 task 在视觉/语音中已有，但 (1) 结合 SigLIP 对比 + 重建 + 去噪的三重损失未见先例，(2) 在单导联 MI 检测中未被探索 |
| **可行性** | **极高**。只需修改 `mamba_2loss_pretrain.py` 中的数据加载和损失函数，不改架构。预计 2-3 天完成代码修改 |
| **预期影响** | AUROC +1-3pp。去噪预训练可提升对噪声样本的鲁棒性，间接提升整体性能 |
| **工作量** | 1-2 周（实现 + 消融实验） |
| **优先级** | **高** |

### 方向 2: Physiology-Informed View Generation for ECG Contrastive Learning

**标题候选**: "PhysioAug: Cardiac Cycle-Aware Data Augmentation for ECG Self-Supervised Learning"

| 维度 | 评估 |
|------|------|
| **新颖性** | **高**。尽管 ECG augmentation 论文很多，但系统性地利用心电生理先验（哪些波形区间可以增强、哪些不能）的工作极少。iAAFT (2025) 保留 R 峰但不考虑 ST 段语义 |
| **可行性** | 中高。需要 R 峰检测和心搏分割（可用 `neurokit2` 库），实现复杂度中等 |
| **预期影响** | AUROC +1-2pp。主要提升来自于避免对比学习中的"假正对"——当 augmentation 破坏 ST 段时，增强后的信号与原始信号不再是同一诊断类别 |
| **工作量** | 2-3 周 |
| **优先级** | **高** |

### 方向 3: Signal Quality-Aware Curriculum Pretraining

**标题候选**: "From Clean to Noisy: Curriculum Learning for Large-Scale ECG Self-Supervised Pretraining"

| 维度 | 评估 |
|------|------|
| **新颖性** | 中。Curriculum learning 在语音中有先例 (accordion annealing)，但在 ECG 预训练中未见。结合 "Ditch the Denoiser" (2025) 的理论框架有新颖性 |
| **可行性** | **极高**。只需修改 DataLoader sampler，预计 1-2 天完成 |
| **预期影响** | AUROC +0.5-1.5pp。MIMIC-IV 中约 10-20% 的低质量样本可能在当前训练中引入噪声梯度 |
| **工作量** | 1 周 |
| **优先级** | **高** |

### 方向 4: Learnable ECG Preprocessing Layer

**标题候选**: "End-to-End Learnable Filtering for ECG Analysis: Beyond Fixed Bandpass"

| 维度 | 评估 |
|------|------|
| **新颖性** | **高**。SincNet 思路在 ECG 中几乎未被探索。可学习滤波器 + Mamba SSM 的组合完全新颖 |
| **可行性** | 中。需要修改模型架构和训练管道，验证可学习滤波器的收敛性 |
| **预期影响** | AUROC +1-3pp。如果当前固定滤波器不是最优的（很可能），自适应滤波可以恢复被错误滤除的诊断信息 |
| **工作量** | 2-3 周 |
| **优先级** | **中高** |

### 方向 5: SSM-Native Masking Pretraining for ECG

**标题候选**: "Beyond [MASK]: Cardiac Cycle-Aligned State Space Masking for ECG Foundation Models"

| 维度 | 评估 |
|------|------|
| **新颖性** | **极高**。当前 masked pretraining 方法（MAE、CREMA）都基于 Transformer。为 SSM 设计的 masking 策略是全新方向。State Reset Masking 概念未见任何文献 |
| **可行性** | 中。需要修改 Mamba 内部状态管理，可能需要 fork mambapy 代码 |
| **预期影响** | AUROC +2-4pp（如果成功）。但风险较高，可能不如 MAE 有效 |
| **工作量** | 3-4 周 |
| **优先级** | **中** |

### 方向 6: Multi-Teacher Progressive Lead Distillation

**标题候选**: "Progressive Lead Degradation: Robust Single-Lead ECG Learning via Adaptive Cross-Lead Knowledge Distillation"

| 维度 | 评估 |
|------|------|
| **新颖性** | **高**。12→1 导联蒸馏已有 (SelfMIS、KD for ECG)，但渐进式导联退化训练未见先例。与 Random Lead Masking (2024) 互补但更系统化 |
| **可行性** | 高。在现有 SigLIP 框架上，只需修改 Teacher 端的输入导联数 |
| **预期影响** | AUROC +1-2pp。使 Student 学习更鲁棒的跨导联不变表征 |
| **工作量** | 1-2 周 |
| **优先级** | **中高** |

### 方向 7: Diffusion-Mamba Hybrid for ECG Synthesis and Augmentation

**标题候选**: "RDiffMamba: Conditional ECG Generation via Diffusion-State Space Model Hybrid"

| 维度 | 评估 |
|------|------|
| **新颖性** | 中高。RDiffGAN-ECG (ScienceDirect, 2025) 已将 S4 和 Mamba 整合到扩散 GAN 中。但专门用于 MI 亚型的 class-conditional 生成尚未探索 |
| **可行性** | 中。需要实现 diffusion pipeline + Mamba backbone，工作量较大 |
| **预期影响** | AUROC +2-4pp（通过生成罕见 MI 亚型样本）。特别是 PMI (N=2)、IPLMI (N=5) 等极端罕见类 |
| **工作量** | 4-6 周 |
| **优先级** | **中** |

### 方向 8: Domain-Adaptive ECG for Non-Standard Leads

**标题候选**: "Lead-Invariant ECG Representations for Wearable Armband Devices via Adversarial Domain Adaptation"

| 维度 | 评估 |
|------|------|
| **新颖性** | **高**。标准导联→非标准导联（上臂）的域适应是新问题。与已有的 12→reduced lead 重建不同 |
| **可行性** | 低-中。需要上臂心电数据（当前无），可用 PTB-XL 的 Lead I vs aVL 做代理实验 |
| **预期影响** | 对产品化非常关键，但短期 AUROC 提升依赖于代理实验的有效性 |
| **工作量** | 4-8 周 |
| **优先级** | **中（长期高）** |

### 综合排序

| 排名 | 方向 | 优先级 | 预期 AUROC 提升 | 实现难度 | 论文新颖性 | 建议时间 |
|------|------|--------|----------------|---------|-----------|---------|
| 1 | 去噪预训练 (DACP) | 高 | +1-3pp | 低 | 中高 | 立即 |
| 2 | 信号质量 Curriculum | 高 | +0.5-1.5pp | 极低 | 中 | 立即 |
| 3 | PhysioAug 增强 | 高 | +1-2pp | 中 | 高 | 1 周内 |
| 4 | 可学习预处理层 | 中高 | +1-3pp | 中 | 高 | 2 周内 |
| 5 | 渐进式导联蒸馏 | 中高 | +1-2pp | 低 | 高 | 2 周内 |
| 6 | SSM-Native Masking | 中 | +2-4pp | 中高 | 极高 | 1 月内 |
| 7 | Diffusion-Mamba 合成 | 中 | +2-4pp | 高 | 中高 | 1-2 月 |
| 8 | 非标准导联域适应 | 中(长期高) | TBD | 高 | 高 | 2-3 月 |

---

## 7. 实施路线图

### Phase 1: 快速验证（第 1-2 周）

```
Week 1:
  Day 1-2: 实现信号质量 Curriculum Learning（修改 DataLoader sampler）
           同时在 800K 数据上计算 SQI 指标
  Day 3-5: 实现 Denoising Pretext Task（添加噪声注入 + 修改损失函数）
           在 216K 子集上快速验证

Week 2:
  Day 1-3: 在 800K 上完整运行 DACP + Curriculum
  Day 4-5: 评估：与 baseline (当前 2-Loss) 对比
           消融：单独 DACP、单独 Curriculum、两者结合
```

### Phase 2: 增强策略优化（第 3-4 周）

```
Week 3:
  Day 1-3: 实现 PhysioAug（R 峰检测 + 心搏分割 + 语义保持增强）
  Day 4-5: 在 SigLIP 预训练中集成 PhysioAug

Week 4:
  Day 1-2: 实现可学习预处理层
  Day 3-5: 对比实验：固定滤波 vs 可学习滤波 vs 无滤波
```

### Phase 3: SSM 特有创新（第 5-8 周）

```
Week 5-6:
  SSM-Native Masking 预训练实验
  渐进式导联蒸馏实验

Week 7-8:
  消融实验和论文撰写
  最优组合验证
```

### Phase 4: 长期方向（第 9-16 周）

```
Week 9-12:
  Diffusion-Mamba ECG 合成（如果前述方向未达目标）

Week 13-16:
  非标准导联域适应（配合上臂心电数据获取）
```

---

## 8. 参考文献

### 预处理与去噪
1. CREMA: Song et al. "A Contrastive Regularized Masked Autoencoder for Robust ECG Diagnostics across Clinical Domains." arXiv:2407.07110 (2024).
2. FGDAE: "Fully-Gated Denoising Auto-Encoder for Artifact Reduction in ECG Signals." Sensors 25(3):801 (2025).
3. Preprocessing Review: "Preprocessing and Denoising Techniques for Electrocardiography and Magnetocardiography: A Review." PMC (2024).
4. MTL-NET: "Multitask Learning-Based Quality Assessment and Denoising of Electrocardiogram Signals." IEEE TBME (2024).

### 噪声鲁棒性与 Curriculum Learning
5. "Ditch the Denoiser: Emergence of Noise Robustness in Self-Supervised Learning from Data Curriculum." arXiv:2505.12191 (2025).
6. Signal Quality Auditing: "Signal Quality Auditing for Time-series Data." arXiv:2402.00803 (2024).

### 对比学习与增强
7. Multi-stage Temporal and Cross-view CL: ScienceDirect (2025).
8. ECG-FM: "An Open Electrocardiogram Foundation Model." JAMIA Open 8(5) (2025).
9. CLOCS: Kiyasseh et al. "CLOCS: Contrastive Learning of Cardiac Signals Across Space, Time, and Patients." ICML (2021).
10. Contrastive Random Lead Coding: arXiv:2410.19842 (2024).
11. SAHN: "Semantic-Aware Hard Negative Mining for Medical Vision-Language Contrastive Pretraining." ACM MM (2025).

### ECG 增强
12. GeoECG: Zhu et al. "Data Augmentation via Wasserstein Geodesic Perturbation for Robust ECG Prediction." PMLR (2022).
13. iAAFT Surrogates: "Augmentation of EEG and ECG Time Series: Integrating Changepoint Detection into iAAFT Surrogates." arXiv:2504.03761 (2025).
14. Specialized ECG Augmentation: "Leveraging Precordial Lead Positional Variability." Biomedical Engineering Letters (2025).
15. RDiffGAN-ECG: "High-quality Specific R-peaks Conditional ECG Generation Using Denoising Diffusion GAN with SSMs." ScienceDirect (2025).
16. Synthetic ECG Scoping Review: ScienceDirect (2024).

### Mamba/SSM
17. ECG-Mamba: "Cardiac Abnormality Classification With Non-Uniform-Mix Augmentation." PMC (2025).
18. MSECG: "Incorporating Mamba for Robust and Efficient ECG Super-Resolution." arXiv:2412.04861 (2024).
19. EEGM2: "An Efficient Mamba-2-Based Self-Supervised Framework for Long-Sequence EEG Modeling." arXiv:2502.17873 (2025).
20. MambaTS: "Improved Selective State Space Models for Long-term Time Series Forecasting." NeurIPS (2024).
21. Adaptive long-range modeling: "Mamba and Dynamic Graph Learning for EEG and ECG." Scientific Reports (2025).

### 知识蒸馏
22. "A Novel Method for Reducing Arrhythmia Classification from 12-Lead to Single-Lead via Teacher-Student KD." Information Sciences (2022).
23. Multi-teacher Decomposed Feature Distillation: ResearchGate (2025).
24. Lightweight KD for Wearable ECG: Sensors 24(24):7896 (2024).

### 可穿戴与域适应
25. FEFSN: "Domain Adaptation of ECG Signals Using a Fuzzy Energy-Frequency Spectrogram Network." Applied Sciences (2025).
26. Wearable Armband ECG: PMC (2025).
27. CNNAED: "Deep Learning for Suppressing EMG and Motion Artifacts in Armband ECG." Biomedical Engineering Letters (2025).
28. ECG Foundation Model at Scale: "An ECG Foundation Model Built on over 10 Million Recordings." PMC (2025).

### Foundation Models
29. CSFM: "Cardiac Health Assessment Across Scenarios Using a Multimodal Foundation Model from 1.7M Individuals." Nature Machine Intelligence (2026).
30. C-MELT: "Contrastive Enhanced Masked Auto-Encoders for ECG-Language Pre-Training." ICML (2025).
31. ExChanGeAI: "End-to-End Platform for ECG Analysis and Model Fine-Tuning." JMIR (2026).
