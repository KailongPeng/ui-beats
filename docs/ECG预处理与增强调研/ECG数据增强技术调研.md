# ECG 数据增强（Data Augmentation）技术综述

**调研日期**: 2026-03-27
**调研目的**: 为 MIMIC-IV-ECG 800K 预训练 + PTB-XL 9 类 MI 亚型分类微调项目提供数据增强技术参考
**项目背景**: Mamba (SSM) 架构, 2-Loss 预训练 (SigLIP 对比 + 重建), 单导联 Lead I, 500Hz/10s, 当前 Macro AUROC 0.8244

---

## 目录

1. [传统信号级增强](#1-传统信号级增强)
2. [频域 / 时频域增强](#2-频域--时频域增强)
3. [基于生成模型的增强](#3-基于生成模型的增强)
4. [自监督 / 对比学习中的增强](#4-自监督--对比学习中的增强)
5. [类别不平衡处理](#5-类别不平衡处理)
6. [Mixup 系列](#6-mixup-系列)
7. [创新机会分析](#7-创新机会分析)
8. [总结与推荐方案](#8-总结与推荐方案)

---

## 1. 传统信号级增强

传统信号级增强是最基础、最常用的 ECG 数据增强方法族。根据 Sensors 2023 系统综述（Alickovic et al.）对 106 篇文献的分析，基础增强方法（Random Transformation, RT）尽管简单，但在提升性能方面最为稳定一致（consistently increasing performances compared to not using any augmentation）。

### 1.1 时间域增强

#### (a) 随机裁剪 (Random Crop)

- **原理**: 从完整 ECG 记录中随机截取固定长度窗口。对于 10 秒 5000 点的信号，可在信号前后留有余量后随机选取起始位置。
- **代表性工作**: Raghu et al., "Data Augmentation for Electrocardiograms", CHIL 2022（MIT）—— 系统比较了多种增强方法，提出 TaskAug 框架，发现增强策略的有效性高度依赖于具体任务。
- **优点**: 实现简单；模拟实际采集中信号起止点的变异性；对单导联尤其有效。
- **缺点**: 可能截断关键波形（如 P 波或 T 波）；裁剪窗口太短会丢失诊断信息。
- **与我们项目的相关性**: ★★★★ — 我们的 10 秒/5000 点信号可以通过 random crop 模拟不同的心搏位置，且实现成本极低。

#### (b) 时间拉伸/压缩 (Time Warping / Stretching)

- **原理**: 通过插值将信号沿时间轴进行非均匀拉伸或压缩，模拟心率变异性（HRV）。具体实现通常使用三次样条（cubic spline）定义平滑的变形路径，使时间步之间产生连续的拉伸与压缩过渡。
- **代表性工作**: Um et al., "Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring Using Convolutional Neural Networks", ICMI 2017; Le Guennec et al., 2016。STAR 方法（Sinusoidal Time-Amplitude Resampling, 2025）是最新的改进变体，在 R-R 间期内进行正弦波形的时间重采样和幅度缩放，保持 P-QRS-T 形态完整性。
- **优点**: 模拟自然心率变异，生理合理性高；STAR 方法在临床 ECG 分类中表现最优（Macro-AUROC 0.90）。
- **缺点**: 过度变形可能产生非生理模式；需控制变形程度以保持 QRS 波群完整性。
- **与我们项目的相关性**: ★★★★★ — MI 亚型诊断依赖 QRS 和 ST-T 形态，STAR 这类 beat-wise 增强可以保护关键形态特征。**强烈推荐尝试。**

#### (c) 随机时移 (Random Shift / Jittering)

- **原理**: 对信号沿时间轴整体移动若干采样点，填充零值或复制边界值。
- **优点**: 极简实现；模拟采集延迟。
- **缺点**: 位移过大时信号截断；对已做中心对齐的数据效果有限。
- **与我们项目的相关性**: ★★★ — 可与 random crop 合并使用。

### 1.2 幅度域增强

#### (a) 幅度缩放 (Amplitude Scaling)

- **原理**: 将 ECG 信号乘以随机缩放因子 α ~ N(1, σ²)，模拟不同患者体型、电极接触阻抗的差异。
- **优点**: 实现极简；生理合理（不同体型确实导致不同 ECG 幅度）。
- **缺点**: 过大的缩放可能模拟病理性改变（如 QRS 幅度异常增大模拟心肌肥厚）。
- **与我们项目的相关性**: ★★★★ — 单导联受电极接触的影响更大，幅度缩放是合理的增强。推荐 σ ∈ [0.05, 0.2]。

#### (b) 高斯噪声注入 (Gaussian Noise Injection)

- **原理**: 添加 ε ~ N(0, σ²) 的白噪声，模拟测量噪声和设备噪声。
- **代表性工具**: MIT-BIH Noise Stress Test Database 提供真实噪声模板（包含肌电噪声、基线漂移、电极运动伪影）。
- **优点**: 增强模型对噪声的鲁棒性；对可穿戴设备场景尤其重要。
- **缺点**: 噪声过大会淹没微弱的 ST-T 改变（MI 诊断关键）。
- **与我们项目的相关性**: ★★★★★ — 上臂心电设备的信号质量远低于标准 12 导联，噪声增强可直接提高模型在真实部署场景的鲁棒性。**必须使用。**

#### (c) 基线偏移注入 (Baseline Wander Injection)

- **原理**: 添加低频正弦信号模拟呼吸运动导致的基线漂移，频率通常在 0.15-0.5 Hz。
- **优点**: 模拟真实采集环境中最常见的伪影之一。
- **缺点**: 过大的基线偏移可能掩盖 ST 段改变。
- **与我们项目的相关性**: ★★★★ — 可穿戴设备中基线漂移是主要噪声源。

#### (d) 工频噪声注入 (Powerline Noise)

- **原理**: 添加 50/60 Hz 正弦波及其谐波，模拟电网干扰。
- **与我们项目的相关性**: ★★★ — 主要在实验室环境中有意义。

### 1.3 信号变换

#### (a) 时间翻转 (Time Reversal)

- **原理**: 将信号序列反转：x' = [x_N, x_{N-1}, ..., x_1]。
- **优点**: 操作简单。
- **缺点**: **不推荐。** 系统综述明确指出翻转后的 ECG 不具备生理合理性（P-QRS-T 顺序破坏），可能引入错误的学习偏差。
- **与我们项目的相关性**: ★ — 不建议用于 MI 分类。

#### (b) 空间翻转 (Amplitude Inversion)

- **原理**: 将信号幅度取反：x' = -x。
- **优点**: 操作简单。
- **缺点**: **不推荐。** 翻转后的 ECG 代表镜像心脏活动，与原始标签不一致。
- **与我们项目的相关性**: ★ — 不建议。

#### (c) 通道置换 (Lead Shuffling)

- **原理**: 随机打乱多导联的顺序。
- **优点**: 增加数据多样性。
- **缺点**: **不推荐用于 MI 定位/分类。** 因为 MI 亚型（前壁、下壁、侧壁等）的诊断依赖于特定导联的特征，打乱导联顺序会破坏空间信息。
- **与我们项目的相关性**: ★ — 我们使用单导联 Lead I，不适用。

### 1.4 窗口级增强

#### (a) Sliding Window

- **原理**: 以固定步长在长信号上滑动固定大小窗口，生成重叠或非重叠的多个样本。
- **优点**: 有效扩大数据量；保持每个窗口的时间连续性。
- **缺点**: 相邻窗口间高度相关，可能导致过拟合。
- **与我们项目的相关性**: ★★★ — 如果原始信号长于 10 秒，可使用 sliding window 提取多个片段。

#### (b) Random Erasing / Dropout

- **原理**: 随机将信号的某一段设为零值（类似图像的 Cutout）。
- **优点**: 迫使模型不依赖于单一时间段的特征。
- **缺点**: 可能恰好抹除关键的 QRS 或 ST 段。
- **与我们项目的相关性**: ★★★ — 需要控制 dropout 的位置和长度。

---

## 2. 频域 / 时频域增强

### 2.1 频率掩码 (Frequency Masking / SpecAugment 思路)

- **原理**: 借鉴语音识别中的 SpecAugment（Park et al., 2019），将 ECG 转换为频谱图（STFT/CWT），然后在频率维度上随机遮蔽若干连续频带，或在时间维度上遮蔽若干帧。
- **代表性工作**:
  - 时频掩码已被用于心律失常分类的 ViT 框架，在 STFT 频谱图上应用时频掩码以解决类别不平衡问题。
  - Frequency-guided Masking（频率引导掩码）选择性地遮蔽信号中信息量最大的片段。
- **优点**: 在频域中操作可以精细控制增强的频率范围；避免直接破坏时域波形形态。
- **缺点**: 需要额外的 STFT/CWT 变换步骤；掩码过宽可能删除关键频率成分。
- **与我们项目的相关性**: ★★★ — 如果模型输入为时域信号（我们的 Mamba 直接处理 1D 序列），则需要在增强后反变换回时域，增加复杂度。但如果未来探索 2D 输入（频谱图），则非常有用。

### 2.2 小波域增强 (Wavelet Domain Augmentation)

- **原理**: 使用连续小波变换 (CWT) 将 ECG 转换为时频表示（小波尺度图），在小波域中进行扰动（如系数缩放、噪声注入），再通过逆变换恢复信号。
- **代表性工作**:
  - ECG scalogram 用于 MAE 预训练，发现 85% 掩码率 + 500 epochs 效果最优（Nature Scientific Reports, 2025）。
  - Multi-Scale Wavelet-Transformer（2025）使用小波分解实现多尺度特征提取。
- **优点**: 小波变换天然适合分析 ECG 这类非平稳信号；可在不同尺度上独立增强。
- **缺点**: 计算成本较高；逆变换可能引入伪影。
- **与我们项目的相关性**: ★★★ — 对 Mamba 直接 1D 输入的场景，小波域增强需额外变换步骤。但作为预训练数据预处理方式值得探索。

### 2.3 STFT 域增强

- **原理**: 将 ECG 信号做短时傅里叶变换后，在幅度谱或相位谱上进行扰动。
- **优点**: 可精确控制增强的频率和时间范围。
- **缺点**: 相位扰动容易产生非自然信号；反变换的完整性依赖于窗函数选择。
- **与我们项目的相关性**: ★★ — 对 1D Mamba 架构不太直接适用。

### 2.4 带通滤波 (Band-Pass Filtering)

- **原理**: 使用 Butterworth 滤波器进行高通（如 0.5 Hz 去基线漂移）、低通（如 47 Hz 去高频噪声）或带通滤波，作为数据预处理或增强手段。
- **与我们项目的相关性**: ★★★ — 更适合作为预处理而非增强。可以在增强管线中随机改变滤波器参数来引入多样性。

---

## 3. 基于生成模型的增强

### 3.1 GAN 生成合成 ECG

GAN 是 ECG 合成研究中最广泛使用的方法，约占合成 ECG 文献的 28.16%（Synthetic ECG Generation: A Scoping Review, 2024）。

#### (a) DCGAN / Standard GAN

- **原理**: 深度卷积 GAN 直接在 1D ECG 信号上训练生成器和判别器。
- **代表性工作**: Delaney et al. (2019) 使用 DCGAN 生成 MIT-BIH 数据集的心律失常 ECG。
- **优点**: 结构简单，训练相对稳定。
- **缺点**: 模式坍塌问题；生成质量有限。

#### (b) WGAN / WGAN-GP

- **原理**: 使用 Wasserstein 距离替代原始 GAN 的 JS 散度，WGAN-GP 进一步添加梯度惩罚以稳定训练。
- **代表性工作**:
  - AC-WGAN-GP（Auxiliary Classifier WGAN with Gradient Penalty）首次以 1D 形式应用于 MIT-BIH 心律失常数据集，单一模型可生成所有类别的合成 ECG。
  - 增强效果：少数类分类改善 0.24%-32%（视具体类别而定）。
- **优点**: 训练更稳定；支持条件生成。
- **缺点**: 仍需大量训练数据；对长序列（如 5000 点）的生成质量可能下降。
- **与我们项目的相关性**: ★★★ — 可用于生成 PMI 等极少数类的合成样本。

#### (c) 条件 GAN (Conditional GAN / CGAN)

- **原理**: 在生成器和判别器中加入类别标签条件，使模型按指定病种生成 ECG。
- **代表性工作**:
  - PCA-CGAN（Principal Component Analysis-based CGAN, 2024）通过 PCA 降维解决长序列 ECG 拟合困难的问题。
  - CECG-GAN（2024, Nature Scientific Reports）专门处理心脏健康诊断中的数据不平衡。
  - ResT-ECGAN 结合 Transformer + ResNet + GAN（DCGAN + LSTM），过滤生成的 ECG 信号以确保质量。
- **优点**: 可按类别生成，直接解决类别不平衡；改善幅度通常为 1.3%-2.6%。
- **缺点**: 条件生成需要足够的每类训练样本来学习条件分布；对于 PMI（仅 2 条正样本）这样的极端少数类，GAN 可能无法学习有意义的分布。
- **与我们项目的相关性**: ★★★ — 对于有一定样本量的 MI 亚型（如 ASMI, IMI）有价值，但对极少数类（PMI, LMI）效果有限。

### 3.2 VAE 生成

#### (a) 标准 VAE

- **原理**: 编码器将 ECG 映射到潜在空间，解码器从潜在空间采样重建 ECG。潜在空间的正则化使其具有连续、平滑的结构，便于生成新样本。
- **代表性工作**:
  - Conditional Nouveau VAE (cNVAE-ECG, 2025) 可生成多种病理的高分辨率 12 导联 ECG。
  - VQ-VAE（Vector Quantized VAE）在 12 导联频谱图像上实现了 6% 的改善。
- **优点**: 训练比 GAN 更稳定；潜在空间结构化，便于插值生成。
- **缺点**: 生成的信号通常比 GAN 更模糊；重建质量有限。
- **与我们项目的相关性**: ★★★ — 我们的 2-Loss 预训练已包含重建损失，VAE 的思路与之相容。可考虑利用预训练编码器的潜在空间进行少数类增强。

#### (b) 条件 VAE (Conditional VAE)

- **原理**: 在编码器和解码器中加入条件信息（类别标签、心脏参数等）。
- **代表性工作**:
  - FEM 仿真器训练的条件 VAE 可基于心脏参数合成 ECG（PMC, 2024）。
  - Hierarchical VAE（2025）使用层级结构实现更精细的条件生成。
  - CNN-VAE 用于 PTB-XL 数据集分类（2026），简化架构下达到 87.01% 准确率。
- **与我们项目的相关性**: ★★★★ — 我们的预训练 encoder 可作为 conditional VAE 的基础，生成条件化的 MI 亚型 ECG。

### 3.3 扩散模型 (Diffusion Model) 生成 ECG

扩散模型是 2023 年以来 ECG 生成领域最活跃的研究方向。

#### (a) DiffECG

- **原理**: 基于 DDPM（Denoising Diffusion Probabilistic Model），使用 U-Net 架构作为去噪网络。将 1D ECG 转换为 2D 频谱图后应用扩散过程，支持生成、补全和预测三种任务。扩散步数 T=1000，噪声调度 β₀=0.0001, β_T=0.02。
- **性能**:
  - FID: 1.29×10⁻²（优于 GAN 基线 1.74×10⁻²）
  - 分类 F1 从 0.91（仅真实数据）提升到 0.96（加入 DiffECG 合成数据）
- **优点**: 生成质量超越 GAN；统一架构支持多种任务；不需要针对每种任务重新训练。
- **缺点**: 推理速度慢（需要 1000 步去噪）；训练计算成本高。
- **与我们项目的相关性**: ★★★★ — 生成质量最高的方法。但计算开销较大，建议在微调阶段离线生成合成样本后加入训练集。

#### (b) 条件扩散模型

- **原理**: 在去噪过程中加入类别嵌入、任务嵌入等条件信息。
- **代表性工作**:
  - ECGTwin（2025）：可控扩散模型生成个性化 ECG 数字孪生。
  - Simulator-Enhanced Diffusion（2025）：结合心脏模拟器经验的扩散模型。
  - MI 合成 ECG 预训练（2025）：合成 MI ECG 预训练后，GRU 模型 AUC 从 83.1% 提升到 86.6%（+3.5pp）。
- **与我们项目的相关性**: ★★★★★ — 与我们的 MI 分类任务高度相关。合成 MI ECG 预训练已被证明可显著提升 MI 检测性能。

### 3.4 质量评估方法

| 指标 | 全称 | 说明 |
|------|------|------|
| FID | Fréchet Inception Distance | 衡量生成分布与真实分布的统计距离，越小越好 |
| MMD | Maximum Mean Discrepancy | 核方法衡量分布差异 |
| DTW | Dynamic Time Warping | 时序对齐质量 |
| EMD | Earth Mover's Distance | 信号分布比较 |
| RMSE | Root Mean Square Error | 重建误差 |
| 临床验证 | — | 心内科医生盲评生成 ECG 的临床合理性 |
| 分类提升 | — | 加入合成数据后下游分类指标的改善幅度（最终评判标准） |

---

## 4. 自监督 / 对比学习中的增强

### 4.1 SimCLR / MoCo 中的 ECG View 生成策略

#### (a) 经典框架适配

- **原理**: SimCLR 框架通过数据增强生成同一样本的两个不同 view，定义正负样本对，使用 NT-Xent 损失进行对比学习。应用于 ECG 时，需要定义适合 ECG 特性的 view 生成策略。
- **常用 ECG 增强组合**:
  1. 高斯噪声 + 幅度缩放
  2. 时间裁剪 + 基线漂移
  3. 时间拉伸 + 噪声注入
  4. 生理噪声变换（肌电噪声、运动伪影等）
- **代表性工作**:
  - Kiyasseh et al., "Self-supervised representation learning from 12-lead ECG data"（2021, Computers in Biology and Medicine）：探索了人工变换与 ECG 特异性生理噪声变换对 SimCLR 性能的影响。
  - Multi-stage temporal and cross-view contrastive learning（2025, Biomedical Signal Processing and Control）：结合多阶段预测和跨视图对比学习。

#### (b) Poly-Window 对比学习

- **原理**: 从每条 ECG 记录中提取多个时间窗口作为正样本对，利用 ECG 信号的时间冗余性。不同于传统两视图方法，多窗口方法更充分利用了临床 ECG 的时间结构。
- **性能**: AUROC 0.891（优于传统两视图方法的 0.888），用于多标签超类分类。
- **代表性工作**: "Learning ECG Representations via Poly-Window Contrastive Learning"（2025）。
- **与我们项目的相关性**: ★★★★ — 我们的 SigLIP 对比预训练可以借鉴多窗口策略，从 10 秒信号中提取多个时间窗口作为正样本对。

#### (c) LEAVES (自动增强搜索)

- **原理**: 自动搜索最优数据增强策略用于对比学习，避免人工选择增强方法的主观性。
- **性能**: 使用 LEAVES 生成的视图在 SimCLR 和 BYOL 中分别保持 96.0% 和 93.1% 的数据质量。
- **与我们项目的相关性**: ★★★★ — TaskAug / LEAVES 这类自动增强搜索方法可以系统地找到最适合我们 MI 分类任务的增强组合。

### 4.2 Masking 策略

#### (a) 随机掩码 (Random Masking)

- **原理**: 随机遮蔽 ECG 信号的一部分，要求模型重建被遮蔽的部分（类似 BERT 的 MLM）。
- **代表性工作**:
  - MaeFE（Masked Autoencoders Family of ECG, 2022）包含三种掩码模式：
    - MTAE（Masked Time Autoencoder）：沿时间轴掩码，关注时间特征
    - MLAE（Masked Lead Autoencoder）：掩码整个导联，关注空间特征
    - MLTAE（Masked Lead and Time Autoencoder）：同时掩码时间和导联，多头架构
  - MAE-ECG（2024, PLOS ONE）：使用 ViT 架构，高掩码率（75%）预训练后在下游任务上超越其他 DNN 模型。
- **关键发现**: 更高的掩码率和随机掩码策略通常能提升性能。ECG scalogram 上 85% 掩码率 + 500 epochs 效果最优。

#### (b) Wave Masking（波形级掩码）

- **原理**: 基于 ECG 先验知识，掩码完整的波形（P 波、QRS 波群、T 波等），而非随机位置。
- **代表性工作**: WMAE（Wave Masked Autoencoder, 2024）：设计波形级掩码策略，考虑 ECG 信号中各波形是诊断心律失常的关键特征。
- **优点**: 利用 ECG 领域知识，使掩码更有意义；迫使模型学习波形间的关系。
- **缺点**: 需要 R 波检测等预处理步骤。
- **与我们项目的相关性**: ★★★★★ — 我们的 2-Loss 预训练包含重建损失，wave masking 可以直接集成。对于 MI 诊断，掩码 ST-T 段并要求重建，可以迫使模型深入理解 ST-T 形态与 MI 的关系。**高度推荐。**

#### (c) Multi-scale Masking

- **原理**: 在不同尺度上同时进行掩码，捕获多粒度特征。
- **代表性工作**: Multi-scale Masked Autoencoder（2025）用于 ECG 异常检测。
- **与我们项目的相关性**: ★★★★ — Mamba 的多尺度特征提取可以与 multi-scale masking 配合使用。

### 4.3 与我们 2-Loss 预训练的关系

我们当前的预训练框架使用 **SigLIP 对比损失 + 重建损失**，这与上述增强策略有以下关联：

| 组件 | 当前状态 | 建议增强 |
|------|---------|---------|
| SigLIP 对比学习 | 使用什么增强策略？ | 建议加入：高斯噪声 + 幅度缩放 + 时间裁剪的组合 |
| 重建损失 | 全信号重建？ | 建议改为 wave masking 或 random masking 后重建 |
| 正样本对构建 | 同一信号两个 view？ | 建议探索 poly-window 策略（多窗口正样本对） |

**核心建议**: SigLIP 的 sigmoid 损失本身允许更大的 batch size 并在小 batch size 下也表现良好，可以利用这一特性增加正样本对的多样性。建议在对比学习中使用 **"强增强 + 弱增强"** 的非对称策略：
- 弱增强 view：仅加轻微高斯噪声
- 强增强 view：时间裁剪 + 幅度缩放 + 基线漂移 + 较大噪声

---

## 5. 类别不平衡处理

### 5.1 SMOTE 及其变体

#### (a) 标准 SMOTE

- **原理**: 在少数类样本的特征空间中，随机选择 k 近邻，在样本与近邻之间线性插值生成新样本。
- **问题**: 直接在原始 ECG 时域上做 SMOTE 可能生成非生理信号，因为线性插值无法保证 P-QRS-T 波形结构的一致性。
- **与我们项目的相关性**: ★★ — 不建议在原始 ECG 时域上使用。

#### (b) T-SMOTE (Temporal-oriented SMOTE)

- **原理**: Microsoft Research 提出的时间导向 SMOTE，充分利用时序数据的时间信息进行过采样。
- **性能**: 在不平衡时序分类任务上一致优于其他 SMOTE 变体。
- **与我们项目的相关性**: ★★★ — 比标准 SMOTE 更适合 ECG 时序数据。

#### (c) 特征空间 SMOTE

- **原理**: 在编码器的潜在空间（而非原始信号空间）中进行 SMOTE 插值，然后通过解码器恢复信号。
- **优点**: 潜在空间的线性插值更有意义；生成的样本更自然。
- **与我们项目的相关性**: ★★★★ — 可利用我们预训练 Mamba 编码器的潜在空间进行特征级插值。**值得探索。**

#### (d) CSMOTE (Contrastive SMOTE)

- **原理**: 将对比学习与 SMOTE 结合，在对比学习的表示空间中进行过采样。
- **代表性工作**: CSMOTE（Springer, 2021）。
- **与我们项目的相关性**: ★★★★ — 与我们的对比预训练框架天然兼容。

### 5.2 Oversampling vs Augmentation

| 方法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| Random Oversampling | 简单复制少数类样本 | 实现最简单 | 严重过拟合风险 |
| SMOTE | 特征空间插值 | 生成新样本 | 时序数据上可能非生理 |
| 数据增强 | 变换原始样本 | 增加真实多样性 | 需要领域知识设计 |
| 生成模型 | GAN/VAE/Diffusion | 最高质量新样本 | 计算成本高；极少类可能学不到分布 |
| 加权采样 | 按类别频率调整采样概率 | 不改变数据本身 | 不增加实际多样性 |

**对于我们的场景（PMI 仅 2 条正样本）**，推荐组合方案：
1. 首先用**加权采样**确保每个 batch 中少数类有足够代表
2. 对少数类样本施加**更强的数据增强**（增加噪声水平、增大变形程度）
3. 如有条件，使用**条件扩散模型**离线生成少数类合成样本

### 5.3 Focal Loss / Class-Weighted Loss 与增强的协同

#### (a) Focal Loss

- **原理**: FL(p_t) = -α_t(1-p_t)^γ log(p_t)，通过 (1-p_t)^γ 降低易分样本权重，聚焦于难分样本。γ 通常取 2。
- **与增强的协同**: 数据增强增加样本多样性，Focal Loss 聚焦于难分样本，两者互补。增强产生的边界样本恰好是 Focal Loss 最关注的对象。

#### (b) Class-Weighted BCE

- **原理**: 对不同类别的二元交叉熵损失乘以与类别频率成反比的权重。
- **权重计算**: w_c = N_total / (N_classes × N_c)，或使用 sqrt 频率加权。

#### (c) 综合策略

最新研究（2025）提出多层次不平衡处理策略的组合：
1. **sqrt-frequency weighted sampling**: 平衡采样
2. **Asymmetric Focal Loss**: 非对称焦点损失
3. **Label Smoothing**: 防止过拟合
4. **Mixup Augmentation**: 数据增强
5. **Per-class Threshold Optimization**: 阈值优化

- **与我们项目的相关性**: ★★★★★ — MI 亚型分类属于极端不平衡的多标签分类问题，必须组合使用多种策略。**建议先实现 Focal Loss + 加权采样 + 数据增强的组合，再逐步添加更复杂的策略。**

---

## 6. Mixup 系列

### 6.1 Mixup

- **原理**: 对两个样本及其标签进行线性插值：
  - x̃ = λx_i + (1-λ)x_j
  - ỹ = λy_i + (1-λ)y_j
  - λ ~ Beta(α, α)，α 通常取 0.2-0.4
- **在 1D ECG 上的适配**: 直接对两条 ECG 信号做线性混合。
- **已有 ECG 研究**: 在 PTB-XL 等数据集上使用 Mixup 通常带来 +0.5-2% 的准确率提升。
- **问题**: **标准 Mixup 对 Mamba 模型效果不佳**。ECG-Mamba 论文（IEEE JBHI, 2025）明确指出 "common data augmentation methods such as MixUp and CutMix do not perform well with Mamba on ECG data"，标准 Mixup 使 AUPRC 从 0.6100 降至 0.6042，AUROC 从 0.9643 降至 0.9616。
- **与我们项目的相关性**: ★★ — **不建议直接使用标准 Mixup**，因为我们也使用 Mamba 架构。需使用 Non-Uniform-Mix 等 Mamba 友好的变体。

### 6.2 CutMix

- **原理**: 从一个样本中剪切一段时间窗口，粘贴到另一个样本的对应位置，标签按时间比例混合。
- **在 1D ECG 上的适配**: 沿时间轴剪切和粘贴，保持粘贴区域内的时间连续性。
- **优点**: 比 Mixup 更好地保持局部波形结构。
- **缺点**: 切割边界处会产生不连续性；同样对 Mamba 效果不佳。
- **与我们项目的相关性**: ★★ — 与 Mixup 相同的 Mamba 兼容性问题。

### 6.3 Manifold Mixup

- **原理**: 在模型的隐藏层特征空间中进行 Mixup，而非在原始输入空间。
- **已有 ECG 研究**: 在 6 个生物医学时序数据集（包含 PTB-XL 和 Apnea-ECG）上实验，CutMix 和 Manifold Mixup 分别带来 +0.5-2% 的准确率提升，优于标准时域增强。
- **优点**: 特征空间的线性插值比信号空间更有意义；不直接破坏波形形态。
- **缺点**: 需要选择合适的隐藏层进行混合。
- **与我们项目的相关性**: ★★★★ — Manifold Mixup 在特征空间操作，可能避免 Mamba 对输入噪声的敏感性问题。**值得测试。**

### 6.4 Non-Uniform-Mix（ECG-Mamba 专用）

- **原理**: 针对 Mamba 模型设计的保守增强策略。在不同 epoch 对数据集的不同比例应用 MixUp：
  - Epoch 1: 20% 数据参与 MixUp
  - Epoch 2: 40%
  - Epoch 3: 60%
  - Epoch 4+: 80%（恒定）

  这种渐进式增强使 Mamba 模型能在早期学习干净特征，后期逐步适应增强数据。

- **数学表达**:
  - x̃ = λx + (1-λ)x̃_random
  - ỹ = λy + (1-λ)y_random
  - λ ~ Beta(α, α)

- **性能**:
  - AUPRC: 0.6271（基线 0.6100，+2.8%）
  - AUROC: 0.9671（基线 0.9643，+0.3%）
  - Challenge Score: 0.7195（基线 0.7013）

  对比之下，标准 MixUp 使 AUPRC 降至 0.6042，AUROC 降至 0.9616（均低于基线）。

- **与我们项目的相关性**: ★★★★★ — **这是目前唯一被验证对 Mamba ECG 模型有效的 Mix 类增强方法。** 必须优先尝试。

---

## 7. 创新机会分析

### 7.1 当前数据增强的未解决问题

1. **极端少数类增强困境**: 当某个类别仅有 2-5 个正样本时（如我们的 PMI），所有生成模型都无法学到有意义的类条件分布。目前没有公认的解决方案。

2. **增强-分类器兼容性**: 不同模型架构对增强方法的响应差异巨大（如 Mamba 对标准 Mixup 负响应），但缺乏系统性的指导原则。

3. **临床合理性保证**: 现有增强方法缺乏系统的临床验证机制，无法保证增强后的 ECG 仍然保持特定病理的关键特征。

4. **单导联限制**: 大多数增强研究基于 12 导联 ECG，单导联（Lead I）场景下的增强策略研究不足。

5. **增强策略的自适应选择**: TaskAug / LEAVES 等自动增强搜索方法仍处于初期阶段，缺乏针对 ECG 的大规模验证。

### 7.2 针对 MI 亚型分类的潜力增强

按照优先级排序：

| 优先级 | 方法 | 理由 | 预期收益 |
|--------|------|------|---------|
| P0 | Non-Uniform-Mix | 唯一验证过对 Mamba 有效的 Mix 增强 | +2-3% AUPRC |
| P0 | 高斯噪声 + 基线漂移 | 实现简单；提升可穿戴鲁棒性 | +1-2% AUROC |
| P1 | STAR (beat-wise 时间-幅度重采样) | 保护 P-QRS-T 形态的时间增强 | +1-3% Macro AUROC |
| P1 | Wave Masking 预训练 | 迫使模型学习 ST-T 形态重建 | 预训练质量提升 |
| P1 | Focal Loss + 加权采样 | 直接解决类别不平衡 | 少数类提升显著 |
| P2 | 条件扩散模型合成 | 为中等少数类生成高质量合成 ECG | +1-4% AUC |
| P2 | Manifold Mixup | 特征空间操作避免输入噪声敏感 | 待验证 |
| P3 | Poly-Window 对比学习 | 多窗口正样本对提升预训练 | 预训练质量提升 |
| P3 | 特征空间 SMOTE | 利用预训练 encoder 的潜在空间插值 | 少数类提升 |

### 7.3 与 Mamba 架构结合的特殊考虑

Mamba（Selective State Space Model）对数据增强有以下特殊要求：

1. **时序连续性敏感**: SSM 通过递归扫描（parallel scan）处理序列，对时间步之间的连续性高度敏感。这解释了为什么标准 CutMix（在切割边界产生不连续性）对 Mamba 有害。

2. **噪声敏感性**: Mamba 的选择性机制（selective scan）旨在过滤无关信息，但过多的噪声可能干扰选择性门控。建议使用较低水平的噪声增强。

3. **渐进式增强有效**: Non-Uniform-Mix 的成功表明 Mamba 需要先在干净数据上学习基础特征，再逐步适应增强数据。

4. **特征空间增强优于输入空间**: 由于 Mamba 对输入噪声敏感，在 Mamba 编码器的中间层或输出层进行增强（如 Manifold Mixup）可能比在输入层增强更安全。

5. **Masking 与 SSM 的兼容性**: 对于 Mamba 的重建预训练，需要注意掩码不应破坏 parallel scan 的连续性。建议使用较大的连续掩码块（block masking）而非随机点掩码。

### 7.4 可以发论文的创新方向

#### 方向 1: Mamba-Aware ECG Augmentation（★★★★★）
- **核心思想**: 系统研究各种 ECG 增强方法与 Mamba 架构的兼容性，提出 Mamba-specific 的增强准则。
- **卖点**: ECG-Mamba 论文仅测试了 Mixup 和 CutMix，缺乏对更多增强方法（STAR, wave masking, manifold mixup, noise injection 等）的系统比较。
- **工作量**: 中等（3-6 月），主要是实验性工作。

#### 方向 2: Progressive Augmentation for SSM-based ECG Classification（★★★★）
- **核心思想**: 扩展 Non-Uniform-Mix 的渐进式理念到其他增强方法——在训练过程中动态调整增强强度和类型（课程学习 + 数据增强）。
- **卖点**: 结合课程学习（curriculum learning）和数据增强，针对 SSM 架构的特性设计增强课程。
- **工作量**: 中等。

#### 方向 3: Latent Space Augmentation for Extreme Class Imbalance in ECG（★★★★★）
- **核心思想**: 利用预训练 Mamba 编码器的潜在空间，对极端少数类（如 PMI, LMI）进行插值或条件生成，然后通过解码器恢复 ECG 信号。
- **卖点**: 直接解决极端不平衡问题（PMI 仅 2 例），避免在信号空间增强的生理不合理性。可与 2-Loss 预训练框架自然结合。
- **工作量**: 中到大（需要设计潜在空间操作并验证生成质量）。

#### 方向 4: Cross-Domain ECG Transfer via Augmentation（★★★★）
- **核心思想**: 通过精心设计的增强管线，使 12 导联预训练的模型更好地迁移到单导联（Lead I）场景，模拟上臂心电设备的信号特性。
- **卖点**: 直接对标可穿戴/上臂设备的应用场景，具有明确的临床转化价值。
- **工作量**: 中等。

#### 方向 5: Physiology-Informed Diffusion for MI Subtype Augmentation（★★★★★）
- **核心思想**: 结合心脏电生理学知识（各 MI 亚型的特征性 ECG 改变），设计生理学引导的条件扩散模型，生成各 MI 亚型的合成 ECG。
- **卖点**: 物理/生理模型 + 深度生成模型的结合；Naghashyar (2025) 已初步验证了合成 MI ECG 预训练的可行性，但仅涵盖常见 MI 类型，未覆盖罕见亚型。
- **工作量**: 大（需要心脏电生理学专业知识）。

---

## 8. 总结与推荐方案

### 8.1 针对我们项目的分层推荐

#### 第一阶段：低成本快速实验（1-2 周）

| 方法 | 实现难度 | 预期效果 |
|------|---------|---------|
| 高斯噪声注入 (σ=0.01-0.05) | ★ | 鲁棒性提升 |
| 幅度缩放 (σ=0.1) | ★ | 多样性增加 |
| 基线漂移注入 (0.15-0.5 Hz) | ★★ | 可穿戴鲁棒性 |
| Non-Uniform-Mix | ★★ | +2-3% AUPRC |
| Focal Loss (γ=2) + 加权采样 | ★★ | 少数类改善 |

#### 第二阶段：中等投入实验（2-4 周）

| 方法 | 实现难度 | 预期效果 |
|------|---------|---------|
| STAR (beat-wise 时间-幅度重采样) | ★★★ | +1-3% Macro AUROC |
| Wave Masking 预训练改进 | ★★★ | 预训练表征质量提升 |
| Manifold Mixup (中间层) | ★★★ | 待验证，可能 +1-2% |
| 多窗口正样本对（Poly-Window） | ★★★ | SigLIP 预训练改进 |

#### 第三阶段：高投入研究（1-3 月）

| 方法 | 实现难度 | 预期效果 |
|------|---------|---------|
| 条件扩散模型合成 MI ECG | ★★★★ | +1-4% AUC |
| 潜在空间 SMOTE/插值 | ★★★★ | 极少类改善 |
| 自动增强搜索 (TaskAug) | ★★★★ | 最优增强组合 |

### 8.2 增强方法与 Mamba 兼容性速查表

| 增强方法 | Mamba 兼容性 | 备注 |
|---------|-------------|------|
| 高斯噪声 | ✅ 兼容（低噪声级别）| σ < 0.05 |
| 幅度缩放 | ✅ 兼容 | 不改变时序结构 |
| 基线漂移 | ✅ 兼容 | 低频成分，不影响 scan |
| 时间裁剪 | ✅ 兼容 | 保持连续性 |
| 时间拉伸 | ✅ 兼容 | 保持连续性 |
| STAR | ✅ 可能兼容 | beat-wise 操作，保持局部连续性 |
| 标准 Mixup | ❌ 不兼容 | 实测降低性能 |
| CutMix | ❌ 不兼容 | 边界不连续干扰 SSM |
| Non-Uniform-Mix | ✅ 兼容 | Mamba 专用变体 |
| Manifold Mixup | ❓ 待验证 | 特征空间操作可能避免问题 |
| 随机掩码 | ⚠️ 需注意 | 建议使用连续块掩码 |
| 时间翻转 | ❌ 不推荐 | 破坏时序因果性 |
| 频率掩码 | ✅ 兼容 | 在频域操作，不直接影响时域连续性 |

### 8.3 关键文献索引

| 文献 | 核心贡献 | 与我们项目的关系 |
|------|---------|----------------|
| ECG-Mamba (IEEE JBHI, 2025) | Non-Uniform-Mix；Mamba+ECG 首次系统研究 | 直接指导增强策略 |
| Sensors 2023 系统综述 (Alickovic et al.) | ECG 增强方法分类体系 | 方法选择参考 |
| STAR (2025) | Beat-wise 时间-幅度重采样 | 生理保真增强 |
| DiffECG (2024) | 扩散模型 ECG 生成 | 合成数据增强 |
| Naghashyar (2025) | 合成 MI ECG 预训练 | 直接相关 |
| TaskAug / CHIL 2022 (Raghu et al.) | 任务自适应增强 | 自动增强搜索 |
| WMAE (2024) | Wave Masking 策略 | 预训练改进 |
| MaeFE (2022) | ECG Masked Autoencoder 体系 | 掩码预训练 |
| Poly-Window CL (2025) | 多窗口对比学习 | 对比预训练改进 |
| T-SMOTE (Microsoft Research) | 时间导向过采样 | 不平衡处理 |
| PCA-CGAN (PLOS ONE, 2024) | PCA 降维条件 GAN | 条件生成 |

### 8.4 最终建议

对于我们的具体场景（Mamba 架构 + SigLIP 2-Loss 预训练 + PTB-XL 9 类 MI 亚型 + 单导联 Lead I + 极端不平衡），建议采用以下**组合增强管线**：

```
预训练阶段:
  SigLIP View 1: 原信号 + 轻微高斯噪声 (σ=0.01)
  SigLIP View 2: 随机裁剪 + 幅度缩放(±10%) + 基线漂移 + 较大噪声(σ=0.03)
  重建分支: Wave Masking (掩码 ST-T 段) → 重建损失

微调阶段:
  数据层: Non-Uniform-Mix (渐进式 20%→80%)
  信号增强: 高斯噪声 + 幅度缩放 + 基线漂移（对少数类增强强度 ×2）
  损失函数: Asymmetric Focal Loss (γ=2) + sqrt-frequency 加权
  采样策略: 类别加权采样 (class-balanced sampling)
  阈值: Per-class threshold optimization
```

预期效果：Macro AUROC 从 0.8244 提升至 0.83-0.85 区间（保守估计），具体取决于增强组合的调优。

---

## 参考文献 (Web Sources)

- [A Systematic Survey of Data Augmentation of ECG Signals for AI Applications (Sensors 2023)](https://www.mdpi.com/1424-8220/23/11/5237)
- [ECG-Mamba: Cardiac Abnormality Classification With Non-Uniform-Mix Augmentation (IEEE JBHI 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12599890/)
- [Comparative Analysis of Data Augmentation for Clinical ECG Classification with STAR (2025)](https://arxiv.org/html/2510.24740v1)
- [DiffECG: A Versatile Probabilistic Diffusion Model for ECG Signals Synthesis](https://arxiv.org/html/2306.01875v2)
- [Improving Myocardial Infarction Detection via Synthetic ECG Pretraining (2025)](https://arxiv.org/html/2506.23259)
- [Data Augmentation for Electrocardiograms (CHIL 2022)](https://arxiv.org/abs/2204.04360)
- [Synthetic ECG signals generation: A scoping review (2024)](https://www.sciencedirect.com/science/article/pii/S0010482524015385)
- [Learning ECG Representations via Poly-Window Contrastive Learning (2025)](https://arxiv.org/html/2508.15225v1)
- [Self-supervised representation learning from 12-lead ECG data (2021)](https://www.sciencedirect.com/science/article/pii/S0010482521009082)
- [Multi-stage temporal and cross-view contrastive learning for ECG (2025)](https://www.sciencedirect.com/science/article/pii/S1746809425018282)
- [Wave Masked Autoencoder for ECG (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0020025524014300)
- [MaeFE: Masked Autoencoders Family of ECG (2022)](https://www.researchgate.net/publication/366230794)
- [T-SMOTE: Temporal-oriented SMOTE (Microsoft Research)](https://www.microsoft.com/en-us/research/publication/t-smote-temporal-oriented-synthetic-minority-oversampling-technique-for-imbalanced-time-series-classification/)
- [PCA-CGAN for imbalanced ECG classification (PLOS ONE 2024)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0330707)
- [Conditional VAE for ECG Generation (2025)](https://arxiv.org/abs/2503.13469)
- [ECG Classification on PTB-XL with CNN-VAE (2026)](https://arxiv.org/html/2603.07558)
- [Efficient pretraining of ECG scalogram images using MAE (2025)](https://www.nature.com/articles/s41598-025-10773-w)
- [Multi-scale Masked Autoencoder for ECG Anomaly Detection (2025)](https://arxiv.org/html/2502.05494v1)
- [Contrastive Learning for Multi-Label ECG with Jaccard Score Sigmoid Loss (2026)](https://arxiv.org/html/2602.10553v1)
- [A survey of transformers and LLMs for ECG diagnosis (2025)](https://link.springer.com/article/10.1007/s10462-025-11259-x)
- [Enhanced Multi-Class Arrhythmia Detection Using GANs (2025)](https://link.springer.com/article/10.1007/s12559-025-10520-3)
- [Horizons in Single-Lead ECG Analysis (2022)](https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2022.866047/full)
- [SigLIP: Sigmoid Loss for Language Image Pre-Training (ICCV 2023)](https://arxiv.org/abs/2303.15343)
