# 非标准/可穿戴导联 ECG 开源数据集调研

> 调研日期：2026-03-31
> 目标：寻找带 QRS/beat 标注的非标准导联 ECG 数据集，尤其是上臂/腕部/手持设备导联

---

## 核心结论

**上臂 ECG 专用带标注数据集：目前不存在开源版本。**

2024 ESC 大会有一篇摘要在近端上臂贴干电极做了评测（QRS 检测 98–99%），但数据未公开。最接近的替代方案是 TELE ECG Database（手持，手到手 Lead-I）和 SHDB-AF（非标准 Holter 导联）。

---

## Tier 1：最相关——非胸部/非标准导联，带 QRS 标注

### 1. TELE ECG Database ⭐ 最推荐

| 字段 | 内容 |
|------|------|
| URL | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QTG0EP |
| 导联 | **单导联 Lead-I，手到手干电极**（TeleMedCare 设备，无胸部贴附） |
| 标注 | ✅ 人工 QRS 标注 + 伪影掩码 |
| 规模 | 120 名 COPD/CHF 患者，250 条 × 30 秒，带质量差样本 |
| 授权 | 完全开放，CC license |
| 备注 | 最接近"手持非接触 ECG"场景，信噪比低，有真实噪声样本 |

### 2. SHDB-AF（PhysioNet，2025年发布）⭐ 非标准 Holter 导联

| 字段 | 内容 |
|------|------|
| URL | https://physionet.org/content/shdb-af/1.0.1/ |
| 导联 | **非标准 2 导联 Holter：CC5 + NASA 导联**（Fukuda Holter，非 12 标准导联） |
| 标注 | ✅ 128 条记录中 98 条有 beat 级节律标注（PhysioZoo 工具，心脏科医生审核） |
| 规模 | 122 名患者，128 条 × ~24 小时，200Hz |
| 授权 | 完全开放，ODC-By 1.0 |
| 备注 | 2025 年新发布；CC5/NASA 是非标准 Holter 导联，信号特性与标准 12 导有差异 |

---

## Tier 2：单导联可穿戴 Holter，带 beat 标注

### 3. Icentia11k

| 字段 | 内容 |
|------|------|
| URL | https://physionet.org/content/icentia11k-continuous-ecg/1.0/ |
| 导联 | 单导联改良 Lead-I，**胸贴**（CardioSTAT patch，非上臂） |
| 标注 | ✅ N/PAC/PVC/Q beat 标注 + NSR/AF/AFL 节律，~28 亿个标注 beat |
| 规模 | 11,000 名患者，最长 2 周连续记录，250Hz |
| 授权 | CC BY-NC-SA 4.0，可免费下载 |
| 备注 | 最大的单导联可穿戴 ECG beat 标注数据集；胸贴非上臂，但单导联特性可参考 |

### 4. CPSC 2020（PVC/SPB 检测）

| 字段 | 内容 |
|------|------|
| URL | http://www.icbeb.org/CPSC2020.html |
| 导联 | 单导联可穿戴 Holter（具体贴附位置未详细说明） |
| 标注 | ✅ QRS beat 标注 + PVC/SPB 位置 |
| 规模 | 10 条 × ~24 小时，400Hz，MATLAB 格式 |
| 授权 | 学术使用，非商业 |

### 5. BUT QDB（Brno University）

| 字段 | 内容 |
|------|------|
| URL | https://physionet.org/content/butqdb/1.0.0/ |
| 导联 | 单导联，Bittium Faros 180 可穿戴设备，自由生活环境 |
| 标注 | ⚠️ 信号质量分级标注（Class 1/2/3），**无 R-peak 时间戳** |
| 规模 | 15 名受试者，18 条 × >24 小时，1000Hz + 3 轴加速度计 |
| 授权 | CC BY 4.0 |
| 备注 | 适合信号质量门控研究，不直接用于 QRS 检测评测 |

---

## Tier 3：手持/远程 ECG，部分标注

### 6. PhysioNet/CinC Challenge 2017（AliveCor）

| 字段 | 内容 |
|------|------|
| URL | https://physionet.org/content/challenge-2017/1.0.0/ |
| 导联 | 单导联 Lead-I，**拇指到拇指**干电极（KardiaMobile） |
| 标注 | ⚠️ 节律级别（Normal/AF/Other/Noise），**无 R-peak 位置** |
| 规模 | 8,528 条 × 9–60 秒，300Hz |
| 授权 | 训练集开放 |
| 备注 | 非胸部，手持，但没有 beat 级标注 |

### 7. SAFER ECG Database

| 字段 | 内容 |
|------|------|
| 导联 | 单导联 Lead-I，拇指到拇指（Zenicor EKG-2） |
| 标注 | ✅ 人工标注 R-peak |
| 规模 | 479 条 × 30 秒 |
| 授权 | **受限**，需发邮件申请（SAFER@medschl.cam.ac.uk） |

---

## 上臂 ECG 专用数据集：现状

| 来源 | 内容 | 是否公开 |
|------|------|---------|
| ESC 2024 大会摘要 | 近端上臂干电极，QRS 检测 98–99%，12 名受试者 | **未公开** |
| MDPI Electronics 论文 | 上臂无线 ECG 传感器系统 | **未公开** |
| Apple Watch HealthKit | 腕部单导联 Lead-I | **无公开数据集** |

**结论：上臂 ECG 带标注数据集目前不存在开源版本。**

---

## 对本项目的建议

```
近期可用（无需等待自采数据）：
  1. SHDB-AF：下载即用，非标准 Holter 导联，24h，beat 标注
  2. TELE ECG：手持非标准导联，有真实噪声，适合噪声鲁棒性评测
  3. LTDB + NSTDB ma 噪声叠加：SNR 扫描，测 F1 曲线

中期（项目深化）：
  自采 10–30 分钟上臂 ECG + 人工标注
  → 唯一真正验证上臂场景的方案
```
