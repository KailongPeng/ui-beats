---
title: PN-QRS 论文不确定性机制精读：U_E、U_A 与代码实现的对应关系
tags: [PN-QRS, uncertainty, U_E, U_A, SQA, analysis]
date: 2026-04-09
type: analysis
up: "[[index]]"
related: "[[PN_QRS_解读]]", "[[PN_QRS_to_ECGFounder_pipeline]]"
---

# PN-QRS 论文不确定性机制精读

> 核心问题：论文里的数据质量评估究竟是"求 U_E 和 U_A 的均值"吗？

**结论：不是。** U_E 和 U_A 在论文中粒度不同、用途不同，从未合并为一个均值指标。
代码里 `mean(eu + au)` 是工程改写，不是论文定义的直接实现。

---

## 一、论文的不确定性定义

### U_E（认知不确定性，Epistemic Uncertainty）

定义来自公式 (5)，是对类偏置变换前后预测分布的 **KL 散度，对全段所有帧取均值**：

$$
\mathcal{U}_\mathcal{E} = \frac{1}{T}\sum_{\tau=1}^{T} D_\text{KL}(P(\mathbf{z}_\tau \mid \phi(\mathbf{z})) \| P(\mathbf{z}_\tau \mid \mathbf{z}))
\approx \frac{1}{T}\sum_{\tau=1}^{T} \left[g_\omega(\mathbf{z}^+_\tau)\log\frac{g_\omega(\mathbf{z}^+_\tau)}{g_\omega(\mathbf{z}_\tau)} + g_\omega(\mathbf{z}^-_\tau)\log\frac{g_\omega(\mathbf{z}^-_\tau)}{g_\omega(\mathbf{z}_\tau)}\right]
$$

关键特征：
- **粒度：窗口级标量**（对 T=625 帧求平均，最终是一个数）
- **物理含义**：整段 ECG 窗口与训练分布的距离——高 U_E 说明整段信号是 OoD 噪声（非 ECG）
- **论文阈值**：α = 0.1

### U_A（偶然不确定性，Aleatoric Uncertainty）

定义来自公式 (6)，是**每帧预测的熵**（相对正负变换，逐帧）：

$$
\mathcal{U}_\mathcal{A} = -\left(g_\omega(\mathbf{z}^+_\tau)\log g_\omega(\mathbf{z}^+_\tau) + (1 - g_\omega(\mathbf{z}^-_\tau))\log(1-g_\omega(\mathbf{z}^-_\tau))\right)
$$

关键特征：
- **粒度：帧级序列**（T 个值，与输入信号时间轴对齐）
- **物理含义**：每一帧的预测模糊程度——高 U_A 帧说明该处 QRS 形态难以判断（可疑心拍）
- **论文阈值**：β = 0.12

---

## 二、论文 Algorithm 1 的完整流程

```
输入: 10 秒 ECG, L 路导联

Step 1: z ← f_θ(x*),  z^{+/-} ← φ^{+/-}(z)
Step 2: ŷ ← g_ω(z_τ),  ŷ^{+/-} ← g_ω(z^{+/-}_τ)    [逐帧 logits]

Step 3: U_E ← (1/T) Σ_τ [ ŷ⁺ log(ŷ⁺/ŷ) + ŷ⁻ log(ŷ⁻/ŷ) ]   ← 窗口级标量
Step 4: U_A ← ŷ log ŷ⁺ + (1−ŷ) log(1−ŷ⁻)                    ← 帧级序列 [T]

Step 5: 若 U_E > α → 丢弃该导联（整段是 OoD 噪声）

Step 6: U ← U_E + U_A     ← 只在多导联选择时用到（下一步）
Step 7-9: for τ = 1 to T:
            选 U^l 最小的导联 l，取该导联的 ŷ⁺_τ 和 U_A_τ
            [多导联逐帧选最优导联]

Step 10: 根据 ŷ⁺ 确定 R-peak 位置 {R_k}

Step 11-14: for 每个 R-peak R_k:
              U_A[R_k] ← (1/7) Σ_{τ ∈ [R_k-3, R_k+4]} U_A_τ   ← 7帧窗口均值(≈112ms)
              若 U_A[R_k] > β → 标记为可疑心拍
```

### 两个指标的用途总结

| 指标 | 粒度 | 阈值 | 用途 |
|------|------|------|------|
| U_E | 窗口级标量 | α = 0.1 | **Step 5**：判断整段是否为 non-ECG（OoD 筛查） |
| U_A | 帧级序列 T | β = 0.12 | **Step 13**：判断每个心拍是否为可疑伪迹 |
| U_E + U_A | 帧级序列 T | — | **Step 6-9**：仅用于多导联逐帧选最优导联，不作为质量分数输出 |

> **关键点**：Step 6 的 `U = U_E + U_A` 是把窗口级标量 U_E 广播到每帧，与帧级 U_A 相加，目的是让多导联选择时把"整段可疑"的导联降权。**不是用来作为窗口质量分数的。**

---

## 三、代码 `uncertain_est()` 的实际实现

```python
def uncertain_est(logits):
    # logits: (n_classes=3, T) — 正常、QRS、噪声三类
    au = en_est(logits)     # 逐帧预测熵，对应 U_A（帧级，连续值）
    eu = mi_est(logits)     # 逐帧 KL 散度，对应 U_E 的逐帧项（未对T求均值）
    eu[eu > 0.12] = 10      # ← 二值化：复用 β=0.12，但作用在 eu 而非 au
    return eu + au           # 返回帧级组合值 [T]
```

调用处：
```python
uc = uncertain_est(logits)
mean_uc = uc.mean()          # 对所有帧取均值 → 窗口质量分数
```

### 代码与论文的三处差异

| 差异 | 论文 | 代码 |
|------|------|------|
| **U_E 粒度** | 对 T 帧求均值后得到标量，然后与 α=0.1 比较 | `mi_est()` 返回逐帧 KL（求均值之前），保留为 [T] 序列 |
| **二值化** | 无此操作；U_E 保持连续值，直接和 α 比较 | `eu[eu > 0.12] = 10`，把噪声帧打成 10，强制双峰分布 |
| **阈值复用** | α=0.1 用于 U_E 的窗口筛查，β=0.12 用于 U_A 的逐拍判断 | 将 β=0.12 用于 eu（认知不确定性），非论文设计 |

---

## 四、`mean_uc` 的实质

经过 `eu[eu > 0.12] = 10` 之后，`mean_uc = mean(eu + au)` 的含义变成：

$$
\text{mean\_uc} \approx \frac{1}{T}\sum_\tau \left[ 10 \cdot \mathbf{1}[eu_\tau > 0.12] + au_\tau \right]
$$

这不是论文的连续不确定性均值，而是**噪声帧占比的加权计数**：

- 干净窗口（无噪声帧）：mean_uc ≈ 0.1–0.5（au 贡献）
- 少量噪声帧（e.g. 5 帧 / 625 帧）：mean_uc ≈ 10 × (5/625) + 0.3 ≈ 0.38（不明显）
- 较多噪声帧（e.g. 50 帧 / 625 帧）：mean_uc ≈ 10 × (50/625) + 0.3 ≈ 1.1（开始超阈值）
- 电极脱落（大部分帧 eu > 0.12）：mean_uc → ~10

正是因为这种二值化放大效应，mean_uc 在干净信号和噪声信号之间形成**悬殊的双峰分布**，Otsu 阈值才可信。

---

## 五、为什么代码不按论文来？

论文的 U_E（窗口级，二分类：是/不是 ECG）在实际工程中有局限：

1. **粒度粗**：U_E 只能给整个 10s 窗口打一个质量标签，无法区分"大部分干净但有一小段噪声"的窗口
2. **阈值 α=0.1 为 iD ECG 标定**：在可穿戴设备的上臂导联上，干净信号的 U_E 本身可能就超过 0.1，直接用论文阈值会过度丢弃
3. **U_A 逐拍判断**：只适用于已检出 R-peak 之后，不适合在 R-peak 检测之前做前置质量筛查

`uncertain_est()` 的二值化改写本质上是把**逐帧 U_E 的相对大小**用一个硬阈值转换成"噪声/干净"的二值信号，然后通过均值把帧级信息汇聚成窗口级分数。功能等价（噪声越多分数越高），但实现路径与论文定义完全不同。

---

## 六、结论

| 问题 | 回答 |
|------|------|
| 论文有没有计算 mean(U_E + U_A)？ | **没有**。论文 U_E 是窗口标量，U_A 是帧级序列，两者用途独立 |
| Step 6 的 U = U_E + U_A 是质量分数吗？ | 不是。它只用于多导联逐帧选最优导联，不输出为质量指标 |
| 代码的 mean_uc 是什么？ | 噪声帧占比的加权计数，是代码层面的工程适应，非论文定义 |
| 用于窗口级质量筛查是否合理？ | **合理**。虽然与论文定义不同，但对可穿戴数据的噪声分布产生了理想的双峰结构，Otsu 阈值可信 |
| `eu[eu > 0.12] = 10` 中 0.12 从哪来？ | 来自论文 β=0.12（原本用于 U_A 逐拍判断），被工程上挪用于 eu 的二值化 |

---

## 相关文档

- [[PN_QRS_on_custom_ECG]] — extract_quality_segments.py 使用方法，含 mean_uc 说明
- [[PN_QRS_to_ECGFounder_pipeline]] — 质量分数的含义（已按本文修正）
- [[PN_QRS_解读]] — 完整代码解读（Section 5: U_E/U_A，Section 21: 多导联融合）
