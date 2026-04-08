---
title: ECG 预处理与增强调研
tags: [index, MOC, ECG, preprocessing, augmentation]
date: 2026-04-08
type: MOC
---

# ECG 预处理与增强调研 · 知识库

> **Map of Content** — 本目录所有文档的导航中心

---

## 核心文档

| 文档 | 行数 | 核心内容 |
|------|------|---------|
| [[ECG预处理技术调研]] | 661行 | 滤波、降噪、重采样、标准化方法对比 |
| [[ECG数据增强技术调研]] | 628行 | Mixup、CutMix、时频增强、Mamba 兼容性分析 |
| [[ECG创新方向分析]] | 579行 | 技术路线评估、创新点定位 |
| [[ECG预处理与增强调研汇总]] | 92行 | 三份调研交叉汇总、行动建议 |

---

## references 子库

> [[references/index|→ 进入 references 索引]]

| 文档 | 类型 | 核心内容 |
|------|------|---------|
| [[references/PN_QRS_解读]] | 笔记 (3315行) | PN-QRS 完整解读：原理、复现、评测、泛化分析 |
| [[references/wearable_ECG_datasets]] | 调研 | 非标准/可穿戴导联 ECG 开源数据集汇总 |
| [[references/PN_QRS_on_custom_ECG]] | **指南** | ⭐ 如何对自采 Excel ECG 数据应用 PN-QRS |
| [PN_QRS_复现报告.html](references/PN_QRS_复现报告.html) | HTML报告 | MIT-BIH / INCART / LTDB 复现结果可视化 |
| [PN_QRS_report_en.html](references/PN_QRS_report_en.html) | HTML报告 | 英文版复现报告 |
| [PN_QRS.pdf](references/PN_QRS.pdf) | 论文原文 | PN-QRS 原始论文 PDF |

---

## 关键结论速查

### 预处理一致性问题
> 预训练（MIMIC-IV）用了带通滤波，微调（PTB-XL）只用了 z-score → **分布偏移，建议统一**

### Mamba + 数据增强
> 标准 Mixup / CutMix 对 Mamba **有害**，唯一验证有效的是 **Non-Uniform-Mix**（AUPRC +2.8%）

### PN-QRS 复现结果
| 数据集 | Se | P+ | F1 | 与论文差距 |
|--------|----|----|-----|---------|
| MIT-BIH (v2) | 99.52% | 99.87% | 99.69% | −0.26pp |
| LTDB (自测) | 99.76% | 99.40% | 99.58% | 论文未测 |

### 自采数据应用
> 有 8 个 Excel 文件（timestamp + CH20），见 [[references/PN_QRS_on_custom_ECG]]

---

## 相关链接

- PN-QRS 代码：`/home/kailong/ECG/ECG/ECGFounder/PN-QRS/`
- 模型权重：`PN-QRS/experiments/logs_real/zy2lki18/models/best_model.pt`
- 数据目录：`/home/kailong/ECG/ECG/data/PN-QRS/`
