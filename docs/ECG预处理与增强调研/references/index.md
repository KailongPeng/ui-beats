---
title: References · 索引
tags: [index, MOC, PN-QRS, references]
date: 2026-04-08
type: MOC
up: "[[../index]]"
---

# References 子库索引

> ← [[../index|返回上级 MOC]]

---

## 文档列表

### Markdown 笔记

- [[PN_QRS_解读]] — PN-QRS 完整解读（3315行）
  - 原理、MC Dropout、U_E/U_A 不确定性
  - 复现：MIT-BIH F1=99.69%，LTDB F1=99.58%
  - Section 25–26：非标准导联数据集调研与扩展 Benchmark
- [[wearable_ECG_datasets]] — 可穿戴/非标准导联开源数据集汇总
- [[PN_QRS_on_custom_ECG]] ⭐ — **自采 Excel ECG 数据应用 PN-QRS 指南**
- [[PN_QRS_to_ECGFounder_pipeline]] — PN-QRS 结果如何为 ECGFounder 提供训练数据（质量过滤 + 滑动窗口切片）

### HTML 报告（点击在浏览器中打开）

- [PN_QRS_复现报告.html](PN_QRS_复现报告.html) — 中文复现报告，含 MIT-BIH / INCART / LTDB 详细结果
- [PN_QRS_report_en.html](PN_QRS_report_en.html) — 英文版复现报告

### PDF 原文

- [PN_QRS.pdf](PN_QRS.pdf) — PN-QRS 论文原文

---

## 知识图谱（本库内链接关系）

```
PN_QRS_解读
  ├── Section 17   → 复现细节
  ├── Section 21   → 多导联 vs 单导联分析
  ├── Section 25   → 非标准导联数据集
  │     └── wearable_ECG_datasets  (详细表格)
  ├── Section 26   → 扩展 Benchmark（EDB / LTSTDB / CUDB）
  └── PN_QRS_on_custom_ECG  → 自采数据应用
        └── PN_QRS_to_ECGFounder_pipeline  → 如何为 ECGFounder 提供数据

PN_QRS_复现报告.html
  └── PN_QRS_report_en.html  (英文镜像)
```

---

## 快速导航

| 我想要… | 去哪里 |
|--------|--------|
| 理解 U_E / U_A 含义 | [[PN_QRS_解读#Section 5]] |
| 查看 MIT-BIH 复现结果 | [[PN_QRS_解读#17.6]] 或 HTML 报告 |
| 找非标准导联数据集 | [[wearable_ECG_datasets]] + [[PN_QRS_解读#Section 25]] |
| 把自采 Excel 数据跑出 R-peak | [[PN_QRS_on_custom_ECG]] |
| 理解为何单导联模型可用于多导联 | [[PN_QRS_解读#Section 21]] |
| 把自采数据接入 ECGFounder 训练 | [[PN_QRS_to_ECGFounder_pipeline]] |
