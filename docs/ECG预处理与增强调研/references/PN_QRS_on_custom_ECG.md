---
title: PN-QRS 应用于自采 Excel ECG 数据（单/多导联通用）
tags: [PN-QRS, ECG, QRS-detection, Excel, multi-lead, upper-arm, guide]
date: 2026-04-08
type: guide
up: "[[index]]"
related: "[[PN_QRS_解读]]"
---

# PN-QRS 应用于自采 Excel ECG 数据

> ← [[index|返回 References 索引]]  |  参考 [[PN_QRS_解读]]

---

## 数据格式说明

| 格式 | 列结构 | 适用场景 |
|------|--------|---------|
| **旧格式** | `timestamp` + `CH20` | 早期单导联采集，上臂导联 |
| **新格式** | `timestamp` + `CH1…CH8` + `CH20` | 多导联采集，CH1–8 为标准导联，CH20 为上臂导联 |

脚本**自动检测**列结构，两种格式无需分别处理。

**信号幅度**：设备输出为 ADC 数值（几百到几千），正负方向因导联不同而异。`preprocess_ecg` 内部会做 z-score 归一化，幅度不影响结果。

---

## 核心概念

```
CH1–CH8（多导联）          CH20（单导联）
──────────────────         ──────────────────
每条导联独立推理            直接单导联推理
       ↓                          ↓
 各导联输出 logits          logits → R-peak
 + U_E 不确定性
       ↓
 逐帧选 min(U_E) 导联
 （论文 Algorithm 1）
       ↓
  融合 R-peak 输出
```

CH20 上臂导联：模型对此类非标准形态的 U_E 可能偏高，结果仍可用，
但若漏检较多可尝试信号取反（`-signal`），见第 6 节。

---

## 1. 环境准备

```bash
cd /home/kailong/ECG/ECG/ECGFounder/PN-QRS
pip install pandas openpyxl scipy numpy torch
```

模型权重：
```
experiments/logs_real/zy2lki18/models/best_model.pt
```

---

## 2. 数据读取

```python
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ECGRecord:
    fs: int                           # 采样率 Hz
    ch_standard: Optional[np.ndarray] # (N, n_leads)，CH1–CH8，None 如不存在
    ch_upper_arm: Optional[np.ndarray]# (N,)，CH20，None 如不存在
    ch_names_standard: list = field(default_factory=list)
    source_file: str = ""


def infer_fs(ts_series: pd.Series) -> int:
    """从 timestamp 列推断采样率，取多个差值的中位数以应对抖动。"""
    if pd.api.types.is_numeric_dtype(ts_series):
        diffs = np.diff(ts_series.values[:50].astype(float))
        median_diff = float(np.median(diffs[diffs > 0]))
        fs = round(1000 / median_diff) if median_diff > 1 else round(1 / median_diff)
    else:
        ts_dt = pd.to_datetime(ts_series)
        diffs = [(ts_dt.iloc[i+1] - ts_dt.iloc[i]).total_seconds()
                 for i in range(min(50, len(ts_dt)-1))]
        median_diff = float(np.median([d for d in diffs if d > 0]))
        fs = round(1 / median_diff)
    return fs


def load_excel_ecg(path: str) -> ECGRecord:
    """
    读取 Excel ECG 文件，自动识别旧格式（仅 CH20）和新格式（CH1–CH8 + CH20）。

    Excel 列结构（顺序不限）：
      timestamp — 时间戳
      CH1–CH8   — 标准/胸前导联（可选）
      CH20      — 上臂导联（可选）
    """
    df = pd.read_excel(path)
    cols = df.columns.tolist()

    # ── 找 timestamp 列 ──────────────────────────────────────
    ts_col = next(
        (c for c in cols if str(c).lower() in ("timestamp", "time", "t")),
        cols[0]   # 默认取第一列
    )
    fs = infer_fs(df[ts_col])

    # ── 找 CH1–CH8 列 ────────────────────────────────────────
    std_cols = [c for c in cols
                if str(c).upper() in {f"CH{i}" for i in range(1, 9)}]
    std_cols_sorted = sorted(std_cols, key=lambda c: int(str(c)[2:]))

    # ── 找 CH20 列 ───────────────────────────────────────────
    upper_col = next(
        (c for c in cols if str(c).upper() == "CH20"), None
    )

    ch_standard = (
        df[std_cols_sorted].values.astype(np.float32)
        if std_cols_sorted else None
    )
    ch_upper_arm = (
        df[upper_col].values.astype(np.float32)
        if upper_col is not None else None
    )

    print(f"[加载] {path}")
    print(f"  采样率: {fs} Hz | 时长: {len(df)/fs:.1f}s | 样本数: {len(df)}")
    if ch_standard is not None:
        print(f"  标准导联: {std_cols_sorted} ({len(std_cols_sorted)} 导联)")
    if ch_upper_arm is not None:
        print(f"  上臂导联: CH20")

    return ECGRecord(
        fs=fs,
        ch_standard=ch_standard,
        ch_upper_arm=ch_upper_arm,
        ch_names_standard=std_cols_sorted,
        source_file=str(path),
    )
```

---

## 3. 模型加载

```python
import sys, torch
from pathlib import Path

PNQRS_ROOT = Path("/home/kailong/ECG/ECG/ECGFounder/PN-QRS")
CKPT_PATH  = PNQRS_ROOT / "experiments/logs_real/zy2lki18/models/best_model.pt"
sys.path.insert(0, str(PNQRS_ROOT))

from dataset.dataset   import preprocess_ecg
from models.qrs_model  import QRSModel
from models.multi_head import encoder4qrs, decoder4qrs, phi_qrs
from utils.qrs_post_process import correct, uncertain_est


def load_model(device) -> QRSModel:
    ckpt  = torch.load(str(CKPT_PATH), map_location="cpu")
    model = QRSModel(encoder4qrs(), decoder4qrs(), phi_qrs()).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model.eval()
```

---

## 4. 推理核心

### 4.1 单导联推理（CH20 上臂）

```python
FS_MODEL    = 50
WIN_SEC     = 10
OVERLAP_SEC = 2
TOL_SEC     = 0.150


def infer_single_lead(model, window_1d: np.ndarray, fs: int, device) -> tuple:
    """
    单条导联推理。
    返回: (r_peaks_in_orig_fs, ue_score)
      r_peaks_in_orig_fs — R-peak 样本索引（原始采样率）
      ue_score           — 该窗口平均 U_E（越高越不确定）
    """
    processed = preprocess_ecg(window_1d, fs=fs)          # (1, 500) or (500,)
    if processed.ndim == 1:
        processed = processed[np.newaxis, :]
    elif processed.shape[0] > processed.shape[1]:
        processed = processed.T

    sig_t = torch.from_numpy(processed).unsqueeze(0).to(device)  # (1,1,500)
    with torch.no_grad():
        logits = model(sig_t, return_projection=True)
        logits = logits.squeeze(-1).squeeze(0).cpu().numpy()      # (3,500)

    uc      = uncertain_est(logits)                               # (500,)
    r_50hz  = correct(logits[0], uc)
    ue_mean = float(np.mean(uc))

    r_orig  = np.round(np.array(r_50hz) * (fs / FS_MODEL)).astype(int)
    return r_orig, ue_mean
```

### 4.2 多导联推理（CH1–CH8，逐帧 min-U_E 融合）

```python
def infer_multi_lead(model, windows: list, fs: int, device) -> tuple:
    """
    多导联推理，实现论文 Algorithm 1 的逐帧 U_E 最小化导联选择。

    windows: list of 1D arrays, 每条导联一个 (WIN_SAMP,) 数组
    返回: (r_peaks_in_orig_fs, best_lead_per_frame, mean_ue_per_lead)
    """
    all_logits = []
    for window_1d in windows:
        processed = preprocess_ecg(window_1d, fs=fs)
        if processed.ndim == 1:
            processed = processed[np.newaxis, :]
        elif processed.shape[0] > processed.shape[1]:
            processed = processed.T

        sig_t = torch.from_numpy(processed).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(sig_t, return_projection=True)
            logits = logits.squeeze(-1).squeeze(0).cpu().numpy()  # (3, 500)
        all_logits.append(logits)

    # 计算每条导联的 U_E 序列
    uc_list = [uncertain_est(lg) for lg in all_logits]   # n_leads × (500,)
    lo_list = [lg[0] for lg in all_logits]               # n_leads × (500,)，QRS logits

    # 逐帧选 U_E 最小的导联
    uc_mat   = np.stack(uc_list, axis=1)                 # (500, n_leads)
    lo_mat   = np.stack(lo_list, axis=1)                 # (500, n_leads)

    best_lead  = np.argmin(uc_mat, axis=1)               # (500,)，每帧最优导联
    T          = lo_mat.shape[0]
    logits_o   = lo_mat[np.arange(T), best_lead]         # (500,) 融合后的 QRS 曲线
    uc_fused   = uc_mat[np.arange(T), best_lead]         # (500,) 融合后的 U_E

    r_50hz = correct(logits_o, uc_fused)
    r_orig = np.round(np.array(r_50hz) * (fs / FS_MODEL)).astype(int)

    mean_ue_per_lead = [float(np.mean(uc)) for uc in uc_list]
    return r_orig, best_lead, mean_ue_per_lead
```

### 4.3 滑窗 NMS 去重

```python
def nms(preds: np.ndarray, tol: int) -> np.ndarray:
    """合并 tol 范围内的重复预测，取每簇中位数。"""
    if len(preds) == 0:
        return np.array([], dtype=int)
    preds = np.sort(preds)
    clusters, cur = [], [preds[0]]
    for p in preds[1:]:
        if p - cur[-1] <= tol:
            cur.append(p)
        else:
            clusters.append(cur)
            cur = [p]
    clusters.append(cur)
    return np.array([int(np.median(c)) for c in clusters])


def sliding_window_detect(
    model,
    signal_or_signals,   # 单导联: (N,)；多导联: (N, n_leads)
    fs: int,
    device,
    multi_lead: bool = False,
) -> dict:
    """
    滑窗检测主函数。
    返回 dict，键：'r_peaks'（样本索引），'r_times'（秒），
                   'mean_hr'，以及多导联时额外的 'lead_usage_pct'
    """
    win_samp  = int(WIN_SEC * fs)
    step_samp = int((WIN_SEC - OVERLAP_SEC) * fs)
    tol_samp  = int(TOL_SEC * fs)

    all_r   = []
    # 多导联统计：每条导联被选中的帧数
    n_leads = signal_or_signals.shape[1] if multi_lead else 1
    lead_frame_count = np.zeros(n_leads, dtype=int)

    pos = 0
    while pos < (len(signal_or_signals) if not multi_lead else signal_or_signals.shape[0]):
        sig_len = len(signal_or_signals) if not multi_lead else signal_or_signals.shape[0]

        if multi_lead:
            chunk = signal_or_signals[pos : pos + win_samp]           # (W, n_leads)
            windows = []
            for i in range(n_leads):
                w = chunk[:, i]
                if len(w) < win_samp:
                    w = np.pad(w, (0, win_samp - len(w)))
                windows.append(w)
            r_rel, best_lead, _ = infer_multi_lead(model, windows, fs, device)
            # 统计每条导联被选中的帧数（只取有效段）
            valid_frames = min(len(best_lead), int((min(pos + win_samp, sig_len) - pos) * FS_MODEL / fs))
            np.add.at(lead_frame_count, best_lead[:valid_frames], 1)
        else:
            w = signal_or_signals[pos : pos + win_samp]
            if len(w) < win_samp:
                w = np.pad(w, (0, win_samp - len(w)))
            r_rel, _ = infer_single_lead(model, w, fs, device)

        r_abs = r_rel + pos
        r_abs = r_abs[r_abs < sig_len]
        all_r.append(r_abs)
        pos += step_samp

    r_peaks = nms(np.concatenate(all_r) if all_r else np.array([]), tol_samp)
    r_times = r_peaks / fs
    mean_hr = float(60 / np.mean(np.diff(r_peaks) / fs)) if len(r_peaks) > 1 else float("nan")

    result = {"r_peaks": r_peaks, "r_times": r_times, "mean_hr": mean_hr}
    if multi_lead:
        total = lead_frame_count.sum()
        result["lead_usage_pct"] = (lead_frame_count / max(total, 1) * 100).tolist()
    return result
```

---

## 5. 批量处理主流程

```python
import glob, os, json
from dataclasses import asdict

def process_file(path: str, model, device) -> dict:
    """处理单个 Excel 文件，返回结构化结果。"""
    rec = load_excel_ecg(path)
    output = {"file": os.path.basename(path), "fs": rec.fs}

    # ── CH1–CH8：多导联 ─────────────────────────────────────
    if rec.ch_standard is not None and rec.ch_standard.shape[1] > 0:
        res = sliding_window_detect(
            model, rec.ch_standard, rec.fs, device, multi_lead=True
        )
        output["standard_leads"] = {
            "n_beats":        int(len(res["r_peaks"])),
            "mean_hr_bpm":    round(res["mean_hr"], 1),
            "r_peaks":        res["r_peaks"].tolist(),
            "r_times_s":      res["r_times"].tolist(),
            "lead_usage_pct": {
                rec.ch_names_standard[i]: round(pct, 1)
                for i, pct in enumerate(res["lead_usage_pct"])
            },
        }
        print(f"  [CH1–8] beats={len(res['r_peaks'])}  HR≈{res['mean_hr']:.0f}bpm")
        print(f"    导联使用率: " +
              " | ".join(f"{k}:{v:.0f}%" for k, v in output["standard_leads"]["lead_usage_pct"].items()))

    # ── CH20：单导联（上臂）────────────────────────────────────
    if rec.ch_upper_arm is not None:
        res = sliding_window_detect(
            model, rec.ch_upper_arm, rec.fs, device, multi_lead=False
        )
        output["upper_arm_ch20"] = {
            "n_beats":     int(len(res["r_peaks"])),
            "mean_hr_bpm": round(res["mean_hr"], 1),
            "r_peaks":     res["r_peaks"].tolist(),
            "r_times_s":   res["r_times"].tolist(),
        }
        print(f"  [CH20]  beats={len(res['r_peaks'])}  HR≈{res['mean_hr']:.0f}bpm")

    return output


def batch_process(data_dir: str, gpu: int = 0):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model  = load_model(device)
    print(f"模型加载完毕，设备: {device}\n")

    files = sorted(glob.glob(os.path.join(data_dir, "*.xlsx")))
    print(f"找到 {len(files)} 个文件\n{'─'*50}")

    all_results = []
    for f in files:
        print(f"\n处理: {os.path.basename(f)}")
        result = process_file(f, model, device)
        all_results.append(result)

        # 保存 R-peak CSV（分标准导联和上臂导联）
        base = f.replace(".xlsx", "")
        if "standard_leads" in result:
            pd.DataFrame({
                "sample_index": result["standard_leads"]["r_peaks"],
                "time_seconds": result["standard_leads"]["r_times_s"],
            }).to_csv(base + "_CH1-8_rpeaks.csv", index=False)

        if "upper_arm_ch20" in result:
            pd.DataFrame({
                "sample_index": result["upper_arm_ch20"]["r_peaks"],
                "time_seconds": result["upper_arm_ch20"]["r_times_s"],
            }).to_csv(base + "_CH20_rpeaks.csv", index=False)

    # 汇总 JSON
    summary_path = os.path.join(data_dir, "rpeaks_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f_out:
        json.dump(all_results, f_out, ensure_ascii=False, indent=2)
    print(f"\n{'─'*50}\n汇总已保存: {summary_path}")
    return all_results
```

**运行**：
```python
results = batch_process("/path/to/your/excel/files", gpu=0)
```

---

## 6. 常见问题

### Q1：CH20 漏检很多 R-peak

上臂导联 QRS 形态非标准，可能倒置或幅度极小。

```python
# 在 load_excel_ecg 末尾，对 CH20 做信号质量预检
if ch_upper_arm is not None:
    # 正向检测
    r_pos, ue_pos = infer_single_lead(model, ch_upper_arm[:int(30*fs)], fs, device)
    # 反向检测
    r_neg, ue_neg = infer_single_lead(model, -ch_upper_arm[:int(30*fs)], fs, device)
    # 以 30s 片段的检测数量和 U_E 综合判断
    if len(r_neg) > len(r_pos) * 1.2 or (len(r_neg) >= len(r_pos) and ue_neg < ue_pos):
        print("  [CH20] 检测到信号倒置，自动取反")
        ch_upper_arm = -ch_upper_arm
```

### Q2：采样率推断错误

```python
# 在 load_excel_ecg 里强制指定
FS_OVERRIDE = 250  # 查设备手册
```

### Q3：CH1–8 中某些导联始终被忽略（lead_usage_pct ≈ 0%）

说明该导联 U_E 始终很高——可能是：
- 导联脱落 / 接触不良
- 信号极性严重不匹配

U_E 机制会自动排除该导联，不影响最终结果。这个统计数据可以用来判断哪个导联质量差。

### Q4：信号是 ADC 计数而非 mV，需要转换吗？

不需要。`preprocess_ecg` 内部调用 `preprocessing.scale()`（z-score），
将任何幅度范围的信号归一化到均值为 0、标准差为 1，模型只看相对形态。

---

## 7. 输出结构

每个文件产生最多两个 CSV + 一个汇总 JSON：

```
data_dir/
├── recording1_CH1-8_rpeaks.csv   ← 多导联融合结果
├── recording1_CH20_rpeaks.csv    ← 上臂导联结果
├── recording2_CH1-8_rpeaks.csv
├── recording2_CH20_rpeaks.csv
├── ...
└── rpeaks_summary.json           ← 所有文件的汇总（beats数、心率、导联使用率）
```

`rpeaks_summary.json` 结构示例：
```json
[
  {
    "file": "recording1.xlsx",
    "fs": 500,
    "standard_leads": {
      "n_beats": 1243,
      "mean_hr_bpm": 72.4,
      "r_peaks": [...],
      "r_times_s": [...],
      "lead_usage_pct": {
        "CH1": 12.3, "CH2": 45.1, "CH3": 8.2,
        "CH4": 3.0,  "CH5": 18.7, "CH6": 6.1,
        "CH7": 4.9,  "CH8": 1.7
      }
    },
    "upper_arm_ch20": {
      "n_beats": 1238,
      "mean_hr_bpm": 72.1,
      "r_peaks": [...],
      "r_times_s": [...]
    }
  }
]
```

---

## 8. 验证可视化

```python
import matplotlib.pyplot as plt

def visualize_rpeaks(excel_path: str, csv_path: str, channel: str = "CH20",
                      fs: int = 500, duration_s: int = 10):
    df     = pd.read_excel(excel_path)
    signal = df[channel].values
    rp     = pd.read_csv(csv_path)

    n    = duration_s * fs
    mask = rp["sample_index"] < n

    fig, ax = plt.subplots(figsize=(15, 3))
    ax.plot(np.arange(n) / fs, signal[:n], lw=0.8, color="steelblue", label=channel)
    ax.scatter(
        rp.loc[mask, "sample_index"] / fs,
        signal[rp.loc[mask, "sample_index"].values],
        color="red", s=40, zorder=5, label="R-peak"
    )
    ax.set_xlabel("Time (s)")
    ax.set_title(f"PN-QRS R-peak 检测 — {channel}（前{duration_s}秒）")
    ax.legend()
    plt.tight_layout()
    plt.savefig(excel_path.replace(".xlsx", f"_{channel}_preview.png"), dpi=150)
    plt.show()

# 用法：
visualize_rpeaks("recording1.xlsx", "recording1_CH20_rpeaks.csv", channel="CH20", fs=500)
```

---

## 9. 下一步

| 任务 | 说明 |
|------|------|
| **R-peak 分割** | 以 R-peak 为中心切单心跳（前后各 400ms）→ 送 ECGFounder 分类 |
| **导联质量评估** | 用 `lead_usage_pct` 排查接触不良导联 |
| **上臂 vs 标准导联对比** | 对比 CH20 和 CH1–8 的检测结果一致性，量化上臂导联的误差率 |
| **SQI 筛选** | 对 U_E 超阈值的窗口标记为低质量，跳过下游分析 |

参见 [[PN_QRS_解读]] Section 18（ECGFounder 接入管线）和 Section 22（SQI 实验结果）。
