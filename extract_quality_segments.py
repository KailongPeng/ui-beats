#!/usr/bin/env python3
"""
extract_quality_segments.py -- 基于 PN-QRS 不确定性评估 CH20 信号质量，提取高质量 10 秒片段

原理：
  PN-QRS 在对每个 10 秒窗口推理时，模型内部同时计算两种不确定性：
    U_E (认知不确定性 Epistemic) = mi_est(logits)  模型对预测的自信程度
    U_A (偶然不确定性 Aleatoric) = en_est(logits)  信号本身的噪声程度
  mean(U_E + U_A) 低 → 信号干净 → 高质量窗口
  mean(U_E + U_A) 高 → 信号嘈杂 → 低质量窗口

  此方法只需要单路 CH20 信号，无需 12 导联参考。

用法：
  python extract_quality_segments.py --csv /path/to/data.csv --fs 1000
  python extract_quality_segments.py --csv /path/to/data.csv --fs 1000 --uc_thr 0.5
  python extract_quality_segments.py --csv /path/to/data.csv --fs 1000 --out_dir /path/to/output
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch

PNQRS_ROOT = Path(__file__).parent
CKPT_PATH  = PNQRS_ROOT / "experiments/logs_real/zy2lki18/models/best_model.pt"
sys.path.insert(0, str(PNQRS_ROOT))

from dataset.dataset import preprocess_ecg
from models.multi_head import decoder4qrs, encoder4qrs, phi_qrs
from models.qrs_model import QRSModel
from utils.qrs_post_process import correct, uncertain_est

FS_MODEL      = 50
WIN_SEC       = 10
STEP_SEC      = 8          # 滑动步长 (WIN_SEC - OVERLAP_SEC，2 秒 overlap)
UC_THR_DEF    = 1.0        # mean(U_E+U_A) 阈值，高于此值视为低质量
BEAT_MIN      = 5          # 10s 内最少心拍 (~30 bpm)
BEAT_MAX      = 25         # 10s 内最多心拍 (~150 bpm)
COLOR_GOOD    = "#2ecc71"
COLOR_BAD     = "#e74c3c"


# ──────────────────────────────────────────────
# 模型加载
# ──────────────────────────────────────────────

def load_model(device):
    ckpt  = torch.load(str(CKPT_PATH), map_location="cpu")
    model = QRSModel(encoder4qrs(), decoder4qrs(), phi_qrs()).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model.eval()


# ──────────────────────────────────────────────
# 单窗口推理（返回 R-peak + 不确定性）
# ──────────────────────────────────────────────

def infer_window(model, window_1d: np.ndarray, fs: int, device):
    """
    返回 (r_peaks_原始采样率, mean_uc, uc_per_frame)
    uc_per_frame: shape [T_50hz]，逐帧不确定性 (U_E + U_A)
    """
    std = window_1d.std()
    if std > 1:
        window_1d = (window_1d - window_1d.mean()) / std
    proc = preprocess_ecg(window_1d, fs=fs)
    if proc.ndim == 1:
        proc = proc[np.newaxis, :]
    elif proc.shape[0] > proc.shape[1]:
        proc = proc.T
    sig_t = torch.from_numpy(proc).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(sig_t, return_projection=True)
        logits = logits.squeeze(-1).squeeze(0).cpu().numpy()
    uc     = uncertain_est(logits)               # [T_50hz]
    r_50hz = correct(logits[0], uc)
    r_orig = np.round(np.array(r_50hz) * (fs / FS_MODEL)).astype(int)
    return r_orig, float(np.mean(uc)), uc


# ──────────────────────────────────────────────
# 质量评估主函数
# ──────────────────────────────────────────────

def assess_quality(signal: np.ndarray, fs: int, model, device, uc_thr: float):
    """
    滑动窗口对整段信号评估质量。
    返回 windows 列表，每项包含：
      start_samp, end_samp, start_s, end_s,
      n_beats, mean_uc, is_good, r_peaks_abs
    """
    ws      = int(WIN_SEC  * fs)
    ss      = int(STEP_SEC * fs)
    sig_len = len(signal)
    windows = []
    pos     = 0

    while pos < sig_len:
        actual_end = min(pos + ws, sig_len)
        w = signal[pos: pos + ws]
        w = np.pad(w, (0, max(0, ws - len(w))))   # 末尾不足则补零

        r_rel, mean_uc, _ = infer_window(model, w, fs, device)
        # 只保留实际信号范围内的 R-peak（去掉 padding 区）
        r_rel   = r_rel[r_rel < (actual_end - pos)]
        n_beats = len(r_rel)
        r_abs   = r_rel + pos

        is_good = (mean_uc <= uc_thr) and (BEAT_MIN <= n_beats <= BEAT_MAX)

        windows.append(dict(
            start_samp  = pos,
            end_samp    = actual_end,
            start_s     = pos / fs,
            end_s       = actual_end / fs,
            n_beats     = n_beats,
            mean_uc     = round(mean_uc, 4),
            is_good     = is_good,
            r_peaks_abs = r_abs,
        ))
        pos += ss

    return windows


# ──────────────────────────────────────────────
# 保存高质量片段 (NPZ)
# ──────────────────────────────────────────────

def save_segments(signal, windows, fs, out_dir, base_name):
    """高质量片段各存一个 NPZ 文件。"""
    os.makedirs(out_dir, exist_ok=True)
    good = [w for w in windows if w["is_good"]]
    for i, w in enumerate(good):
        seg   = signal[w["start_samp"]: w["end_samp"]]
        fname = f"{base_name}_seg{i:03d}_{int(w['start_s'])}s.npz"
        np.savez(
            os.path.join(out_dir, fname),
            signal  = seg,
            fs      = np.array(fs),
            start_s = np.array(w["start_s"]),
            mean_uc = np.array(w["mean_uc"]),
            n_beats = np.array(w["n_beats"]),
            r_peaks = w["r_peaks_abs"] - w["start_samp"],   # 相对于片段起点
        )
    return good


# ──────────────────────────────────────────────
# 可视化 1：全局概览（完整信号 + 窗口颜色 + 不确定性柱状图）
# ──────────────────────────────────────────────

def plot_overview(signal, windows, fs, uc_thr, out_path):
    t = np.arange(len(signal)) / fs

    fig, (ax_ecg, ax_uc) = plt.subplots(
        2, 1, figsize=(20, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2]}
    )

    # ECG 信号 + 背景色块
    ax_ecg.plot(t, signal, lw=0.35, color="steelblue", alpha=0.85, rasterized=True)
    for w in windows:
        ax_ecg.axvspan(
            w["start_s"], w["end_s"],
            alpha=0.13 if w["is_good"] else 0.08,
            color=COLOR_GOOD if w["is_good"] else COLOR_BAD,
            lw=0
        )
    n_good  = sum(w["is_good"] for w in windows)
    n_total = len(windows)
    ax_ecg.set_ylabel("CH20 (ADC counts)", fontsize=9)
    ax_ecg.set_title(
        f"{os.path.basename(out_path)}  |  "
        f"绿色=高质量窗口 ({n_good}/{n_total})  红色=低质量  "
        f"阈值 mean(U_E+U_A) <= {uc_thr}",
        fontsize=8.5, pad=5
    )
    ax_ecg.tick_params(labelsize=8)
    ax_ecg.spines["top"].set_visible(False)
    ax_ecg.spines["right"].set_visible(False)

    # 不确定性柱状图
    xs     = [(w["start_s"] + w["end_s"]) / 2 for w in windows]
    # 截断到 3 倍阈值，防止坏窗口撑高 y 轴
    ucs    = [min(w["mean_uc"], uc_thr * 3) for w in windows]
    colors = [COLOR_GOOD if w["is_good"] else COLOR_BAD for w in windows]
    bar_w  = STEP_SEC * 0.75
    ax_uc.bar(xs, ucs, width=bar_w, color=colors, alpha=0.75, edgecolor="none")
    ax_uc.axhline(uc_thr, color="gray", lw=1.0, linestyle="--",
                  label=f"阈值 {uc_thr}")
    ax_uc.set_ylabel("mean(U_E+U_A)", fontsize=8)
    ax_uc.set_xlabel("Time (s)", fontsize=9)
    ax_uc.tick_params(labelsize=8)
    ax_uc.legend(fontsize=7, loc="upper right")
    ax_uc.spines["top"].set_visible(False)
    ax_uc.spines["right"].set_visible(False)

    plt.tight_layout(h_pad=0.5)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [概览图] {out_path}")


# ──────────────────────────────────────────────
# 可视化 2：高质量片段网格（每片段一个子图，带 R-peak 红点）
# ──────────────────────────────────────────────

def plot_good_segments(signal, good_windows, fs, out_path, max_cols=3):
    if not good_windows:
        print("  [片段图] 无高质量片段，跳过绘图")
        return

    n     = len(good_windows)
    ncols = min(n, max_cols)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(6 * ncols, 3 * nrows))
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.55, wspace=0.3)

    for idx, w in enumerate(good_windows):
        r = idx // ncols
        c = idx  % ncols
        ax = fig.add_subplot(gs[r, c])

        seg = signal[w["start_samp"]: w["end_samp"]]
        t_ax = np.arange(len(seg)) / fs

        ax.plot(t_ax, seg, lw=0.6, color="steelblue", alpha=0.9)

        # R-peak 红点（相对于片段起点）
        rp_rel = w["r_peaks_abs"] - w["start_samp"]
        rp_rel = rp_rel[(rp_rel >= 0) & (rp_rel < len(seg))]
        if len(rp_rel):
            ax.scatter(rp_rel / fs, seg[rp_rel],
                       color="red", s=20, zorder=5, linewidths=0)

        # 心率标注
        hr_str = ""
        if len(rp_rel) > 1:
            mean_hr = 60 / np.mean(np.diff(rp_rel) / fs)
            hr_str  = f"  {mean_hr:.0f} bpm"

        ax.set_title(
            f"seg{idx:03d}  {w['start_s']:.0f}-{w['end_s']:.0f}s\n"
            f"beats={w['n_beats']}{hr_str}  uc={w['mean_uc']:.3f}",
            fontsize=7.5
        )
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # 隐藏多余子图格子
    total_cells = nrows * ncols
    for idx in range(n, total_cells):
        r = idx // ncols
        c = idx  % ncols
        fig.add_subplot(gs[r, c]).set_visible(False)

    fig.suptitle(
        f"高质量片段 (共 {n} 个)  |  mean(U_E+U_A) 低 + 心拍数正常  |  红点=R-peak",
        fontsize=9, y=1.01
    )
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [片段图] {out_path}")


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="基于 PN-QRS 不确定性提取 CH20 高质量 10 秒片段"
    )
    ap.add_argument("--csv",     required=True,              help="原始 ECG CSV/Excel 文件路径")
    ap.add_argument("--fs",      required=True, type=int,    help="采样率 Hz")
    ap.add_argument("--uc_thr",  default=UC_THR_DEF, type=float,
                    help=f"不确定性阈值，默认 {UC_THR_DEF}（越低越严格；建议从 1.0 开始调）")
    ap.add_argument("--out_dir", default=None,
                    help="高质量片段 NPZ 保存目录（默认：CSV 同目录下 quality_segments/）")
    ap.add_argument("--gpu",     default="0")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"加载模型... 设备: {device}")
    model = load_model(device)

    # 读取 CH20 信号
    if args.csv.endswith(".csv"):
        df = pd.read_csv(args.csv)
    else:
        df = pd.read_excel(args.csv)
    upper_col = next((c for c in df.columns if str(c).upper() == "CH20"), None)
    if upper_col is None:
        raise ValueError(f"未在 {args.csv} 中找到 CH20 列，可用列: {df.columns.tolist()}")
    signal = df[upper_col].values.astype(np.float32)

    base_name = os.path.basename(args.csv).replace(".csv", "").replace(".xlsx", "")
    csv_dir   = os.path.dirname(os.path.abspath(args.csv))
    out_dir   = args.out_dir or os.path.join(csv_dir, "quality_segments")

    print(f"\n文件  : {args.csv}")
    print(f"采样率: {args.fs} Hz  时长: {len(signal)/args.fs:.1f}s  阈值: {args.uc_thr}")
    print(f"窗口  : {WIN_SEC}s  步长: {STEP_SEC}s  心拍范围: {BEAT_MIN}-{BEAT_MAX}/10s\n")

    # 逐窗口质量评估
    windows = assess_quality(signal, args.fs, model, device, args.uc_thr)

    n_good  = sum(w["is_good"] for w in windows)
    n_total = len(windows)
    print(f"评估完成：{n_total} 个窗口  高质量: {n_good}  低质量: {n_total - n_good}")

    # 保存 NPZ
    good_windows = save_segments(signal, windows, args.fs, out_dir, base_name)
    print(f"高质量片段已保存至: {out_dir}  ({len(good_windows)} 个 .npz)")

    # 保存质量报告 CSV
    report_path = os.path.join(csv_dir, base_name + "_quality_report.csv")
    pd.DataFrame([
        {k: v for k, v in w.items() if k != "r_peaks_abs"}
        for w in windows
    ]).to_csv(report_path, index=False)
    print(f"质量报告: {report_path}")

    # 可视化
    overview_path = os.path.join(csv_dir, base_name + "_quality_overview.png")
    segments_path = os.path.join(csv_dir, base_name + "_quality_segments.png")
    plot_overview(signal, windows, args.fs, args.uc_thr, overview_path)
    plot_good_segments(signal, good_windows, args.fs, segments_path)

    print(f"\n完成。高质量率: {n_good}/{n_total} = {n_good/max(n_total,1)*100:.1f}%")


if __name__ == "__main__":
    main()
