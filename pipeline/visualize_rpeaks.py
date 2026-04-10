#!/usr/bin/env python3
"""
visualize_rpeaks.py -- 画出 ECG 信号 + 检测到的 R-peak 标记

用法：
  # 单文件
  python visualize_rpeaks.py --csv /path/to/data.csv --fs 1000
  python visualize_rpeaks.py --csv /path/to/data.csv --fs 1000 --start 10 --duration 20
  python visualize_rpeaks.py --csv /path/to/data.csv --fs 1000 --channels CH20

  # 批量（递归扫描子目录，每个 CSV 各生成一张图）
  python visualize_rpeaks.py --batch --data_dir /path/to/dir --fs 1000
  python visualize_rpeaks.py --batch --data_dir /path/to/dir --fs 1000 --channels CH20 --duration 20
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CH_STANDARD  = [f"CH{i}" for i in range(1, 9)]
CH_UPPER_ARM = "CH20"

SKIP_SUFFIXES = (
    "_rpeaks.csv", "_CH1-8_rpeaks.csv", "_CH20_rpeaks.csv",
    "_quality_report.csv", "_wave_sqi.csv", "_wave_sqi_detail.csv",
)


def load_rpeaks(csv_path: str, suffix: str) -> np.ndarray:
    rp_path = csv_path.replace(".csv", f"_{suffix}_rpeaks.csv") \
                      .replace(".xlsx", f"_{suffix}_rpeaks.csv")
    if os.path.exists(rp_path):
        return pd.read_csv(rp_path)["sample_index"].values
    return np.array([], dtype=int)


def plot_channel(ax, signal, r_peaks, fs, start_samp, label, color="steelblue"):
    t = np.arange(len(signal)) / fs + start_samp / fs
    ax.plot(t, signal, lw=0.6, color=color, alpha=0.85)

    mask = (r_peaks >= start_samp) & (r_peaks < start_samp + len(signal))
    rp_local = r_peaks[mask] - start_samp
    if len(rp_local):
        ax.scatter(rp_local / fs + start_samp / fs,
                   signal[rp_local],
                   color="red", s=25, zorder=5, linewidths=0)

    ax.set_ylabel(label, fontsize=8, labelpad=2)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def visualize_one(csv_path: str, fs: int, start: float, duration: float,
                  channels: str, out: str | None = None):
    df = pd.read_csv(csv_path)
    cols = df.columns.tolist()

    if channels == "all":
        show_ch = [c for c in cols if str(c).upper() in set(CH_STANDARD + [CH_UPPER_ARM])]
    else:
        show_ch = [c.strip() for c in channels.split(",")]

    if not show_ch:
        print(f"  [跳过] 未找到可显示的通道: {csv_path}")
        return

    start_samp = int(start * fs)
    end_samp   = min(start_samp + int(duration * fs), len(df))

    rp_standard  = load_rpeaks(csv_path, "CH1-8")
    rp_upper_arm = load_rpeaks(csv_path, "CH20")

    n_plots = len(show_ch)
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, max(2 * n_plots, 4)),
                             sharex=True)
    if n_plots == 1:
        axes = [axes]

    for ax, ch in zip(axes, show_ch):
        signal = df[ch].values[start_samp:end_samp].astype(float)
        rp = rp_upper_arm if str(ch).upper() == CH_UPPER_ARM else rp_standard
        color = "darkorange" if str(ch).upper() == CH_UPPER_ARM else "steelblue"
        plot_channel(ax, signal, rp, fs, start_samp, ch, color=color)

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    axes[0].set_title(
        f"{os.path.basename(csv_path)}  |  "
        f"{start:.0f}–{start + duration:.0f}s  |  "
        f"red dots = R-peaks",
        fontsize=9, pad=6
    )

    for ax, ch in zip(axes, show_ch):
        rp = rp_upper_arm if str(ch).upper() == CH_UPPER_ARM else rp_standard
        mask = (rp >= start_samp) & (rp < end_samp)
        n_beats = mask.sum()
        if n_beats > 1:
            rp_seg = rp[mask]
            mean_hr = 60 / np.mean(np.diff(rp_seg) / fs)
            ax.text(0.99, 0.92, f"{n_beats} beats  {mean_hr:.0f} bpm",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.5, color="red",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    plt.tight_layout()

    if out is None:
        out = csv_path.replace(".csv",  f"_rpeaks_vis_{int(start)}-{int(start+duration)}s.png") \
                      .replace(".xlsx", f"_rpeaks_vis_{int(start)}-{int(start+duration)}s.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  已保存: {out}")
    plt.close()


def visualize_low_amp(csv_path: str, fs: int, channels: str,
                      win_sec: float = 10.0, top_n: int = 6,
                      out: str | None = None):
    """
    找出信号中峰峰值最小的 top_n 个窗口并可视化，帮助评估低幅度区块的 R-peak 检测能力。
    每个子图标注：窗口时间范围、峰峰值、检出心拍数。
    """
    df = pd.read_csv(csv_path)
    cols = df.columns.tolist()

    if channels == "all":
        show_ch = [c for c in cols if str(c).upper() in set(CH_STANDARD + [CH_UPPER_ARM])]
    else:
        show_ch = [c.strip() for c in channels.split(",")]

    if not show_ch:
        print(f"  [跳过] 未找到可显示的通道: {csv_path}")
        return

    # 用第一个通道的信号计算峰峰值（代表整体信号强度）
    ref_ch = show_ch[0]
    sig_full = df[ref_ch].values.astype(float)
    n_total  = len(sig_full)
    win_samp = int(win_sec * fs)

    rp_standard  = load_rpeaks(csv_path, "CH1-8")
    rp_upper_arm = load_rpeaks(csv_path, "CH20")

    # ── 滑动窗口，计算峰峰值 ────────────────────────────────────────────────
    starts = list(range(0, n_total - win_samp + 1, win_samp))  # 非重叠
    ptp_list = []
    for s in starts:
        seg = sig_full[s: s + win_samp]
        ptp_list.append((float(np.ptp(seg)), s))   # (peak-to-peak, start_sample)

    # 按峰峰值升序，取最低的 top_n 个（不超过实际窗口数）
    ptp_list.sort(key=lambda x: x[0])
    low_segs = ptp_list[:min(top_n, len(ptp_list))]
    top_n    = len(low_segs)

    n_cols = min(3, top_n)
    n_rows = (top_n + n_cols - 1) // n_cols
    n_ch   = len(show_ch)

    fig, axes = plt.subplots(
        n_rows * n_ch, n_cols,
        figsize=(6 * n_cols, 2.2 * n_ch * n_rows),
        squeeze=False
    )
    fig.suptitle(
        f"{os.path.basename(csv_path)}  —  lowest {top_n} amplitude windows "
        f"(window={win_sec:.0f}s, ref={ref_ch})",
        fontsize=10, y=1.01
    )

    for seg_idx, (ptp_val, start_s) in enumerate(low_segs):
        col = seg_idx % n_cols
        t_start = start_s / fs
        t_end   = (start_s + win_samp) / fs

        for ch_idx, ch in enumerate(show_ch):
            row = (seg_idx // n_cols) * n_ch + ch_idx
            ax  = axes[row][col]

            seg   = df[ch].values[start_s: start_s + win_samp].astype(float)
            rp    = rp_upper_arm if str(ch).upper() == CH_UPPER_ARM else rp_standard
            color = "darkorange" if str(ch).upper() == CH_UPPER_ARM else "steelblue"
            plot_channel(ax, seg, rp, fs, start_s, ch, color=color)

            # 标注信息（只在第一个通道子图顶部标）
            if ch_idx == 0:
                mask   = (rp >= start_s) & (rp < start_s + win_samp)
                n_beats = mask.sum()
                ax.set_title(
                    f"{t_start:.1f}–{t_end:.1f}s  |  ptp={ptp_val:.4f}  |  {n_beats} beats",
                    fontsize=8
                )

    # 隐藏多余子图
    total_slots = n_rows * n_cols
    for empty in range(len(low_segs), total_slots):
        col = empty % n_cols
        for ch_idx in range(n_ch):
            row = (empty // n_cols) * n_ch + ch_idx
            axes[row][col].set_visible(False)

    plt.tight_layout()

    if out is None:
        out = csv_path.replace(".csv",  "_low_amp_vis.png") \
                      .replace(".xlsx", "_low_amp_vis.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  已保存: {out}")
    plt.close()


def visualize_low_amp_global(all_files: list, data_dir: str, fs: int,
                             channel: str = "CH20", win_sec: float = 10.0,
                             top_n: int = 12, out_dir: str | None = None):
    """
    跨文件全局低幅度分析：
      1. 扫描所有文件，统计指定 channel 每个窗口的峰峰值
      2. 画全局幅度分布直方图（含 25/50 百分位线）
      3. 取全局幅度最低的 top_n 个窗口，网格可视化 + R-peak 标注
    """
    out_dir = out_dir or data_dir

    # ── Pass 1：收集所有窗口的峰峰值 ────────────────────────────────────────
    win_samp = int(win_sec * fs)
    all_windows = []   # list of (ptp, csv_path, start_sample)

    print(f"\nPass 1: computing {channel} amplitude across {len(all_files)} files...")
    for i, fpath in enumerate(all_files, 1):
        print(f"  [{i}/{len(all_files)}] {os.path.relpath(fpath, data_dir)}", flush=True)
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue
        ch_col = next((c for c in df.columns if str(c).upper() == channel.upper()), None)
        if ch_col is None:
            continue
        sig = df[ch_col].values.astype(float)
        for s in range(0, len(sig) - win_samp + 1, win_samp):
            ptp = float(np.ptp(sig[s: s + win_samp]))
            all_windows.append((ptp, fpath, s))

    if not all_windows:
        print(f"未找到含 {channel} 的文件。")
        return

    all_ptp = [w[0] for w in all_windows]
    p25, p50 = float(np.percentile(all_ptp, 25)), float(np.percentile(all_ptp, 50))
    print(f"\n全局统计（{len(all_windows)} 个窗口）：")
    print(f"  min={min(all_ptp):.4f}  p25={p25:.4f}  p50={p50:.4f}  max={max(all_ptp):.4f}")

    # ── 分布直方图 ────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(all_ptp, bins=50, color="steelblue", alpha=0.75, edgecolor="none")
    ax.axvline(p25, color="orange", lw=1.5, label=f"p25={p25:.3f}")
    ax.axvline(p50, color="red",    lw=1.5, label=f"p50={p50:.3f}")
    ax.set_xlabel(f"{channel} peak-to-peak amplitude (raw ADC units)", fontsize=10)
    ax.set_ylabel("Window count", fontsize=10)
    ax.set_title(f"Global {channel} amplitude distribution — {len(all_files)} files, "
                 f"{len(all_windows)} windows ({win_sec:.0f}s each)", fontsize=10)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    hist_path = os.path.join(out_dir, f"global_{channel}_amp_distribution.png")
    plt.savefig(hist_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  分布图 → {hist_path}")

    # ── Pass 2：取低/中/高三组，合并成一张对比图 ─────────────────────────────
    all_windows.sort(key=lambda x: x[0])
    n_avail = len(all_windows)
    n_each  = min(top_n, n_avail // 3)   # 三组各取 n_each 个

    mid_start = (n_avail - n_each) // 2
    groups = [
        ("Low",    all_windows[:n_each]),
        ("Mid",    all_windows[mid_start: mid_start + n_each]),
        ("High",   all_windows[n_avail - n_each:]),
    ]

    n_cols   = min(3, n_each)
    n_rows_g = (n_each + n_cols - 1) // n_cols   # rows per group
    n_rows   = n_rows_g * 3                       # total rows

    rp_suffix = "CH20" if channel.upper() == "CH20" else "CH1-8"
    color     = "darkorange" if channel.upper() == "CH20" else "steelblue"

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 3 * n_rows),
                             squeeze=False)
    fig.suptitle(
        f"Global {channel} amplitude — Low / Mid / High  "
        f"(window={win_sec:.0f}s, n={n_each} each)  |  red dots = R-peaks  |  "
        f"cv_rr = std(RR)/mean(RR), low=regular/good, high=noisy/bad",
        fontsize=10, y=1.005
    )

    # 左侧标注组名
    group_label_rows = [0, n_rows_g, n_rows_g * 2]
    group_colors     = ["#d62728", "#ff7f0e", "#2ca02c"]   # 红/橙/绿

    # 全局窗口编号索引（用于控制台对照）
    global_idx = {id(w): i for i, w in enumerate(all_windows, 1)}

    for g_idx, (g_name, segs) in enumerate(groups):
        row_offset = g_idx * n_rows_g
        for s_idx, (ptp_val, fpath, start_s) in enumerate(segs):
            row = row_offset + s_idx // n_cols
            col = s_idx % n_cols
            ax  = axes[row][col]

            # 组名标记（每组第一个子图左上角）
            if s_idx == 0:
                ax.text(-0.18, 0.5, g_name, transform=ax.transAxes,
                        fontsize=13, fontweight="bold", va="center", ha="center",
                        rotation=90, color=group_colors[g_idx])

            try:
                df = pd.read_csv(fpath)
            except Exception:
                ax.set_visible(False)
                continue
            ch_col = next((c for c in df.columns if str(c).upper() == channel.upper()), None)
            if ch_col is None:
                ax.set_visible(False)
                continue

            seg = df[ch_col].values[start_s: start_s + win_samp].astype(float)
            rp  = load_rpeaks(fpath, rp_suffix)
            plot_channel(ax, seg, rp, fs, start_s, channel, color=color)

            mask    = (rp >= start_s) & (rp < start_s + win_samp)
            n_beats = int(mask.sum())
            t_start = start_s / fs
            fname   = os.path.basename(fpath)   # ASCII only, no Chinese

            # CV_RR: coefficient of variation of RR intervals
            # low (~0.02-0.10) = regular rhythm / reliable detection
            # high (>0.20)     = irregular / likely false peaks in noisy signal
            win_rp = rp[mask]
            if len(win_rp) >= 3:
                rr = np.diff(win_rp.astype(float)) / fs   # RR intervals in seconds
                cv_rr_str = f"{np.std(rr) / (np.mean(rr) + 1e-9):.3f}"
            else:
                cv_rr_str = "--"

            # 控制台打印完整路径供对照
            print(f"    [{g_name}#{s_idx+1}] {os.path.relpath(fpath, data_dir)}  "
                  f"{t_start:.1f}-{t_start+win_sec:.1f}s  ptp={ptp_val:.4f}  "
                  f"cv_rr={cv_rr_str}  {n_beats} beats")
            ax.set_title(
                f"[{g_name}#{s_idx+1}] {fname}\n"
                f"{t_start:.1f}-{t_start+win_sec:.1f}s  "
                f"ptp={ptp_val:.4f}  cv_rr={cv_rr_str}  {n_beats}beats",
                fontsize=7
            )
            # 组别背景色（淡）
            ax.set_facecolor({0: "#fff0f0", 1: "#fffaf0", 2: "#f0fff0"}[g_idx])

        # 隐藏本组多余格子
        for empty in range(len(segs), n_rows_g * n_cols):
            r = row_offset + empty // n_cols
            c = empty % n_cols
            axes[r][c].set_visible(False)

    plt.tight_layout()
    grid_path = os.path.join(out_dir, f"global_{channel}_amp_comparison.png")
    plt.savefig(grid_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  对比图（低/中/高）→ {grid_path}")


def main():
    ap = argparse.ArgumentParser()
    # 单文件 / 批量互斥
    ap.add_argument("--csv",      help="单文件模式：原始 ECG CSV 路径")
    ap.add_argument("--data_dir", help="批量模式：根目录（递归扫描所有 CSV）")
    ap.add_argument("--batch",    action="store_true", help="开启批量模式")
    # 共用参数
    ap.add_argument("--fs",       required=True, type=int,   help="采样率 Hz")
    ap.add_argument("--start",    default=0,     type=float, help="起始时间（秒，默认0）")
    ap.add_argument("--duration", default=30,    type=float, help="显示时长（秒，默认30）")
    ap.add_argument("--channels", default="all",
                    help="要显示的通道，逗号分隔，如 CH20 或 CH1,CH2,CH20，默认 all")
    ap.add_argument("--out",      default=None,  help="单文件模式：输出图片路径")
    ap.add_argument("--low_amp",  action="store_true",
                    help="找出峰峰值最小的窗口并可视化（评估低幅度区块的检测能力）")
    ap.add_argument("--win_sec",  default=10.0, type=float, help="--low_amp 模式的窗口长度（秒，默认10）")
    ap.add_argument("--top_n",    default=6,    type=int,   help="--low_amp 模式显示最低幅度的前 N 个窗口（默认6）")
    args = ap.parse_args()

    if args.batch or args.data_dir:
        if not args.data_dir:
            ap.error("批量模式需要 --data_dir")
        all_files = sorted(
            glob.glob(os.path.join(args.data_dir, "**", "*.csv"),  recursive=True) +
            glob.glob(os.path.join(args.data_dir, "**", "*.xlsx"), recursive=True)
        )
        all_files = [f for f in all_files
                     if not any(f.endswith(s) for s in SKIP_SUFFIXES)]
        if not all_files:
            print(f"未找到 CSV/Excel 文件：{args.data_dir}")
            return
        print(f"找到 {len(all_files)} 个文件\n{'─'*50}")
        if args.low_amp:
            # 批量 + low_amp → 全局排序模式
            ch = args.channels if args.channels != "all" else "CH20"
            visualize_low_amp_global(all_files, args.data_dir, args.fs,
                                     channel=ch, win_sec=args.win_sec,
                                     top_n=args.top_n)
        else:
            for i, f in enumerate(all_files, 1):
                print(f"\n[{i}/{len(all_files)}] {os.path.relpath(f, args.data_dir)}", flush=True)
                visualize_one(f, args.fs, args.start, args.duration, args.channels)

    elif args.csv:
        if args.low_amp:
            visualize_low_amp(args.csv, args.fs, args.channels,
                              args.win_sec, args.top_n, args.out)
        else:
            visualize_one(args.csv, args.fs, args.start, args.duration,
                          args.channels, args.out)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
