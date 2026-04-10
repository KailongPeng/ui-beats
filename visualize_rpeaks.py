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
        for i, f in enumerate(all_files, 1):
            print(f"\n[{i}/{len(all_files)}] {os.path.relpath(f, args.data_dir)}", flush=True)
            visualize_one(f, args.fs, args.start, args.duration, args.channels)

    elif args.csv:
        visualize_one(args.csv, args.fs, args.start, args.duration,
                      args.channels, args.out)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
