#!/usr/bin/env python3
"""
visualize_rpeaks.py -- 画出 ECG 信号 + 检测到的 R-peak 标记

用法：
  python visualize_rpeaks.py --csv /path/to/data.csv --fs 1000
  python visualize_rpeaks.py --csv /path/to/data.csv --fs 1000 --start 10 --duration 20
  python visualize_rpeaks.py --csv /path/to/data.csv --fs 1000 --channels CH20
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CH_STANDARD  = [f"CH{i}" for i in range(1, 9)]
CH_UPPER_ARM = "CH20"


def load_rpeaks(csv_path: str, suffix: str) -> np.ndarray:
    """尝试读取对应的 _rpeaks.csv，不存在则返回空数组。"""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",      required=True,  help="原始 ECG CSV 文件路径")
    ap.add_argument("--fs",       required=True,  type=int, help="采样率 Hz")
    ap.add_argument("--start",    default=0,      type=float, help="起始时间（秒，默认0）")
    ap.add_argument("--duration", default=30,     type=float, help="显示时长（秒，默认30）")
    ap.add_argument("--channels", default="all",
                    help="要显示的通道，逗号分隔，如 CH20 或 CH1,CH2,CH20，默认 all")
    ap.add_argument("--out",      default=None,   help="输出图片路径（默认与 CSV 同目录）")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    cols = df.columns.tolist()

    # 确定要显示的通道
    if args.channels == "all":
        show_ch = [c for c in cols if str(c).upper() in set(CH_STANDARD + [CH_UPPER_ARM])]
    else:
        show_ch = [c.strip() for c in args.channels.split(",")]

    if not show_ch:
        print("未找到可显示的通道，请检查列名")
        return

    # 切取时间段
    start_samp = int(args.start * args.fs)
    end_samp   = min(start_samp + int(args.duration * args.fs), len(df))
    n_samp     = end_samp - start_samp

    # 读取 R-peak 结果
    rp_standard  = load_rpeaks(args.csv, "CH1-8")
    rp_upper_arm = load_rpeaks(args.csv, "CH20")

    # 画图
    n_plots = len(show_ch)
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, max(2 * n_plots, 4)),
                             sharex=True)
    if n_plots == 1:
        axes = [axes]

    for ax, ch in zip(axes, show_ch):
        signal = df[ch].values[start_samp:end_samp].astype(float)
        # CH20 用上臂 R-peak，其余用标准导联 R-peak
        rp = rp_upper_arm if str(ch).upper() == CH_UPPER_ARM else rp_standard
        color = "darkorange" if str(ch).upper() == CH_UPPER_ARM else "steelblue"
        plot_channel(ax, signal, rp, args.fs, start_samp, ch, color=color)

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    axes[0].set_title(
        f"{os.path.basename(args.csv)}  |  "
        f"{args.start:.0f}–{args.start + args.duration:.0f}s  |  "
        f"red dots = R-peaks",
        fontsize=9, pad=6
    )

    # 在 CH20 子图右上角标注 beats 和 HR
    for ax, ch in zip(axes, show_ch):
        rp = rp_upper_arm if str(ch).upper() == CH_UPPER_ARM else rp_standard
        mask = (rp >= start_samp) & (rp < end_samp)
        n_beats = mask.sum()
        if n_beats > 1:
            rp_seg = rp[mask]
            mean_hr = 60 / np.mean(np.diff(rp_seg) / args.fs)
            ax.text(0.99, 0.92, f"{n_beats} beats  {mean_hr:.0f} bpm",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.5, color="red",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    plt.tight_layout()

    out_path = args.out
    if out_path is None:
        out_path = args.csv.replace(".csv", f"_rpeaks_vis_{int(args.start)}-{int(args.start+args.duration)}s.png") \
                           .replace(".xlsx", f"_rpeaks_vis_{int(args.start)}-{int(args.start+args.duration)}s.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"已保存: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
