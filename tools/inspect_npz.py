#!/usr/bin/env python3
"""
inspect_npz.py — 可视化 quality_segments 目录下的 NPZ 文件

用法：
  python tools/inspect_npz.py path/to/file.npz
  python tools/inspect_npz.py path/to/quality_segments/   # 批量浏览目录
  python tools/inspect_npz.py path/to/dir/ --save_dir ./plots  # 批量保存图片
"""
import sys
import os
import argparse
import glob
import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # 无 GUI 服务器请改成 "Agg" 并加 --save_dir
import matplotlib.pyplot as plt


# ── 单文件可视化 ──────────────────────────────────────────────────────────────

def plot_npz(path: str, show: bool = True, save_path: str = None):
    d = np.load(path)   # NPZ 无需 pickle，allow_pickle=False 更安全
    keys = list(d.files)

    fs       = int(d["fs"])        if "fs"       in keys else 1000
    start_s  = float(d["start_s"]) if "start_s"  in keys else 0.0
    mean_uc  = float(d["mean_uc"]) if "mean_uc"  in keys else float("nan")
    n_beats  = int(d["n_beats"])   if "n_beats"  in keys else -1
    r_peaks  = d["r_peaks"]        if "r_peaks"  in keys else np.array([])
    ch20     = d["signal"]         if "signal"   in keys else None

    # 标准导联 CH1~CH8（新格式才有）
    std_keys = sorted(
        [k for k in keys if k.upper().startswith("CH") and k.upper() != "CH20"],
        key=lambda k: int(k[2:])
    )

    # ── 打印 metadata ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  文件   : {os.path.basename(path)}")
    print(f"  Keys   : {keys}")
    print(f"  fs     : {fs} Hz")
    print(f"  start  : {start_s:.1f} s")
    print(f"  mean_uc: {mean_uc:.4f}")
    print(f"  n_beats: {n_beats}")
    print(f"  r_peaks: {len(r_peaks)} 个  → {r_peaks.tolist()}")
    if std_keys:
        print(f"  标准导联: {std_keys}")
    print(f"{'─'*60}")

    # ── 绘图 ────────────────────────────────────────────────────────────────
    n_panels = 1 + len(std_keys)   # CH20 + 各标准导联
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(14, 2.8 * n_panels),
                             sharex=True)
    if n_panels == 1:
        axes = [axes]

    t = np.arange(len(ch20)) / fs if ch20 is not None else np.array([])

    # CH20
    ax = axes[0]
    if ch20 is not None:
        ax.plot(t, ch20, color="#2196F3", lw=0.8, label="CH20 (上臂)")
        if len(r_peaks):
            ax.scatter(r_peaks / fs, ch20[r_peaks],
                       color="red", s=30, zorder=5, label="R-peak")
    ax.set_ylabel("CH20", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(
        f"{os.path.basename(path)}\n"
        f"start={start_s:.1f}s  fs={fs}Hz  mean_uc={mean_uc:.3f}  beats={n_beats}",
        fontsize=9
    )
    ax.grid(True, lw=0.3, alpha=0.5)

    # 标准导联
    colors = plt.cm.tab10.colors
    for i, k in enumerate(std_keys):
        sig = d[k].astype(float)
        ax  = axes[i + 1]
        t_k = np.arange(len(sig)) / fs
        ax.plot(t_k, sig, color=colors[i % 10], lw=0.8, label=k)
        ax.set_ylabel(k, fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, lw=0.3, alpha=0.5)

    axes[-1].set_xlabel("时间 (s)", fontsize=9)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  已保存: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ── 目录批量浏览 ──────────────────────────────────────────────────────────────

def browse_dir(data_dir: str, save_dir: str = None):
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True))
    if not files:
        print(f"目录 {data_dir} 下没有找到 .npz 文件")
        return
    print(f"找到 {len(files)} 个 NPZ 文件")

    for path in files:
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            png = os.path.join(save_dir,
                               os.path.splitext(os.path.basename(path))[0] + ".png")
            plot_npz(path, show=False, save_path=png)
        else:
            plot_npz(path, show=True)
            inp = input("  [Enter] 下一个 / [q] 退出: ").strip().lower()
            if inp == "q":
                break


# ── 入口 ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target",
                    help="单个 .npz 文件路径，或包含 .npz 的目录")
    ap.add_argument("--save_dir", default=None,
                    help="批量模式：把图保存到此目录（不弹窗）")
    args = ap.parse_args()

    if os.path.isdir(args.target):
        browse_dir(args.target, args.save_dir)
    elif args.target.endswith(".npz"):
        plot_npz(args.target, show=True)
    else:
        print("请传入 .npz 文件或包含 .npz 的目录")
        sys.exit(1)


if __name__ == "__main__":
    main()
