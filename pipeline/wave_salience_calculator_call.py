#!/usr/bin/env python3
"""
wave_salience_calculator_.py -- 调用 _wave_salience_calculator.py 的接口封装

从 _wave_salience_calculator 导入 P/Q/S/T/综合显著性计算器，
加载 CSV 信号 + 已有 rpeaks.csv，批量计算各波 SQI 并格式化输出。

用法：
  python wave_salience_calculator_.py --csv /path/to/data.csv --fs 1000
  python wave_salience_calculator_.py --batch --data_dir /path/to/dir --fs 1000
"""

import argparse
import glob as _glob
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── 导入工具库 ──────────────────────────────────────────────────────────────
from _wave_salience_calculator import (
    PWaveSalienceCalculator,
    QWaveSalienceCalculator,
    SWaveSalienceCalculator,
    TWaveSalienceCalculator,
    WaveSalienceCalculator,
)

# 各子计算器（用于展示分波明细）
SUB_CALCULATORS = [
    ("P", PWaveSalienceCalculator),
    ("Q", QWaveSalienceCalculator),
    ("S", SWaveSalienceCalculator),
    ("T", TWaveSalienceCalculator),
]

SKIP_SUFFIXES = (
    "_wave_sqi.csv", "_wave_sqi_detail.csv",
    "_quality_report.csv", "_rpeaks.csv",
    "_CH1-8_rpeaks.csv", "_CH20_rpeaks.csv",
)


# ── CSV 读取（兼容末尾多余列） ──────────────────────────────────────────────
def _read_csv_robust(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.readlines()
    if not raw:
        return pd.DataFrame()
    ncols = len(raw[0].split(","))
    fixed = []
    for line in raw:
        fields = line.rstrip("\n").split(",")
        if len(fields) != ncols:
            fields = fields[:ncols]
        fixed.append(",".join(fields))
    return pd.read_csv(io.StringIO("\n".join(fixed)))


def load_signal_and_rpeaks(csv_path: str, fs: int):
    """
    读取 CSV 中的 CH20 信号，以及同目录下已有的 rpeaks CSV。
    返回 (signal_1d, r_peaks_indices) 或 (None, None)。
    """
    df = _read_csv_robust(csv_path)

    # 找 CH20
    upper_col = next((c for c in df.columns if str(c).upper() == "CH20"), None)
    if upper_col is None:
        print(f"  [跳过] 没有 CH20 列: {csv_path}")
        return None, None
    signal = df[upper_col].values.astype(np.float32)

    # 优先用 CH20 rpeaks，否则用 CH1-8 rpeaks
    base = csv_path.replace(".csv", "").replace(".xlsx", "")
    for suffix in ("_CH20_rpeaks.csv", "_CH1-8_rpeaks.csv"):
        rp_path = base + suffix
        if os.path.exists(rp_path):
            r_peaks = pd.read_csv(rp_path)["sample_index"].values
            return signal, r_peaks

    print(f"  [跳过] 找不到 rpeaks CSV（请先运行 apply_pnqrs.py）: {csv_path}")
    return None, None


# ── 单文件计算 ──────────────────────────────────────────────────────────────
def process_file(csv_path: str, fs: int, detail: bool = False) -> dict | None:
    signal, r_peaks = load_signal_and_rpeaks(csv_path, fs)
    if signal is None:
        return None

    duration = len(signal) / fs
    print(f"  fs={fs}Hz  duration={duration:.1f}s  r_peaks={len(r_peaks)}")

    # ── 综合 SQI ────────────────────────────────────────────────────────────
    try:
        composite_result = WaveSalienceCalculator().calculate(signal, r_peaks, fs)
    except Exception as e:
        print(f"  [错误] WaveSalienceCalculator: {e}")
        return None

    row = {
        "file":        os.path.basename(csv_path),
        "duration_s":  round(duration, 1),
        "n_rpeaks":    len(r_peaks),
        "composite_value":  round(float(composite_result.value),      4),
        "composite_score":  round(float(composite_result.score),      4),
        "composite_conf":   round(float(composite_result.confidence), 4),
    }

    # ── 分波明细 ────────────────────────────────────────────────────────────
    sub_rows = []
    for wave_name, CalcClass in SUB_CALCULATORS:
        try:
            res = CalcClass().calculate(signal, r_peaks, fs)
            print(
                f"    {wave_name}: value={res.value:.3f}  "
                f"confidence={res.confidence:.1%}  score={res.score:.3f}"
                + (f"  | {res.description}" if res.description else "")
            )
            row[f"{wave_name}_value"]      = round(float(res.value),      4)
            row[f"{wave_name}_confidence"] = round(float(res.confidence), 4)
            row[f"{wave_name}_score"]      = round(float(res.score),      4)
            if detail:
                sub_rows.append({
                    "file": os.path.basename(csv_path),
                    "wave": wave_name,
                    "value":      round(float(res.value),      4),
                    "confidence": round(float(res.confidence), 4),
                    "score":      round(float(res.score),      4),
                    "description": res.description or "",
                })
        except Exception as e:
            print(f"    {wave_name}: [错误] {e}")
            row[f"{wave_name}_value"] = row[f"{wave_name}_confidence"] = row[f"{wave_name}_score"] = None

    print(
        f"  → composite  value={composite_result.value:.3f}  "
        f"score={composite_result.score:.3f}  conf={composite_result.confidence:.1%}"
        + (f"\n     {composite_result.description}" if composite_result.description else "")
    )

    if detail:
        row["_detail"] = sub_rows
    return row


# ── 输出格式化 ──────────────────────────────────────────────────────────────
def print_batch_table(stats: list[dict]):
    header = f"{'file':<40}  {'dur':>6}  {'P_val':>6} {'P_conf':>7} {'T_val':>6} {'T_conf':>7}  {'comp':>6}"
    print(f"\n{'─'*len(header)}")
    print(header)
    print(f"{'─'*len(header)}")
    for s in stats:
        print(
            f"{s['file']:<40}  {s['duration_s']:>6.1f}"
            f"  {s.get('P_value', float('nan')):>6.3f} {s.get('P_confidence', float('nan')):>7.1%}"
            f"  {s.get('T_value', float('nan')):>6.3f} {s.get('T_confidence', float('nan')):>7.1%}"
            f"  {s['composite_score']:>6.3f}"
        )
    print(f"{'─'*len(header)}")


def save_results(stats: list[dict], out_dir: str, detail: bool):
    os.makedirs(out_dir, exist_ok=True)

    # 汇总表（去掉 _detail 字段）
    summary_rows = [{k: v for k, v in s.items() if k != "_detail"} for s in stats]
    summary_path = os.path.join(out_dir, "wave_sqi_summary_.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\n汇总 CSV → {summary_path}")

    if detail:
        detail_rows = [r for s in stats for r in s.get("_detail", [])]
        if detail_rows:
            detail_path = os.path.join(out_dir, "wave_sqi_detail_.csv")
            pd.DataFrame(detail_rows).to_csv(detail_path, index=False)
            print(f"明细 CSV → {detail_path}")


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",      help="单文件模式：CSV 路径")
    ap.add_argument("--data_dir", help="批量模式：根目录")
    ap.add_argument("--batch",    action="store_true")
    ap.add_argument("--fs",       required=True, type=int)
    ap.add_argument("--detail",   action="store_true", help="输出分波明细 CSV")
    ap.add_argument("--out_dir",  default=None, help="输出目录（默认与输入同目录）")
    args = ap.parse_args()

    if args.batch or args.data_dir:
        if not args.data_dir:
            sys.exit("批量模式需要 --data_dir")
        patterns = ["**/*.csv", "**/*.xlsx"]
        all_files = []
        for pat in patterns:
            all_files.extend(_glob.glob(os.path.join(args.data_dir, pat), recursive=True))
        all_files = sorted(set(
            f for f in all_files
            if not any(f.endswith(s) for s in SKIP_SUFFIXES)
        ))
        if not all_files:
            sys.exit(f"未找到 CSV/Excel 文件：{args.data_dir}")

        print(f"找到 {len(all_files)} 个文件\n{'─'*60}")
        stats = []
        for i, f in enumerate(all_files, 1):
            print(f"\n[{i}/{len(all_files)}] {os.path.relpath(f, args.data_dir)}", flush=True)
            r = process_file(f, args.fs, args.detail)
            if r:
                stats.append(r)

        if stats:
            print_batch_table(stats)
            out_dir = args.out_dir or args.data_dir
            save_results(stats, out_dir, args.detail)

    elif args.csv:
        print(f"\n>> {os.path.basename(args.csv)}")
        r = process_file(args.csv, args.fs, args.detail)
        if r:
            out_dir = args.out_dir or os.path.dirname(args.csv)
            save_results([r], out_dir, args.detail)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
