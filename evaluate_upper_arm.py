#!/usr/bin/env python3
"""
evaluate_upper_arm.py -- 以 12 导联检测结果为参考，评估上臂导联（CH20）的 R-peak 检测质量

用法：
  python evaluate_upper_arm.py --data_dir /path/to/data --fs 1000
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

TOL_SEC = 0.150   # 150ms 容差，与论文一致


def match_peaks(ref: np.ndarray, pred: np.ndarray, tol: int):
    """
    贪心匹配：每个 ref peak 最多匹配一个 pred peak（距离 <= tol）。
    返回 (TP, FP, FN)
    """
    ref  = np.sort(ref)
    pred = np.sort(pred)
    used = np.zeros(len(pred), dtype=bool)
    tp   = 0

    for r in ref:
        dists = np.abs(pred - r)
        dists[used] = tol + 1          # 已匹配的不再参与
        idx = int(np.argmin(dists)) if len(dists) else -1
        if idx >= 0 and dists[idx] <= tol:
            tp += 1
            used[idx] = True

    fp = int((~used).sum())            # pred 中未被匹配的
    fn = len(ref) - tp                 # ref 中未被匹配的
    return tp, fp, fn


def evaluate_file(base_path: str, fs: int):
    """
    base_path: CSV 文件路径去掉扩展名（如 /data/recording）
    """
    ref_path  = base_path + "_CH1-8_rpeaks.csv"
    pred_path = base_path + "_CH20_rpeaks.csv"

    if not os.path.exists(ref_path):
        print(f"  [跳过] 找不到参考文件: {ref_path}")
        return None
    if not os.path.exists(pred_path):
        print(f"  [跳过] 找不到预测文件: {pred_path}")
        return None

    ref  = pd.read_csv(ref_path)["sample_index"].values
    pred = pd.read_csv(pred_path)["sample_index"].values
    tol  = int(TOL_SEC * fs)

    tp, fp, fn = match_peaks(ref, pred, tol)
    se  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pp  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1  = 2 * se * pp / (se + pp) if (se + pp) > 0 else 0.0

    name = os.path.basename(base_path)
    print(f"  {name}")
    print(f"    ref(12导联)={len(ref)}  pred(CH20)={len(pred)}")
    print(f"    TP={tp}  FP={fp}  FN={fn}")
    print(f"    Se={se*100:.2f}%  P+={pp*100:.2f}%  F1={f1*100:.2f}%")

    return dict(name=name, ref=len(ref), pred=len(pred),
                tp=tp, fp=fp, fn=fn, se=se, pp_=pp, f1=f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="包含 *_rpeaks.csv 的目录")
    ap.add_argument("--fs",       required=True, type=int, help="采样率 Hz")
    args = ap.parse_args()

    # 找所有有 CH1-8 结果的文件（说明这个文件跑过 apply_pnqrs.py）
    ref_files = sorted(glob.glob(os.path.join(args.data_dir, "*_CH1-8_rpeaks.csv")))
    if not ref_files:
        print("未找到 *_CH1-8_rpeaks.csv，请先运行 apply_pnqrs.py")
        return

    print(f"找到 {len(ref_files)} 个文件\n{'─'*50}")
    results = []
    for f in ref_files:
        base = f.replace("_CH1-8_rpeaks.csv", "")
        print(f"\n>> {os.path.basename(base)}")
        r = evaluate_file(base, args.fs)
        if r:
            results.append(r)

    if not results:
        return

    # 汇总
    total_tp  = sum(r["tp"]  for r in results)
    total_fp  = sum(r["fp"]  for r in results)
    total_fn  = sum(r["fn"]  for r in results)
    total_se  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_pp  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_f1  = 2 * total_se * total_pp / (total_se + total_pp) if (total_se + total_pp) > 0 else 0

    print(f"\n{'─'*50}")
    print(f"汇总（{len(results)} 个文件）")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"  Se={total_se*100:.2f}%  P+={total_pp*100:.2f}%  F1={total_f1*100:.2f}%")
    print()
    print("指标含义（以 12 导联结果为参考）：")
    print(f"  Se  = CH20 检出了多少 12 导联检到的心拍（漏检率 = {(1-total_se)*100:.2f}%）")
    print(f"  P+  = CH20 检出的心拍有多少是真实的（误报率 = {(1-total_pp)*100:.2f}%）")
    print(f"  F1  = 综合指标")


if __name__ == "__main__":
    main()
