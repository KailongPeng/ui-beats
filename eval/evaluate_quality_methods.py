#!/usr/bin/env python3
"""
evaluate_quality_methods.py -- 以 12 导联 R-peak 为 ground truth，对比三种 CH20 质量评估方法

方法：
  A  : mean(U_E+U_A) + Otsu 阈值（当前方案）
  B1 : 两阶段，论文固定阈值（U_E > 0.1 → 坏；U_A > 0.12 → 坏）
  B2 : 两阶段，Otsu 自适应阈值

评价指标（以每窗口 CH20 R-peak F1 >= f1_thr 为"真实好窗口"）：
  AUC   用于 A（连续分数）
  Acc / Prec / Rec / F1  用于所有方法

用法：
  python eval/evaluate_quality_methods.py --data_dir data/0413_real/ --fs 1000
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── 默认参数 ─────────────────────────────────────────────────────────────────
TOL_SEC    = 0.150   # R-peak 匹配容差
F1_GOOD_THR = 0.85   # 窗口内 F1 >= 此值 → 视为"真实好窗口"
UE_ALPHA   = 0.10    # 论文 Stage-1 U_E 阈值
UA_BETA    = 0.12    # 论文 Stage-2 U_A 阈值


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def otsu_1d(values: np.ndarray) -> float:
    arr = np.sort(values)
    n = len(arr)
    if n < 2:
        return float(arr[0])
    cs = np.cumsum(arr)
    best_thr, best_var = arr[0], -1.0
    for i in range(1, n):
        w1, w2 = i / n, 1 - i / n
        if w1 == 0 or w2 == 0:
            continue
        mu1 = cs[i - 1] / i
        mu2 = (cs[-1] - cs[i - 1]) / (n - i)
        var = w1 * w2 * (mu1 - mu2) ** 2
        if var > best_var:
            best_var = var
            best_thr = float((arr[i - 1] + arr[i]) / 2)
    return best_thr


def match_peaks(ref: np.ndarray, pred: np.ndarray, tol: int):
    ref, pred = np.sort(ref), np.sort(pred)
    used = np.zeros(len(pred), dtype=bool)
    tp = 0
    for r in ref:
        dists = np.abs(pred - r)
        dists[used] = tol + 1
        idx = int(np.argmin(dists)) if len(dists) else -1
        if idx >= 0 and dists[idx] <= tol:
            tp += 1
            used[idx] = True
    return tp, int((~used).sum()), len(ref) - tp   # TP, FP, FN


def window_f1(ref_peaks: np.ndarray, pred_peaks: np.ndarray,
              start_s: float, end_s: float, fs: int) -> float:
    tol = int(TOL_SEC * fs)
    s, e = int(start_s * fs), int(end_s * fs)
    ref_w  = ref_peaks[(ref_peaks  >= s) & (ref_peaks  < e)]
    pred_w = pred_peaks[(pred_peaks >= s) & (pred_peaks < e)]
    if len(ref_w) == 0 and len(pred_w) == 0:
        return 1.0
    if len(ref_w) == 0 or len(pred_w) == 0:
        return 0.0
    tp, fp, fn = match_peaks(ref_w, pred_w, tol)
    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pp = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return 2 * se * pp / (se + pp) if (se + pp) > 0 else 0.0


def binary_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(( true &  pred).sum())
    fp = int((~true &  pred).sum())
    fn = int(( true & ~pred).sum())
    tn = int((~true & ~pred).sum())
    acc  = (tp + tn) / len(true) if len(true) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def auc_score(true: np.ndarray, score: np.ndarray) -> float:
    """AUC via trapezoidal rule (no sklearn dependency)."""
    thresholds = np.unique(score)
    tprs, fprs = [0.0], [0.0]
    for thr in thresholds:
        pred = score <= thr   # lower score = better quality → predict good
        tp = int(( true &  pred).sum())
        fp = int((~true &  pred).sum())
        fn = int(( true & ~pred).sum())
        tn = int((~true & ~pred).sum())
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    tprs.append(1.0); fprs.append(1.0)
    tprs, fprs = np.array(tprs), np.array(fprs)
    order = np.argsort(fprs)
    return float(np.trapz(tprs[order], fprs[order]))


# ── 主逻辑 ───────────────────────────────────────────────────────────────────

def collect_windows(data_dir: str, fs: int) -> pd.DataFrame:
    qr_files = sorted(glob.glob(
        os.path.join(data_dir, "**", "*_quality_report.csv"), recursive=True
    ))
    if not qr_files:
        print("未找到 *_quality_report.csv，请先运行 Step 3")
        sys.exit(1)

    rows = []
    for qr_path in qr_files:
        base      = qr_path.replace("_quality_report.csv", "")
        ref_path  = base + "_CH1-8_rpeaks.csv"
        pred_path = base + "_CH20_rpeaks.csv"

        if not os.path.exists(ref_path) or not os.path.exists(pred_path):
            print(f"  [跳过] {os.path.basename(base)}（缺少 rpeaks CSV）")
            continue

        if os.path.getsize(qr_path) == 0:
            print(f"  [跳过] {os.path.basename(base)}（quality_report 为空文件）")
            continue
        qr = pd.read_csv(qr_path)
        if qr.empty or not {"mean_ue", "mean_ua", "start_s", "end_s", "mean_uc"}.issubset(qr.columns):
            print(f"  [跳过] {os.path.basename(base)}（quality_report 缺少 mean_ue/mean_ua，"
                  f"请重新运行 Step 3）")
            continue

        ref_peaks  = pd.read_csv(ref_path)["sample_index"].values
        pred_peaks = pd.read_csv(pred_path)["sample_index"].values

        for _, w in qr.iterrows():
            f1 = window_f1(ref_peaks, pred_peaks, w["start_s"], w["end_s"], fs)
            rows.append({
                "file"    : os.path.basename(base),
                "start_s" : w["start_s"],
                "end_s"   : w["end_s"],
                "f1_gt"   : round(f1, 4),
                "mean_uc" : w["mean_uc"],
                "mean_ue" : w["mean_ue"],
                "mean_ua" : w["mean_ua"],
            })

    if not rows:
        print("没有可用数据（所有文件都缺少必要列，请重新运行 Step 3）")
        sys.exit(1)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--fs",       required=True, type=int)
    ap.add_argument("--f1_thr",   default=F1_GOOD_THR, type=float,
                    help=f"F1 阈值（窗口内 CH20 R-peak F1 >= 此值视为好窗口，默认 {F1_GOOD_THR}）")
    args = ap.parse_args()

    print(f"\n收集窗口数据（容差 {TOL_SEC*1000:.0f}ms）…")
    df = collect_windows(args.data_dir, args.fs)

    true_good = (df["f1_gt"] >= args.f1_thr).values
    n_total   = len(df)
    n_good    = int(true_good.sum())

    print(f"总窗口数 : {n_total}")
    print(f"真实好窗口 (CH20 F1 >= {args.f1_thr}): {n_good} ({n_good/n_total*100:.1f}%)")
    print(f"涉及文件 : {df['file'].nunique()} 个")

    uc_arr = df["mean_uc"].values
    ue_arr = df["mean_ue"].values
    ua_arr = df["mean_ua"].values

    # ── 方法 A：mean_uc + Otsu ──────────────────────────────────────────────
    uc_thr   = otsu_1d(uc_arr)
    pred_A   = uc_arr <= uc_thr
    mA       = binary_metrics(true_good, pred_A)
    auc_A    = auc_score(true_good, uc_arr)

    # ── 方法 B1：两阶段固定阈值（论文）────────────────────────────────────────
    pred_B1  = (ue_arr <= UE_ALPHA) & (ua_arr <= UA_BETA)
    mB1      = binary_metrics(true_good, pred_B1)
    # 连续分数：max(UE/alpha, UA/beta)，值越大越差
    score_B  = np.maximum(ue_arr / UE_ALPHA, ua_arr / UA_BETA)
    auc_B1   = auc_score(true_good, score_B)

    # ── 方法 B2：两阶段 Otsu 自适应────────────────────────────────────────────
    ue_thr   = otsu_1d(ue_arr)
    ua_thr   = otsu_1d(ua_arr)
    pred_B2  = (ue_arr <= ue_thr) & (ua_arr <= ua_thr)
    mB2      = binary_metrics(true_good, pred_B2)
    score_B2 = np.maximum(ue_arr / ue_thr, ua_arr / ua_thr)
    auc_B2   = auc_score(true_good, score_B2)

    # ── 打印对比表 ───────────────────────────────────────────────────────────
    W = 44
    print(f"\n{'─'*80}")
    print(f"{'方法':<{W}} {'AUC':>5} {'Acc':>5} {'Prec':>5} {'Rec':>5} {'F1':>5}  阈值")
    print(f"{'─'*80}")

    def row(name, m, auc, thr_str):
        print(f"{name:<{W}} {auc:>5.3f} {m['acc']:>5.3f} {m['prec']:>5.3f} "
              f"{m['rec']:>5.3f} {m['f1']:>5.3f}  {thr_str}")

    row("A  : mean(U_E+U_A) + Otsu",
        mA, auc_A, f"uc <= {uc_thr:.3f}")
    row(f"B1 : 两阶段 固定阈值 (论文 α={UE_ALPHA}, β={UA_BETA})",
        mB1, auc_B1, f"ue <= {UE_ALPHA}, ua <= {UA_BETA}")
    row(f"B2 : 两阶段 Otsu 自适应",
        mB2, auc_B2, f"ue <= {ue_thr:.3f}, ua <= {ua_thr:.3f}")
    print(f"{'─'*80}")

    print(f"\n指标含义：")
    print(f"  AUC  — 连续分数区分好/坏窗口的能力（越高越好）")
    print(f"  Prec — 标为好的窗口中真正好的比例（避免假阳性）")
    print(f"  Rec  — 真正好的窗口被标为好的比例（避免漏检）")
    print(f"  F1   — Prec 与 Rec 的调和均值")

    # ── 保存详细结果 ─────────────────────────────────────────────────────────
    df["pred_A"]  = pred_A
    df["pred_B1"] = pred_B1
    df["pred_B2"] = pred_B2
    out_path = os.path.join(args.data_dir, "quality_method_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"\n详细逐窗口结果: {out_path}")


if __name__ == "__main__":
    main()
