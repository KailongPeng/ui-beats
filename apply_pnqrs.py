#!/usr/bin/env python3
"""
apply_pnqrs.py -- 对自采 Excel ECG 文件批量检测 R-peak

支持格式：
  旧格式: timestamp + CH20
  新格式: timestamp + CH1~CH8 + CH20

用法：
  python apply_pnqrs.py --data_dir /path/to/xlsx --gpu 0
  python apply_pnqrs.py --data_dir /path/to/xlsx --fs 500
"""
import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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

FS_MODEL    = 50
WIN_SEC     = 10
OVERLAP_SEC = 2
TOL_SEC     = 0.150


@dataclass
class ECGRecord:
    fs: int
    ch_standard:       Optional[np.ndarray]
    ch_upper_arm:      Optional[np.ndarray]
    ch_names_standard: list = field(default_factory=list)
    source_file:       str  = ""


def infer_fs(ts: pd.Series) -> int:
    if pd.api.types.is_numeric_dtype(ts):
        diffs = np.diff(ts.values[:50].astype(float))
        med   = float(np.median(diffs[diffs > 0]))
        # diff < 0.1  → 秒单位（如 0.001s → 1000Hz）
        # diff >= 0.1 → 毫秒单位（如 1ms → 1000Hz，2ms → 500Hz，4ms → 250Hz）
        return round(1 / med) if med < 0.1 else round(1000 / med)
    ts_dt = pd.to_datetime(ts)
    diffs = [(ts_dt.iloc[i+1] - ts_dt.iloc[i]).total_seconds()
             for i in range(min(50, len(ts_dt)-1))]
    return round(1 / float(np.median([d for d in diffs if d > 0])))


def load_excel_ecg(path: str, fs_override: Optional[int] = None) -> ECGRecord:
    if path.endswith(".csv"):
        # engine='python' 对末尾多余逗号/空列更宽容，不会因列数不一致而崩溃
        df = pd.read_csv(path, engine="python")
    else:
        df = pd.read_excel(path)
    cols = df.columns.tolist()
    ts_col = next(
        (c for c in cols if str(c).lower() in ("timestamp", "time", "t")),
        cols[0]
    )
    fs = fs_override if fs_override else infer_fs(df[ts_col])
    std_cols = sorted(
        [c for c in cols if str(c).upper() in {f"CH{i}" for i in range(1, 9)}],
        key=lambda c: int(str(c)[2:])
    )
    upper_col    = next((c for c in cols if str(c).upper() == "CH20"), None)
    ch_standard  = df[std_cols].values.astype(np.float32)  if std_cols    else None
    ch_upper_arm = df[upper_col].values.astype(np.float32) if upper_col   else None

    print(f"[加载] {os.path.basename(path)}")
    print(f"  fs={fs}Hz  时长={len(df)/fs:.1f}s  样本={len(df)}")
    if ch_standard  is not None: print(f"  标准导联: {std_cols}")
    if ch_upper_arm is not None: print(f"  上臂导联: CH20")
    return ECGRecord(fs, ch_standard, ch_upper_arm, std_cols, str(path))


def load_model(device) -> QRSModel:
    ckpt  = torch.load(str(CKPT_PATH), map_location="cpu")
    model = QRSModel(encoder4qrs(), decoder4qrs(), phi_qrs()).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model.eval()


def _run_window(model, window_1d: np.ndarray, fs: int, device):
    # preprocess_ecg 内部的 pp() 函数设计用于 mV 单位信号（阈值 10mV）。
    # 若信号为 ADC 计数（幅度 >>10），pp() 会把整段信号抹平成常数导致检测失败。
    # 预先 z-score 使幅度落在 ±3 左右，pp() 不再触发，后续 preprocess_ecg
    # 内部会再做一次 z-score，两次叠加不影响结果。
    std = window_1d.std()
    if std > 1:   # std <= 1 说明已是 mV 量级，不需要预处理
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
    return logits, uncertain_est(logits)


def infer_single_lead(model, window_1d, fs, device):
    logits, uc = _run_window(model, window_1d, fs, device)
    r_50hz = correct(logits[0], uc)
    r_orig = np.round(np.array(r_50hz) * (fs / FS_MODEL)).astype(int)
    return r_orig, float(np.mean(uc))


def infer_multi_lead(model, windows, fs, device):
    all_logits, all_uc = [], []
    for w in windows:
        lg, uc = _run_window(model, w, fs, device)
        all_logits.append(lg)
        all_uc.append(uc)
    uc_mat   = np.stack(all_uc, axis=1)
    lo_mat   = np.stack([l[0] for l in all_logits], axis=1)
    best     = np.argmin(uc_mat, axis=1)
    T        = lo_mat.shape[0]
    logits_o = lo_mat[np.arange(T), best]
    uc_fused = uc_mat[np.arange(T), best]
    r_50hz   = correct(logits_o, uc_fused)
    r_orig   = np.round(np.array(r_50hz) * (fs / FS_MODEL)).astype(int)
    return r_orig, best, [float(np.mean(u)) for u in all_uc]


def _nms(preds: np.ndarray, tol: int) -> np.ndarray:
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


def detect(signal, fs: int, model, device, multi_lead: bool = False) -> dict:
    ws      = int(WIN_SEC * fs)
    ss      = int((WIN_SEC - OVERLAP_SEC) * fs)
    tol     = int(TOL_SEC * fs)
    sig_len = signal.shape[0]
    n_leads = signal.shape[1] if multi_lead else 1
    lead_cnt = np.zeros(n_leads, dtype=int)
    all_r    = []

    pos = 0
    while pos < sig_len:
        if multi_lead:
            chunk   = signal[pos: pos + ws]
            windows = [np.pad(chunk[:, i], (0, max(0, ws - len(chunk)))) for i in range(n_leads)]
            r_rel, best, _ = infer_multi_lead(model, windows, fs, device)
            valid = min(len(best), int((min(pos + ws, sig_len) - pos) * FS_MODEL / fs))
            np.add.at(lead_cnt, best[:valid], 1)
        else:
            w     = signal[pos: pos + ws]
            w     = np.pad(w, (0, max(0, ws - len(w))))
            r_rel, _ = infer_single_lead(model, w, fs, device)

        r_abs = r_rel + pos
        all_r.append(r_abs[r_abs < sig_len])
        pos += ss

    r_peaks = _nms(np.concatenate(all_r) if all_r else np.array([]), tol)
    r_times = r_peaks / fs
    mean_hr = float(60 / np.mean(np.diff(r_peaks) / fs)) if len(r_peaks) > 1 else float("nan")
    out = {"r_peaks": r_peaks, "r_times": r_times, "mean_hr": mean_hr}
    if multi_lead:
        total = lead_cnt.sum()
        out["lead_usage_pct"] = (lead_cnt / max(total, 1) * 100).tolist()
    return out


def process_file(path: str, model, device, fs_override=None) -> dict:
    rec    = load_excel_ecg(path, fs_override)
    result = {"file": os.path.basename(path), "fs": rec.fs}

    if rec.ch_standard is not None:
        res = detect(rec.ch_standard, rec.fs, model, device, multi_lead=True)
        result["standard_leads"] = {
            "n_beats":        int(len(res["r_peaks"])),
            "mean_hr_bpm":    round(res["mean_hr"], 1),
            "r_peaks":        res["r_peaks"].tolist(),
            "r_times_s":      res["r_times"].tolist(),
            "lead_usage_pct": {
                rec.ch_names_standard[i]: round(p, 1)
                for i, p in enumerate(res["lead_usage_pct"])
            },
        }
        usage = " | ".join(f"{k}:{v:.0f}%"
                           for k, v in result["standard_leads"]["lead_usage_pct"].items())
        print(f"  [CH1-8] beats={len(res['r_peaks'])}  HR={res['mean_hr']:.0f}bpm  {usage}")

    if rec.ch_upper_arm is not None:
        res = detect(rec.ch_upper_arm, rec.fs, model, device, multi_lead=False)
        result["upper_arm_ch20"] = {
            "n_beats":     int(len(res["r_peaks"])),
            "mean_hr_bpm": round(res["mean_hr"], 1),
            "r_peaks":     res["r_peaks"].tolist(),
            "r_times_s":   res["r_times"].tolist(),
        }
        print(f"  [CH20]  beats={len(res['r_peaks'])}  HR={res['mean_hr']:.0f}bpm")

    base = path.replace(".xlsx", "").replace(".csv", "")
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

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--gpu",  default="0")
    ap.add_argument("--fs",   default=None, type=int, help="手动指定采样率")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model  = load_model(device)
    print(f"模型就绪，设备: {device}\n{'─'*50}")

    files = sorted(
        glob.glob(os.path.join(args.data_dir, "**", "*.xlsx"), recursive=True) +
        glob.glob(os.path.join(args.data_dir, "**", "*.csv"),  recursive=True)
    )
    # 跳过脚本自身生成的结果文件
    files = [f for f in files if not any(
        f.endswith(s) for s in ("_rpeaks.csv", "_quality_report.csv",
                                "_wave_sqi.csv", "_wave_sqi_detail.csv",
                                "rpeaks_summary.json")
    )]
    print(f"找到 {len(files)} 个文件\n")

    all_results = []
    for f in files:
        print(f"\n>> {os.path.basename(f)}")
        all_results.append(process_file(f, model, device, args.fs))

    summary = os.path.join(args.data_dir, "rpeaks_summary.json")
    with open(summary, "w", encoding="utf-8") as fp:
        json.dump(all_results, fp, ensure_ascii=False, indent=2)
    print(f"\n汇总: {summary}")


if __name__ == "__main__":
    main()
