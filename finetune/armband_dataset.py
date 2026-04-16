#!/usr/bin/env python3
"""
armband_dataset.py -- CH20 上臂导联微调数据集

数据源：
  <data_dir>/<subject>/<activity>/rec*.csv
  <data_dir>/<subject>/<activity>/rec*_CH1-8_rpeaks.csv     (教师伪标签，由 apply_pnqrs.py 产生)
  <data_dir>/<subject>/<activity>/rec*_quality_report.csv   (窗口质量，由 extract_quality_segments.py 产生)

每个样本 = 一个 is_good=True 的 10 秒窗口：
  - CH20 信号 → preprocess_ecg → (1, T_200hz) tensor
  - CH1-8 伪 R 峰 截取到该窗口 → r_peaks_to_mask → (T_50hz,) mask
  - 元信息：subject, activity, file, start_s
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

PNQRS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PNQRS_ROOT))

from dataset.dataset import preprocess_ecg, r_peaks_to_mask


WIN_SEC = 10
MASK_RATE = 50


def _read_csv_robust(path: Path) -> pd.DataFrame:
    """与 pipeline/extract_quality_segments.py 相同的截断行尾空列的读取方式。"""
    import io
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


def scan_records(data_dir: Path, subjects: Optional[List[str]] = None) -> List[dict]:
    """
    扫描 <data_dir>/<subject>/<activity>/rec*.csv，返回 record 描述列表。

    每个 record 是一个原始 CSV 文件（例如 rec01.csv），聚合了路径信息。
    subjects=None 表示所有被试；否则只保留指定被试目录名。
    """
    data_dir = Path(data_dir).resolve()
    records: List[dict] = []

    for csv_path in sorted(data_dir.glob("*/*/*.csv")):
        name = csv_path.name
        # 跳过脚本产生的辅助 CSV
        skip_suffixes = (
            "_quality_report.csv", "_CH1-8_rpeaks.csv", "_CH20_rpeaks.csv",
            "_wave_sqi.csv",
        )
        if any(name.endswith(s) for s in skip_suffixes):
            continue
        if name in {"batch_quality_summary.csv", "batch_wave_sqi_summary.csv"}:
            continue

        rel_parts = csv_path.relative_to(data_dir).parts
        if len(rel_parts) < 3:
            continue
        subject, activity = rel_parts[0], rel_parts[1]
        if subjects is not None and subject not in subjects:
            continue

        stem = csv_path.stem
        rpeaks_path  = csv_path.with_name(f"{stem}_CH1-8_rpeaks.csv")
        quality_path = csv_path.with_name(f"{stem}_quality_report.csv")

        if not rpeaks_path.exists() or not quality_path.exists():
            continue

        records.append(dict(
            csv_path=csv_path,
            rpeaks_path=rpeaks_path,
            quality_path=quality_path,
            subject=subject,
            activity=activity,
            stem=stem,
        ))

    return records


class ArmbandWindowDataset(Dataset):
    """
    以"单个高质量 10 秒窗口"为样本粒度。

    初始化时读一次每个 record 的 signal / rpeaks / quality_report，展开成窗口列表；
    __getitem__ 时从已缓存的信号里切片，避免重复 IO。
    """

    def __init__(
        self,
        data_dir: Path,
        fs: int,
        subjects: Optional[List[str]] = None,
        filter_quality: bool = True,
        verbose: bool = True,
    ) -> None:
        self.fs = fs
        self.filter_quality = filter_quality

        records = scan_records(Path(data_dir), subjects=subjects)
        if not records:
            raise RuntimeError(
                f"No valid records under {data_dir}"
                + (f" for subjects={subjects}" if subjects else "")
            )

        self.records = records
        self._signals: List[np.ndarray] = []
        self._windows: List[dict] = []   # 每个元素：dict(rec_idx, start_samp, end_samp, r_peaks, meta...)

        win_samp = int(WIN_SEC * fs)
        n_good = n_skip = 0

        for rec in records:
            df_sig = _read_csv_robust(rec["csv_path"])
            col = next((c for c in df_sig.columns if str(c).upper() == "CH20"), None)
            if col is None:
                if verbose:
                    print(f"[skip] 无 CH20 列: {rec['csv_path'].name}")
                continue
            signal = df_sig[col].values.astype(np.float32)
            self._signals.append(signal)
            sig_idx = len(self._signals) - 1   # 当前信号在 _signals 中的下标

            r_peaks = pd.read_csv(rec["rpeaks_path"])["sample_index"].values.astype(int)
            qr = pd.read_csv(rec["quality_path"])

            for _, row in qr.iterrows():
                is_good = bool(row["is_good"]) if "is_good" in qr.columns else True
                if filter_quality and not is_good:
                    n_skip += 1
                    continue

                s, e = int(row["start_samp"]), int(row["end_samp"])
                if e - s != win_samp or s < 0 or e > len(signal):
                    n_skip += 1
                    continue

                # 截取到窗口内的 r_peaks，转为相对窗口起点的索引
                in_win = r_peaks[(r_peaks >= s) & (r_peaks < e)]
                if len(in_win) < 3:            # 极端情况：窗口内几乎没检到 R 峰，跳过
                    n_skip += 1
                    continue

                self._windows.append(dict(
                    rec_idx=sig_idx,
                    start_samp=s,
                    end_samp=e,
                    r_peaks_rel=(in_win - s).astype(int),
                    subject=rec["subject"],
                    activity=rec["activity"],
                    stem=rec["stem"],
                ))
                n_good += 1

        if verbose:
            print(f"[ArmbandWindowDataset] subjects={subjects or 'ALL'}  "
                  f"records={len(records)}  windows_kept={n_good}  windows_skipped={n_skip}")

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int):
        w = self._windows[idx]
        signal = self._signals[w["rec_idx"]][w["start_samp"]: w["end_samp"]]

        # z-score（与 extract_quality_segments 一致）
        std = signal.std()
        if std > 1:
            signal = (signal - signal.mean()) / std

        processed = preprocess_ecg(signal.astype(np.float32), self.fs)
        if processed.ndim == 1:
            processed = processed[np.newaxis, :]
        elif processed.shape[0] > processed.shape[1]:
            processed = processed.T
        signal_tensor = torch.from_numpy(processed.astype(np.float32))   # (1, T_200hz)

        mask_np = r_peaks_to_mask(
            r_peaks=w["r_peaks_rel"],
            fs=self.fs,
            signal_length=w["end_samp"] - w["start_samp"],
            mask_sampling_rate=MASK_RATE,
        )
        mask_tensor = torch.from_numpy(mask_np.astype(np.float32))       # (T_50hz,)

        return {
            "signal": signal_tensor,
            "mask":   mask_tensor,
            "subject":  w["subject"],
            "activity": w["activity"],
            "stem":     w["stem"],
            "start_s":  w["start_samp"] / self.fs,
        }


def armband_collate_fn(batch):
    signals = torch.stack([b["signal"] for b in batch])
    masks   = torch.stack([b["mask"]   for b in batch])
    meta    = [{k: b[k] for k in ("subject", "activity", "stem", "start_s")} for b in batch]
    return signals, masks, meta


__all__ = ["ArmbandWindowDataset", "armband_collate_fn", "scan_records"]


if __name__ == "__main__":
    # 自测：扫一遍 dummy 数据，统计窗口数
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--fs", type=int, required=True)
    ap.add_argument("--subject", default=None, help="只加载此被试；留空则全部")
    args = ap.parse_args()

    subj = [args.subject] if args.subject else None
    ds = ArmbandWindowDataset(Path(args.data_dir), fs=args.fs, subjects=subj, verbose=True)
    print(f"\nTotal windows: {len(ds)}")
    if len(ds):
        s = ds[0]
        print(f"signal: {tuple(s['signal'].shape)}  mask: {tuple(s['mask'].shape)}")
        print(f"subject={s['subject']}  activity={s['activity']}  stem={s['stem']}  start_s={s['start_s']}")
        print(f"mask positives: {(s['mask'] > 0).sum().item()}")
