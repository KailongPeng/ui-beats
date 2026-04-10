#!/usr/bin/env python3
"""
wave_salience_calculator.py -- ECG 波形显著性 SQI 计算框架

原理：
  用 NeuroKit2 检测 P/Q/S/T 各波的峰值位置，计算各波相对于 R 波的
  幅度比值（salience score）以及检出率（detection rate）。
  综合评分 = 以检出率为权重的加权平均。

  适用于单导联可穿戴 ECG（如上臂 CH20），P 波和 T 波可能很弱。

用法：
  python wave_salience_calculator.py --csv /path/to/data.csv --fs 1000
  python wave_salience_calculator.py --csv /path/to/data.csv --fs 1000 --detail
  python wave_salience_calculator.py --batch --data_dir /path/to/data_dir --fs 1000
"""

import argparse
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import neurokit2 as nk
except ImportError:
    sys.exit("需要安装 neurokit2：pip install neurokit2")

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
MIN_BEATS       = 3        # 最少心拍数才能做有意义的 SQI 计算
SEGMENT_SEC     = 10       # 默认分析片段长度
DELINEATE_METHOD = "dwt"   # NeuroKit2 波形检测方法
SCORE_CLIP      = 1.0      # salience 分数上限

SKIP_SUFFIXES = ("_wave_sqi.csv", "_wave_sqi_detail.csv",
                 "_quality_report.csv", "_rpeaks.csv")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class SalienceResult:
    """单波（或综合）显著性结果"""
    wave_name: str
    score: float               # 0-1，幅度相对 R 波的中位数比值
    detection_rate: float      # 0-1，检出率
    n_detected: int
    n_total: int
    per_beat_scores: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# 抽象接口
# ---------------------------------------------------------------------------
class SQICalculatorRole(ABC):
    """所有 SQI 计算器的统一接口"""

    @abstractmethod
    def compute(self, ecg: np.ndarray, fs: int, **kwargs) -> dict:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ---------------------------------------------------------------------------
# 波段显著性基类
# ---------------------------------------------------------------------------
class _BaseSalienceCalculator(SQICalculatorRole):
    """波段显著性计算基类，提供振幅提取与得分映射公用方法。

    方法分工：
        delineate()              → 对 ECG 做一次完整的波形检测（静态，可复用）
        _get_wave_array()        → 从 NeuroKit2 结果中提取当前波段的峰值位置数组
        _get_amplitudes()        → 从信号中提取各峰值位置的振幅
        _filter_intervals()      → 过滤不合理的间期（如 PR > 200ms），剔除异常拍
        _compute_salience_score()→ 振幅 → 0-1 显著性分数
        _segment_by_gaps()       → 按 R-peak 间隔的不连续性把信号分段，分段评分
        compute()                → 串联以上方法的主流程
    """

    def __init__(self, wave_key: str, wave_name: str):
        self._wave_key = wave_key    # NeuroKit2 信号列名，如 "ECG_P_Peaks"
        self._wave_name = wave_name  # 人可读名，如 "P"

    @property
    def name(self) -> str:
        return self._wave_name

    # =====================================================================
    # 1. 波形检测（单次调用，各计算器可复用）
    # =====================================================================
    @staticmethod
    def delineate(ecg: np.ndarray, fs: int,
                  rpeaks: Optional[np.ndarray] = None
                  ) -> Tuple[dict, np.ndarray, np.ndarray]:
        """
        对 ECG 做一次完整的波形检测。

        返回:
            signals_dict : NeuroKit2 信号 DataFrame（含 ECG_P_Peaks 等列）
            r_peaks      : R-peak 采样点索引
            cleaned      : 清洁后的 ECG 信号
        """
        cleaned = nk.ecg_clean(ecg, sampling_rate=fs)

        if rpeaks is None:
            _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
            r_peaks = info["ECG_R_Peaks"]
        else:
            r_peaks = rpeaks

        if len(r_peaks) < MIN_BEATS:
            return {}, r_peaks, cleaned

        signals, _waves = nk.ecg_delineate(
            cleaned, r_peaks, sampling_rate=fs, method=DELINEATE_METHOD,
            show=False
        )
        return signals, r_peaks, cleaned

    # =====================================================================
    # 2. _get_wave_array：从 NeuroKit2 结果中提取当前波段的峰值位置
    # =====================================================================
    def _get_wave_array(self, signals_dict: dict,
                        r_peaks: np.ndarray,
                        cleaned: np.ndarray,
                        fs: int) -> np.ndarray:
        """
        返回 per-beat 峰值位置数组（长度 = len(r_peaks)），
        未检出的拍对应 NaN。
        """
        if self._wave_key not in signals_dict:
            return np.full(len(r_peaks), np.nan)

        peak_col = signals_dict[self._wave_key]
        if isinstance(peak_col, pd.Series):
            peak_col = peak_col.values

        # NeuroKit2 可能返回 0/1 mask 或 索引列表，统一处理
        if len(peak_col) == len(cleaned):
            wave_positions = np.where(peak_col == 1)[0]
        else:
            wave_positions = np.array(peak_col, dtype=float)
            wave_positions = wave_positions[~np.isnan(wave_positions)]

        return self._match_to_beats(r_peaks, wave_positions.astype(int),
                                    max_dist_samples=int(fs * 0.5))

    # =====================================================================
    # 3. _get_amplitudes：从信号中提取各峰值位置的振幅
    # =====================================================================
    @staticmethod
    def _get_amplitudes(ecg: np.ndarray,
                        peak_indices: np.ndarray) -> np.ndarray:
        """给定峰值位置数组（可含 NaN），返回对应振幅（NaN 保留）。"""
        amps = np.full(len(peak_indices), np.nan, dtype=np.float64)
        for i, idx in enumerate(peak_indices):
            if np.isnan(idx):
                continue
            idx_int = int(idx)
            if 0 <= idx_int < len(ecg):
                amps[i] = ecg[idx_int]
        return amps

    # =====================================================================
    # 4. _filter_intervals：过滤生理不合理的间期，剔除异常拍
    # =====================================================================
    def _filter_intervals(self, per_beat_peaks: np.ndarray,
                          r_peaks: np.ndarray,
                          fs: int) -> np.ndarray:
        """
        过滤不合理的波段-R 峰间期。
        例如 P-R 间期应在 80-300ms，T 峰距 R 应在 100-500ms。
        超出范围的拍标记为 NaN（视为未检出）。

        子类可覆盖 _interval_range_ms() 定制范围。
        """
        lo_ms, hi_ms = self._interval_range_ms()
        if lo_ms is None:
            return per_beat_peaks   # 无需过滤

        filtered = per_beat_peaks.copy()
        for i, (wp, rp) in enumerate(zip(per_beat_peaks, r_peaks)):
            if np.isnan(wp):
                continue
            interval_ms = abs(wp - rp) / fs * 1000
            if interval_ms < lo_ms or interval_ms > hi_ms:
                filtered[i] = np.nan
        return filtered

    def _interval_range_ms(self) -> Tuple[Optional[float], Optional[float]]:
        """返回 (min_ms, max_ms) 或 (None, None) 表示不过滤。
        子类按各波段的生理范围覆盖。"""
        return (None, None)

    # =====================================================================
    # 5. _compute_salience_score：振幅 → 0-1 分数
    # =====================================================================
    @staticmethod
    def _compute_salience_score(wave_amps: np.ndarray,
                                r_amp_ref: float
                                ) -> Tuple[float, np.ndarray]:
        """
        per-beat 分数 = |wave_amp| / r_amp_ref（clip 到 [0, SCORE_CLIP]）
        汇总分数 = per-beat 有效值的中位数
        """
        valid = ~np.isnan(wave_amps)
        per_beat = np.full_like(wave_amps, np.nan)
        if valid.sum() == 0:
            return 0.0, per_beat
        per_beat[valid] = np.clip(
            np.abs(wave_amps[valid]) / (r_amp_ref + 1e-9),
            0, SCORE_CLIP
        )
        score = float(np.nanmedian(per_beat))
        return score, per_beat

    # =====================================================================
    # 6. _segment_by_gaps：按 R-peak 间隔的不连续性分段
    # =====================================================================
    @staticmethod
    def _segment_by_gaps(r_peaks: np.ndarray, fs: int,
                         max_gap_sec: float = 3.0
                         ) -> List[np.ndarray]:
        """
        按相邻 R-peak 间隔把心拍分成连续段。
        间隔 > max_gap_sec 处切开。返回心拍索引的分段列表。

        用途：长信号中间可能有电极脱落产生的空洞，分段后只在
        连续段内计算 SQI，避免空洞干扰统计。
        """
        if len(r_peaks) < 2:
            return [np.arange(len(r_peaks))]

        rr = np.diff(r_peaks) / fs
        splits = np.where(rr > max_gap_sec)[0] + 1  # 间隔处切开
        return np.split(np.arange(len(r_peaks)), splits)

    # =====================================================================
    # 7. compute：主流程
    # =====================================================================
    def compute(self, ecg: np.ndarray, fs: int, **kwargs) -> SalienceResult:
        """
        计算该波段的显著性。

        可选 kwargs（避免重复波形检测）：
            signals_dict, r_peaks, cleaned, r_amp_ref
        """
        signals_dict = kwargs.get("signals_dict")
        r_peaks = kwargs.get("r_peaks")
        cleaned = kwargs.get("cleaned")
        r_amp_ref = kwargs.get("r_amp_ref")

        # 若未提供预计算结果，自行做波形检测
        if signals_dict is None or r_peaks is None or cleaned is None:
            signals_dict, r_peaks, cleaned = self.delineate(ecg, fs)

        n_total = len(r_peaks)
        if n_total < MIN_BEATS:
            return SalienceResult(self._wave_name, 0.0, 0.0, 0, n_total)

        # R 波幅度基准
        if r_amp_ref is None:
            r_amp_ref = float(np.median(np.abs(cleaned[r_peaks])))

        # Step A: 提取该波段的 per-beat 峰值位置
        per_beat_peaks = self._get_wave_array(signals_dict, r_peaks,
                                              cleaned, fs)

        # Step B: 过滤生理不合理的间期
        per_beat_peaks = self._filter_intervals(per_beat_peaks, r_peaks, fs)

        # Step C: 统计检出率
        n_detected = int(np.sum(~np.isnan(per_beat_peaks)))
        detection_rate = n_detected / n_total if n_total > 0 else 0.0

        # Step D: 提取振幅
        wave_amps = self._get_amplitudes(cleaned, per_beat_peaks)

        # Step E: 计算显著性分数
        score, per_beat_scores = self._compute_salience_score(
            wave_amps, r_amp_ref)

        return SalienceResult(
            wave_name=self._wave_name,
            score=round(score, 4),
            detection_rate=round(detection_rate, 4),
            n_detected=n_detected,
            n_total=n_total,
            per_beat_scores=per_beat_scores,
        )

    # =====================================================================
    # 内部辅助
    # =====================================================================
    @staticmethod
    def _match_to_beats(r_peaks: np.ndarray,
                        wave_positions: np.ndarray,
                        max_dist_samples: int) -> np.ndarray:
        """
        为每个 R-peak 匹配最近的波形峰值位置。
        距离 > max_dist_samples → NaN（该拍未检出此波）。
        """
        result = np.full(len(r_peaks), np.nan)
        if len(wave_positions) == 0:
            return result
        for i, r in enumerate(r_peaks):
            dists = np.abs(wave_positions.astype(float) - r)
            j_min = int(np.argmin(dists))
            if dists[j_min] <= max_dist_samples:
                result[i] = wave_positions[j_min]
        return result


# ---------------------------------------------------------------------------
# 各波段计算器
# ---------------------------------------------------------------------------
class PWaveSalienceCalculator(_BaseSalienceCalculator):
    """P波显著性计算器：衡量P波相对于R波的突出程度及检出质量。

    生理约束：P 波应出现在 R 峰之前 80–300ms。
    上臂导联 P 波通常很弱（salience 0.05-0.15），低检出率是正常现象。"""
    def __init__(self):
        super().__init__("ECG_P_Peaks", "P")

    def _interval_range_ms(self):
        return (80, 300)   # P 在 R 前 80-300ms


class QWaveSalienceCalculator(_BaseSalienceCalculator):
    """Q波显著性计算器：Q 波是 QRS 起始的小向下偏转。

    生理约束：Q 波距 R 峰 10–80ms。"""
    def __init__(self):
        super().__init__("ECG_Q_Peaks", "Q")

    def _interval_range_ms(self):
        return (10, 80)


class SWaveSalienceCalculator(_BaseSalienceCalculator):
    """S波显著性计算器：S 波是 R 波之后的向下偏转。

    生理约束：S 波距 R 峰 10–80ms。"""
    def __init__(self):
        super().__init__("ECG_S_Peaks", "S")

    def _interval_range_ms(self):
        return (10, 80)


class TWaveSalienceCalculator(_BaseSalienceCalculator):
    """T波显著性计算器：T 波是心室复极波，幅度仅次于 QRS。

    生理约束：T 峰距 R 峰 100–500ms。
    上臂导联 T 波可能减弱（salience 0.15-0.35），检出率应 > 80%。"""
    def __init__(self):
        super().__init__("ECG_T_Peaks", "T")

    def _interval_range_ms(self):
        return (100, 500)


# ---------------------------------------------------------------------------
# 综合计算器
# ---------------------------------------------------------------------------
class WaveSalienceCalculator(_BaseSalienceCalculator):
    """波形显著性综合指标：汇总 P/Q/S/T 各段相对于 R 波的显著性得分，
    以各子指标的置信度（检出率）作为权重进行加权平均。"""

    def __init__(self):
        # 不调用 super().__init__，因为自己不对应单一波段
        self._calculators = [
            PWaveSalienceCalculator(),
            QWaveSalienceCalculator(),
            SWaveSalienceCalculator(),
            TWaveSalienceCalculator(),
        ]

    @property
    def name(self) -> str:
        return "composite"

    def compute(self, ecg: np.ndarray, fs: int, **kwargs) -> dict:
        """
        一次波形检测，四路评分，加权汇总。

        返回:
            {
              "composite_salience": float,
              "wave_results": {"P": SalienceResult, "Q": ..., "S": ..., "T": ...},
              "n_beats": int,
              "r_amplitude_median": float,
              "status": "ok" | "delineation_failed" | "insufficient_beats",
            }
        """
        # Step 1：波形检测（单次）
        try:
            signals_dict, r_peaks, cleaned = _BaseSalienceCalculator.delineate(
                ecg, fs, rpeaks=kwargs.get("rpeaks"))
        except Exception as e:
            print(f"  [WARN] 波形检测失败: {e}")
            return self._empty_result("delineation_failed")

        n_beats = len(r_peaks)
        if n_beats < MIN_BEATS:
            return self._empty_result("insufficient_beats", n_beats=n_beats)

        # Step 2：R 波幅度基准
        r_amp = float(np.median(np.abs(cleaned[r_peaks])))

        # Step 3：各波评分（复用波形检测结果）
        wave_results: Dict[str, SalienceResult] = {}
        for calc in self._calculators:
            res = calc.compute(ecg, fs,
                               signals_dict=signals_dict,
                               r_peaks=r_peaks,
                               cleaned=cleaned,
                               r_amp_ref=r_amp)
            wave_results[calc.name] = res

        # Step 4：加权综合
        scores = [r.score for r in wave_results.values()]
        weights = [r.detection_rate for r in wave_results.values()]
        total_w = sum(weights)
        composite = (sum(s * w for s, w in zip(scores, weights)) / total_w
                     if total_w > 0 else 0.0)

        return {
            "composite_salience": round(composite, 4),
            "wave_results": wave_results,
            "n_beats": n_beats,
            "r_amplitude_median": round(r_amp, 4),
            "status": "ok",
        }

    @staticmethod
    def _empty_result(status: str = "delineation_failed",
                      n_beats: int = 0) -> dict:
        empty = SalienceResult("—", 0.0, 0.0, 0, n_beats)
        return {
            "composite_salience": 0.0,
            "wave_results": {"P": empty, "Q": empty, "S": empty, "T": empty},
            "n_beats": n_beats,
            "r_amplitude_median": 0.0,
            "status": status,
        }


# ---------------------------------------------------------------------------
# 文件 I/O（与 extract_quality_segments.py 保持一致）
# ---------------------------------------------------------------------------
def load_signal(csv_path: str):
    """读取文件，返回 (signal, df) 或 (None, None) 若无 CH20 列。"""
    if csv_path.endswith(".csv"):
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_excel(csv_path)
    upper_col = next((c for c in df.columns if str(c).upper() == "CH20"), None)
    if upper_col is None:
        return None, None
    return df[upper_col].values.astype(np.float32), df


def _should_skip(path: str) -> bool:
    return any(path.endswith(s) for s in SKIP_SUFFIXES)


# ---------------------------------------------------------------------------
# 单文件处理
# ---------------------------------------------------------------------------
def process_one_file(csv_path: str, fs: int,
                     segment_sec: int = SEGMENT_SEC,
                     detail: bool = False,
                     out_dir: Optional[str] = None) -> Optional[dict]:
    """
    对单个 CSV/Excel 文件做波形显著性分析。

    将信号按 segment_sec 切片，对每片计算 SQI，取中位数汇总。
    返回汇总 dict 或 None（跳过）。
    """
    signal, _ = load_signal(csv_path)
    if signal is None:
        print(f"  [跳过] 未找到 CH20 列：{csv_path}")
        return None

    duration_s = len(signal) / fs
    seg_len = segment_sec * fs
    calculator = WaveSalienceCalculator()

    # 按片段切分
    segment_results = []
    start = 0
    while start + seg_len <= len(signal):
        seg = signal[start: start + seg_len]
        res = calculator.compute(seg, fs)
        if res["status"] == "ok":
            res["start_s"] = start / fs
            res["end_s"] = (start + seg_len) / fs
            segment_results.append(res)
        start += seg_len

    # 处理尾部不足整段的片段（至少 5 秒）
    if start < len(signal) and (len(signal) - start) >= fs * 5:
        seg = signal[start:]
        res = calculator.compute(seg, fs)
        if res["status"] == "ok":
            res["start_s"] = start / fs
            res["end_s"] = len(signal) / fs
            segment_results.append(res)

    if not segment_results:
        print(f"  [跳过] 无可分析片段：{csv_path}")
        return None

    # 汇总（中位数）
    def median_of(key):
        vals = [r[key] for r in segment_results]
        return round(float(np.median(vals)), 4)

    def wave_median(wave_name, attr):
        vals = [r["wave_results"][wave_name].__dict__[attr]
                for r in segment_results]
        return round(float(np.median(vals)), 4)

    summary = {
        "file": os.path.basename(csv_path),
        "duration_s": round(duration_s, 1),
        "n_segments": len(segment_results),
        "composite_salience": median_of("composite_salience"),
        "r_amplitude_median": median_of("r_amplitude_median"),
    }
    for w in ["P", "Q", "S", "T"]:
        summary[f"{w}_salience"] = wave_median(w, "score")
        summary[f"{w}_detection_rate"] = wave_median(w, "detection_rate")

    # 终端输出
    base = os.path.basename(csv_path)
    print(f"\n>> {base}")
    print(f"   fs={fs}Hz  duration={duration_s:.1f}s  segments={len(segment_results)}")
    for w in ["P", "Q", "S", "T"]:
        s = summary[f"{w}_salience"]
        d = summary[f"{w}_detection_rate"]
        n_det = sum(r["wave_results"][w].n_detected for r in segment_results)
        n_tot = sum(r["wave_results"][w].n_total for r in segment_results)
        print(f"   {w}: salience={s:.3f}  detection={d*100:.1f}%  ({n_det}/{n_tot})")
    print(f"   composite={summary['composite_salience']:.3f}")

    # 保存汇总 CSV
    save_dir = out_dir or os.path.dirname(csv_path)
    os.makedirs(save_dir, exist_ok=True)
    stem = Path(csv_path).stem
    summary_path = os.path.join(save_dir, f"{stem}_wave_sqi.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"   → {summary_path}")

    # 保存逐段明细
    if detail:
        rows = []
        for r in segment_results:
            row = {"start_s": r["start_s"], "end_s": r["end_s"],
                   "n_beats": r["n_beats"],
                   "composite": r["composite_salience"]}
            for w in ["P", "Q", "S", "T"]:
                wr = r["wave_results"][w]
                row[f"{w}_salience"] = wr.score
                row[f"{w}_detection"] = wr.detection_rate
            rows.append(row)
        detail_path = os.path.join(save_dir, f"{stem}_wave_sqi_detail.csv")
        pd.DataFrame(rows).to_csv(detail_path, index=False)
        print(f"   → {detail_path}")

    return summary


# ---------------------------------------------------------------------------
# 批量处理
# ---------------------------------------------------------------------------
def process_batch(data_dir: str, fs: int,
                  segment_sec: int = SEGMENT_SEC,
                  detail: bool = False,
                  out_dir: str | None = None):
    """递归扫描目录下所有 CSV/Excel 文件，按行为分组汇总。"""
    import glob as _glob
    from collections import defaultdict

    patterns = ["**/*.csv", "**/*.xlsx", "**/*.xls"]
    all_files = []
    for pat in patterns:
        all_files.extend(_glob.glob(os.path.join(data_dir, pat), recursive=True))
    all_files = sorted(set(f for f in all_files if not _should_skip(f)))

    if not all_files:
        print(f"在 {data_dir} 下未找到 CSV/Excel 文件。")
        return

    print(f"找到 {len(all_files)} 个文件，开始分析…\n")

    # 按行为（第一级子目录）分组
    activity_results = defaultdict(list)
    all_summaries = []

    for fpath in all_files:
        rel = os.path.relpath(fpath, data_dir)
        parts = Path(rel).parts
        activity = parts[0] if len(parts) > 1 else "root"

        summary = process_one_file(fpath, fs, segment_sec, detail)
        if summary is None:
            continue
        summary["activity"] = activity
        summary["rel_path"] = rel
        activity_results[activity].append(summary)
        all_summaries.append(summary)

    if not all_summaries:
        print("\n无可分析文件。")
        return

    # 汇总表
    print("\n" + "─" * 90)
    header = (f"{'activity':<15} {'file':<20} {'dur(s)':>7} {'segs':>5} "
              f"{'P_sal':>6} {'P_det%':>6} {'T_sal':>6} {'T_det%':>6} "
              f"{'comp':>6}")
    print(header)
    print("─" * 90)

    for act in sorted(activity_results.keys()):
        items = activity_results[act]
        # 行为小计
        p_sal = np.median([s["P_salience"] for s in items])
        p_det = np.median([s["P_detection_rate"] for s in items])
        t_sal = np.median([s["T_salience"] for s in items])
        t_det = np.median([s["T_detection_rate"] for s in items])
        comp = np.median([s["composite_salience"] for s in items])
        dur = sum(s["duration_s"] for s in items)
        print(f"{act:<15} {'':20} {dur:>7.0f} {'':>5} "
              f"{p_sal:>6.3f} {p_det*100:>5.1f}% {t_sal:>6.3f} {t_det*100:>5.1f}% "
              f"{comp:>6.3f}")
        for s in items:
            print(f"{'':15} {'└ ' + s['file']:<20} {s['duration_s']:>7.1f} "
                  f"{s['n_segments']:>5} "
                  f"{s['P_salience']:>6.3f} {s['P_detection_rate']*100:>5.1f}% "
                  f"{s['T_salience']:>6.3f} {s['T_detection_rate']*100:>5.1f}% "
                  f"{s['composite_salience']:>6.3f}")

    print("─" * 90)
    comp_all = np.median([s["composite_salience"] for s in all_summaries])
    print(f"{'TOTAL':<15} {'':20} "
          f"{sum(s['duration_s'] for s in all_summaries):>7.0f} "
          f"{sum(s['n_segments'] for s in all_summaries):>5} "
          f"{'':>6} {'':>6} {'':>6} {'':>6} {comp_all:>6.3f}")

    # 保存批量汇总 CSV
    save_dir = out_dir or data_dir
    os.makedirs(save_dir, exist_ok=True)
    batch_csv = os.path.join(save_dir, "batch_wave_sqi_summary.csv")
    pd.DataFrame(all_summaries).to_csv(batch_csv, index=False)
    print(f"\n→ {batch_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="ECG 波形显著性 SQI 计算（P/Q/S/T 相对 R 波的幅度比值）")
    ap.add_argument("--csv",         type=str, help="单文件：CSV/Excel 路径")
    ap.add_argument("--data_dir",    type=str, help="批量模式：根目录路径")
    ap.add_argument("--batch",       action="store_true", help="开启批量模式")
    ap.add_argument("--fs",          type=int, required=True, help="采样率 Hz")
    ap.add_argument("--segment_sec", type=int, default=SEGMENT_SEC,
                    help=f"分析片段长度（秒），默认 {SEGMENT_SEC}s")
    ap.add_argument("--detail",      action="store_true",
                    help="输出逐片段明细 CSV")
    ap.add_argument("--out_dir",     type=str, default=None,
                    help="输出目录（不指定则放在 CSV 旁）")
    args = ap.parse_args()

    if args.batch:
        if not args.data_dir:
            ap.error("批量模式需要 --data_dir")
        process_batch(args.data_dir, args.fs, args.segment_sec, args.detail, args.out_dir)
    else:
        if not args.csv:
            ap.error("单文件模式需要 --csv")
        process_one_file(args.csv, args.fs, args.segment_sec,
                         args.detail, args.out_dir)


if __name__ == "__main__":
    main()
