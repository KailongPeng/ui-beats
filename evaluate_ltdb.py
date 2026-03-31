"""
LTDB (MIT-BIH Long-Term ECG Database) Single-Lead QRS Evaluation
=================================================================
Tests PN-QRS on 24-hour Holter data — a dataset NOT evaluated in the paper.
Comparison baseline: MIT-BIH v2 result (F1=99.69%).

Key differences from MIT-BIH / INCART:
  - LTDB sampling rate: 128 Hz (vs 360 / 257)
  - Record duration: 20-24 hours (vs 30 min)
  - Memory-safe chunk processing (1-hour chunks)
  - Single lead only (lead 0)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import wfdb

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from dataset.dataset import preprocess_ecg
from models.qrs_model import QRSModel
from models.multi_head import encoder4qrs, decoder4qrs, phi_qrs
from utils.qrs_post_process import uncertain_est, correct

# constants
LTDB_DIR   = Path("/home/kailong/ECG/ECG/data/PN-QRS/LTDB")
CKPT_PATH  = Path("/home/kailong/ECG/ECG/ECGFounder/PN-QRS/experiments"
                  "/logs_real/zy2lki18/models/best_model.pt")
LTDB_FS    = 128
FS_MODEL   = 50
WIN_SEC    = 10
OVERLAP_SEC = 2
WIN_SAMP   = LTDB_FS * WIN_SEC           # 1280 samples
STRIDE     = (WIN_SEC - OVERLAP_SEC) * LTDB_FS  # 1024 samples
TOL_SAMP   = int(0.150 * LTDB_FS)       # 19 samples @ 128Hz
CHUNK_SEC  = 3600                        # process 1 hour at a time
CHUNK_SAMP = LTDB_FS * CHUNK_SEC
BEAT_SYMS  = set("NLRBaAJSVrFejnE/fQ!|")


def load_model(device):
    ckpt  = torch.load(CKPT_PATH, map_location="cpu")
    model = QRSModel(encoder4qrs(), decoder4qrs(), phi_qrs()).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def run_model_get_peaks(model, window_1d: np.ndarray, fs: int, device):
    """Return R-peak positions at original fs, relative to window start."""
    processed = preprocess_ecg(window_1d.astype(np.float32), fs=fs)
    if processed.ndim == 1:
        processed = processed[np.newaxis, :]
    elif processed.shape[0] > processed.shape[1]:
        processed = processed.T
    sig_t = torch.from_numpy(processed).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(sig_t, return_projection=True)
        logits = logits.squeeze(-1).squeeze(0).cpu().numpy()  # (3, T_model)
    uc   = uncertain_est(logits)
    prob = logits[0]
    r_model = correct(prob, uc)
    if len(r_model) == 0:
        return []
    return np.round(np.array(r_model) * (fs / FS_MODEL)).astype(int).tolist()


def deduplicate_peaks(peaks: list, tol: int) -> list:
    if not peaks:
        return []
    peaks = sorted(peaks)
    result = [peaks[0]]
    for p in peaks[1:]:
        if p - result[-1] > tol:
            result.append(p)
    return result


def match_peaks(pred: list, ref: list, tol: int):
    """Greedy match; each ref peak matched at most once."""
    matched_ref  = set()
    matched_pred = set()
    pred_arr = np.array(pred, dtype=int) if pred else np.array([], dtype=int)
    for j, r in enumerate(ref):
        if len(pred_arr) == 0:
            break
        dists = np.abs(pred_arr - r)
        hits  = [h for h in np.where(dists <= tol)[0] if h not in matched_pred]
        if hits:
            best = min(hits, key=lambda h: dists[h])
            matched_ref.add(j)
            matched_pred.add(best)
    TP = len(matched_ref)
    FP = len(pred) - len(matched_pred)
    FN = len(ref)  - len(matched_ref)
    return TP, FP, FN


def process_record(rec_id: str, model, device):
    rec_path = str(LTDB_DIR / rec_id)

    hdr = wfdb.rdheader(rec_path)
    total_samp = hdr.sig_len
    fs         = int(hdr.fs)
    duration_h = total_samp / fs / 3600

    try:
        ann      = wfdb.rdann(rec_path, "atr")
        ref_peaks = [s for s, sym in zip(ann.sample, ann.symbol)
                     if sym in BEAT_SYMS]
    except Exception as e:
        print(f"  [SKIP] {rec_id}: no atr annotation ({e})", flush=True)
        return None

    all_pred = []
    chunk_idx = 0
    for chunk_start in range(0, total_samp, CHUNK_SAMP):
        chunk_end = min(chunk_start + CHUNK_SAMP + WIN_SAMP, total_samp)
        try:
            rec = wfdb.rdrecord(rec_path,
                                sampfrom=chunk_start,
                                sampto=chunk_end,
                                channels=[0])
        except Exception as e:
            print(f"  [WARN] chunk {chunk_idx} read error: {e}", flush=True)
            chunk_idx += 1
            continue
        sig = rec.p_signal[:, 0].astype(np.float32)

        for win_start in range(0, len(sig) - WIN_SAMP + 1, STRIDE):
            abs_start = chunk_start + win_start
            # skip windows that belong to the next chunk's overlap zone
            if (abs_start + WIN_SAMP > chunk_start + CHUNK_SAMP
                    and chunk_start + CHUNK_SAMP < total_samp):
                continue
            window = sig[win_start: win_start + WIN_SAMP]
            peaks_local = run_model_get_peaks(model, window, fs, device)
            all_pred.extend([p + abs_start for p in peaks_local])

        chunk_idx += 1
        # Progress: print every hour
        hours_done = (chunk_start + CHUNK_SAMP) / fs / 3600
        tp_so_far = sum(
            1 for r in ref_peaks
            if any(abs(p - r) <= TOL_SAMP for p in all_pred)
        ) if chunk_idx % 4 == 0 else -1
        if tp_so_far >= 0:
            print(f"    {rec_id} chunk {chunk_idx}: "
                  f"{min(hours_done, duration_h):.0f}/{duration_h:.0f}h  "
                  f"pred_so_far={len(all_pred)}", flush=True)

    pred_peaks = deduplicate_peaks(sorted(all_pred), tol=TOL_SAMP)
    TP, FP, FN = match_peaks(pred_peaks, ref_peaks, tol=TOL_SAMP)
    return TP, FP, FN, len(ref_peaks), duration_h


def get_records():
    """Return record stems with both .dat and .atr files."""
    records = []
    for hea in sorted(LTDB_DIR.glob("*.hea")):
        stem = hea.stem
        if (LTDB_DIR / (stem + ".dat")).exists():
            records.append(stem)
    return records


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", nargs="+", default=None,
                        help="Record IDs to process (default: all)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"LTDB dir: {LTDB_DIR}")
    print(f"WIN={WIN_SEC}s  STRIDE={WIN_SEC-OVERLAP_SEC}s  "
          f"OVERLAP={OVERLAP_SEC}s  TOL=150ms={TOL_SAMP}samp @ {LTDB_FS}Hz")
    print(f"Chunk size: {CHUNK_SEC}s = 1 hour\n")

    model   = load_model(device)
    all_records = get_records()
    if args.records:
        records = [r for r in args.records if r in all_records]
        missing = [r for r in args.records if r not in all_records]
        if missing:
            print(f"  [WARN] records not found / missing .dat: {missing}")
    else:
        records = all_records
    print(f"Records with .dat: {len(all_records)} total, processing {len(records)}: {records}\n")

    total_TP = total_FP = total_FN = 0

    for rec_id in records:
        print(f"  Processing {rec_id}...", flush=True)
        result = process_record(rec_id, model, device)
        if result is None:
            continue
        TP, FP, FN, n_ref, dur_h = result
        total_TP += TP
        total_FP += FP
        total_FN += FN
        Se_r = TP / (n_ref + 1e-9)
        Pp_r = TP / (TP + FP + 1e-9)
        F1_r = 2 * Se_r * Pp_r / (Se_r + Pp_r + 1e-9)
        print(f"  {rec_id} ({dur_h:.0f}h): "
              f"TP={TP}  FP={FP}  FN={FN}  ref={n_ref}  "
              f"Se={Se_r*100:.2f}%  P+={Pp_r*100:.2f}%  F1={F1_r*100:.2f}%",
              flush=True)

    Se = total_TP / (total_TP + total_FN + 1e-9)
    Pp = total_TP / (total_TP + total_FP + 1e-9)
    F1 = 2 * Se * Pp / (Se + Pp + 1e-9)

    print("\n===== LTDB Single-Lead QRS Evaluation =====")
    print(f"  Records: {len(records)}")
    print(f"  TP={total_TP}  FP={total_FP}  FN={total_FN}")
    print(f"  Se  = {Se:.4f}  ({Se*100:.2f}%)")
    print(f"  P+  = {Pp:.4f}  ({Pp*100:.2f}%)")
    print(f"  F1  = {F1:.4f}  ({F1*100:.2f}%)")
    print("===========================================")
    print(f"\n  MIT-BIH v2 (reference): Se=99.51%  P+=99.88%  F1=99.69%")
    print(f"  LTDB (this run):         Se={Se*100:.2f}%  P+={Pp*100:.2f}%  F1={F1*100:.2f}%")
    print(f"  dF1 = {(F1-0.9969)*100:+.2f}pp vs MIT-BIH v2")


if __name__ == "__main__":
    main()
