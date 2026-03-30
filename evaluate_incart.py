"""
INCART 12-Lead QRS Evaluation
================================
Reproduces PN-QRS paper Table 2 (INCART dataset).

Protocol (Algorithm 1 from paper):
  1. Slide 10-s windows with 2-s overlap over each 30-min record
  2. Run model on all 12 leads independently per window
  3. Select the lead with minimum mean U_E
  4. Use that lead's R-peak predictions
  5. Deduplicate across overlapping windows (150ms NMS)
  6. Match predictions to ground-truth annotations (tolerance = 150ms)

Paper target: FP=15, FN=9, F1≈99.99%
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

# ── constants ──────────────────────────────────────────────────────────────
INCART_DIR  = Path("/home/kailong/ECG/ECG/data/PN-QRS/INCART")
CKPT_PATH   = Path("/home/kailong/ECG/ECG/ECGFounder/PN-QRS/experiments"
                   "/logs_real/zy2lki18/models/best_model.pt")
INCART_FS   = 257
FS_MODEL    = 50
WIN_SEC     = 10
OVERLAP_SEC = 2
WIN_SAMP    = WIN_SEC * INCART_FS          # 2570
STRIDE      = (WIN_SEC - OVERLAP_SEC) * INCART_FS  # 2056
TOL_SAMP    = int(0.150 * INCART_FS)      # 38 samples @ 257Hz
BEAT_SYMS   = set("NLRBaAJSVrFejnE/fQ!|")
N_LEADS     = 12


def load_model(device):
    ckpt  = torch.load(CKPT_PATH, map_location="cpu")
    model = QRSModel(encoder4qrs(), decoder4qrs(), phi_qrs()).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def preprocess_lead(lead_sig: np.ndarray, fs: int) -> np.ndarray:
    """Preprocess one 1-D lead signal -> (1, T_model) float32."""
    out = preprocess_ecg(lead_sig.astype(np.float32), fs=fs)
    if out.ndim == 1:
        out = out[np.newaxis, :]
    elif out.shape[0] > out.shape[1]:
        out = out.T
    return out  # (1, T_model)


def run_window_all_leads(model, window: np.ndarray, fs: int, device):
    """
    window: (WIN_SAMP, 12) raw signal
    Returns: best_ue (float), best_peaks (list[int] in local 257Hz coords)
    """
    # Preprocess all 12 leads and stack into a batch
    lead_tensors = []
    valid_idx    = []
    for li in range(N_LEADS):
        sig = window[:, li]
        if np.isnan(sig).any() or np.std(sig) < 1e-6:
            continue  # skip dead/NaN leads
        proc = preprocess_lead(sig, fs)    # (1, T_model)
        lead_tensors.append(proc)
        valid_idx.append(li)

    if not lead_tensors:
        return float('inf'), []

    # Batch inference: (n_valid, 1, T_model)
    batch = torch.from_numpy(
        np.stack(lead_tensors, axis=0)     # (n_valid, 1, T_model)
    ).to(device)

    with torch.no_grad():
        logits_batch = model(batch, return_projection=True)
        # shape: (n_valid, 3, T_model, 1) or (n_valid, 3, T_model)
        logits_batch = logits_batch.squeeze(-1).cpu().numpy()  # (n_valid, 3, T_model)

    best_ue    = float('inf')
    best_peaks = []

    for k in range(len(valid_idx)):
        logits = logits_batch[k]          # (3, T_model)
        uc     = uncertain_est(logits)    # (T_model,)
        mean_ue = float(np.mean(uc))

        if mean_ue < best_ue:
            prob = logits[0]              # (T_model,) QRS probability
            r_model = correct(prob, uc)  # positions at FS_MODEL=50Hz
            if len(r_model) > 0:
                r_local = np.round(
                    np.array(r_model) * (fs / FS_MODEL)
                ).astype(int)
            else:
                r_local = np.array([], dtype=int)
            best_ue    = mean_ue
            best_peaks = r_local.tolist()

    return best_ue, best_peaks


def nms_peaks(peaks: list, tol: int) -> list:
    """Remove duplicates within tol samples; keep first occurrence."""
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
    pred_arr = np.array(pred) if pred else np.array([], dtype=int)
    for j, r in enumerate(ref):
        if len(pred_arr) == 0:
            break
        dists = np.abs(pred_arr - r)
        hits  = np.where(dists <= tol)[0]
        hits  = [h for h in hits if h not in matched_pred]
        if hits:
            best = min(hits, key=lambda h: dists[h])
            matched_ref.add(j)
            matched_pred.add(best)
    TP = len(matched_ref)
    FP = len(pred) - len(matched_pred)
    FN = len(ref)  - len(matched_ref)
    return TP, FP, FN


def get_records(db_dir: Path):
    """Return record stems that have both .dat and .atr files."""
    records = []
    for hea in sorted(db_dir.glob("I*.hea")):
        stem = hea.stem
        if (db_dir / (stem + ".dat")).exists() and \
           (db_dir / (stem + ".atr")).exists():
            records.append(stem)
    return records


def run_record(record_id: str, model, device):
    rec_path = str(INCART_DIR / record_id)
    rec      = wfdb.rdrecord(rec_path)                # all 12 leads
    signal   = rec.p_signal.astype(np.float32)        # (N, 12)
    n_samp   = signal.shape[0]
    actual_fs = int(rec.fs)

    ann      = wfdb.rdann(rec_path, "atr")
    ref_abs  = np.array(
        [s for s, sym in zip(ann.sample, ann.symbol) if sym in BEAT_SYMS],
        dtype=int
    )

    # Build window starts (2s overlap)
    starts = list(range(0, n_samp - WIN_SAMP + 1, STRIDE))
    last   = n_samp - WIN_SAMP
    if last >= 0 and (not starts or starts[-1] < last):
        starts.append(last)

    all_peaks = []
    for start in starts:
        win = signal[start: start + WIN_SAMP, :]   # (WIN_SAMP, 12)
        _, peaks_local = run_window_all_leads(model, win, actual_fs, device)
        all_peaks.extend([p + start for p in peaks_local])

    pred_abs = nms_peaks(all_peaks, tol=TOL_SAMP)
    return pred_abs, ref_abs.tolist()


def main():
    import os
    # Use CUDA_VISIBLE_DEVICES=3 → device cuda:0; fallback to CPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"INCART dir: {INCART_DIR}")
    print(f"WIN={WIN_SEC}s  STRIDE={WIN_SEC-OVERLAP_SEC}s  "
          f"OVERLAP={OVERLAP_SEC}s  TOL=150ms={TOL_SAMP}samp\n")

    model   = load_model(device)
    records = get_records(INCART_DIR)
    print(f"Records with .dat+.atr: {len(records)}\n")

    total_TP = total_FP = total_FN = 0
    all_pred, all_ref = [], []

    for rec_id in records:
        try:
            pred, ref = run_record(rec_id, model, device)
            all_pred.append(pred)
            all_ref.append(ref)
            tp_r = sum(1 for r in ref if
                       any(abs(p - r) <= TOL_SAMP for p in pred))
            TP_r = tp_r
            FP_r = max(len(pred) - tp_r, 0)
            FN_r = len(ref) - tp_r
            total_TP += TP_r
            total_FP += FP_r
            total_FN += FN_r
            Se_r = TP_r / (len(ref) + 1e-9)
            print(f"  {rec_id}: pred={len(pred):5d}  ref={len(ref):5d}  "
                  f"TP={TP_r}  FP={FP_r}  FN={FN_r}  Se={Se_r:.4f}",
                  flush=True)
        except Exception as e:
            print(f"  [ERROR] {rec_id}: {e}", flush=True)

    Se = total_TP / (total_TP + total_FN + 1e-9)
    Pp = total_TP / (total_TP + total_FP + 1e-9)
    F1 = 2 * Se * Pp / (Se + Pp + 1e-9)

    print("\n===== INCART 12-Lead QRS Evaluation =====")
    print(f"  Records: {len(all_pred)}")
    print(f"  TP={total_TP}  FP={total_FP}  FN={total_FN}")
    print(f"  Se  = {Se:.4f}  ({Se*100:.2f}%)")
    print(f"  P+  = {Pp:.4f}  ({Pp*100:.2f}%)")
    print(f"  F1  = {F1:.4f}  ({F1*100:.2f}%)")
    print("=========================================")
    print(f"\n  Paper: FP=15  FN=9  F1≈99.99%")
    print(f"  This:  FP={total_FP}  FN={total_FN}  F1={F1*100:.2f}%")
    print(f"  Gap:   dFP={total_FP-15:+d}  dFN={total_FN-9:+d}  "
          f"dF1={F1*100-99.99:+.2f}pp")


if __name__ == "__main__":
    main()
