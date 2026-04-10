import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""
MIT-BIH Arrhythmia Database evaluation for PN-QRS model.

Signal:      360 Hz, 2 leads (MLII + V5), ~30-min records
Annotations: wfdb beat annotations (all beat types)
Protocol:
  - Cut each record into non-overlapping 10-second windows
  - Preprocess: bandpass -> downsample to 200 Hz -> z-score
  - Feed single-lead (lead 0, MLII) through model
  - Convert 50 Hz predictions back to 360 Hz sample space
  - Tolerance = 150 ms = 54 samples @ 360 Hz
  - Global Se / P+ / F1 across all records

Paced records excluded (standard practice): 102, 104, 107, 217
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import wfdb

sys.path.insert(0, str(Path(__file__).parent))
from dataset.dataset import preprocess_ecg
from models.qrs_model import QRSModel
from models.multi_head import encoder4qrs, decoder4qrs, phi_qrs
from utils.qrs_post_process import correct, uncertain_est


FS_MITBIH  = 360
FS_MODEL   = 50
WIN_SEC    = 10
WIN_SAMP   = FS_MITBIH * WIN_SEC   # 3600 samples
TOLERANCE  = int(0.150 * FS_MITBIH)  # 150 ms -> 54 samples

PACED_RECORDS = {"102", "104", "107", "217"}

ALL_RECORDS = [
    "100","101","102","103","104","105","106","107","108","109",
    "111","112","113","114","115","116","117","118","119",
    "121","122","123","124",
    "200","201","202","203","205","207","208","209","210",
    "212","213","214","215","217","219","220","221","222","223",
    "228","230","231","232","233","234",
]

BEAT_SYMS = set("NLRBaAJSVrFejnE/fQ!|")


def find_best_checkpoint(log_dir: Path) -> Path:
    candidates = sorted(log_dir.rglob("best_model.pt"))
    if not candidates:
        raise FileNotFoundError(f"No best_model.pt found under {log_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def predict_window(model, window_1d: np.ndarray, device) -> np.ndarray:
    """
    window_1d: (WIN_SAMP,) raw signal at 360 Hz
    Returns: R-peak positions in 360 Hz space, relative to window start.
    """
    processed = preprocess_ecg(window_1d, fs=FS_MITBIH)
    if processed.ndim == 1:
        processed = processed[np.newaxis, :]
    elif processed.shape[0] > processed.shape[1]:
        processed = processed.T

    sig_t = torch.from_numpy(processed).unsqueeze(0).to(device)  # (1,1,2000)

    model.eval()
    with torch.no_grad():
        logits = model(sig_t, return_projection=True)        # (1,3,500,1)
        logits = logits.squeeze(-1).squeeze(0).cpu().numpy()  # (3,500)

    uc     = uncertain_est(logits)
    prob   = logits[0]
    r_50hz = correct(prob, uc)

    if len(r_50hz) == 0:
        return np.array([], dtype=int)

    r_360hz = np.round(np.array(r_50hz) * (FS_MITBIH / FS_MODEL)).astype(int)
    return r_360hz


def run_record(record_id: str, db_dir: Path, model, device):
    """Returns (pred_abs, ref_abs) both in absolute 360 Hz sample positions."""
    rec_path = str(db_dir / record_id)
    record   = wfdb.rdrecord(rec_path, channels=[0])
    signal   = record.p_signal[:, 0].astype(np.float32)
    n_samp   = len(signal)

    ann = wfdb.rdann(rec_path, "atr")
    ref_abs = np.array(
        [s for s, sym in zip(ann.sample, ann.symbol) if sym in BEAT_SYMS],
        dtype=int
    )

    preds = []
    for start in range(0, n_samp - WIN_SAMP + 1, WIN_SAMP):
        r_rel = predict_window(model, signal[start: start + WIN_SAMP], device)
        preds.append(r_rel + start)

    pred_abs = np.concatenate(preds) if preds else np.array([], dtype=int)
    return pred_abs, ref_abs


def compute_metrics(all_pred, all_ref):
    TP_total = FP_total = FN_total = 0
    for pred, ref in zip(all_pred, all_ref):
        if len(ref) == 0:
            FP_total += len(pred)
            continue
        matched_pred = set()
        TP = 0
        for r in ref:
            hits = np.where(np.abs(pred - r) <= TOLERANCE)[0]
            if len(hits):
                best = hits[np.argmin(np.abs(pred[hits] - r))]
                if best not in matched_pred:
                    TP += 1
                    matched_pred.add(best)
        FN = len(ref) - TP
        FP = max(len(pred) - len(matched_pred), 0)
        TP_total += TP
        FP_total += FP
        FN_total += FN
    Se = TP_total / (TP_total + FN_total + 1e-9)
    Pp = TP_total / (TP_total + FP_total + 1e-9)
    F1 = 2 * Se * Pp / (Se + Pp + 1e-9)
    return Se, Pp, F1, TP_total, FP_total, FN_total


def main(args):
    device  = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    db_dir  = Path(args.db_dir)
    print(f"Device:      {device}")
    print(f"MIT-BIH dir: {db_dir}")
    print(f"Tolerance:   {TOLERANCE} samples = 150 ms @ {FS_MITBIH} Hz")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_best_checkpoint(
        Path("/home/kailong/ECG/ECG/ECGFounder/PN-QRS/experiments")
    )
    print(f"Checkpoint:  {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model = QRSModel(encoder4qrs(), decoder4qrs(), phi_qrs()).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    records = [r for r in ALL_RECORDS if r not in PACED_RECORDS]
    print(f"Records: {len(records)} (paced excluded: {sorted(PACED_RECORDS)})\n")

    all_pred, all_ref = [], []
    for rec_id in records:
        if not (db_dir / (rec_id + ".dat")).exists():
            print(f"  [SKIP] {rec_id}")
            continue
        try:
            pred, ref = run_record(rec_id, db_dir, model, device)
            all_pred.append(pred)
            all_ref.append(ref)
            tp_r = sum(1 for r in ref if np.any(np.abs(pred - r) <= TOLERANCE))
            se_r = tp_r / (len(ref) + 1e-9)
            print(f"  {rec_id}: pred={len(pred):5d}  ref={len(ref):5d}  Se={se_r:.4f}")
        except Exception as exc:
            print(f"  [ERROR] {rec_id}: {exc}")

    Se, Pp, F1, TP, FP, FN = compute_metrics(all_pred, all_ref)
    print("\n========== MIT-BIH Evaluation (thr=150ms, paced excl.) ==========")
    print(f"  Records evaluated : {len(all_pred)}")
    print(f"  TP={TP}  FP={FP}  FN={FN}")
    print(f"  Se  (Sensitivity)       = {Se:.4f}")
    print(f"  P+  (Pos. Predictivity) = {Pp:.4f}")
    print(f"  F1                      = {F1:.4f}")
    print("===================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--db_dir",     type=str,
                        default="/home/kailong/ECG/ECG/data/PN-QRS/MITBIH")
    parser.add_argument("--gpu",        type=int, default=0)
    args = parser.parse_args()
    main(args)
