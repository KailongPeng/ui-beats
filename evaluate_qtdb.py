"""
QT Database (QTDB) evaluation for PN-QRS model.

Signal:      250 Hz, 2 leads, ~15-min records (105 records)
Annotations: q1c (manually corrected beat annotations, lead 1)
Protocol:
  - Non-overlapping 10-second windows (2500 samples @ 250 Hz)
  - preprocess_ecg(window, fs=250): bandpass -> 200 Hz -> z-score
  - 50 Hz predictions × (250/50) = 5.0 -> 250 Hz sample space
  - Tolerance = 150 ms = 37 samples @ 250 Hz
  - Global Se / P+ / F1 across all records
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

FS_DB      = 250
FS_MODEL   = 50
WIN_SEC    = 10
WIN_SAMP   = FS_DB * WIN_SEC        # 2500
TOLERANCE  = int(0.150 * FS_DB)    # 37 samples
SCALE      = FS_DB / FS_MODEL       # 5.0

BEAT_SYMS = set("NLRBaAJSVrFejnE/fQ!|")
ANN_EXTS  = ["atr"]   # only full beat annotations; skip q1c-only records


def find_best_checkpoint(log_dir: Path) -> Path:
    candidates = sorted(log_dir.rglob("best_model.pt"))
    if not candidates:
        raise FileNotFoundError(f"No best_model.pt found under {log_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_annotations(rec_path: str, exts: list):
    """Try annotation extensions in order, return first that works."""
    for ext in exts:
        try:
            ann = wfdb.rdann(rec_path, ext)
            beats = np.array(
                [s for s, sym in zip(ann.sample, ann.symbol) if sym in BEAT_SYMS],
                dtype=int
            )
            if len(beats) > 0:
                return beats, ext
        except Exception:
            continue
    return np.array([], dtype=int), None


def predict_window(model, window_1d: np.ndarray, device) -> np.ndarray:
    processed = preprocess_ecg(window_1d, fs=FS_DB)
    if processed.ndim == 1:
        processed = processed[np.newaxis, :]
    elif processed.shape[0] > processed.shape[1]:
        processed = processed.T
    sig_t = torch.from_numpy(processed).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(sig_t, return_projection=True)
        logits = logits.squeeze(-1).squeeze(0).cpu().numpy()
    uc     = uncertain_est(logits)
    prob   = logits[0]
    r_50hz = correct(prob, uc)
    if len(r_50hz) == 0:
        return np.array([], dtype=int)
    return np.round(np.array(r_50hz) * SCALE).astype(int)


def run_record(record_id: str, db_dir: Path, model, device):
    rec_path = str(db_dir / record_id)
    record   = wfdb.rdrecord(rec_path, channels=[0])
    signal   = record.p_signal[:, 0].astype(np.float32)
    n_samp   = len(signal)

    ref_abs, ann_ext = load_annotations(rec_path, ANN_EXTS)

    preds = []
    for start in range(0, n_samp - WIN_SAMP + 1, WIN_SAMP):
        r_rel = predict_window(model, signal[start: start + WIN_SAMP], device)
        preds.append(r_rel + start)
    pred_abs = np.concatenate(preds) if preds else np.array([], dtype=int)
    return pred_abs, ref_abs, ann_ext


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
        TP_total += TP; FP_total += FP; FN_total += FN
    Se = TP_total / (TP_total + FN_total + 1e-9)
    Pp = TP_total / (TP_total + FP_total + 1e-9)
    F1 = 2 * Se * Pp / (Se + Pp + 1e-9)
    return Se, Pp, F1, TP_total, FP_total, FN_total


def main(args):
    device  = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    db_dir  = Path(args.db_dir)
    print(f"Device:    {device}")
    print(f"QTDB dir:  {db_dir}")
    print(f"Tolerance: {TOLERANCE} samples = 150 ms @ {FS_DB} Hz")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_best_checkpoint(
        Path("/home/kailong/ECG/ECG/ECGFounder/PN-QRS/experiments")
    )
    print(f"Checkpoint: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model = QRSModel(encoder4qrs(), decoder4qrs(), phi_qrs()).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # QTDB record names from RECORDS file or glob .dat
    records = sorted(set(p.stem for p in db_dir.glob("*.dat")))
    print(f"Records found: {len(records)}\n")

    all_pred, all_ref = [], []
    for rec_id in records:
        try:
            pred, ref, ann_ext = run_record(rec_id, db_dir, model, device)
            if len(ref) == 0:
                print(f"  [SKIP] {rec_id}: no beat annotations found")
                continue
            all_pred.append(pred)
            all_ref.append(ref)
            tp_r = sum(1 for r in ref if np.any(np.abs(pred - r) <= TOLERANCE))
            se_r = tp_r / (len(ref) + 1e-9)
            print(f"  {rec_id:12s} [{ann_ext}]: pred={len(pred):5d}  ref={len(ref):5d}  Se={se_r:.4f}")
        except Exception as exc:
            print(f"  [ERROR] {rec_id}: {exc}")

    Se, Pp, F1, TP, FP, FN = compute_metrics(all_pred, all_ref)
    print(f"\n========== QTDB Evaluation (thr=150ms) ==========")
    print(f"  Records evaluated : {len(all_pred)}")
    print(f"  TP={TP}  FP={FP}  FN={FN}")
    print(f"  Se  (Sensitivity)       = {Se:.4f}")
    print(f"  P+  (Pos. Predictivity) = {Pp:.4f}")
    print(f"  F1                      = {F1:.4f}")
    print(f"==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--db_dir",     type=str,
                        default="/home/kailong/ECG/ECG/data/PN-QRS/QTDB")
    parser.add_argument("--gpu",        type=int, default=0)
    args = parser.parse_args()
    main(args)
