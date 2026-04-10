import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""
Lobachevsky University ECG Database (LUDB) evaluation for PN-QRS model.

Signal:      500 Hz, 12 leads, 10-second records (200 records)
Annotations: per-lead wfdb files ('i','ii',... one per lead)
             Each annotation has: ( N ) for QRS, ( t ) for T-wave, ( p ) for P-wave
             We extract only 'N' symbols as R-peak references.
Protocol:
  - Each record is exactly 10 seconds -> single window, no cutting needed
  - Lead 0 = Lead I (signal channel 0), annotation file extension = 'i'
  - preprocess_ecg(signal, fs=500): bandpass -> 200 Hz -> z-score -> (1, 2000)
  - 50 Hz predictions × (500/50) = 10 -> 500 Hz sample space
  - Tolerance = 150 ms = 75 samples @ 500 Hz
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

FS_DB      = 500
FS_MODEL   = 50
TOLERANCE  = int(0.150 * FS_DB)   # 75 samples
SCALE      = FS_DB / FS_MODEL      # 10.0

# LUDB uses 'N' for QRS peaks; wave-delineation markers (, ), t, p are not beats
LUDB_BEAT_SYM = 'N'


def find_best_checkpoint(log_dir: Path) -> Path:
    candidates = sorted(log_dir.rglob("best_model.pt"))
    if not candidates:
        raise FileNotFoundError(f"No best_model.pt found under {log_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_annotations(rec_path: str):
    """LUDB: annotation extension = 'i' (Lead I). Extract only N symbols."""
    ann = wfdb.rdann(rec_path, 'i')
    ref = np.array(
        [s for s, sym in zip(ann.sample, ann.symbol) if sym == LUDB_BEAT_SYM],
        dtype=int
    )
    return ref


def predict_single(model, signal_1d: np.ndarray, device) -> np.ndarray:
    processed = preprocess_ecg(signal_1d, fs=FS_DB)
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
    ref_abs  = load_annotations(rec_path)
    pred_abs = predict_single(model, signal, device)
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
        TP_total += TP; FP_total += FP; FN_total += FN
    Se = TP_total / (TP_total + FN_total + 1e-9)
    Pp = TP_total / (TP_total + FP_total + 1e-9)
    F1 = 2 * Se * Pp / (Se + Pp + 1e-9)
    return Se, Pp, F1, TP_total, FP_total, FN_total


def main(args):
    device  = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    db_dir  = Path(args.db_dir)
    print(f"Device:    {device}")
    print(f"LUDB dir:  {db_dir}")
    print(f"Tolerance: {TOLERANCE} samples = 150 ms @ {FS_DB} Hz")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_best_checkpoint(
        Path("/home/kailong/ECG/ECG/ECGFounder/PN-QRS/experiments")
    )
    print(f"Checkpoint: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model = QRSModel(encoder4qrs(), decoder4qrs(), phi_qrs()).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    records = sorted(set(p.stem for p in db_dir.glob("*.dat")),
                     key=lambda x: int(x) if x.isdigit() else x)
    print(f"Records found: {len(records)}\n")

    all_pred, all_ref = [], []
    for rec_id in records:
        try:
            pred, ref = run_record(rec_id, db_dir, model, device)
            if len(ref) == 0:
                print(f"  [SKIP] {rec_id}: 0 N-beats in annotation")
                continue
            all_pred.append(pred)
            all_ref.append(ref)
            tp_r = sum(1 for r in ref if np.any(np.abs(pred - r) <= TOLERANCE))
            se_r = tp_r / (len(ref) + 1e-9)
            print(f"  {rec_id:4s}: pred={len(pred):3d}  ref={len(ref):3d}  Se={se_r:.4f}")
        except Exception as exc:
            print(f"  [ERROR] {rec_id}: {exc}")

    Se, Pp, F1, TP, FP, FN = compute_metrics(all_pred, all_ref)
    print(f"\n========== LUDB Evaluation (thr=150ms) ==========")
    print(f"  Records evaluated : {len(all_pred)}")
    print(f"  TP={TP}  FP={FP}  FN={FN}")
    print(f"  Se  (Sensitivity)       = {Se:.4f}")
    print(f"  P+  (Pos. Predictivity) = {Pp:.4f}")
    print(f"  F1                      = {F1:.4f}")
    print(f"=================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--db_dir",     type=str,
                        default="/home/kailong/ECG/ECG/data/PN-QRS/LUDB/data")
    parser.add_argument("--gpu",        type=int, default=0)
    args = parser.parse_args()
    main(args)
