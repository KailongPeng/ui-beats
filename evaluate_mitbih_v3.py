"""
MIT-BIH Evaluation v3: Improvement 2+3 combined.
  - 2s overlap (from v2)
  - R-peak refinement: after coarse 50Hz prediction, search for true
    signal maximum within ±25ms in the original 360Hz signal.
    Uses |signal| so it works for both upward and inverted QRS.

Checkpoint: same synthetic-data checkpoint as v1/v2.
Output:     mitbih_eval_v3.log  (does not overwrite v1/v2)
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import torch
import wfdb

sys.path.insert(0, str(Path(__file__).parent))
from dataset.dataset import preprocess_ecg
from models.qrs_model import QRSModel
from models.multi_head import encoder4qrs, decoder4qrs, phi_qrs
from utils.qrs_post_process import correct, uncertain_est

FS_MITBIH   = 360
FS_MODEL    = 50
WIN_SEC     = 10
OVERLAP_SEC = 2
WIN_SAMP    = FS_MITBIH * WIN_SEC
STEP_SAMP   = FS_MITBIH * (WIN_SEC - OVERLAP_SEC)
TOLERANCE   = int(0.150 * FS_MITBIH)   # 54 samples
REFINE_RAD  = int(0.025 * FS_MITBIH)   # ±25ms = ±9 samples search radius

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
        raise FileNotFoundError(f"No best_model.pt under {log_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def nms_predictions(preds: np.ndarray, tol: int) -> np.ndarray:
    if len(preds) == 0:
        return preds
    preds = np.sort(preds)
    clusters, cluster = [], [preds[0]]
    for p in preds[1:]:
        if p - cluster[-1] <= tol:
            cluster.append(p)
        else:
            clusters.append(cluster)
            cluster = [p]
    clusters.append(cluster)
    return np.array([int(np.median(c)) for c in clusters])


def refine_peaks(signal_full: np.ndarray, peaks: np.ndarray, radius: int) -> np.ndarray:
    """Snap each coarse R-peak to the nearest |signal| maximum within radius."""
    refined = np.empty_like(peaks)
    n = len(signal_full)
    abs_sig = np.abs(signal_full)
    for i, p in enumerate(peaks):
        lo = max(0, p - radius)
        hi = min(n, p + radius + 1)
        refined[i] = lo + int(np.argmax(abs_sig[lo:hi]))
    return refined


def predict_window(model, window_1d: np.ndarray, device) -> np.ndarray:
    processed = preprocess_ecg(window_1d, fs=FS_MITBIH)
    if processed.ndim == 1:
        processed = processed[np.newaxis, :]
    elif processed.shape[0] > processed.shape[1]:
        processed = processed.T
    sig_t = torch.from_numpy(processed).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(sig_t, return_projection=True)
        logits = logits.squeeze(-1).squeeze(0).cpu().numpy()
    uc     = uncertain_est(logits)
    prob   = logits[0]
    r_50hz = correct(prob, uc)
    if len(r_50hz) == 0:
        return np.array([], dtype=int)
    return np.round(np.array(r_50hz) * (FS_MITBIH / FS_MODEL)).astype(int)


def run_record(record_id, db_dir, model, device):
    rec_path = str(db_dir / record_id)
    record   = wfdb.rdrecord(rec_path, channels=[0])
    signal   = record.p_signal[:, 0].astype(np.float32)
    n_samp   = len(signal)

    ann = wfdb.rdann(rec_path, "atr")
    ref_abs = np.array(
        [s for s, sym in zip(ann.sample, ann.symbol) if sym in BEAT_SYMS],
        dtype=int,
    )

    starts = list(range(0, n_samp - WIN_SAMP + 1, STEP_SAMP))
    last_possible = n_samp - WIN_SAMP
    if last_possible >= 0 and (not starts or starts[-1] < last_possible):
        starts.append(last_possible)

    preds = []
    for start in starts:
        r_rel = predict_window(model, signal[start: start + WIN_SAMP], device)
        if len(r_rel) > 0:
            preds.append(r_rel + start)

    if preds:
        pred_coarse = nms_predictions(np.concatenate(preds), TOLERANCE)
        pred_abs    = refine_peaks(signal, pred_coarse, REFINE_RAD)
        pred_abs    = nms_predictions(pred_abs, TOLERANCE)
    else:
        pred_abs = np.array([], dtype=int)

    return pred_abs, ref_abs


def compute_metrics(all_pred, all_ref):
    TP = FP = FN = 0
    for pred, ref in zip(all_pred, all_ref):
        if len(ref) == 0:
            FP += len(pred)
            continue
        matched = set()
        tp_r = 0
        for r in ref:
            hits = np.where(np.abs(pred - r) <= TOLERANCE)[0]
            if len(hits):
                best = hits[np.argmin(np.abs(pred[hits] - r))]
                if best not in matched:
                    tp_r += 1
                    matched.add(best)
        FN += len(ref) - tp_r
        FP += max(len(pred) - len(matched), 0)
        TP += tp_r
    Se = TP / (TP + FN + 1e-9)
    Pp = TP / (TP + FP + 1e-9)
    F1 = 2 * Se * Pp / (Se + Pp + 1e-9)
    return Se, Pp, F1, TP, FP, FN


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    db_dir = Path(args.db_dir)

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else Path(
        "/home/kailong/ECG/ECG/ECGFounder/PN-QRS/experiments"
    )
    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_best_checkpoint(ckpt_dir)

    print(f"[v3] overlap={OVERLAP_SEC}s + R-peak refinement (radius={REFINE_RAD} samp = 25ms)")
    print(f"Device:     {device}")
    print(f"Checkpoint: {ckpt_path}\n")

    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model = QRSModel(encoder4qrs(), decoder4qrs(), phi_qrs()).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    records = [r for r in ALL_RECORDS if r not in PACED_RECORDS]
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
    print("\n===== MIT-BIH v3 (2s overlap + peak refinement, tol=150ms, paced excl.) =====")
    print(f"  TP={TP}  FP={FP}  FN={FN}")
    print(f"  Se  = {Se:.4f}")
    print(f"  P+  = {Pp:.4f}")
    print(f"  F1  = {F1:.4f}")
    print("=================================================================================")
    Se_v1, F1_v1 = 0.9753, 0.9868
    Se_v2, F1_v2 = 0.9951, 0.9969
    print(f"  vs v1: dSe={Se-Se_v1:+.4f}  dF1={F1-F1_v1:+.4f}")
    print(f"  vs v2: dSe={Se-Se_v2:+.4f}  dF1={F1-F1_v2:+.4f}")
    print(f"  target gap remaining: {0.9995-F1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--ckpt_dir",   default=None)
    parser.add_argument("--db_dir", default="/home/kailong/ECG/ECG/data/PN-QRS/MITBIH")
    parser.add_argument("--gpu",    type=int, default=0)
    args = parser.parse_args()
    main(args)
