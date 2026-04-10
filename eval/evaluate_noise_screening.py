import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""
PN-QRS Noise Screening Evaluation
===================================
Tests whether the model's window-level epistemic uncertainty (U_E = mean of uncertain_est())
can distinguish clean ECG from non-ECG/noise signals.

Datasets
--------
- CPSC2019-Test  : 200 clean 10-s records @ 500 Hz  (calibration baseline)
- INCART          : 75 12-lead Holter records @ 257 Hz (clean clinical ECG, OOD)
- NSTDB pure noise: em / ma / bw records @ 360 Hz     (ground-truth noise)

Protocol
--------
1. Compute U_E = mean(uncertain_est(logits)) for every 10-s window.
2. Calibrate threshold t_alpha at (1-alpha) quantile of CPSC2019-Test U_E.
3. Predict: noise if U_E > t_alpha, else ECG.
4. Compute Se (noise recall), Sp (ECG specificity), accuracy.
5. Save histogram and results to noise_screening_eval.log.

Model: experiments/logs_real/zy2lki18/models/best_model.pt
Device: CPU (local GPUs occupied by vLLM).
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from dataset.dataset import preprocess_ecg
from models.qrs_model import QRSModel
from models.multi_head import encoder4qrs, decoder4qrs, phi_qrs
from utils.qrs_post_process import uncertain_est

# constants
WIN_SEC   = 10
FS_MODEL  = 50
ALPHA     = 0.1          # FPR budget on clean ECG

CPSC2019_TEST_DIR = Path("/home/kailong/ECG/ECG/data/PN-QRS/CPSC2019_real_data/cpsc2019_test")
INCART_DIR        = Path("/home/kailong/ECG/ECG/data/PN-QRS/INCART")
NSTDB_DIR         = Path("/home/kailong/ECG/ECG/data/PN-QRS/NSTDB")
CKPT_PATH         = Path("/home/kailong/ECG/ECG/ECGFounder/PN-QRS/experiments"
                         "/logs_real/zy2lki18/models/best_model.pt")

# Pure noise record names in NSTDB
NSTDB_NOISE_RECORDS = ["em", "ma", "bw"]
NSTDB_FS = 360
INCART_FS = 257


def load_model(ckpt_path: Path, device) -> QRSModel:
    torch.set_num_threads(4)   # prevent cache thrashing on many-core CPU
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model = QRSModel(encoder4qrs(), decoder4qrs(), phi_qrs()).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def window_ue(model, window_1d: np.ndarray, fs: int, device) -> float:
    """Return mean uncertain_est score for a single 10-s window (1-D signal)."""
    processed = preprocess_ecg(window_1d, fs=fs)
    if processed.ndim == 1:
        processed = processed[np.newaxis, :]
    elif processed.shape[0] > processed.shape[1]:
        processed = processed.T

    sig_t = torch.from_numpy(processed).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(sig_t, return_projection=True)
        logits = logits.squeeze(-1).squeeze(0).cpu().numpy()  # (3, T_model)

    uc = uncertain_est(logits)   # (T_model,)
    return float(np.mean(uc))


def windows_from_signal(signal_1d: np.ndarray, fs: int):
    """Yield non-overlapping WIN_SEC windows from a 1-D signal."""
    win_samp = fs * WIN_SEC
    n = len(signal_1d)
    for start in range(0, n - win_samp + 1, win_samp):
        yield signal_1d[start: start + win_samp]


def score_cpsc2019_test(model, device):
    """200 clean 10-s .mat records -> list of window U_E scores."""
    import scipy.io as sio
    data_dir = CPSC2019_TEST_DIR / "data"
    mat_files = sorted(data_dir.glob("data_*.mat"))
    if not mat_files:
        print("  [WARN] No CPSC2019-Test .mat files found.")
        return []
    scores = []
    for i, f in enumerate(mat_files):
        try:
            ecg = sio.loadmat(str(f))["ecg"][:, 0].astype(np.float32)
            scores.append(window_ue(model, ecg, fs=500, device=device))
            if (i + 1) % 20 == 0:
                print(f"  CPSC [{i+1}/{len(mat_files)}] last_ue={scores[-1]:.4f}",
                      flush=True)
        except Exception as e:
            print(f"  [SKIP] {f.name}: {e}", flush=True)
    return scores


def score_wfdb_records(model, db_dir: Path, record_ids: list, fs_default: int,
                       device, max_windows_per_record: int = 10):
    """Score wfdb records: returns per-window U_E scores."""
    import wfdb
    scores = []
    for rid in record_ids:
        rec_path = str(db_dir / rid)
        try:
            rec    = wfdb.rdrecord(rec_path, channels=[0])
            signal = rec.p_signal[:, 0].astype(np.float32)
            actual_fs = int(rec.fs)
            count = 0
            rec_scores = []
            for win in windows_from_signal(signal, actual_fs):
                if count >= max_windows_per_record:
                    break
                rec_scores.append(window_ue(model, win, fs=actual_fs, device=device))
                count += 1
            scores.extend(rec_scores)
            if rec_scores:
                print(f"  {rid}: {len(rec_scores)} windows  "
                      f"mean_ue={np.mean(rec_scores):.4f}", flush=True)
        except Exception as e:
            print(f"  [SKIP] {rid}: {e}", flush=True)
    return scores


def get_incart_records(db_dir: Path):
    """Return list of INCART record IDs."""
    records = [p.stem for p in sorted(db_dir.glob("I*.hea"))]
    if not records:
        records = [p.stem for p in sorted(db_dir.glob("*.hea"))]
    return records


def get_nstdb_noise_records(db_dir: Path):
    return [rid for rid in NSTDB_NOISE_RECORDS if (db_dir / (rid + ".hea")).exists()]


def calibrate_threshold(clean_scores: list, alpha: float = ALPHA) -> float:
    return float(np.quantile(clean_scores, 1 - alpha))


def compute_metrics(scores, labels, threshold):
    """labels: 1=noise, 0=clean ECG"""
    preds  = [1 if s > threshold else 0 for s in scores]
    labels = list(labels)
    TP = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    FN = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
    TN = sum(p == 0 and l == 0 for p, l in zip(preds, labels))
    FP = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
    Se  = TP / (TP + FN + 1e-9)
    Sp  = TN / (TN + FP + 1e-9)
    Acc = (TP + TN) / (TP + TN + FP + FN + 1e-9)
    return Se, Sp, Acc, TP, FP, TN, FN


def save_histogram(cpsc_scores, incart_scores, noise_scores, threshold, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        kw = dict(bins=40, alpha=0.55, density=True)
        if cpsc_scores:
            ax.hist(cpsc_scores, label=f"CPSC2019-Test (n={len(cpsc_scores)})",
                    color="steelblue", **kw)
        if incart_scores:
            ax.hist(incart_scores, label=f"INCART Holter (n={len(incart_scores)})",
                    color="forestgreen", **kw)
        if noise_scores:
            ax.hist(noise_scores, label=f"NSTDB noise (n={len(noise_scores)})",
                    color="crimson", **kw)
        ax.axvline(threshold, color="black", linestyle="--",
                   label=f"threshold alpha={ALPHA} -> {threshold:.4f}")
        ax.set_xlabel("Mean U_E per 10-s window")
        ax.set_ylabel("Density")
        ax.set_title("PN-QRS Noise Screening: U_E distributions")
        ax.legend()
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=120)
        plt.close()
        print(f"  Histogram saved -> {out_path}")
    except ImportError:
        print("  [WARN] matplotlib not available, skipping histogram.")


def main(args):
    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {CKPT_PATH}\n")

    model = load_model(CKPT_PATH, device)

    # Step 1: CPSC2019-Test (calibration)
    print("=== CPSC2019-Test (clean ECG, threshold calibration) ===")
    cpsc_scores = score_cpsc2019_test(model, device)
    print(f"  Windows: {len(cpsc_scores)}")
    if cpsc_scores:
        print(f"  U_E  mean={np.mean(cpsc_scores):.4f}  "
              f"std={np.std(cpsc_scores):.4f}  "
              f"p90={np.quantile(cpsc_scores, 0.90):.4f}  "
              f"p99={np.quantile(cpsc_scores, 0.99):.4f}")
    else:
        print("[FATAL] No CPSC2019-Test scores; cannot calibrate threshold.")
        return

    threshold = calibrate_threshold(cpsc_scores, alpha=ALPHA)
    print(f"\n  Threshold (alpha={ALPHA}) = {threshold:.4f} "
          f"[p{int((1-ALPHA)*100)} of clean ECG]\n")

    # Step 2: INCART Holter
    print("=== INCART 12-lead Holter (clean clinical ECG, OOD) ===")
    incart_records = get_incart_records(INCART_DIR)
    if not incart_records:
        print("  [WARN] No INCART records found -- data may still be downloading.")
        incart_scores = []
    else:
        print(f"  Records: {len(incart_records)}")
        incart_scores = score_wfdb_records(
            model, INCART_DIR, incart_records, fs_default=INCART_FS,
            device=device, max_windows_per_record=5)
        print(f"  Windows: {len(incart_scores)}")
        if incart_scores:
            print(f"  U_E  mean={np.mean(incart_scores):.4f}  std={np.std(incart_scores):.4f}")
            flagged = sum(s > threshold for s in incart_scores)
            print(f"  Flagged as noise: {flagged}/{len(incart_scores)} "
                  f"({100*flagged/len(incart_scores):.1f}%)  [target <= {ALPHA*100:.0f}%]")

    # Step 3: NSTDB pure noise
    print("\n=== NSTDB pure noise (em / ma / bw) ===")
    noise_records = get_nstdb_noise_records(NSTDB_DIR)
    if not noise_records:
        print("  [WARN] No NSTDB noise records found -- data may still be downloading.")
        noise_scores = []
    else:
        print(f"  Records: {noise_records}")
        noise_scores = score_wfdb_records(
            model, NSTDB_DIR, noise_records, fs_default=NSTDB_FS,
            device=device, max_windows_per_record=15)
        print(f"  Windows: {len(noise_scores)}")
        if noise_scores:
            print(f"  U_E  mean={np.mean(noise_scores):.4f}  std={np.std(noise_scores):.4f}")

    # Step 4: Combined metrics
    print("\n=== Combined Noise-Screening Metrics ===")
    all_scores = cpsc_scores + incart_scores + noise_scores
    all_labels = [0]*len(cpsc_scores) + [0]*len(incart_scores) + [1]*len(noise_scores)

    if noise_scores and (cpsc_scores or incart_scores):
        Se, Sp, Acc, TP, FP, TN, FN = compute_metrics(all_scores, all_labels, threshold)
        print(f"  Threshold = {threshold:.4f}")
        print(f"  TP={TP}  FP={FP}  TN={TN}  FN={FN}")
        print(f"  Se  (noise recall)    = {Se:.4f}")
        print(f"  Sp  (ECG specificity) = {Sp:.4f}")
        print(f"  Acc                   = {Acc:.4f}")
        if cpsc_scores:
            fp_c = sum(s > threshold for s in cpsc_scores)
            print(f"\n  CPSC2019-Test FPR : {fp_c}/{len(cpsc_scores)} "
                  f"({100*fp_c/len(cpsc_scores):.1f}%)")
        if incart_scores:
            fp_i = sum(s > threshold for s in incart_scores)
            print(f"  INCART Holter FPR : {fp_i}/{len(incart_scores)} "
                  f"({100*fp_i/len(incart_scores):.1f}%)")
        if noise_scores:
            tp_n = sum(s > threshold for s in noise_scores)
            print(f"  NSTDB noise TPR   : {tp_n}/{len(noise_scores)} "
                  f"({100*tp_n/len(noise_scores):.1f}%)")
        print(f"\n  Paper Table I: ECG-set Acc = 96.84%")
        print(f"  This run  Acc = {Acc*100:.2f}%")
    else:
        print("  Insufficient data for combined metrics (check downloads).")

    hist_path = Path(__file__).parent / "noise_screening_ue_hist.png"
    save_histogram(cpsc_scores, incart_scores, noise_scores, threshold, hist_path)
    print("\n=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=ALPHA)
    args = parser.parse_args()
    ALPHA = args.alpha
    main(args)
