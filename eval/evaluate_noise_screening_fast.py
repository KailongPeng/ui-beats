import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""
Fast noise screening: uses pre-computed CPSC2019-Test stats, only runs INCART+NSTDB.

CPSC2019-Test results (from full run, 200 records):
  mean_ue=1.3877  std=0.5016  p90=1.9967  p99=2.5922
  Threshold (alpha=0.1) = 1.9967

Already scored INCART records (I01-I15, 20 windows each, from partial run):
  Stored below as INCART_PARTIAL.
"""
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

INCART_DIR = Path("/home/kailong/ECG/ECG/data/PN-QRS/INCART")
NSTDB_DIR  = Path("/home/kailong/ECG/ECG/data/PN-QRS/NSTDB")
CKPT_PATH  = Path("/home/kailong/ECG/ECG/ECGFounder/PN-QRS/experiments"
                  "/logs_real/zy2lki18/models/best_model.pt")

WIN_SEC  = 10
ALPHA    = 0.1
NSTDB_FS = 360
INCART_FS = 257

# Pre-computed CPSC2019-Test (200 windows, full run)
CPSC_MEAN      = 1.3877
CPSC_STD       = 0.5016
CPSC_THRESHOLD = 1.9967   # p90

# Partially scored INCART records (20 windows each, mean_ue from partial run)
INCART_PARTIAL = {
    "I01": 1.1767, "I02": 1.0491, "I03": 0.8286, "I04": 0.8620,
    "I05": 0.6101, "I06": 1.6129, "I07": 1.0383, "I08": 1.0743,
    "I09": 1.0035, "I10": 1.4178, "I11": 1.4611, "I12": 2.8166,
    "I13": 0.7404, "I14": 1.0143, "I15": 1.0756,
}

# Already-evaluated: reconstruct per-window scores (using mean as proxy for 20 windows)
# More importantly, for threshold-crossing we just need the mean per record
INCART_DONE_SCORES = list(INCART_PARTIAL.values())  # 15 mean values


def load_model():
    torch.set_num_threads(4)
    ckpt  = torch.load(CKPT_PATH, map_location="cpu")
    model = QRSModel(encoder4qrs(), decoder4qrs(), phi_qrs())
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def window_ue(model, window_1d, fs):
    processed = preprocess_ecg(window_1d.astype(np.float32), fs=fs)
    if processed.ndim == 1:
        processed = processed[np.newaxis, :]
    elif processed.shape[0] > processed.shape[1]:
        processed = processed.T
    sig_t = torch.from_numpy(processed).unsqueeze(0)
    with torch.no_grad():
        logits = model(sig_t, return_projection=True)
        logits = logits.squeeze(-1).squeeze(0).cpu().numpy()
    return float(np.mean(uncertain_est(logits)))


def score_wfdb(model, db_dir, record_ids, fs_default, max_win=10):
    import wfdb
    results = {}
    for rid in record_ids:
        try:
            rec    = wfdb.rdrecord(str(db_dir / rid), channels=[0])
            signal = rec.p_signal[:, 0].astype(np.float32)
            fs     = int(rec.fs)
            win    = fs * WIN_SEC
            scores = []
            for s in range(0, len(signal) - win + 1, win):
                if len(scores) >= max_win:
                    break
                scores.append(window_ue(model, signal[s:s+win], fs))
            results[rid] = scores
            print(f"  {rid}: {len(scores)} windows  mean_ue={np.mean(scores):.4f}",
                  flush=True)
        except Exception as e:
            print(f"  [SKIP] {rid}: {e}", flush=True)
    return results


def main():
    print(f"=== Pre-computed CPSC2019-Test (200 windows) ===")
    print(f"  mean={CPSC_MEAN:.4f}  std={CPSC_STD:.4f}  threshold(p90)={CPSC_THRESHOLD:.4f}")
    print(f"  Simulated per-window scores: {200} records\n")

    model = load_model()

    # INCART remaining (I16-I24)
    all_incart_hea = [p.stem for p in sorted(INCART_DIR.glob("I*.hea"))]
    remaining_incart = [r for r in all_incart_hea if r not in INCART_PARTIAL]
    print(f"=== INCART remaining records: {remaining_incart} ===")
    new_incart = score_wfdb(model, INCART_DIR, remaining_incart,
                            INCART_FS, max_win=5)
    new_incart_scores = [np.mean(v) for v in new_incart.values() if v]

    all_incart_scores = INCART_DONE_SCORES + new_incart_scores
    print(f"\n  INCART total: {len(all_incart_scores)} record-means")
    print(f"  U_E  mean={np.mean(all_incart_scores):.4f}  std={np.std(all_incart_scores):.4f}")
    flagged = sum(s > CPSC_THRESHOLD for s in all_incart_scores)
    print(f"  Flagged as noise: {flagged}/{len(all_incart_scores)} "
          f"({100*flagged/len(all_incart_scores):.1f}%)  [target <= 10%]")

    # NSTDB pure noise
    print("\n=== NSTDB pure noise (em / ma / bw) ===")
    nstdb_records = [r for r in ["em", "ma", "bw"]
                     if (NSTDB_DIR / (r + ".hea")).exists()]
    print(f"  Records found: {nstdb_records}")
    nstdb_results = score_wfdb(model, NSTDB_DIR, nstdb_records,
                               NSTDB_FS, max_win=15)
    noise_scores = [s for v in nstdb_results.values() for s in v]
    print(f"\n  Noise windows total: {len(noise_scores)}")
    if noise_scores:
        print(f"  U_E  mean={np.mean(noise_scores):.4f}  std={np.std(noise_scores):.4f}")
        print(f"  min={np.min(noise_scores):.4f}  max={np.max(noise_scores):.4f}")

    # Combined metrics
    print("\n=== Combined Noise-Screening Metrics ===")
    threshold = CPSC_THRESHOLD

    # Use CPSC2019 window-level FPR (by definition alpha=0.1, so 10% of 200 = 20 FP)
    cpsc_fp_count = int(ALPHA * 200)  # expected at p90 threshold
    incart_fp_count = sum(s > threshold for s in all_incart_scores)
    noise_tp_count  = sum(s > threshold for s in noise_scores)
    noise_fn_count  = len(noise_scores) - noise_tp_count
    clean_total = 200 + len(all_incart_scores)
    clean_fp    = cpsc_fp_count + incart_fp_count
    clean_tn    = clean_total - clean_fp

    Se  = noise_tp_count / (len(noise_scores) + 1e-9)
    Sp  = clean_tn / (clean_total + 1e-9)
    Acc = (noise_tp_count + clean_tn) / (clean_total + len(noise_scores) + 1e-9)

    print(f"  Threshold = {threshold:.4f}  (alpha={ALPHA})")
    print(f"  Clean ECG windows: {clean_total}  FP: {clean_fp}  TN: {clean_tn}")
    print(f"  Noise windows:     {len(noise_scores)}  TP: {noise_tp_count}  FN: {noise_fn_count}")
    print(f"  Se  (noise recall)    = {Se:.4f}  ({Se*100:.2f}%)")
    print(f"  Sp  (ECG specificity) = {Sp:.4f}  ({Sp*100:.2f}%)")
    print(f"  Acc                   = {Acc:.4f}  ({Acc*100:.2f}%)")
    print(f"\n  Paper Table I (ECG-set Acc) = 96.84%")
    print(f"  This run Acc = {Acc*100:.2f}%")
    print("\n=== NSTDB per-record breakdown ===")
    for rid, scores in nstdb_results.items():
        if scores:
            tp = sum(s > threshold for s in scores)
            print(f"  {rid}: {tp}/{len(scores)} detected  "
                  f"({100*tp/len(scores):.0f}%)  mean_ue={np.mean(scores):.4f}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
