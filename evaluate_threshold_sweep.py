"""
PN-QRS Noise Screening — Threshold Sweep
==========================================
Two-stage pipeline:
  Stage 1  Run model once, cache per-window U_E values to ue_cache.json.
  Stage 2  Load cache, sweep candidate thresholds, print Se/Sp/Acc table.

Key improvements vs evaluate_noise_screening.py:
  - Uses GPU (CUDA_VISIBLE_DEVICES selects which card)
  - 18 windows per INCART record (covers ~3 min of 30-min Holter)
  - Full threshold sweep: P80 to P99.9 of CPSC2019-Test U_E
  - Caching avoids re-running the network on subsequent runs

Usage:
  CUDA_VISIBLE_DEVICES=3 python -u evaluate_threshold_sweep.py \
    > threshold_sweep.log 2>&1

  # Skip model inference on subsequent runs
  python -u evaluate_threshold_sweep.py --no-recompute \
    > threshold_sweep.log 2>&1
"""
import argparse
import json
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

# ── paths ────────────────────────────────────────────────────────────────────
CPSC2019_TEST_DIR = Path("/home/kailong/ECG/ECG/data/PN-QRS/CPSC2019_real_data/cpsc2019_test")
INCART_DIR        = Path("/home/kailong/ECG/ECG/data/PN-QRS/INCART")
NSTDB_DIR         = Path("/home/kailong/ECG/ECG/data/PN-QRS/NSTDB")
CKPT_PATH         = Path("/home/kailong/ECG/ECG/ECGFounder/PN-QRS/experiments"
                         "/logs_real/zy2lki18/models/best_model.pt")
CACHE_PATH        = Path("/home/kailong/ECG/ECG/ECGFounder/PN-QRS/ue_cache.json")

WIN_SEC  = 10
FS_MODEL = 50
NSTDB_NOISE_RECORDS = ["em", "ma", "bw"]


# ── model helpers ────────────────────────────────────────────────────────────

def load_model(device):
    torch.set_num_threads(4)
    ckpt  = torch.load(str(CKPT_PATH), map_location="cpu")
    model = QRSModel(encoder4qrs(), decoder4qrs(), phi_qrs()).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.train(mode=False)   # inference mode (same as model.eval())
    return model


def window_ue(model, window_1d: np.ndarray, fs: int, device) -> float:
    processed = preprocess_ecg(window_1d, fs=fs)
    if processed.ndim == 1:
        processed = processed[np.newaxis, :]
    elif processed.shape[0] > processed.shape[1]:
        processed = processed.T
    sig_t = torch.from_numpy(processed).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(sig_t, return_projection=True)
        logits = logits.squeeze(-1).squeeze(0).cpu().numpy()
    return float(np.mean(uncertain_est(logits)))


def windows_from_signal(signal_1d, fs):
    win = fs * WIN_SEC
    for s in range(0, len(signal_1d) - win + 1, win):
        yield signal_1d[s: s + win]


# ── per-dataset scoring ──────────────────────────────────────────────────────

def score_cpsc2019(model, device):
    import scipy.io as sio
    data_dir = CPSC2019_TEST_DIR / "data"
    files = sorted(data_dir.glob("data_*.mat"))
    scores = []
    for i, f in enumerate(files):
        try:
            ecg = sio.loadmat(str(f))["ecg"][:, 0].astype(np.float32)
            scores.append(window_ue(model, ecg, fs=500, device=device))
            if (i + 1) % 50 == 0:
                print(f"  CPSC2019 [{i+1}/{len(files)}]", flush=True)
        except Exception as e:
            print(f"  [SKIP] {f.name}: {e}", flush=True)
    print(f"  CPSC2019 done: {len(scores)} windows", flush=True)
    return scores


def score_incart(model, device, max_win_per_rec=18):
    """18 windows x 10s = 3 min per 30-min Holter record."""
    import wfdb
    records = sorted(p.stem for p in INCART_DIR.glob("I*.hea"))
    if not records:
        records = sorted(p.stem for p in INCART_DIR.glob("*.hea"))
    scores, rec_info = [], []
    for rid in records:
        try:
            rec = wfdb.rdrecord(str(INCART_DIR / rid), channels=[0])
            sig = rec.p_signal[:, 0].astype(np.float32)
            fs  = int(rec.fs)
            wins = list(windows_from_signal(sig, fs))[:max_win_per_rec]
            s = [window_ue(model, w, fs, device) for w in wins]
            scores.extend(s)
            rec_info.append({"id": rid, "wins": len(s),
                             "mean_ue": float(np.mean(s)), "max_ue": float(np.max(s))})
            print(f"  INCART {rid}: {len(s)} wins  mean={np.mean(s):.3f}  "
                  f"max={np.max(s):.3f}", flush=True)
        except Exception as e:
            print(f"  [SKIP] {rid}: {e}", flush=True)
    return scores, rec_info


def score_nstdb(model, device, max_win=15):
    import wfdb
    scores_by_type = {}
    for rid in NSTDB_NOISE_RECORDS:
        hea = NSTDB_DIR / (rid + ".hea")
        if not hea.exists():
            continue
        try:
            rec = wfdb.rdrecord(str(NSTDB_DIR / rid), channels=[0])
            sig = rec.p_signal[:, 0].astype(np.float32)
            fs  = int(rec.fs)
            wins = list(windows_from_signal(sig, fs))[:max_win]
            s = [window_ue(model, w, fs, device) for w in wins]
            scores_by_type[rid] = s
            print(f"  NSTDB {rid}: {len(s)} wins  mean={np.mean(s):.3f}  "
                  f"min={np.min(s):.3f}", flush=True)
        except Exception as e:
            print(f"  [SKIP] {rid}: {e}", flush=True)
    return scores_by_type


# ── threshold sweep ──────────────────────────────────────────────────────────

def sweep_thresholds(cpsc_scores, incart_scores, noise_scores):
    """
    Sweep candidate thresholds; for each compute Se / Sp / Acc / F1.
    Clean ECG = CPSC2019 + INCART (label 0), Noise = NSTDB (label 1).
    """
    clean = list(cpsc_scores) + list(incart_scores)
    noisy = list(noise_scores)
    all_scores = clean + noisy
    all_labels = [0] * len(clean) + [1] * len(noisy)

    cpsc_arr = np.array(cpsc_scores)
    pcts = [80, 85, 88, 90, 92, 94, 95, 96, 97, 98, 99, 99.5, 99.9]
    thresholds = sorted(set(
        [float(np.quantile(cpsc_arr, p / 100)) for p in pcts]
        + list(np.arange(1.5, max(all_scores) + 0.1, 0.05))
    ))

    print("\n" + "=" * 80)
    print(f"{'Threshold':>10} {'Se%':>7} {'Sp%':>7} {'Acc%':>7} "
          f"{'F1_noise%':>10} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
    print("=" * 80)

    results = []
    prev_se, prev_sp = -1, -1
    for t in thresholds:
        preds = [1 if s > t else 0 for s in all_scores]
        TP = sum(p == 1 and l == 1 for p, l in zip(preds, all_labels))
        FN = sum(p == 0 and l == 1 for p, l in zip(preds, all_labels))
        TN = sum(p == 0 and l == 0 for p, l in zip(preds, all_labels))
        FP = sum(p == 1 and l == 0 for p, l in zip(preds, all_labels))
        Se  = TP / (TP + FN + 1e-9)
        Sp  = TN / (TN + FP + 1e-9)
        Acc = (TP + TN) / (len(all_scores) + 1e-9)
        F1  = 2 * TP / (2 * TP + FP + FN + 1e-9)
        if abs(Se - prev_se) > 0.005 or abs(Sp - prev_sp) > 0.005:
            print(f"{t:10.4f} {Se*100:7.2f} {Sp*100:7.2f} {Acc*100:7.2f} "
                  f"{F1*100:10.2f} {TP:5d} {FP:5d} {TN:5d} {FN:5d}", flush=True)
            prev_se, prev_sp = Se, Sp
        results.append({"t": t, "Se": Se, "Sp": Sp, "Acc": Acc, "F1": F1,
                         "TP": TP, "FP": FP, "TN": TN, "FN": FN})

    best_f1    = max(results, key=lambda r: r["F1"])
    sp96 = [r for r in results if r["Sp"] >= 0.96]
    best_sp96  = max(sp96, key=lambda r: r["Se"]) if sp96 else None
    se99 = [r for r in results if r["Se"] >= 0.99]
    best_se99  = max(se99, key=lambda r: r["Sp"]) if se99 else None

    print("\n── Recommended Operating Points ─────────────────────────────────────")
    for label, r in [("Max F1-noise", best_f1),
                     ("Sp>=96%  max Se", best_sp96),
                     ("Se>=99%  max Sp", best_se99)]:
        if r is None:
            print(f"  {label}: N/A")
            continue
        print(f"  {label}: t={r['t']:.4f}  Se={r['Se']*100:.2f}%  "
              f"Sp={r['Sp']*100:.2f}%  Acc={r['Acc']*100:.2f}%  "
              f"F1={r['F1']*100:.2f}%  "
              f"TP={r['TP']} FP={r['FP']} TN={r['TN']} FN={r['FN']}")

    return results, best_f1, best_sp96, best_se99


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-recompute", action="store_true",
                        help="Skip model inference, use cached ue_cache.json")
    args = parser.parse_args()

    if args.no_recompute and CACHE_PATH.exists():
        print(f"Loading cached U_E values from {CACHE_PATH}")
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        cpsc_scores   = cache["cpsc"]
        incart_scores = cache["incart"]
        incart_info   = cache.get("incart_info", [])
        noise_by_type = cache["nstdb"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}", flush=True)
        print(f"Checkpoint: {CKPT_PATH}\n", flush=True)
        model = load_model(device)

        print("=== CPSC2019-Test (clean ECG) ===", flush=True)
        cpsc_scores = score_cpsc2019(model, device)
        print(f"  mean={np.mean(cpsc_scores):.4f}  std={np.std(cpsc_scores):.4f}  "
              f"p90={np.quantile(cpsc_scores, 0.90):.4f}  "
              f"p95={np.quantile(cpsc_scores, 0.95):.4f}  "
              f"p99={np.quantile(cpsc_scores, 0.99):.4f}", flush=True)

        print("\n=== INCART 12-lead Holter (18 windows/record) ===", flush=True)
        incart_scores, incart_info = score_incart(model, device)
        print(f"  Total windows: {len(incart_scores)}", flush=True)
        if incart_scores:
            print(f"  mean={np.mean(incart_scores):.4f}  std={np.std(incart_scores):.4f}  "
                  f"p90={np.quantile(incart_scores, 0.90):.4f}", flush=True)

        print("\n=== NSTDB pure noise ===", flush=True)
        noise_by_type = score_nstdb(model, device)
        for k, v in noise_by_type.items():
            print(f"  {k}: {len(v)} wins  mean={np.mean(v):.4f}  min={np.min(v):.4f}",
                  flush=True)

        cache = {"cpsc": cpsc_scores, "incart": incart_scores,
                 "incart_info": incart_info, "nstdb": noise_by_type}
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"\nCached U_E values -> {CACHE_PATH}", flush=True)

    noise_scores = [s for v in noise_by_type.values() for s in v]

    print(f"\n── Distribution Summary ─────────────────────────────────────────────")
    print(f"  CPSC2019-Test : n={len(cpsc_scores)}  "
          f"mean={np.mean(cpsc_scores):.4f}  "
          f"p90={np.quantile(cpsc_scores,0.9):.4f}  "
          f"p99={np.quantile(cpsc_scores,0.99):.4f}")
    print(f"  INCART Holter : n={len(incart_scores)}  "
          f"mean={np.mean(incart_scores):.4f}  "
          f"p90={np.quantile(incart_scores,0.9):.4f}")
    if noise_scores:
        print(f"  NSTDB noise   : n={len(noise_scores)}  "
              f"mean={np.mean(noise_scores):.4f}  min={np.min(noise_scores):.4f}")

    if not noise_scores:
        print("No noise scores — cannot compute metrics.")
        return

    sweep_thresholds(cpsc_scores, incart_scores, noise_scores)


if __name__ == "__main__":
    main()
