import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""
MIMIC-IV ECG Noise Proportion Estimation via PN-QRS U_E
Uses 200 randomly sampled 10-s records; thresholds t=2.45 (aggressive) and t=3.20 (conservative).
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import torch
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from dataset.dataset import preprocess_ecg
from models.qrs_model import QRSModel
from models.multi_head import encoder4qrs, decoder4qrs, phi_qrs
from utils.qrs_post_process import uncertain_est

CKPT = Path("/home/kailong/ECG/ECG/ECGFounder/PN-QRS/experiments"
            "/logs_real/zy2lki18/models/best_model.pt")
SAMPLE_LIST = Path("/tmp/mimic_sample_list.txt")
N_SAMPLE = 200
T_AGG  = 2.45   # aggressive threshold (t=noise_min from NSTDB)
T_CONS = 3.20   # conservative threshold (NSTDB em mean=3.11)

def load_model():
    torch.set_num_threads(4)
    ckpt  = torch.load(CKPT, map_location="cpu")
    model = QRSModel(encoder4qrs(), decoder4qrs(), phi_qrs())
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def infer_ue(model, signal_1d, fs=500):
    processed = preprocess_ecg(signal_1d.astype(np.float32), fs=fs)
    if processed.ndim == 1:
        processed = processed[np.newaxis, :]
    elif processed.shape[0] > processed.shape[1]:
        processed = processed.T
    sig_t = torch.from_numpy(processed).unsqueeze(0)
    with torch.no_grad():
        logits = model(sig_t, return_projection=True)
        logits = logits.squeeze(-1).squeeze(0).cpu().numpy()
    return float(np.mean(uncertain_est(logits)))

def main():
    import wfdb
    hea_paths = [l.strip() for l in open(SAMPLE_LIST) if l.strip()][:N_SAMPLE]
    print(f"Sampling {len(hea_paths)} MIMIC-IV records (CPU inference)...", flush=True)

    model = load_model()
    scores = []
    errors = 0
    for i, hea_path in enumerate(hea_paths):
        rec_path = hea_path.replace(".hea", "")
        try:
            rec    = wfdb.rdrecord(rec_path, channels=[0])
            signal = rec.p_signal[:, 0].astype(np.float32)
            fs     = int(rec.fs)
            ue     = infer_ue(model, signal, fs=fs)
            scores.append(ue)
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(hea_paths)}]  last_ue={ue:.4f}  "
                      f"running_noise_pct(t={T_AGG})={100*sum(s>T_AGG for s in scores)/len(scores):.1f}%",
                      flush=True)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [SKIP] {rec_path}: {e}", flush=True)

    print(f"\n=== MIMIC-IV Noise Estimation ({len(scores)} records) ===")
    print(f"  U_E  mean={np.mean(scores):.4f}  std={np.std(scores):.4f}")
    print(f"  p50={np.quantile(scores,0.50):.4f}  p75={np.quantile(scores,0.75):.4f}")
    print(f"  p90={np.quantile(scores,0.90):.4f}  p95={np.quantile(scores,0.95):.4f}")
    print(f"  p99={np.quantile(scores,0.99):.4f}  max={np.max(scores):.4f}")

    for t, label in [(T_AGG, "激进 (t=2.45)"), (T_CONS, "保守 (t=3.20)")]:
        flagged = sum(s > t for s in scores)
        pct = 100 * flagged / len(scores)
        kept_800k = int(800035 * (1 - flagged / len(scores)))
        print(f"\n  阈值 {label}:")
        print(f"    噪声比例 = {flagged}/{len(scores)} = {pct:.2f}%")
        print(f"    保留率   = {100-pct:.2f}%")
        print(f"    推算800K后保留 ≈ {kept_800k:,} 条")

    # Save for cache
    result = {"mimic_sample": scores, "n": len(scores), "errors": errors}
    cache_path = Path(__file__).parent / "ue_cache.json"
    with open(cache_path) as f:
        cache = json.load(f)
    cache["mimic_sample"] = scores
    with open(cache_path, "w") as f:
        json.dump(cache, f)
    print(f"\n  Scores saved to ue_cache.json ['mimic_sample']")
    print(f"  Errors/skipped: {errors}")
    print("=== Done ===")

if __name__ == "__main__":
    main()
