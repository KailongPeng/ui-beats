"""
CPSC2019 官方评测协议：Se / P+ / F1
- 模型输出 50Hz mask -> correct() 后处理得到 R-peak 位置（50Hz空间）
- x10 转换回 500Hz 空间
- 与标注比较，tolerance = 75ms = 37 samples at 500Hz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).parent))
from dataset.dataset import CPSC2019Dataset, cpsc2019_collate_fn
from models.qrs_model import QRSModel
from models.multi_head import encoder4qrs, decoder4qrs, phi_qrs
from utils.qrs_post_process import correct, uncertain_est


FS_MODEL  = 50    # mask sampling rate (model output)
FS_SIGNAL = 500   # original signal sampling rate
SCALE     = FS_SIGNAL // FS_MODEL   # 10: convert 50Hz to 500Hz
TOLERANCE = int(0.075 * FS_SIGNAL)  # 37 samples at 500Hz = 75ms


def find_best_checkpoint(log_dir: Path) -> Path:
    candidates = sorted(log_dir.rglob("best_model.pt"))
    if not candidates:
        raise FileNotFoundError(f"No best_model.pt found under {log_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def predict_r_peaks(model, signal_tensor, device):
    """
    signal_tensor: (1, 1, 2000) single record
    Returns: predicted R-peak positions in 500Hz space (numpy array)

    Uses return_projection=True to get [logits_o, logits_p, logits_n],
    then applies uncertainty-aware post-processing (uncertain_est).
    """
    model.eval()
    with torch.no_grad():
        signal_tensor = signal_tensor.to(device)
        # shape: (1, 3, 500, 1) — stack of [logits_o, logits_p, logits_n]
        logits = model(signal_tensor, return_projection=True)
        logits = logits.squeeze(-1).squeeze(0).cpu().numpy()   # (3, 500)

    # uncertain_est expects (C, T) with C=3
    uc = uncertain_est(logits)          # (500,) uncertainty sequence
    prob = logits[0]                    # (500,) QRS probability from logits_o

    r_peaks_50hz = correct(prob, uc)

    if len(r_peaks_50hz) == 0:
        return np.array([], dtype=int)

    r_peaks_500hz = (np.array(r_peaks_50hz) * SCALE).astype(int)
    return r_peaks_500hz


def compute_metrics(all_pred, all_ref):
    """Global Se, P+, F1 over all records."""
    TP_total = FP_total = FN_total = 0

    for pred, ref in zip(all_pred, all_ref):
        ref = ref[(ref >= int(0.5 * FS_SIGNAL)) & (ref <= int(9.5 * FS_SIGNAL))]
        matched_pred = set()
        TP = 0
        for r in ref:
            hits = np.where(np.abs(pred - r) <= TOLERANCE)[0]
            if len(hits) > 0:
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
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_best_checkpoint(
        Path("/home/kailong/ECG/ECG/ECGFounder/PN-QRS/experiments")
    )
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = QRSModel(encoder4qrs(), decoder4qrs(), phi_qrs()).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_root = Path("/home/kailong/ECG/ECG/data/PN-QRS/CPSC2019_real_data/cpsc2019_test")
    dataset = CPSC2019Dataset(root=test_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=cpsc2019_collate_fn)
    print(f"Test records: {len(dataset)}")

    all_pred, all_ref = [], []

    for signals, masks, record_ids in loader:
        record_id = record_ids[0]
        ref_path = test_root / "ref" / f"R_{record_id}.mat"
        r_ref = loadmat(str(ref_path))["R_peak"].flatten()
        r_ref = np.unique(np.round(r_ref).astype(int))

        r_pred = predict_r_peaks(model, signals, device)
        all_pred.append(r_pred)
        all_ref.append(r_ref)

    Se, Pp, F1, TP, FP, FN = compute_metrics(all_pred, all_ref)

    print("\n========== CPSC2019 Evaluation (thr=75ms) ==========")
    print(f"  TP={TP}  FP={FP}  FN={FN}")
    print(f"  Se  (Sensitivity)       = {Se:.4f}")
    print(f"  P+  (Pos. Predictivity) = {Pp:.4f}")
    print(f"  F1                      = {F1:.4f}")
    print("=====================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=2)
    args = parser.parse_args()
    main(args)
