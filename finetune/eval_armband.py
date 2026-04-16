#!/usr/bin/env python3
"""
eval_armband.py -- Paired evaluation of baseline vs fine-tuned checkpoint
on arm-band CH20 R-peak detection.

Reference labels: *_CH1-8_rpeaks.csv  (PN-QRS pseudo-labels from 12-lead).
Metric: Se / P+ / F1 @ 150 ms tolerance (match_peaks from evaluate_upper_arm).

Usage:
  python finetune/eval_armband.py \\
    --data_dir data/0410_real --fs 1000 \\
    --test_subject subject01 \\
    --baseline_ckpt  experiments/logs_real/zy2lki18/models/best_model.pt \\
    --finetuned_ckpt experiments/logs_armband/20250416_120000_subject01/models/best_model.pt
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

UI_BEAT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(UI_BEAT_ROOT))

from models.multi_head import encoder4qrs, decoder4qrs, phi_qrs
from models.qrs_model import QRSModel
from pipeline.extract_quality_segments import run_inference
from pipeline.evaluate_upper_arm import match_peaks, TOL_SEC


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_from_ckpt(ckpt_path: Path, device) -> QRSModel:
    """Load any QRSModel-format checkpoint into a fresh QRSModel and set eval mode."""
    ckpt  = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = QRSModel(encoder4qrs(), decoder4qrs(), phi_qrs()).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference: CH20 R-peaks for a single CSV
# ─────────────────────────────────────────────────────────────────────────────

def predict_rpeaks(model: QRSModel, csv_path: Path, fs: int, device) -> np.ndarray:
    """
    Sliding-window inference on CH20 column, returns sorted absolute R-peak indices
    at original fs.  Reuses run_inference from pipeline/extract_quality_segments.py
    (which handles batching, preprocessing, and post-processing).
    """
    df  = pd.read_csv(csv_path)
    col = next((c for c in df.columns if str(c).upper() == "CH20"), None)
    if col is None:
        return np.array([], dtype=int)
    signal = df[col].values.astype(np.float32)

    windows = run_inference(signal, fs, model, device)
    if not windows:
        return np.array([], dtype=int)

    all_peaks: list = []
    for w in windows:
        all_peaks.extend(w["r_peaks_abs"].tolist())

    return np.array(sorted(set(all_peaks)), dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# Per-file evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_file(model: QRSModel, csv_path: Path, fs: int, device):
    """
    Returns a metrics dict, or None if the reference CSV does not exist.
    """
    stem     = csv_path.stem
    ref_path = csv_path.with_name(f"{stem}_CH1-8_rpeaks.csv")
    if not ref_path.exists():
        return None

    ref  = pd.read_csv(ref_path)["sample_index"].values.astype(int)
    pred = predict_rpeaks(model, csv_path, fs, device)
    tol  = int(TOL_SEC * fs)

    tp, fp, fn = match_peaks(ref, pred, tol)
    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pp = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * se * pp / (se + pp) if (se + pp) > 0 else 0.0
    return dict(tp=tp, fp=fp, fn=fn,
                se=round(se, 4), pp=round(pp, 4), f1=round(f1, 4),
                n_ref=len(ref), n_pred=len(pred))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline vs fine-tuned checkpoint on arm-band CH20"
    )
    parser.add_argument("--data_dir",       required=True)
    parser.add_argument("--fs",             type=int, required=True)
    parser.add_argument("--test_subject",   required=True)
    parser.add_argument("--baseline_ckpt",  required=True)
    parser.add_argument("--finetuned_ckpt", required=True)
    parser.add_argument("--out_dir",        default=None,
                        help="Output directory; defaults to <finetuned_ckpt>/../eval/")
    parser.add_argument("--gpu",            default="0")
    args = parser.parse_args()

    # ── device ──────────────────────────────────────────────────────────
    if torch.cuda.is_available() and args.gpu != "cpu":
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── load models ─────────────────────────────────────────────────────
    print("Loading baseline  ...")
    baseline  = load_model_from_ckpt(Path(args.baseline_ckpt),  device)
    print("Loading fine-tuned...")
    finetuned = load_model_from_ckpt(Path(args.finetuned_ckpt), device)

    # ── find CSV files ───────────────────────────────────────────────────
    skip_sfx = (
        "_quality_report.csv", "_CH1-8_rpeaks.csv", "_CH20_rpeaks.csv", "_wave_sqi.csv",
    )
    data_dir = Path(args.data_dir)
    subj_dir = data_dir / args.test_subject
    if not subj_dir.exists():
        raise ValueError(f"Subject directory not found: {subj_dir}")

    csv_files = sorted(
        p for p in subj_dir.rglob("*.csv")
        if not any(p.name.endswith(s) for s in skip_sfx)
           and p.name not in {"batch_quality_summary.csv", "batch_wave_sqi_summary.csv"}
    )
    if not csv_files:
        raise ValueError(f"No CSV files found under {subj_dir}")
    print(f"Found {len(csv_files)} CSV files for subject '{args.test_subject}'")

    # ── evaluate ─────────────────────────────────────────────────────────
    rows: list = []
    for csv_path in csv_files:
        rel_parts = csv_path.relative_to(subj_dir).parts
        activity  = rel_parts[0] if len(rel_parts) >= 2 else "unknown"
        stem      = csv_path.stem

        print(f"  [{activity}/{stem}] ", end="", flush=True)
        b_res = eval_file(baseline,  csv_path, args.fs, device)
        f_res = eval_file(finetuned, csv_path, args.fs, device)

        if b_res is None:
            print("skip (no reference *_CH1-8_rpeaks.csv)")
            continue

        print(f"base F1={b_res['f1']:.3f}  fine F1={f_res['f1'] if f_res else 'N/A':.3f}")

        rows.append(dict(
            subject=args.test_subject, activity=activity, stem=stem,
            b_se=b_res["se"],  b_pp=b_res["pp"],  b_f1=b_res["f1"],
            f_se=f_res["se"] if f_res else None,
            f_pp=f_res["pp"] if f_res else None,
            f_f1=f_res["f1"] if f_res else None,
            b_tp=b_res["tp"], b_fp=b_res["fp"], b_fn=b_res["fn"],
            f_tp=f_res["tp"] if f_res else None,
            f_fp=f_res["fp"] if f_res else None,
            f_fn=f_res["fn"] if f_res else None,
        ))

    if not rows:
        print("No evaluable files found (missing *_CH1-8_rpeaks.csv for all).")
        return

    # ── output directory ─────────────────────────────────────────────────
    if args.out_dir is None:
        out_dir = Path(args.finetuned_ckpt).parent.parent / "eval"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── eval_results.csv ─────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    res_path = out_dir / "eval_results.csv"
    df.to_csv(res_path, index=False)
    print(f"\nSaved : {res_path}")

    # ── per-activity summary ─────────────────────────────────────────────
    activities = sorted(df["activity"].unique())
    summary: list = []
    for act in activities:
        sub  = df[df["activity"] == act]
        b_f1 = float(sub["b_f1"].mean())
        f_f1 = float(sub["f_f1"].mean())
        summary.append(dict(activity=act,
                            baseline_F1=round(b_f1, 4),
                            finetuned_F1=round(f_f1, 4),
                            delta=round(f_f1 - b_f1, 4),
                            n_files=len(sub)))
    b_overall = float(df["b_f1"].mean())
    f_overall = float(df["f_f1"].mean())
    summary.append(dict(activity="OVERALL",
                        baseline_F1=round(b_overall, 4),
                        finetuned_F1=round(f_overall, 4),
                        delta=round(f_overall - b_overall, 4),
                        n_files=len(df)))

    # ── eval_summary.md ──────────────────────────────────────────────────
    md_path = out_dir / "eval_summary.md"
    with open(md_path, "w") as fh:
        fh.write(f"# Eval Summary — Subject: {args.test_subject}\n\n")
        fh.write(
            "> **Caveat**: CH1-8 labels are PN-QRS pseudo-labels produced by the 12-lead teacher "
            "model, **not** human-annotated ground truth. An improvement in F1 means the fine-tuned "
            "student better imitates the teacher on CH20, not necessarily that it is closer to the "
            "true R-peaks. Arm-band vs 12-lead differences in lead polarity, morphology, or "
            "timing are not captured by this metric. True gold-standard evaluation requires manual "
            "annotation of a held-out subset.\n\n"
        )
        fh.write("## Per-activity mean F1 @ 150 ms\n\n")
        fh.write("| Activity | Baseline F1 | Fine-tuned F1 | Δ | n_files |\n")
        fh.write("|----------|------------|--------------|---|--------|\n")
        for r in summary:
            delta_str = f"{r['delta']:+.4f}"
            fh.write(
                f"| {r['activity']} | {r['baseline_F1']:.4f} | "
                f"{r['finetuned_F1']:.4f} | {delta_str} | {r['n_files']} |\n"
            )
        fh.write(f"\n**Baseline ckpt**: `{args.baseline_ckpt}`\n")
        fh.write(f"**Fine-tuned ckpt**: `{args.finetuned_ckpt}`\n")
        fh.write(f"**Tolerance**: {TOL_SEC * 1000:.0f} ms\n")
    print(f"Saved : {md_path}")

    print(
        f"\nOverall  baseline F1={b_overall:.4f}  "
        f"fine-tuned F1={f_overall:.4f}  "
        f"Δ={f_overall - b_overall:+.4f}"
    )


if __name__ == "__main__":
    main()
