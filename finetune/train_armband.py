#!/usr/bin/env python3
"""
train_armband.py -- AEU fine-tune PN-QRS on arm-band CH20 data.

Leave-One-Subject-Out (LOSO): --test_subject is held out as validation;
all other subjects form the training set.

Usage:
  python finetune/train_armband.py \
    --data_dir data/0410_real --fs 1000 \
    --test_subject subject01 --gpu 0
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

UI_BEAT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(UI_BEAT_ROOT))

from models.multi_head import encoder4qrs, decoder4qrs, phi_qrs
from utils.loss import bce_loss_func, sim_loss_func
from finetune.armband_dataset import ArmbandWindowDataset, armband_collate_fn


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def estimate_r(z: Tensor, mask: Tensor) -> Tensor:
    """
    Compute per-region prototype by averaging encoder features z over mask-selected
    positions, then broadcast back to (B, T, D).
    Mirrors BeatTrainer.estimate_r (training/beat_trainer.py:275-285).
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)      # (B, T, 1)
    mask = mask.to(z.dtype)
    denominator = mask.sum()
    if denominator.item() == 0:
        base = torch.zeros(z.size(-1), device=z.device, dtype=z.dtype)
    else:
        weighted = (z * mask).sum(dim=(0, 1))
        base = weighted / denominator
    return base.unsqueeze(0).unsqueeze(0).expand(z.size(0), z.size(1), -1)


def aeu_step(en, de, phi,
             x: Tensor, y: Tensor,
             alpha_opt, theta_opt):
    """
    One AEU optimisation step:
      α branch — encoder + decoder updated via BCE loss
      θ branch — phi updated via similarity loss on detached z

    Args:
        x : (B, 1, T_200hz)  — preprocessed signal
        y : (B, T_50hz)      — binary R-peak mask

    Returns:
        (alpha_loss_val, theta_loss_val)
    """
    y3 = y.unsqueeze(-1)       # (B, T_50, 1)  for BCE and estimate_r

    # ── α branch ──────────────────────────────────────────────────────────
    en.train()
    de.train()
    alpha_opt.zero_grad()
    z   = en(x)                # (B, T_enc, D)
    lgt = de(z)                # (B, T_50, 1)
    alpha_loss = bce_loss_func(lgt, y3)
    alpha_loss.backward()
    alpha_opt.step()

    z_det = z.detach()

    # ── θ branch ──────────────────────────────────────────────────────────
    phi.train()
    theta_opt.zero_grad()
    mask_q  = y3               # QRS region
    mask_nq = 1.0 - mask_q    # non-QRS region

    z_p, z_n = phi(z_det)
    r_p = estimate_r(z_det, mask_q)
    r_n = estimate_r(z_det, mask_nq)

    # z_p pulls towards z outside QRS, towards r_p inside QRS
    sim_loss_p = (sim_loss_func(z_det, z_p, mask_nq) +
                  sim_loss_func(r_p,   z_p, mask_q))
    # z_n pulls towards z inside QRS, towards r_n outside QRS
    sim_loss_n = (sim_loss_func(z_det, z_n, mask_q) +
                  sim_loss_func(r_n,   z_n, mask_nq))

    theta_loss = sim_loss_p + sim_loss_n
    theta_loss.backward()
    theta_opt.step()

    return alpha_loss.item(), theta_loss.item()


@torch.no_grad()
def evaluate_bce(en, de, loader, device) -> float:
    """Compute mean BCE on loader (no_grad). Used for early stopping."""
    en.eval()
    de.eval()
    total, count = 0.0, 0
    for x, y, _ in loader:
        x = x.to(device)
        y = y.unsqueeze(-1).to(device)
        lgt = de(en(x))
        total += bce_loss_func(lgt, y).item()
        count += 1
    en.train()
    de.train()
    return total / count if count else float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_qrsmodel_state(ckpt_path: Path, en, de, phi, device) -> None:
    """
    Load a QRSModel-format checkpoint and dispatch weights to the three
    sub-modules.  Checkpoint keys follow the pattern:
        encoder.<k>            → en
        decoder.<k>            → de
        projection_head.<k>    → phi
    """
    ckpt  = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)

    en.load_state_dict( {k[len("encoder."):]:         v for k, v in state.items()
                         if k.startswith("encoder.")})
    de.load_state_dict( {k[len("decoder."):]:         v for k, v in state.items()
                         if k.startswith("decoder.")})
    phi.load_state_dict({k[len("projection_head."):]: v for k, v in state.items()
                         if k.startswith("projection_head.")})


def save_qrsmodel_ckpt(path: Path, en, de, phi, meta: dict) -> None:
    """
    Merge sub-module state dicts back into QRSModel format and save.
    Compatible with pipeline/extract_quality_segments.py:load_model.
    """
    merged: dict = {}
    for k, v in en.state_dict().items():
        merged[f"encoder.{k}"] = v
    for k, v in de.state_dict().items():
        merged[f"decoder.{k}"] = v
    for k, v in phi.state_dict().items():
        merged[f"projection_head.{k}"] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": merged, "meta": meta}, str(path))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    default_ckpt = str(UI_BEAT_ROOT / "experiments/logs_real/zy2lki18/models/best_model.pt")

    parser = argparse.ArgumentParser(
        description="AEU fine-tune PN-QRS on arm-band CH20 data (LOSO)"
    )
    parser.add_argument("--data_dir",     required=True,
                        help="Root dir: <data_dir>/<subject>/<activity>/rec*.csv")
    parser.add_argument("--fs",           type=int, required=True,
                        help="Sampling rate of raw CSV (e.g. 1000)")
    parser.add_argument("--test_subject", required=True,
                        help="Name of the hold-out subject directory")
    parser.add_argument("--init_ckpt",    default=default_ckpt,
                        help="Initial QRSModel checkpoint (.pt)")
    parser.add_argument("--save_dir",     default=None,
                        help="Directory for model files; auto-generated if omitted")
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--alpha_lr",     type=float, default=5e-5)
    parser.add_argument("--theta_lr",     type=float, default=5e-5)
    parser.add_argument("--early_stop",   type=int,   default=10)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--gpu",          default="0",
                        help="GPU index, or 'cpu'")
    parser.add_argument("--num_workers",  type=int,   default=0)
    args = parser.parse_args()

    # ── reproducibility ──────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── device ───────────────────────────────────────────────────────────
    if torch.cuda.is_available() and args.gpu != "cpu":
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── subjects ─────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    all_subjects = sorted(
        d.name for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if args.test_subject not in all_subjects:
        raise ValueError(
            f"test_subject '{args.test_subject}' not found in {data_dir}. "
            f"Available: {all_subjects}"
        )
    train_subjects = [s for s in all_subjects if s != args.test_subject]
    val_subjects   = [args.test_subject]
    print(f"Train subjects : {train_subjects}")
    print(f"Val subjects   : {val_subjects}")

    # ── datasets ─────────────────────────────────────────────────────────
    train_ds = ArmbandWindowDataset(data_dir, args.fs, subjects=train_subjects)
    val_ds   = ArmbandWindowDataset(data_dir, args.fs, subjects=val_subjects)

    if len(train_ds) == 0:
        raise RuntimeError(
            "Training dataset is empty. Check data_dir structure and "
            "that *_quality_report.csv files exist."
        )
    if len(val_ds) == 0:
        raise RuntimeError(
            "Validation dataset is empty. Check test_subject directory and "
            "that *_quality_report.csv files exist."
        )

    train_ld = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=armband_collate_fn,
        num_workers=args.num_workers, drop_last=True,
    )
    val_ld = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=armband_collate_fn,
        num_workers=args.num_workers,
    )

    # ── model ────────────────────────────────────────────────────────────
    en  = encoder4qrs().to(device)
    de  = decoder4qrs().to(device)
    phi = phi_qrs().to(device)
    load_qrsmodel_state(Path(args.init_ckpt), en, de, phi, device)
    print(f"Loaded weights : {args.init_ckpt}")

    # ── optimizers ───────────────────────────────────────────────────────
    alpha_opt = torch.optim.Adam(
        list(en.parameters()) + list(de.parameters()),
        lr=args.alpha_lr, betas=(0.9, 0.9), eps=1e-8, amsgrad=True,
    )
    theta_opt = torch.optim.Adam(
        phi.parameters(),
        lr=args.theta_lr, betas=(0.9, 0.9), eps=1e-8, amsgrad=True,
    )

    # ── save directory ───────────────────────────────────────────────────
    if args.save_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        save_dir = (UI_BEAT_ROOT / "experiments" / "logs_armband"
                    / f"{ts}_{args.test_subject}" / "models")
    else:
        save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to      : {save_dir}")

    # ── training loop ────────────────────────────────────────────────────
    best_val = float("inf")
    patience = 0
    history  = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        alpha_losses: list = []
        theta_losses: list = []

        en.train()
        de.train()
        phi.train()
        for x, y, _ in train_ld:
            x = x.to(device)
            y = y.to(device)
            a_l, t_l = aeu_step(en, de, phi, x, y, alpha_opt, theta_opt)
            alpha_losses.append(a_l)
            theta_losses.append(t_l)

        mean_alpha = float(np.mean(alpha_losses)) if alpha_losses else float("inf")
        mean_theta = float(np.mean(theta_losses)) if theta_losses else float("inf")
        val_bce    = evaluate_bce(en, de, val_ld, device)
        elapsed    = time.time() - t0

        if val_bce < best_val:
            best_val = val_bce
            patience = 0
            save_qrsmodel_ckpt(save_dir / "best_model.pt", en, de, phi, vars(args))
            note = "  ★ new best"
        else:
            patience += 1
            note = ""

        save_qrsmodel_ckpt(save_dir / "last_model.pt", en, de, phi, vars(args))

        print(
            f"[epoch {epoch:3d}/{args.epochs}]  "
            f"α={mean_alpha:.4f}  θ={mean_theta:.4f}  "
            f"val_bce={val_bce:.4f}  ({elapsed:.0f}s){note}"
        )

        history.append(dict(
            epoch=epoch,
            train_alpha_loss=mean_alpha,
            train_theta_loss=mean_theta,
            val_bce=val_bce,
            epoch_time_s=round(elapsed, 1),
        ))

        if patience >= args.early_stop:
            print(f"Early stop at epoch {epoch} (patience={args.early_stop}).")
            break

    # ── write history + args ─────────────────────────────────────────────
    exp_dir = save_dir.parent
    history_path = exp_dir / "history.csv"
    with open(history_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"History saved  : {history_path}")

    args_path = exp_dir / "args.json"
    with open(args_path, "w") as fh:
        json.dump(vars(args), fh, indent=2)

    print(f"\nDone.  Best val_bce = {best_val:.4f}")
    print(f"Best model     : {save_dir / 'best_model.pt'}")
    print(f"\nNext step — run evaluation:")
    print(f"  python finetune/eval_armband.py \\")
    print(f"    --data_dir {args.data_dir} --fs {args.fs} \\")
    print(f"    --test_subject {args.test_subject} \\")
    print(f"    --baseline_ckpt experiments/logs_real/zy2lki18/models/best_model.pt \\")
    print(f"    --finetuned_ckpt {save_dir / 'best_model.pt'} \\")
    print(f"    --gpu {args.gpu}")


if __name__ == "__main__":
    main()
