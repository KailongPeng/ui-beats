import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""
Train PN-QRS on real CPSC2019 data.
Saves to experiments/logs_real/ - does NOT overwrite existing checkpoints.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import wandb
from torch.utils.data import DataLoader
from config.BeatConfig import BeatConfig
from dataset.dataset import CPSC2019Dataset, cpsc2019_collate_fn
from models.multi_head import encoder4qrs, decoder4qrs, phi_qrs
from models.qrs_model import QRSModel
from training.beat_trainer import BeatTrainer

REAL_DATA_ROOT = Path("/home/kailong/ECG/ECG/data/PN-QRS/CPSC2019_real_data")
LOG_DIR        = Path("/home/kailong/ECG/ECG/ECGFounder/PN-QRS/experiments/logs_real")

def main():
    os.environ.setdefault("WANDB_MODE", "offline")

    config = BeatConfig(
        project_name="pnqrs_real_cpsc2019",
        run_name="real_data_retrain",
        offline=True,
        training_type="qrs",
        batch_size=64,
        epochs=100,
        early_stop_patience=15,
        alpha_lr=1e-3,
        theta_lr=1e-3,
        dataset_root=REAL_DATA_ROOT,
        log_dir=LOG_DIR,
    )

    run = wandb.init(
        project=config.project_name,
        name=config.run_name,
        config=config.to_wandb_config(),
        reinit=True,
    )

    model_save_dir = LOG_DIR / run.id / "models"
    model_save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = CPSC2019Dataset(root=REAL_DATA_ROOT / "cpsc2019_train")
    test_dataset  = CPSC2019Dataset(root=REAL_DATA_ROOT / "cpsc2019_test")
    print(f"Train: {len(train_dataset)}  Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=4, collate_fn=cpsc2019_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=4, collate_fn=cpsc2019_collate_fn,
    )

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = BeatTrainer(
        batch_size=config.batch_size,
        encoder_qrs=encoder4qrs(),
        decoder_qrs=decoder4qrs(),
        phi_qrs=phi_qrs(),
        alpha_lr=config.alpha_lr,
        theta_lr=config.theta_lr,
        early_stop_patience=config.early_stop_patience,
        model_save_path=model_save_dir,
        device=device,
    )

    def log_cb(epoch, step, total_steps, metrics, phase):
        prefix = "train" if phase in {"batch", "epoch"} else phase
        log_d  = {f"{prefix}/{k}": v for k, v in metrics.items()}
        log_d["epoch"] = epoch + 1
        wandb.log(log_d, commit=(phase != "batch"))

    best_path = model_save_dir / "best_model.pt"
    trainer.train(
        dataloader=train_loader,
        epochs=config.epochs,
        log_interval=20,
        log_callback=log_cb,
        val_loader=test_loader,
        best_model_path=best_path,
    )

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        full_model = QRSModel(encoder4qrs(), decoder4qrs(), phi_qrs()).to(device)
        full_model.load_state_dict(ckpt["model_state"])
        full_model.eval()
        test_metrics = trainer.evaluate_full_model(test_loader, full_model)
        wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
        print(f"\nBest: {best_path}\nTest: {test_metrics}")

    run.finish()

if __name__ == "__main__":
    main()
