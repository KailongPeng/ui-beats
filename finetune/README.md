# finetune/ — Arm-band CH20 Fine-tuning Pipeline

Fine-tune a pre-trained PN-QRS model on arm-band (CH20) data using the original
**AEU training framework** (BCE + similarity loss, two optimizers), guided by
pseudo-labels produced by the 12-lead PN-QRS teacher model.

```
12-lead PN-QRS (teacher)
    ↓ generates *_CH1-8_rpeaks.csv  (pseudo-labels)
CH20 model (student) fine-tuned via AEU loss
    ↓ evaluated against same pseudo-labels
eval_summary.md  (baseline vs fine-tuned F1 per activity)
```

---

## Prerequisites

The following pipeline outputs must already exist alongside each raw CSV:

| File pattern | Produced by |
|---|---|
| `<stem>_CH1-8_rpeaks.csv` | `pipeline/apply_pnqrs.py` |
| `<stem>_quality_report.csv` | `pipeline/extract_quality_segments.py` |

If these files are missing, run:
```bash
# Step 1: 12-lead inference (produces *_CH1-8_rpeaks.csv)
conda run -n ECGFounder python pipeline/apply_pnqrs.py \
    --data_dir data/0410_real --fs 1000 --gpu 0

# Step 2: Quality assessment (produces *_quality_report.csv)
conda run -n ECGFounder python pipeline/extract_quality_segments.py \
    --batch --data_dir data/0410_real --fs 1000 --gpu 0
```

---

## Quick Start — Full LOSO in One Command

```bash
bash finetune/run_finetune.sh --data_dir data/0410_real --fs 1000
```

This will:
1. Iterate over every subject directory under `data_dir`
2. For each subject: train with all *other* subjects, hold this one out as validation
3. Evaluate baseline vs fine-tuned, write `eval_summary.md` per fold

All outputs land under `experiments/logs_armband/`.

### Multi-GPU note

`run_finetune.sh` runs one GPU at a time (sequentially per fold).
To parallelise across GPUs, launch separate folds manually (see §Manual Steps).

---

## Manual Steps

### Step 1: Train one LOSO fold

```bash
conda run -n ECGFounder python finetune/train_armband.py \
    --data_dir   data/0410_real \
    --fs         1000 \
    --test_subject subject01 \
    --gpu        0
```

### Step 2: Evaluate

```bash
conda run -n ECGFounder python finetune/eval_armband.py \
    --data_dir       data/0410_real \
    --fs             1000 \
    --test_subject   subject01 \
    --baseline_ckpt  experiments/logs_real/zy2lki18/models/best_model.pt \
    --finetuned_ckpt experiments/logs_armband/<timestamp>_subject01/models/best_model.pt \
    --gpu            0
```

---

## Parameters

### train_armband.py

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | required | Root dir: `<data_dir>/<subject>/<activity>/rec*.csv` |
| `--fs` | required | Raw CSV sampling rate (Hz) |
| `--test_subject` | required | Hold-out subject directory name |
| `--init_ckpt` | CPSC2019 best | Initial QRSModel checkpoint |
| `--save_dir` | auto-generated | Where to save model files |
| `--epochs` | 30 | Maximum training epochs |
| `--batch_size` | 8 | Batch size |
| `--alpha_lr` | 5e-5 | Learning rate for encoder+decoder (α branch) |
| `--theta_lr` | 5e-5 | Learning rate for projection head (θ branch) |
| `--early_stop` | 10 | Patience epochs before early stopping |
| `--seed` | 42 | Random seed |
| `--gpu` | `"0"` | GPU index, or `"cpu"` |

### eval_armband.py

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | required | Same root as training |
| `--fs` | required | Sampling rate |
| `--test_subject` | required | Subject to evaluate |
| `--baseline_ckpt` | required | Path to CPSC2019-trained model |
| `--finetuned_ckpt` | required | Path to fine-tuned best_model.pt |
| `--out_dir` | `<ckpt>/../eval/` | Where to write results |

---

## Output Files

```
experiments/logs_armband/<timestamp>_<subject>/
├── models/
│   ├── best_model.pt      ← QRSModel checkpoint (lowest val_bce)
│   └── last_model.pt      ← QRSModel checkpoint (last epoch)
├── history.csv            ← epoch, train_alpha_loss, train_theta_loss, val_bce, epoch_time_s
├── args.json              ← all CLI arguments used
└── eval/
    ├── eval_results.csv   ← per-file Se / P+ / F1 (baseline + fine-tuned)
    └── eval_summary.md    ← per-activity mean F1 table with Δ column
```

The `best_model.pt` is **compatible with `pipeline/extract_quality_segments.py`** — drop it in
as a replacement for `experiments/logs_real/zy2lki18/models/best_model.pt`.

---

## Training Diagnostics

**Healthy training** — `val_bce` should decrease in the first ~10 epochs then plateau.
If it keeps rising from epoch 1, the learning rate may be too high; try `--alpha_lr 1e-5`.

**Early stopping** fires when `val_bce` fails to improve for `--early_stop` (default 10) epochs.
If your dataset is small and the curve is noisy, increase patience: `--early_stop 20`.

**`history.csv`** — plot `val_bce` vs epoch to check for overfitting:
```python
import pandas as pd, matplotlib.pyplot as plt
h = pd.read_csv("experiments/logs_armband/.../history.csv")
h[["train_alpha_loss","val_bce"]].plot(); plt.show()
```

---

## Limitations & Caveats

- **Pseudo ground truth**: `*_CH1-8_rpeaks.csv` are outputs of the PN-QRS teacher model
  on the 12-lead signal, not human-annotated peaks.  An increase in F1 means the student
  is better imitating the teacher on CH20 — it is not a guarantee of improved accuracy
  relative to true R-peak positions.

- **Lead differences**: The arm-band CH20 signal may differ from 12-lead in polarity,
  amplitude, and morphology.  Pseudo-labels are aligned by time, not by waveform shape.

- **LOSO scope**: The evaluation measures cross-subject generalisation within your dataset.
  Performance on a fully held-out population may differ.

- **Gold-standard eval**: For publication, manually annotate a small held-out subset and
  evaluate with those labels instead of the pseudo-labels.
