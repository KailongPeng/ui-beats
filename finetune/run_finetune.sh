#!/usr/bin/env bash
# run_finetune.sh -- LOSO fine-tune + evaluate PN-QRS on arm-band data
#
# Usage:
#   bash finetune/run_finetune.sh --data_dir data/0410_real --fs 1000
#   bash finetune/run_finetune.sh --data_dir data/0410_real --fs 1000 \
#     --gpu 0 --epochs 30 --conda_env ECGFounder

set -euo pipefail

# ── default parameters ───────────────────────────────────────────────────────
DATA_DIR=""
FS=""
GPU="0"
EPOCHS="30"
BATCH_SIZE="8"
CONDA_ENV="ECGFounder"

# ── parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_dir)   DATA_DIR="$2";   shift 2 ;;
        --fs)         FS="$2";         shift 2 ;;
        --gpu)        GPU="$2";        shift 2 ;;
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --conda_env)  CONDA_ENV="$2";  shift 2 ;;
        -h|--help)
            echo "Usage: bash finetune/run_finetune.sh --data_dir <dir> --fs <hz> [options]"
            echo ""
            echo "Options:"
            echo "  --data_dir   <path>   Root data directory (required)"
            echo "  --fs         <int>    Sampling rate in Hz (required)"
            echo "  --gpu        <int>    GPU index (default: 0)"
            echo "  --epochs     <int>    Max training epochs (default: 30)"
            echo "  --batch_size <int>    Batch size (default: 8)"
            echo "  --conda_env  <name>   Conda env name (default: ECGFounder)"
            exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$DATA_DIR" || -z "$FS" ]]; then
    echo "[ERROR] --data_dir and --fs are required."
    echo "Usage: bash finetune/run_finetune.sh --data_dir <dir> --fs <hz>"
    exit 1
fi

# ── resolve paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UI_BEAT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$UI_BEAT_ROOT"

# ── collect subject directories ───────────────────────────────────────────────
mapfile -t SUBJECTS < <(
    ls -d "$DATA_DIR"/*/ 2>/dev/null | while read -r d; do basename "$d"; done
)
if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
    echo "[ERROR] No subject directories found in: $DATA_DIR"
    exit 1
fi
echo "Found ${#SUBJECTS[@]} subjects: ${SUBJECTS[*]}"
echo ""

# ── LOSO loop ─────────────────────────────────────────────────────────────────
for SUBJ in "${SUBJECTS[@]}"; do
    echo "=========================================="
    echo "  LOSO fold: test_subject = $SUBJ"
    echo "=========================================="

    # train
    conda run -n "$CONDA_ENV" --no-capture-output \
        python finetune/train_armband.py \
            --data_dir     "$DATA_DIR" \
            --fs           "$FS" \
            --test_subject "$SUBJ" \
            --epochs       "$EPOCHS" \
            --batch_size   "$BATCH_SIZE" \
            --gpu          "$GPU"

    # find the best checkpoint produced by this fold
    BEST_CKPT=$(ls -t experiments/logs_armband/*_"${SUBJ}"/models/best_model.pt 2>/dev/null | head -1 || true)
    if [[ -z "$BEST_CKPT" ]]; then
        echo "[WARN] No best_model.pt found for subject '$SUBJ' -- skipping eval."
        echo ""
        continue
    fi

    # evaluate
    conda run -n "$CONDA_ENV" --no-capture-output \
        python finetune/eval_armband.py \
            --data_dir       "$DATA_DIR" \
            --fs             "$FS" \
            --test_subject   "$SUBJ" \
            --baseline_ckpt  "experiments/logs_real/zy2lki18/models/best_model.pt" \
            --finetuned_ckpt "$BEST_CKPT" \
            --gpu            "$GPU"

    echo ""
done

echo "=========================================="
echo "  All LOSO folds complete."
echo "  Results: experiments/logs_armband/"
echo "=========================================="
