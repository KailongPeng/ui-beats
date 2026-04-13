#!/usr/bin/env bash
# run_pipeline.sh — 一键运行 PN-QRS 完整 5 步流程
#
# 用法：
#   bash pipeline/run_pipeline.sh --data_dir /path/to/data --fs 1000
#   bash pipeline/run_pipeline.sh --data_dir /path/to/data --fs 1000 --gpu 1 --top_n 9
#
# 参数：
#   --data_dir   数据根目录（必填）
#   --fs         采样率 Hz（必填）
#   --gpu        GPU 编号（默认 0）
#   --conda_env  conda 环境名（默认 ECGFounder）
#   --top_n      幅度对比图每组窗口数（默认 9）
#   --uc_thr     质量片段不确定性阈值，auto 或数字（默认 auto）
#   --skip       跳过指定步骤，逗号分隔，如 --skip 2,4

set -e  # 任意一步报错立即停止

# ── 默认参数 ─────────────────────────────────────────────────────────────────
DATA_DIR=""
FS=""
GPU=0
CONDA_ENV="ECGFounder"
TOP_N=9
UC_THR="auto"
SKIP=""

# ── 解析参数 ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_dir)  DATA_DIR="$2"; shift 2 ;;
        --fs)        FS="$2";       shift 2 ;;
        --gpu)       GPU="$2";      shift 2 ;;
        --conda_env) CONDA_ENV="$2";shift 2 ;;
        --top_n)     TOP_N="$2";    shift 2 ;;
        --uc_thr)    UC_THR="$2";   shift 2 ;;
        --skip)      SKIP="$2";     shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$DATA_DIR" || -z "$FS" ]]; then
    echo "Usage: bash pipeline/run_pipeline.sh --data_dir /path/to/data --fs 1000"
    exit 1
fi

# ── 路径设置 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

CONDA="conda run -n $CONDA_ENV"

should_skip() { echo ",$SKIP," | grep -q ",$1,"; }

# ── 工具函数 ─────────────────────────────────────────────────────────────────
step_banner() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Step $1: $2"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ── 开始 ─────────────────────────────────────────────────────────────────────
echo ""
echo "PN-QRS Pipeline"
echo "  data_dir : $DATA_DIR"
echo "  fs       : $FS Hz"
echo "  gpu      : $GPU"
echo "  conda    : $CONDA_ENV"
echo "  uc_thr   : $UC_THR"
echo "  top_n    : $TOP_N"
[[ -n "$SKIP" ]] && echo "  skip     : $SKIP"
START_ALL=$(date +%s)

# ── Step 1：R-peak 检测 ───────────────────────────────────────────────────────
if should_skip 1; then
    echo "  [skip] Step 1"
else
    step_banner 1 "R-peak detection  (apply_pnqrs.py)"
    T0=$(date +%s)
    $CONDA python pipeline/apply_pnqrs.py \
        --data_dir "$DATA_DIR" --fs "$FS" --gpu "$GPU"
    echo "  done in $(($(date +%s) - T0))s"
fi

# ── Step 2：可视化 + 幅度-质量分析 ───────────────────────────────────────────
if should_skip 2; then
    echo "  [skip] Step 2"
else
    step_banner 2 "Visualization + amplitude-quality analysis  (visualize_rpeaks.py)"
    T0=$(date +%s)
    python pipeline/visualize_rpeaks.py \
        --batch --data_dir "$DATA_DIR" --fs "$FS" \
        --low_amp --top_n "$TOP_N"
    echo "  done in $(($(date +%s) - T0))s"
fi

# ── Step 3：高质量片段提取 ────────────────────────────────────────────────────
if should_skip 3; then
    echo "  [skip] Step 3"
else
    step_banner 3 "Quality segmentation  (extract_quality_segments.py)"
    T0=$(date +%s)
    $CONDA python pipeline/extract_quality_segments.py \
        --batch --data_dir "$DATA_DIR" --fs "$FS" \
        --uc_thr "$UC_THR" --gpu "$GPU"
    echo "  done in $(($(date +%s) - T0))s"
fi

# ── Step 4：上臂导联精度评估 ──────────────────────────────────────────────────
if should_skip 4; then
    echo "  [skip] Step 4"
else
    step_banner 4 "Upper arm evaluation  (evaluate_upper_arm.py)"
    T0=$(date +%s)
    python pipeline/evaluate_upper_arm.py \
        --data_dir "$DATA_DIR" --fs "$FS"
    echo "  done in $(($(date +%s) - T0))s"
fi

# ── Step 5：波形显著性 SQI ────────────────────────────────────────────────────
if should_skip 5; then
    echo "  [skip] Step 5"
else
    step_banner 5 "Wave salience SQI  (wave_salience_calculator.py)"
    T0=$(date +%s)
    python pipeline/wave_salience_calculator.py \
        --batch --data_dir "$DATA_DIR" --fs "$FS"
    echo "  done in $(($(date +%s) - T0))s"
fi

# ── 完成 ─────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  All steps done  (total: $(($(date +%s) - START_ALL))s)"
echo "  Results in: $DATA_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
