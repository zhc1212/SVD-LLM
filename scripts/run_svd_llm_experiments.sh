#!/bin/bash
# SVD-LLM (Phase 2) experiments: Sequential LoRA fine-tuning + evaluation
# Run 4 compression ratios serially on a single GPU
#
# Usage:
#   bash scripts/run_svd_llm_experiments.sh [MODEL_PATH] [GPU_ID]
#
# Example:
#   bash scripts/run_svd_llm_experiments.sh jeffwan/llama-7b-hf 0

set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <MODEL_PATH> [GPU_ID]}"
GPU_ID="${2:-0}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

RATIOS=(0.2 0.4 0.6 0.8)
OUTPUT_BASE="outputs"
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

echo "============================================"
echo "SVD-LLM Sequential LoRA Experiments"
echo "Model: $MODEL_PATH"
echo "GPU: $GPU_ID"
echo "Ratios: ${RATIOS[*]}"
echo "============================================"

for RATIO in "${RATIOS[@]}"; do
    RATIO_PCT=$(echo "$RATIO * 100" | bc | cut -d. -f1)
    SAVE_PATH="${OUTPUT_BASE}/llama7b_svd_llm_${RATIO_PCT}"
    LOG_FILE="${LOG_DIR}/svd_llm_${RATIO_PCT}.log"

    echo ""
    echo "============================================"
    echo "[${RATIO_PCT}%] Starting fine-tuning (ratio=${RATIO})"
    echo "  Save: $SAVE_PATH"
    echo "  Log:  $LOG_FILE"
    echo "============================================"

    # Fine-tune
    python scripts/finetune.py \
        --model_path "$MODEL_PATH" \
        --ratio "$RATIO" \
        --save_path "$SAVE_PATH" \
        --lora_r 32 \
        --lora_alpha 64 \
        --lr 2e-4 \
        --epochs 1 \
        --batch_size 4 \
        --grad_accum 4 \
        --max_length 256 \
        2>&1 | tee "$LOG_FILE"

    echo "[${RATIO_PCT}%] Fine-tuning complete. Starting evaluation..."

    # Perplexity eval
    python scripts/eval_model.py \
        --model_path "$SAVE_PATH" \
        --eval perplexity \
        --datasets wikitext2 c4 \
        2>&1 | tee -a "$LOG_FILE"

    # Downstream eval
    python scripts/eval_model.py \
        --model_path "$SAVE_PATH" \
        --eval downstream \
        2>&1 | tee -a "$LOG_FILE"

    echo "[${RATIO_PCT}%] Done."
done

echo ""
echo "============================================"
echo "All experiments complete!"
echo "Collecting results..."
echo "============================================"

python scripts/collect_results.py
