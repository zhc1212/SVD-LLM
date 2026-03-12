#!/bin/bash
# 在 GPU 0 和 GPU 2 上并行运行实验
# GPU 0: svd_llm_w (所有压缩比)
# GPU 2: svd_llm (所有压缩比)

set -e

source /home/xiyaofeng/ENTER/etc/profile.d/conda.sh && conda activate compactifai
cd /home/xiyaofeng/huicheng/SVD_LLM

MODEL=/home/xiyaofeng/.cache/huggingface/hub/models--jeffwan--llama-7b-hf/snapshots/82eb0e6908390680598ca3ec1d77adfc5e1b24aa/

RATIOS="0.2 0.4 0.6 0.8"

mkdir -p outputs/original logs

# ============================================================
# GPU 0: 先评估原始模型，然后跑 svd_llm_w
# ============================================================
run_gpu0() {
    echo "[GPU0] === Evaluating Original Model ==="
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_model.py \
        --model_path $MODEL \
        --eval perplexity \
        --datasets wikitext2 c4 \
        --output_file outputs/original/eval_results.json

    for ratio in $RATIOS; do
        ratio_pct=$(echo "$ratio * 100" | bc | cut -d. -f1)
        save_dir="outputs/llama7b_svd_llm_w_${ratio_pct}"

        echo ""
        echo "[GPU0] === svd_llm_w @ ${ratio_pct}% compression ==="

        CUDA_VISIBLE_DEVICES=0 python scripts/compress.py \
            --model_path $MODEL \
            --method svd_llm_w \
            --ratio $ratio \
            --save_path $save_dir \
            --device cuda

        CUDA_VISIBLE_DEVICES=0 python scripts/eval_model.py \
            --model_path $save_dir \
            --eval perplexity \
            --datasets wikitext2 c4 \
            --device cuda

        echo "[GPU0] Done: svd_llm_w @ ${ratio_pct}%"
    done

    # 下游任务评估 (20%)
    echo "[GPU0] === svd_llm_w downstream (20%) ==="
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_model.py \
        --model_path outputs/llama7b_svd_llm_w_20 \
        --eval downstream \
        --device cuda

    echo "[GPU0] === ALL DONE ==="
}

# ============================================================
# GPU 2: 跑 svd_llm (sequential update)
# ============================================================
run_gpu2() {
    for ratio in $RATIOS; do
        ratio_pct=$(echo "$ratio * 100" | bc | cut -d. -f1)
        save_dir="outputs/llama7b_svd_llm_${ratio_pct}"

        echo ""
        echo "[GPU2] === svd_llm @ ${ratio_pct}% compression ==="

        CUDA_VISIBLE_DEVICES=2 python scripts/compress.py \
            --model_path $MODEL \
            --method svd_llm \
            --ratio $ratio \
            --save_path $save_dir \
            --device cuda

        CUDA_VISIBLE_DEVICES=2 python scripts/eval_model.py \
            --model_path $save_dir \
            --eval perplexity \
            --datasets wikitext2 c4 \
            --device cuda

        echo "[GPU2] Done: svd_llm @ ${ratio_pct}%"
    done

    # 下游任务评估 (20%)
    echo "[GPU2] === svd_llm downstream (20%) ==="
    CUDA_VISIBLE_DEVICES=2 python scripts/eval_model.py \
        --model_path outputs/llama7b_svd_llm_20 \
        --eval downstream \
        --device cuda

    echo "[GPU2] === ALL DONE ==="
}

# 并行启动两个 GPU
echo "Starting experiments on GPU 0 and GPU 2..."
echo "Logs: logs/gpu0.log, logs/gpu2.log"

run_gpu0 > logs/gpu0.log 2>&1 &
PID0=$!

run_gpu2 > logs/gpu2.log 2>&1 &
PID2=$!

echo "GPU 0 PID: $PID0"
echo "GPU 2 PID: $PID2"
echo ""
echo "Monitor with:"
echo "  tail -f logs/gpu0.log"
echo "  tail -f logs/gpu2.log"
echo ""
echo "Waiting for both to complete..."

wait $PID0
echo "GPU 0 finished (exit code: $?)"

wait $PID2
echo "GPU 2 finished (exit code: $?)"

echo ""
echo "=== All experiments complete ==="
python scripts/collect_results.py
