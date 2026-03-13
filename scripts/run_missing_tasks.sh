#!/bin/bash
# 补充缺失的下游任务: mathqa + gsm8k (batch_size=1 避免 padding CUDA 错误)
set -e

source /home/xiyaofeng/ENTER/etc/profile.d/conda.sh && conda activate compactifai
cd /home/xiyaofeng/huicheng/SVD_LLM

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# 1. mathqa_local for all ratios (MC task, batch_size=64)
for ratio in 20 40 60 80; do
    echo ""
    echo "$(date) === mathqa_local: SVD-LLM(W) ${ratio}% ==="
    python scripts/eval_model.py \
        --model_path outputs/llama7b_svd_llm_w_${ratio} \
        --eval downstream \
        --tasks mathqa_local \
        --batch_size 64 \
        --device cuda \
        --output_file outputs/llama7b_svd_llm_w_${ratio}/eval_mathqa.json
    echo "$(date) Done mathqa ${ratio}%"
    cat outputs/llama7b_svd_llm_w_${ratio}/eval_mathqa.json
done

# 2. gsm8k for all ratios (generation task, batch_size=1)
for ratio in 20 40 60 80; do
    echo ""
    echo "$(date) === gsm8k: SVD-LLM(W) ${ratio}% ==="
    python scripts/eval_model.py \
        --model_path outputs/llama7b_svd_llm_w_${ratio} \
        --eval downstream \
        --tasks gsm8k \
        --batch_size 1 \
        --device cuda \
        --output_file outputs/llama7b_svd_llm_w_${ratio}/eval_gsm8k.json
    echo "$(date) Done gsm8k ${ratio}%"
    cat outputs/llama7b_svd_llm_w_${ratio}/eval_gsm8k.json
done

echo ""
echo "$(date) All missing tasks complete!"
