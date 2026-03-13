#!/bin/bash
# 串行跑所有压缩模型的 downstream 评估
set -e

source /home/xiyaofeng/ENTER/etc/profile.d/conda.sh && conda activate compactifai
cd /home/xiyaofeng/huicheng/SVD_LLM

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

for ratio in 20 40 60 80; do
    echo ""
    echo "$(date) === Downstream eval: SVD-LLM(W) ${ratio}% ==="
    python scripts/eval_model.py \
        --model_path outputs/llama7b_svd_llm_w_${ratio} \
        --eval downstream \
        --batch_size 64 \
        --device cuda

    echo "$(date) Done: ${ratio}%"
    if [ -f "outputs/llama7b_svd_llm_w_${ratio}/eval_results.json" ]; then
        cat "outputs/llama7b_svd_llm_w_${ratio}/eval_results.json"
    fi
done

echo ""
echo "$(date) All downstream evaluations complete!"
