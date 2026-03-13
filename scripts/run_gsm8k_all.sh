#!/bin/bash
# gsm8k for all ratios, batch_size=64
set -e

source /home/xiyaofeng/ENTER/etc/profile.d/conda.sh && conda activate compactifai
cd /home/xiyaofeng/huicheng/SVD_LLM

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

for ratio in 20 40 60 80; do
    echo ""
    echo "$(date) === gsm8k: SVD-LLM(W) ${ratio}% ==="
    python scripts/eval_model.py \
        --model_path outputs/llama7b_svd_llm_w_${ratio} \
        --eval downstream \
        --tasks gsm8k \
        --batch_size 64 \
        --device cuda \
        --output_file outputs/llama7b_svd_llm_w_${ratio}/eval_gsm8k.json
    echo "$(date) Done gsm8k ${ratio}%"
    cat outputs/llama7b_svd_llm_w_${ratio}/eval_gsm8k.json
done

echo ""
echo "$(date) All gsm8k evaluations complete!"
