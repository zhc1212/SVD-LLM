#!/bin/bash
# Phase 1: SVD-LLM(W) 全部实验 + Phase 1.5: 下游任务 (20%)
# 在 GPU 2 上串行运行，预计总耗时 ~3-4 小时
#
# 流程:
#   1. SVD-LLM(W) 压缩 × 4 ratios (20%, 40%, 60%, 80%)
#   2. 每个 ratio 压缩完后立即评估 WikiText-2 和 C4 Perplexity
#   3. 20% ratio 下游任务评估 (8 tasks)
#   4. 汇总结果

set -e

source /home/xiyaofeng/ENTER/etc/profile.d/conda.sh && conda activate compactifai
cd /home/xiyaofeng/huicheng/SVD_LLM

export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1

MODEL=/home/xiyaofeng/.cache/huggingface/hub/models--jeffwan--llama-7b-hf/snapshots/82eb0e6908390680598ca3ec1d77adfc5e1b24aa/

RATIOS="0.2 0.4 0.6 0.8"

mkdir -p outputs/original logs

echo "$(date) =========================================="
echo "$(date) Starting Phase 1: SVD-LLM(W) experiments"
echo "$(date) =========================================="

# Step 0: 原始模型 Perplexity (已完成，跳过)
# WikiText-2: 5.67, C4: 7.20
echo "$(date) Skipping original model eval (already done: wt2=5.67, c4=7.20)"

# Step 1: SVD-LLM(W) × 4 ratios
for ratio in $RATIOS; do
    ratio_pct=$(echo "$ratio * 100" | bc | cut -d. -f1)
    save_dir="outputs/llama7b_svd_llm_w_${ratio_pct}"

    echo ""
    echo "$(date) === SVD-LLM(W) @ ${ratio_pct}% compression ==="

    # 压缩
    python scripts/compress.py \
        --model_path $MODEL \
        --method svd_llm_w \
        --ratio $ratio \
        --save_path $save_dir \
        --device cuda

    # 评估 Perplexity
    python scripts/eval_model.py \
        --model_path $save_dir \
        --eval perplexity \
        --datasets wikitext2 c4 \
        --device cuda

    echo "$(date) Done: SVD-LLM(W) @ ${ratio_pct}%"

    # 打印中间结果
    if [ -f "${save_dir}/eval_results.json" ]; then
        echo "$(date) Results:"
        cat "${save_dir}/eval_results.json"
    fi
done

# Step 2: 下游任务评估 (仅 20% SVD-LLM(W))
echo ""
echo "$(date) === Downstream Tasks (SVD-LLM(W) 20%) ==="
python scripts/eval_model.py \
    --model_path outputs/llama7b_svd_llm_w_20 \
    --eval downstream \
    --device cuda

# Step 3: 汇总结果
echo ""
echo "$(date) === Collecting Results ==="
python scripts/collect_results.py

echo ""
echo "$(date) =========================================="
echo "$(date) Phase 1 complete! All experiments done."
echo "$(date) =========================================="
