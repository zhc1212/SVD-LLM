#!/bin/bash
# scripts/run_all_experiments.sh
# 运行 Table 1 实验: svd_llm_w + svd_llm × 4 ratios

set -e

source /home/xiyaofeng/ENTER/etc/profile.d/conda.sh && conda activate compactifai
cd /home/xiyaofeng/huicheng/SVD_LLM

MODEL=/home/xiyaofeng/.cache/huggingface/hub/models--jeffwan--llama-7b-hf/snapshots/82eb0e6908390680598ca3ec1d77adfc5e1b24aa/

METHODS="svd_llm_w svd_llm"
RATIOS="0.2 0.4 0.6 0.8"

# Step 1: 评估原始模型 Perplexity
echo "=== Evaluating Original Model ==="
mkdir -p outputs/original
python scripts/eval_model.py \
    --model_path $MODEL \
    --eval perplexity \
    --datasets wikitext2 c4 \
    --output_file outputs/original/eval_results.json

# Step 2: 压缩 + 评估 Perplexity (每个 method × ratio)
for method in $METHODS; do
    for ratio in $RATIOS; do
        ratio_pct=$(echo "$ratio * 100" | bc | cut -d. -f1)
        save_dir="outputs/llama7b_${method}_${ratio_pct}"

        echo ""
        echo "=== ${method} @ ${ratio_pct}% compression ==="

        # 压缩
        python scripts/compress.py \
            --model_path $MODEL \
            --method $method \
            --ratio $ratio \
            --save_path $save_dir

        # 评估 Perplexity
        python scripts/eval_model.py \
            --model_path $save_dir \
            --eval perplexity \
            --datasets wikitext2 c4

        echo "Done: ${method} @ ${ratio_pct}%"
    done
done

# Step 3: 下游任务评估 (仅 20% 压缩比)
echo ""
echo "=== Downstream Task Evaluation (20% compression) ==="
for method in $METHODS; do
    save_dir="outputs/llama7b_${method}_20"
    echo "Evaluating downstream: ${method}"
    python scripts/eval_model.py \
        --model_path $save_dir \
        --eval downstream
done

echo ""
echo "=== All experiments complete ==="

# Step 4: 汇总结果
python scripts/collect_results.py
