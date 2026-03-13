"""SVD-LLM 主评估脚本

用法:
    python scripts/evaluate.py \
        --model_path outputs/llama7b_svdllm_20 \
        --eval perplexity \
        --datasets wikitext2 c4

    python scripts/evaluate.py \
        --model_path outputs/llama7b_svdllm_20 \
        --eval downstream \
        --tasks openbookqa arc_easy winogrande hellaswag piqa mathqa truthfulqa_gen gsm8k
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.loader import load_model
from src.eval.perplexity import evaluate_perplexity
from src.eval.downstream import evaluate_downstream, format_downstream_results


def main():
    parser = argparse.ArgumentParser(description="SVD-LLM Evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--eval", type=str, required=True, choices=["perplexity", "downstream", "all"])
    parser.add_argument("--datasets", nargs="+", default=["wikitext2", "c4"])
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    print("=== SVD-LLM Evaluation ===")
    print(f"Model: {args.model_path}")

    model, tokenizer = load_model(args.model_path, device_map="auto")

    results = {}

    if args.eval in ["perplexity", "all"]:
        print("\n--- Perplexity Evaluation ---")
        for ds in args.datasets:
            ppl = evaluate_perplexity(model, tokenizer, ds, args.device)
            results[f"ppl_{ds}"] = ppl
            print(f"  {ds}: {ppl:.2f}")

    if args.eval in ["downstream", "all"]:
        print("\n--- Downstream Task Evaluation ---")
        raw_results = evaluate_downstream(
            model, tokenizer, args.tasks, args.batch_size, args.device
        )
        formatted = format_downstream_results(raw_results)
        results["downstream"] = formatted
        for task, score in formatted.items():
            print(f"  {task}: {score:.4f}")

    if args.output_file:
        output_path = args.output_file
    else:
        output_path = os.path.join(args.model_path, "eval_results.json")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
