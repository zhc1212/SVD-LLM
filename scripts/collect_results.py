"""收集所有实验结果，生成 Table 1 对比表"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "outputs"
METHODS = ["svd_llm_w", "svd_llm"]
RATIOS = [20, 40, 60, 80]

# 论文 Table 1 参考值 (LLaMA-7B)
PAPER_RESULTS = {
    ("original", 0): {"wikitext2": 5.68, "c4": 7.34},
    ("svd_llm_w", 20): {"wikitext2": 7.94, "c4": 15.84},
    ("svd_llm_w", 40): {"wikitext2": 13.73, "c4": 75.42},
    ("svd_llm_w", 60): {"wikitext2": 66.62, "c4": 471.83},
    ("svd_llm_w", 80): {"wikitext2": 1349, "c4": 6224},
    ("svd_llm", 20): {"wikitext2": 7.73, "c4": 12.23},
    ("svd_llm", 40): {"wikitext2": 9.27, "c4": 15.63},
    ("svd_llm", 60): {"wikitext2": 15.00, "c4": 26.26},
    ("svd_llm", 80): {"wikitext2": 31.79, "c4": 43.71},
}


def fmt(val):
    if isinstance(val, float):
        if val > 100:
            return f"{val:.0f}"
        return f"{val:.2f}"
    return str(val)


def main():
    print("=" * 90)
    print("SVD-LLM Reproduction Results vs Paper (Table 1: LLaMA-7B)")
    print("=" * 90)

    # Original model
    orig_path = os.path.join(OUTPUT_DIR, "original", "eval_results.json")
    if os.path.exists(orig_path):
        with open(orig_path) as f:
            orig = json.load(f)
        print("\nOriginal Model:")
        print(f"  WikiText-2: {fmt(orig.get('ppl_wikitext2', 'N/A'))}  (paper: 5.68)")
        print(f"  C4:         {fmt(orig.get('ppl_c4', 'N/A'))}  (paper: 7.34)")

    # Compressed models
    print(f"\n{'Method':<12} {'Ratio':<8} {'WikiText-2':>12} {'(Paper)':>12} {'C4':>12} {'(Paper)':>12}")
    print("-" * 70)

    for method in METHODS:
        for ratio in RATIOS:
            result_path = os.path.join(OUTPUT_DIR, f"llama7b_{method}_{ratio}", "eval_results.json")

            if os.path.exists(result_path):
                with open(result_path) as f:
                    results = json.load(f)
                wt2 = results.get("ppl_wikitext2", "N/A")
                c4 = results.get("ppl_c4", "N/A")
            else:
                wt2 = "N/A"
                c4 = "N/A"

            paper = PAPER_RESULTS.get((method, ratio), {})
            paper_wt2 = paper.get("wikitext2", "N/A")
            paper_c4 = paper.get("c4", "N/A")

            print(f"{method:<12} {ratio}%{'':<4} {fmt(wt2):>12} {fmt(paper_wt2):>12} {fmt(c4):>12} {fmt(paper_c4):>12}")
        print()

    # Downstream results (20% only)
    print("\n" + "=" * 90)
    print("Downstream Tasks (20% compression)")
    print("=" * 90)

    paper_downstream = {
        "svd_llm_w": {
            "openbookqa": 0.31, "arc_easy": 0.62, "winogrande": 0.61,
            "hellaswag": 0.45, "piqa": 0.71, "mathqa": 0.21,
        },
        "svd_llm": {
            "openbookqa": 0.33, "arc_easy": 0.67, "winogrande": 0.69,
            "hellaswag": 0.55, "piqa": 0.79, "mathqa": 0.26,
        },
    }

    for method in METHODS:
        result_path = os.path.join(OUTPUT_DIR, f"llama7b_{method}_20", "eval_results.json")
        print(f"\n{method}:")

        if os.path.exists(result_path):
            with open(result_path) as f:
                results = json.load(f)
            downstream = results.get("downstream", {})

            paper_ds = paper_downstream.get(method, {})
            for task in ["openbookqa", "arc_easy", "winogrande", "hellaswag", "piqa", "mathqa"]:
                ours = downstream.get(task, "N/A")
                theirs = paper_ds.get(task, "N/A")
                print(f"  {task:<15} Ours: {fmt(ours):>8}  Paper: {fmt(theirs):>8}")
        else:
            print("  No results yet")


if __name__ == "__main__":
    main()
