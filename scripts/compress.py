"""SVD-LLM 主压缩脚本

用法:
    python scripts/compress.py \
        --model_path <path> \
        --method svd_llm \
        --ratio 0.2 \
        --save_path outputs/llama7b_svdllm_20
"""
import argparse
import os
import sys
import torch
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.loader import load_model
from src.data.calibration import get_calibration_data
from src.compress.sequential_update import (
    compress_model_whitening_only,
    compress_model_sequential,
)


def main():
    parser = argparse.ArgumentParser(description="SVD-LLM Compression")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--method", type=str, required=True,
                        choices=["svd_llm_w", "svd_llm"])
    parser.add_argument("--ratio", type=float, required=True,
                        help="Compression ratio (0.0-1.0)")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--calib_dataset", type=str, default="wikitext2")
    parser.add_argument("--calib_nsamples", type=int, default=256)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"=== SVD-LLM Compression ===")
    print(f"Method: {args.method}")
    print(f"Ratio: {args.ratio}")
    print(f"Model: {args.model_path}")

    print("Loading model...")
    model, tokenizer = load_model(args.model_path)

    print(f"Loading calibration data ({args.calib_nsamples} samples)...")
    calibration_data = get_calibration_data(
        tokenizer, args.calib_dataset, args.calib_nsamples, args.seqlen, args.seed
    )

    start_time = time.time()
    print(f"Compressing with {args.method}...")

    if args.method == "svd_llm_w":
        model = compress_model_whitening_only(
            model, tokenizer, calibration_data, args.ratio, args.device
        )
    elif args.method == "svd_llm":
        model = compress_model_sequential(
            model, tokenizer, calibration_data, args.ratio, args.device
        )

    elapsed = time.time() - start_time
    print(f"Compression done in {elapsed:.1f}s")

    print(f"Saving to {args.save_path}...")
    os.makedirs(args.save_path, exist_ok=True)
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    config = {
        "method": args.method,
        "ratio": args.ratio,
        "calib_dataset": args.calib_dataset,
        "calib_nsamples": args.calib_nsamples,
        "seqlen": args.seqlen,
        "compression_time_seconds": elapsed,
    }
    with open(os.path.join(args.save_path, "compression_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
