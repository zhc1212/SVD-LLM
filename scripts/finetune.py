"""SVD-LLM Sequential LoRA fine-tuning script.

Usage:
    python scripts/finetune.py \
        --model_path /path/to/original/llama-7b \
        --ratio 0.2 \
        --save_path outputs/llama7b_svd_llm_20 \
        --lora_r 32 --lora_alpha 64 --lr 2e-4 --epochs 1
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.loader import load_model
from src.data.calibration import get_calibration_data
from src.finetune.sequential_lora import finetune_sequential_lora
from src.model.replace import merge_compressed_model


def main():
    parser = argparse.ArgumentParser(description="SVD-LLM Sequential LoRA Fine-tuning")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the original (uncompressed) model")
    parser.add_argument("--ratio", type=float, required=True,
                        help="Compression ratio (0.0-1.0)")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to save the fine-tuned model")

    # LoRA hyperparameters
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs per stage")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length for Alpaca")

    # Calibration
    parser.add_argument("--calib_dataset", type=str, default="wikitext2")
    parser.add_argument("--calib_nsamples", type=int, default=256)
    parser.add_argument("--seqlen", type=int, default=2048)

    # Training options
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("SVD-LLM Sequential LoRA Fine-tuning")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Compression ratio: {args.ratio}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Training: lr={args.lr}, epochs={args.epochs}, batch={args.batch_size}×{args.grad_accum}")
    print(f"Save to: {args.save_path}")

    # Load model
    print("\nLoading model...")
    device_map = "auto" if args.device == "cuda" else args.device
    model, tokenizer = load_model(args.model_path, device_map=device_map)

    # Load calibration data
    print(f"Loading calibration data ({args.calib_nsamples} samples)...")
    calibration_data = get_calibration_data(
        tokenizer, args.calib_dataset, args.calib_nsamples, args.seqlen, args.seed
    )

    # Run sequential LoRA fine-tuning
    start_time = time.time()
    tmp_output_dir = args.save_path + "_finetune_tmp"

    model = finetune_sequential_lora(
        model, tokenizer, args.ratio,
        output_dir=tmp_output_dir,
        calibration_data=calibration_data,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        fp16=not args.no_fp16,
        seed=args.seed,
        device=args.device,
    )

    elapsed = time.time() - start_time
    print(f"\nFine-tuning done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Merge CompressedLinear → nn.Linear for saving
    print("Merging compressed layers...")
    merge_compressed_model(model)

    # Save
    print(f"Saving to {args.save_path}...")
    os.makedirs(args.save_path, exist_ok=True)
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    # Save config
    config = {
        "method": "svd_llm",
        "ratio": args.ratio,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "calib_dataset": args.calib_dataset,
        "calib_nsamples": args.calib_nsamples,
        "seqlen": args.seqlen,
        "finetune_time_seconds": elapsed,
    }
    with open(os.path.join(args.save_path, "compression_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
