"""Sequential LoRA fine-tuning for SVD-LLM.

Two-stage LoRA training on a CompressedLinear model:
  Stage 1 (LoRA_u): freeze .first (W'_v), LoRA fine-tune .second (W'_u)
  Stage 2 (LoRA_v): freeze .second (W'_u), LoRA fine-tune .first (W'_v)
"""

import os
import torch
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

from src.data.alpaca import get_alpaca_dataset
from src.data.calibration import get_calibration_data
from src.compress.compress_model import compress_model_whitening_only
from src.model.replace import CompressedLinear


def _find_compressed_target_modules(model, target_attr):
    """Find all module names matching CompressedLinear sub-layers.

    Args:
        model: model (possibly PEFT-wrapped)
        target_attr: "first" or "second"

    Returns:
        list of target module name patterns for LoraConfig
    """
    names = set()
    base = model.base_model if hasattr(model, "base_model") else model
    for name, module in base.named_modules():
        if isinstance(module, CompressedLinear):
            names.add(f"{name}.{target_attr}")
    return sorted(names)


def _freeze_compressed_attr(model, attr_to_freeze):
    """Freeze .first or .second in all CompressedLinear modules.

    Args:
        model: the model
        attr_to_freeze: "first" or "second"
    """
    for module in model.modules():
        if isinstance(module, CompressedLinear):
            sub = getattr(module, attr_to_freeze)
            for param in sub.parameters():
                param.requires_grad = False


def _unfreeze_compressed_attr(model, attr_to_unfreeze):
    """Unfreeze .first or .second in all CompressedLinear modules."""
    for module in model.modules():
        if isinstance(module, CompressedLinear):
            sub = getattr(module, attr_to_unfreeze)
            for param in sub.parameters():
                param.requires_grad = True


def _run_lora_stage(model, dataset, target_modules, stage_name, output_dir,
                    lora_r=32, lora_alpha=64, lora_dropout=0.05,
                    lr=2e-4, epochs=1, batch_size=4, grad_accum=4,
                    max_length=256, gradient_checkpointing=True, fp16=True,
                    seed=42):
    """Run a single LoRA training stage.

    Args:
        model: HuggingFace model with CompressedLinear layers
        dataset: tokenized dataset
        target_modules: list of module name patterns for LoRA
        stage_name: "stage1_lora_u" or "stage2_lora_v"
        output_dir: directory for training outputs
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: dropout rate
        lr: learning rate
        epochs: number of training epochs
        batch_size: per-device batch size
        grad_accum: gradient accumulation steps
        max_length: unused, kept for API consistency
        gradient_checkpointing: enable gradient checkpointing
        fp16: enable fp16 training

    Returns:
        model: model with LoRA merged back
    """
    print(f"\n[{stage_name}] Applying LoRA to {len(target_modules)} modules...")
    print(f"[{stage_name}] Target modules: {target_modules[:3]}...")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    stage_output_dir = os.path.join(output_dir, stage_name)

    training_args = TrainingArguments(
        output_dir=stage_output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=fp16,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=seed,
        data_seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print(f"[{stage_name}] Training...")
    trainer.train()

    print(f"[{stage_name}] Merging LoRA weights...")
    model = model.merge_and_unload()

    torch.cuda.empty_cache()
    return model


def finetune_sequential_lora(model, tokenizer, ratio, output_dir="outputs/finetune_tmp",
                              calibration_data=None, lora_r=32, lora_alpha=64,
                              lora_dropout=0.05, lr=2e-4, epochs=1, batch_size=4,
                              grad_accum=4, max_length=256,
                              gradient_checkpointing=True, fp16=True,
                              seed=42, device="cuda"):
    """Full Sequential LoRA fine-tuning pipeline.

    1. Re-compress model (Stage A) to get CompressedLinear structure
    2. Load Alpaca dataset
    3. Stage 1: LoRA on .second (W'_u), freeze .first (W'_v)
    4. Stage 2: LoRA on .first (W'_v), freeze .second (W'_u)

    Args:
        model: original HuggingFace CausalLM (uncompressed)
        tokenizer: tokenizer
        ratio: compression ratio (0.0-1.0)
        output_dir: directory for intermediate outputs
        calibration_data: pre-loaded calibration data (if None, will load)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        lr: learning rate
        epochs: training epochs per stage
        batch_size: per-device batch size
        grad_accum: gradient accumulation steps
        max_length: Alpaca sequence length
        gradient_checkpointing: enable gradient checkpointing
        fp16: enable fp16 training
        device: compute device

    Returns:
        model: fine-tuned model (still in CompressedLinear form)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Stage A: Re-compress to get CompressedLinear structure
    print("=" * 60)
    print("[Stage A] Compressing model with whitening + SVD...")
    print("=" * 60)

    if calibration_data is None:
        print("Loading calibration data...")
        calibration_data = get_calibration_data(tokenizer)

    model = compress_model_whitening_only(model, tokenizer, calibration_data, ratio, device=device)

    # Verify CompressedLinear structure
    compressed_count = sum(
        1 for m in model.modules() if isinstance(m, CompressedLinear)
    )
    print(f"[Stage A] Done. {compressed_count} CompressedLinear layers created.")

    # Load Alpaca dataset
    print("\nLoading Alpaca dataset...")
    dataset = get_alpaca_dataset(tokenizer, max_length=max_length, seed=seed)
    print(f"Alpaca dataset: {len(dataset)} examples")

    lora_kwargs = dict(
        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        lr=lr, epochs=epochs, batch_size=batch_size, grad_accum=grad_accum,
        max_length=max_length, gradient_checkpointing=gradient_checkpointing,
        fp16=fp16, seed=seed,
    )

    # Stage 1: LoRA on .second (W'_u), freeze .first (W'_v)
    print("\n" + "=" * 60)
    print("[Stage B-1] LoRA fine-tuning W'_u (.second)")
    print("=" * 60)

    _freeze_compressed_attr(model, "first")
    target_modules_stage1 = _find_compressed_target_modules(model, "second")
    model = _run_lora_stage(
        model, dataset, target_modules_stage1,
        stage_name="stage1_lora_u",
        output_dir=output_dir,
        **lora_kwargs,
    )
    _unfreeze_compressed_attr(model, "first")

    # Stage 2: LoRA on .first (W'_v), freeze .second (W'_u)
    print("\n" + "=" * 60)
    print("[Stage B-2] LoRA fine-tuning W'_v (.first)")
    print("=" * 60)

    _freeze_compressed_attr(model, "second")
    target_modules_stage2 = _find_compressed_target_modules(model, "first")
    model = _run_lora_stage(
        model, dataset, target_modules_stage2,
        stage_name="stage2_lora_v",
        output_dir=output_dir,
        **lora_kwargs,
    )
    _unfreeze_compressed_attr(model, "second")

    print("\n" + "=" * 60)
    print("[Done] Sequential LoRA fine-tuning complete.")
    print("=" * 60)

    return model
