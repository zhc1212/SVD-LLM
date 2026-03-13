"""Alpaca dataset loading for Sequential LoRA fine-tuning."""

from datasets import load_dataset


ALPACA_PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)

ALPACA_PROMPT_TEMPLATE_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)


def _format_prompt(example):
    """Format a single Alpaca example into prompt + response."""
    if example.get("input", "").strip():
        prompt = ALPACA_PROMPT_TEMPLATE_WITH_INPUT.format(
            instruction=example["instruction"],
            input=example["input"],
        )
    else:
        prompt = ALPACA_PROMPT_TEMPLATE.format(instruction=example["instruction"])
    return prompt, example["output"]


def get_alpaca_dataset(tokenizer, max_length=256, seed=42):
    """Load and tokenize the Alpaca-cleaned dataset for causal LM training.

    Prompt tokens are masked with -100 in labels so loss is only computed
    on the response portion.

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: maximum sequence length
        seed: random seed for shuffling

    Returns:
        Dataset with columns: input_ids, attention_mask, labels
    """
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    dataset = dataset.shuffle(seed=seed)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize_fn(examples):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for i in range(len(examples["instruction"])):
            example = {
                "instruction": examples["instruction"][i],
                "input": examples["input"][i],
                "output": examples["output"][i],
            }
            prompt, response = _format_prompt(example)
            full_text = prompt + response + tokenizer.eos_token

            # Tokenize full text
            full = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None,
            )

            # Tokenize prompt only to find boundary
            prompt_tokens = tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
            prompt_len = len(prompt_tokens["input_ids"])

            # Build labels: mask prompt tokens with -100
            labels = list(full["input_ids"])
            for j in range(min(prompt_len, len(labels))):
                labels[j] = -100
            # Also mask padding tokens
            for j in range(len(labels)):
                if full["attention_mask"][j] == 0:
                    labels[j] = -100

            input_ids_list.append(full["input_ids"])
            attention_mask_list.append(full["attention_mask"])
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }

    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
        desc="Tokenizing Alpaca",
    )

    # Filter out samples where all labels are -100 (prompt truncated to max_length)
    total_before = len(dataset)
    dataset = dataset.filter(
        lambda x: any(label != -100 for label in x["labels"]),
        desc="Filtering all-masked samples",
    )
    filtered = total_before - len(dataset)
    if filtered > 0:
        print(f"[Alpaca] Filtered {filtered}/{total_before} samples with no response tokens")

    dataset.set_format("torch")
    return dataset
