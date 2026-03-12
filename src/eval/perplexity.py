import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss


def evaluate_perplexity(model, tokenizer, dataset_name="wikitext2", device="cuda"):
    """评估模型在指定数据集上的 Perplexity

    使用滑动窗口方法，序列长度 = model.config.max_position_embeddings (2048)

    Args:
        model: HuggingFace CausalLM
        tokenizer: tokenizer
        dataset_name: "wikitext2" 或 "c4"
        device: 计算设备

    Returns:
        perplexity: float
    """
    model.eval()
    seqlen = model.config.max_position_embeddings  # 2048

    if dataset_name == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(dataset["text"])
    elif dataset_name == "c4":
        dataset = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        text = "\n\n".join(dataset["text"][:1100])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    nll_sum = 0.0
    n_tokens = 0
    loss_fn = CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for start in range(0, len(input_ids) - 1, seqlen):
            end = min(start + seqlen, len(input_ids))
            ids = input_ids[start:end].unsqueeze(0).to(device)

            outputs = model(ids)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = ids[:, 1:].contiguous()

            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            nll_sum += loss.item()
            n_tokens += shift_labels.numel()

    avg_nll = nll_sum / n_tokens
    perplexity = torch.exp(torch.tensor(avg_nll)).item()

    return perplexity
