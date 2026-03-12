import torch
from datasets import load_dataset


def get_calibration_data(tokenizer, dataset_name="wikitext2", nsamples=256, seqlen=2048, seed=42):
    """从数据集加载校准数据，返回 token ids list

    Args:
        tokenizer: HuggingFace tokenizer
        dataset_name: "wikitext2"
        nsamples: 校准样本数
        seqlen: 序列长度
        seed: 随机种子

    Returns:
        list of torch.LongTensor, each shape (1, seqlen)
    """
    if dataset_name == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text_column = "text"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    full_text = "\n\n".join([t for t in dataset[text_column] if t.strip()])
    tokens = tokenizer(full_text, return_tensors="pt").input_ids[0]

    import random
    random.seed(seed)

    samples = []
    max_start = len(tokens) - seqlen - 1
    for _ in range(nsamples):
        start = random.randint(0, max_start)
        sample = tokens[start : start + seqlen].unsqueeze(0)
        samples.append(sample)

    return samples


def collect_linear_input_activations(model, calibration_data, layer_idx,
                                      linear_name, device="cuda"):
    """收集指定线性层的输入激活

    Args:
        model: HuggingFace CausalLM model
        calibration_data: list of (1, seqlen) tensors
        layer_idx: transformer 层索引
        linear_name: 如 "self_attn.q_proj", "mlp.gate_proj"
        device: 计算设备

    Returns:
        activations: tensor of shape (nsamples * seqlen, in_features)
    """
    layer = model.model.layers[layer_idx]
    parts = linear_name.split(".")
    target = layer
    for p in parts:
        target = getattr(target, p)

    activations = []

    def hook_fn(module, input, output):
        inp = input[0].detach()
        activations.append(inp.reshape(-1, inp.shape[-1]).to(torch.float32))

    handle = target.register_forward_hook(hook_fn)

    with torch.no_grad():
        for sample in calibration_data:
            sample = sample.to(device)
            model(sample)

    handle.remove()

    return torch.cat(activations, dim=0)
