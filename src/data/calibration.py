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


def _get_linear_module(model, layer_idx, linear_name):
    """定位模型中指定的线性层"""
    layer = model.model.layers[layer_idx]
    parts = linear_name.split(".")
    target = layer
    for p in parts:
        target = getattr(target, p)
    return target


def collect_layer_covariances(model, calibration_data, layer_idx, linear_names, device="cuda"):
    """收集一个 transformer 层中多个线性层的输入协方差矩阵（流式累加）

    不存储原始激活，只累加 X^T X 和样本计数。
    一次 forward pass 同时收集所有指定线性层的协方差。

    内存占用: 每个线性层一个 n×n 矩阵（n=in_features）
    - 4096×4096 = 64MB per layer (q/k/v/o/gate/up)
    - 11008×11008 = 485MB per layer (down)
    - 一个 transformer 层 7 个线性层共 ~870MB

    Args:
        model: HuggingFace CausalLM model
        calibration_data: list of (1, seqlen) tensors
        layer_idx: transformer 层索引
        linear_names: list of str, 如 ["self_attn.q_proj", "mlp.gate_proj"]
        device: 计算设备

    Returns:
        covariances: dict {linear_name: (XtX, N)}
            XtX: tensor (n, n) — 累加的 X^T X
            N: int — 累加的样本数（token 数）
    """
    # 初始化累加器
    accumulators = {}
    handles = []

    for name in linear_names:
        target = _get_linear_module(model, layer_idx, name)
        in_features = target.in_features if hasattr(target, 'in_features') else target.weight.shape[1]

        accumulators[name] = {
            "XtX": torch.zeros(in_features, in_features, dtype=torch.float64, device="cpu"),
            "N": 0,
        }

        def make_hook(acc_name):
            def hook_fn(module, input, output):
                inp = input[0].detach().float()  # (batch, seqlen, n)
                x = inp.reshape(-1, inp.shape[-1])  # (batch*seqlen, n)
                # 累加到 CPU，避免 GPU 内存压力
                xtx = (x.T @ x).to(torch.float64).cpu()
                accumulators[acc_name]["XtX"] += xtx
                accumulators[acc_name]["N"] += x.shape[0]
            return hook_fn

        handle = target.register_forward_hook(make_hook(name))
        handles.append(handle)

    # 前向传播收集
    with torch.no_grad():
        for sample in calibration_data:
            sample = sample.to(device)
            model(sample)

    # 移除 hooks
    for h in handles:
        h.remove()

    # 返回 (XtX, N)
    return {name: (acc["XtX"], acc["N"]) for name, acc in accumulators.items()}


def collect_linear_input_activations(model, calibration_data, layer_idx,
                                      linear_name, device="cuda"):
    """收集指定线性层的输入激活（保留用于测试和调试）

    WARNING: 此函数保存完整激活，大规模配置下会爆内存。
    生产代码请使用 collect_layer_covariances。

    Args:
        model: HuggingFace CausalLM model
        calibration_data: list of (1, seqlen) tensors
        layer_idx: transformer 层索引
        linear_name: 如 "self_attn.q_proj"
        device: 计算设备

    Returns:
        activations: tensor of shape (nsamples * seqlen, in_features)
    """
    target = _get_linear_module(model, layer_idx, linear_name)
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
