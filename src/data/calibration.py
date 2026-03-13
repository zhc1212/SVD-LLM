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


class _EarlyExitException(Exception):
    """用于 early exit forward pass 的异常"""
    pass


def _make_xtx_hook(accumulators, acc_key, device):
    """创建累加 X^T X 的 forward hook (在 GPU 上累加 float32，减少同步)"""
    def hook_fn(module, input, output):
        inp = input[0].detach().float()  # (batch, seqlen, n) on GPU
        x = inp.reshape(-1, inp.shape[-1])  # (batch*seqlen, n)
        xtx = x.T @ x  # (n, n) 在 GPU 上计算，留在 GPU
        accumulators[acc_key]["XtX"] += xtx
        accumulators[acc_key]["N"] += x.shape[0]
    return hook_fn


def _batch_samples(calibration_data, batch_size):
    """将 (1, seqlen) 样本打包成 (batch_size, seqlen) 批次"""
    batches = []
    for i in range(0, len(calibration_data), batch_size):
        batch = torch.cat(calibration_data[i:i + batch_size], dim=0)
        batches.append(batch)
    return batches


def collect_layer_covariances(model, calibration_data, layer_idx, linear_names,
                               device="cuda", batch_size=8):
    """收集一个 transformer 层中多个线性层的输入协方差矩阵

    优化:
    - 批量 forward pass (batch_size=8, forward 次数从 256→32)
    - X^T X 在 GPU 上以 float32 累加，最后一次性搬到 CPU 转 float64
    - Early exit: forward pass 到目标层之后立即停止，不跑后续层

    Args:
        model: HuggingFace CausalLM model
        calibration_data: list of (1, seqlen) tensors
        layer_idx: transformer 层索引
        linear_names: list of str
        device: 计算设备
        batch_size: 每次 forward pass 的样本数

    Returns:
        covariances: dict {linear_name: (XtX, N)}
    """
    accumulators = {}
    handles = []

    for name in linear_names:
        target = _get_linear_module(model, layer_idx, name)
        in_features = target.in_features if hasattr(target, 'in_features') else target.weight.shape[1]

        accumulators[name] = {
            "XtX": torch.zeros(in_features, in_features, dtype=torch.float32, device=device),
            "N": 0,
        }
        handle = target.register_forward_hook(_make_xtx_hook(accumulators, name, device))
        handles.append(handle)

    # Early exit hook: 在目标层之后的层停止 forward pass
    # 处理第 l 层时只跑 0..l，跳过 l+1..31，平均省 50% forward 计算
    early_exit_handle = None
    num_layers = model.config.num_hidden_layers
    if layer_idx < num_layers - 1:
        next_layer = model.model.layers[layer_idx + 1]
        def _early_exit_pre_hook(module, args):
            raise _EarlyExitException()
        early_exit_handle = next_layer.register_forward_pre_hook(_early_exit_pre_hook)

    # 批量 forward pass
    batches = _batch_samples(calibration_data, batch_size)
    with torch.no_grad():
        for i, batch in enumerate(batches):
            batch = batch.to(device)
            try:
                model(batch)
            except _EarlyExitException:
                pass  # 预期行为: 到目标层后停止
            if (i + 1) % 4 == 0 or i == len(batches) - 1:
                print(f"    Calibration: {(i+1)*batch_size}/{len(calibration_data)}")

    # 移除 hooks
    for h in handles:
        h.remove()
    if early_exit_handle is not None:
        early_exit_handle.remove()

    # GPU float32 → CPU float64
    result = {}
    for name, acc in accumulators.items():
        result[name] = (acc["XtX"].to(torch.float64).cpu(), acc["N"])
        del acc["XtX"]
    torch.cuda.empty_cache()

    return result


def collect_all_layers_covariances(model, calibration_data, num_layers, linear_names,
                                    device="cuda", batch_size=4):
    """一次 forward pass 收集所有 transformer 层所有线性层的协方差

    适用于 SVD-LLM(W): 模型不更新，所有层可一次性收集。
    X^T X 在 GPU 上以 float32 累加。

    内存占用 (LLaMA-7B, float32 on GPU):
    - 6 linears × 32 layers × 4096² × 4 bytes = ~12GB
    - 1 linear × 32 layers × 11008² × 4 bytes = ~15GB
    - 总计 ~27GB GPU RAM (float32, 最后转 float64 到 CPU)

    Args:
        model: HuggingFace CausalLM model
        calibration_data: list of (1, seqlen) tensors
        num_layers: transformer 层数
        linear_names: list of str
        device: 计算设备
        batch_size: 每次 forward pass 的样本数

    Returns:
        dict {(layer_idx, linear_name): (XtX, N)}
    """
    accumulators = {}
    handles = []

    for layer_idx in range(num_layers):
        for name in linear_names:
            key = (layer_idx, name)
            target = _get_linear_module(model, layer_idx, name)
            in_features = target.in_features if hasattr(target, 'in_features') else target.weight.shape[1]

            accumulators[key] = {
                "XtX": torch.zeros(in_features, in_features, dtype=torch.float32, device=device),
                "N": 0,
            }
            handle = target.register_forward_hook(_make_xtx_hook(accumulators, key, device))
            handles.append(handle)

    batches = _batch_samples(calibration_data, batch_size)
    with torch.no_grad():
        for i, batch in enumerate(batches):
            batch = batch.to(device)
            model(batch)
            if (i + 1) % 4 == 0 or i == len(batches) - 1:
                print(f"  Calibration: {(i+1)*batch_size}/{len(calibration_data)} samples")

    for h in handles:
        h.remove()

    # GPU float32 → CPU float64
    result = {}
    for key, acc in accumulators.items():
        result[key] = (acc["XtX"].to(torch.float64).cpu(), acc["N"])
        del acc["XtX"]
    torch.cuda.empty_cache()

    return result


def collect_linear_input_activations(model, calibration_data, layer_idx,
                                      linear_name, device="cuda"):
    """收集指定线性层的输入激活（保留用于测试和调试）

    WARNING: 此函数保存完整激活，大规模配置下会爆内存。
    生产代码请使用 collect_layer_covariances。
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
