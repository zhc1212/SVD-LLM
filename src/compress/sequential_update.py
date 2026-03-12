# src/compress/sequential_update.py
import torch
from src.data.calibration import collect_linear_input_activations
from src.compress.whitening import compress_linear_whitening
from src.model.replace import replace_linear_with_compressed
from src.model.loader import compute_rank


# 每个 transformer 层内需要压缩的线性层，按执行顺序排列
LINEAR_LAYERS_ORDER = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def compress_model_sequential(model, tokenizer, calibration_data, ratio, device="cuda"):
    """SVD-LLM 完整压缩: 白化 + Sequential Update

    逐层逐线性层处理:
    1. 收集当前（更新后的）输入激活
    2. 白化 SVD 压缩
    3. 替换为压缩层
    4. 继续下一个（激活会反映之前的压缩）

    Args:
        model: HuggingFace CausalLM model
        tokenizer: tokenizer (unused, kept for API consistency)
        calibration_data: list of (1, seqlen) tensors
        ratio: 压缩比 (0.0 - 1.0)
        device: 计算设备

    Returns:
        model: 压缩后的模型 (in-place)
    """
    num_layers = model.config.num_hidden_layers

    for layer_idx in range(num_layers):
        print(f"[SVD-LLM] Processing layer {layer_idx}/{num_layers - 1}...")

        for linear_name in LINEAR_LAYERS_ORDER:
            layer = model.model.layers[layer_idx]
            parts = linear_name.split(".")
            target = layer
            for p in parts:
                target = getattr(target, p)

            weight = target.weight.data
            d, n = weight.shape
            rank = compute_rank(d, n, ratio)

            # 用当前模型收集激活（反映之前层的压缩）
            X = collect_linear_input_activations(
                model, calibration_data, layer_idx, linear_name, device=device
            )

            A, B = compress_linear_whitening(weight.float(), X, rank)
            replace_linear_with_compressed(model, layer_idx, linear_name, A, B)

            del X, A, B
            torch.cuda.empty_cache()

    return model


def compress_model_whitening_only(model, tokenizer, calibration_data, ratio, device="cuda"):
    """SVD-LLM(W): 仅白化，无 sequential update

    用原始模型的激活一次性白化压缩所有层。
    所有激活都来自原始未压缩模型。

    Args:
        同 compress_model_sequential

    Returns:
        model: 压缩后的模型
    """
    num_layers = model.config.num_hidden_layers

    # 预收集所有激活（来自原始模型）
    all_activations = {}
    print("[SVD-LLM(W)] Collecting activations from original model...")
    for layer_idx in range(num_layers):
        for linear_name in LINEAR_LAYERS_ORDER:
            print(f"  Layer {layer_idx}, {linear_name}")
            X = collect_linear_input_activations(
                model, calibration_data, layer_idx, linear_name, device=device
            )
            all_activations[(layer_idx, linear_name)] = X

    # 用预收集的激活压缩所有层
    print("[SVD-LLM(W)] Compressing...")
    for layer_idx in range(num_layers):
        print(f"  Compressing layer {layer_idx}/{num_layers - 1}...")
        for linear_name in LINEAR_LAYERS_ORDER:
            layer = model.model.layers[layer_idx]
            parts = linear_name.split(".")
            target = layer
            for p in parts:
                target = getattr(target, p)

            weight = target.weight.data
            d, n = weight.shape
            rank = compute_rank(d, n, ratio)

            X = all_activations[(layer_idx, linear_name)]
            A, B = compress_linear_whitening(weight.float(), X, rank)
            replace_linear_with_compressed(model, layer_idx, linear_name, A, B)

            del X

    del all_activations
    torch.cuda.empty_cache()
    return model
