import torch
from src.data.calibration import collect_layer_covariances
from src.compress.whitening import compress_linear_whitening_from_covariance
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


def _get_linear_module(model, layer_idx, linear_name):
    """定位模型中指定的线性层"""
    layer = model.model.layers[layer_idx]
    parts = linear_name.split(".")
    target = layer
    for p in parts:
        target = getattr(target, p)
    return target


def compress_model_sequential(model, tokenizer, calibration_data, ratio, device="cuda"):
    """SVD-LLM 完整压缩: 白化 + Sequential Update

    逐层处理，每层一次 forward pass 收集 7 个线性层的协方差（流式 X^T X），
    然后压缩并替换。后续层使用更新后模型的激活。

    内存: 模型本体 + 每层 7 个协方差矩阵 (~870MB) + SVD 工作区

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

        # 一次 forward pass 收集当前层所有 7 个线性层的协方差
        cov_dict = collect_layer_covariances(
            model, calibration_data, layer_idx, LINEAR_LAYERS_ORDER, device=device
        )

        for linear_name in LINEAR_LAYERS_ORDER:
            target = _get_linear_module(model, layer_idx, linear_name)
            weight = target.weight.data
            d, n = weight.shape
            rank = compute_rank(d, n, ratio)

            XtX, N = cov_dict[linear_name]
            covariance = (XtX / N).float()

            A, B = compress_linear_whitening_from_covariance(weight.float(), covariance, rank)
            replace_linear_with_compressed(model, layer_idx, linear_name, A, B)

        del cov_dict
        torch.cuda.empty_cache()

    return model


def compress_model_whitening_only(model, tokenizer, calibration_data, ratio, device="cuda"):
    """SVD-LLM(W): 仅白化，无 sequential update

    用原始模型的激活，逐层收集协方差 → 压缩 → 存储结果。
    全部层处理完后一次性替换，保证所有协方差来自原始模型。

    内存: 模型本体 + 224 个 (A,B) 矩阵对 + 每批 7 个协方差

    Args:
        同 compress_model_sequential

    Returns:
        model: 压缩后的模型
    """
    num_layers = model.config.num_hidden_layers

    # 存储所有压缩结果，最后统一替换
    compressed_weights = {}

    for layer_idx in range(num_layers):
        print(f"[SVD-LLM(W)] Collecting & compressing layer {layer_idx}/{num_layers - 1}...")

        # 一次 forward pass 收集当前层 7 个线性层的协方差
        cov_dict = collect_layer_covariances(
            model, calibration_data, layer_idx, LINEAR_LAYERS_ORDER, device=device
        )

        for linear_name in LINEAR_LAYERS_ORDER:
            target = _get_linear_module(model, layer_idx, linear_name)
            weight = target.weight.data
            d, n = weight.shape
            rank = compute_rank(d, n, ratio)

            XtX, N = cov_dict[linear_name]
            covariance = (XtX / N).float()

            A, B = compress_linear_whitening_from_covariance(weight.float(), covariance, rank)
            # 存储到 CPU 减少 GPU 内存
            compressed_weights[(layer_idx, linear_name)] = (A.cpu(), B.cpu())

        del cov_dict
        torch.cuda.empty_cache()

    # 统一替换所有层
    print("[SVD-LLM(W)] Replacing all layers...")
    for layer_idx in range(num_layers):
        for linear_name in LINEAR_LAYERS_ORDER:
            A, B = compressed_weights[(layer_idx, linear_name)]
            replace_linear_with_compressed(model, layer_idx, linear_name, A, B)

    del compressed_weights
    torch.cuda.empty_cache()
    return model
