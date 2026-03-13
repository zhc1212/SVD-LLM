import torch
import os
import tempfile
from src.data.calibration import collect_all_layers_covariances
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


def compress_model_whitening_only(model, tokenizer, calibration_data, ratio, device="cuda"):
    """SVD-LLM(W): 白化 + SVD 截断 (阶段 A)

    一次 forward pass 收集所有 32 层 × 7 线性层的协方差，
    然后逐层压缩并落盘，最后统一替换。

    这是 SVD-LLM(W) 的完整流程，也是 SVD-LLM 阶段 A 的实现。
    SVD-LLM 在此基础上还需要阶段 B (Sequential LoRA 微调)。

    内存: 模型本体 + 所有层协方差 (~54GB CPU) + SVD 工作区

    Args:
        model: HuggingFace CausalLM model
        tokenizer: tokenizer (unused, kept for API consistency)
        calibration_data: list of (1, seqlen) tensors
        ratio: 压缩比 (0.0 - 1.0)
        device: 计算设备

    Returns:
        model: 压缩后的模型
    """
    num_layers = model.config.num_hidden_layers

    # 一次性收集所有层所有线性层的协方差 (256 次 forward pass 而非 32×256)
    print(f"[SVD-LLM(W)] Collecting covariances for all {num_layers} layers in one pass...")
    all_cov = collect_all_layers_covariances(
        model, calibration_data, num_layers, LINEAR_LAYERS_ORDER, device=device
    )
    print("[SVD-LLM(W)] Covariance collection done. Compressing...")

    with tempfile.TemporaryDirectory(prefix="svd_llm_w_") as tmpdir:
        compressed_paths = {}

        for layer_idx in range(num_layers):
            print(f"[SVD-LLM(W)] Compressing layer {layer_idx}/{num_layers - 1}...")

            for linear_name in LINEAR_LAYERS_ORDER:
                target = _get_linear_module(model, layer_idx, linear_name)
                weight = target.weight.data
                d, n = weight.shape
                rank = compute_rank(d, n, ratio)

                XtX, N = all_cov[(layer_idx, linear_name)]
                covariance = (XtX / N).float()
                A, B = compress_linear_whitening_from_covariance(weight.float(), covariance, rank)

                key = (layer_idx, linear_name)
                filename = f"layer{layer_idx}_{linear_name.replace('.', '_')}.pt"
                save_path = os.path.join(tmpdir, filename)
                torch.save({"A": A.cpu(), "B": B.cpu()}, save_path)
                compressed_paths[key] = save_path
                del A, B

            torch.cuda.empty_cache()

        del all_cov

        # 统一替换所有层
        print("[SVD-LLM(W)] Replacing all layers...")
        for layer_idx in range(num_layers):
            for linear_name in LINEAR_LAYERS_ORDER:
                key = (layer_idx, linear_name)
                tensors = torch.load(compressed_paths[key], map_location="cpu")
                replace_linear_with_compressed(model, layer_idx, linear_name, tensors["A"], tensors["B"])

    torch.cuda.empty_cache()
    return model
