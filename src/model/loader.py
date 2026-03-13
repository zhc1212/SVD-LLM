import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaTokenizer


def _load_tokenizer(model_path: str):
    """Load tokenizer with a compatibility fallback for local LLaMA checkpoints."""
    try:
        return AutoTokenizer.from_pretrained(model_path, use_fast=False, legacy=True)
    except (RecursionError, ValueError, TypeError):
        # Fallback for environments where AutoTokenizer + local LLaMA snapshots recurse.
        return LlamaTokenizer.from_pretrained(model_path, legacy=True)


def load_model(model_path: str, device_map: str = "auto", dtype=torch.float16):
    """加载 HuggingFace 模型，返回 (model, tokenizer)"""
    tokenizer = _load_tokenizer(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def get_model_config(model_path: str):
    """获取模型配置（不加载权重）"""
    return AutoConfig.from_pretrained(model_path)


def get_linear_layers(model):
    """返回所有需要压缩的线性层及其名称
    Returns: list of (name, module) tuples
    跳过 lm_head 和 embed_tokens
    """
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
            linear_layers.append((name, module))
    return linear_layers


def compute_rank(d: int, n: int, ratio: float) -> int:
    """根据压缩比计算保留的秩 r
    R_w = 1 - (d + n) * r / (d * n)
    => r = (1 - R_w) * d * n / (d + n)
    """
    r = int((1 - ratio) * d * n / (d + n))
    return max(1, r)
