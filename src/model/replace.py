import torch
import torch.nn as nn


class CompressedLinear(nn.Module):
    """压缩后的线性层: 两个串联的小线性层

    原始: y = Wx + b, W ∈ R^{d×n}
    压缩: y = A(Bx) + b, A ∈ R^{d×r}, B ∈ R^{r×n}
    """

    def __init__(self, A, B, bias=None):
        super().__init__()
        d, r = A.shape
        _, n = B.shape

        self.first = nn.Linear(n, r, bias=False)
        self.second = nn.Linear(r, d, bias=bias is not None)

        self.first.weight = nn.Parameter(B.clone())
        self.second.weight = nn.Parameter(A.clone())
        if bias is not None:
            self.second.bias = nn.Parameter(bias.clone())

    def forward(self, x):
        return self.second(self.first(x))


def merge_compressed_model(model):
    """将模型中所有 CompressedLinear 合并回标准 nn.Linear

    save_pretrained() 前调用，确保保存的 state_dict 与 from_pretrained() 兼容。
    合并后 W_approx = A @ B 烘焙到单个权重矩阵中。

    Args:
        model: 包含 CompressedLinear 的 HuggingFace model

    Returns:
        model: 所有 CompressedLinear 已替换为 nn.Linear (in-place)
    """
    count = 0
    for layer in model.model.layers:
        for name, module in list(layer.named_modules()):
            if isinstance(module, CompressedLinear):
                # 合并: W = A @ B (second.weight @ first.weight)
                A = module.second.weight.data  # (d, r)
                B = module.first.weight.data   # (r, n)
                W_merged = A @ B               # (d, n)

                d, n = W_merged.shape
                has_bias = module.second.bias is not None

                new_linear = nn.Linear(n, d, bias=has_bias, device=W_merged.device, dtype=W_merged.dtype)
                new_linear.weight = nn.Parameter(W_merged)
                if has_bias:
                    new_linear.bias = nn.Parameter(module.second.bias.data.clone())

                # 定位并替换
                parts = name.split(".")
                parent = layer
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], new_linear)
                count += 1

    print(f"[merge] Converted {count} CompressedLinear → nn.Linear")
    return model


def replace_linear_with_compressed(model, layer_idx, linear_name, A, B):
    """替换模型中指定的线性层为压缩版本

    Args:
        model: HuggingFace model
        layer_idx: transformer 层索引
        linear_name: 如 "self_attn.q_proj"
        A: tensor (d, r)
        B: tensor (r, n)
    """
    layer = model.model.layers[layer_idx]
    parts = linear_name.split(".")

    parent = layer
    for p in parts[:-1]:
        parent = getattr(parent, p)

    original = getattr(parent, parts[-1])
    bias = original.bias.data if original.bias is not None else None

    compressed = CompressedLinear(
        A.to(original.weight.dtype).to(original.weight.device),
        B.to(original.weight.dtype).to(original.weight.device),
        bias=bias.to(original.weight.device) if bias is not None else None,
    )

    setattr(parent, parts[-1], compressed)
    del original
    torch.cuda.empty_cache()
