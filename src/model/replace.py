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
