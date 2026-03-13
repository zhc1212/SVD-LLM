import torch
from src.model.replace import CompressedLinear


def test_compressed_linear_forward():
    """测试压缩层的前向传播"""
    A = torch.randn(64, 20)
    B = torch.randn(20, 128)
    bias = torch.randn(64)

    layer = CompressedLinear(A, B, bias)
    x = torch.randn(4, 128)
    y = layer(x)

    assert y.shape == (4, 64)
    y_manual = x @ (A @ B).T + bias
    assert torch.allclose(y, y_manual, atol=1e-4)


def test_compressed_linear_no_bias():
    """测试无 bias 的压缩层"""
    A = torch.randn(64, 20)
    B = torch.randn(20, 128)

    layer = CompressedLinear(A, B, bias=None)
    x = torch.randn(4, 128)
    y = layer(x)

    assert y.shape == (4, 64)
    assert layer.second.bias is None
