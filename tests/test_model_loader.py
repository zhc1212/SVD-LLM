import pytest
import torch
from src.model.loader import compute_rank, get_model_config, load_model, get_linear_layers


def test_compute_rank():
    """测试压缩比到秩的转换 (纯计算, 无外部依赖)"""
    assert compute_rank(4096, 4096, 0.2) == 1638
    assert compute_rank(4096, 11008, 0.4) == 1791
    assert compute_rank(4096, 4096, 0.999) >= 1
    assert compute_rank(64, 64, 0.5) == 16


@pytest.mark.integration
def test_get_model_config(model_path):
    config = get_model_config(model_path)
    assert config.hidden_size == 4096
    assert config.intermediate_size == 11008
    assert config.num_hidden_layers == 32


@pytest.mark.integration
def test_load_model(model_path):
    model, tokenizer = load_model(model_path)
    assert model is not None
    assert tokenizer is not None
    layers = get_linear_layers(model)
    assert len(layers) == 224
    names = [n for n, _ in layers]
    assert "model.layers.0.self_attn.q_proj" in names
    assert "model.layers.31.mlp.down_proj" in names
    del model
    torch.cuda.empty_cache()
