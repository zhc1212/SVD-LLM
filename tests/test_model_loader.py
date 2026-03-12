import pytest
import torch
from src.model.loader import load_model, get_linear_layers, compute_rank, get_model_config

MODEL_PATH = "/home/xiyaofeng/.cache/huggingface/hub/models--jeffwan--llama-7b-hf/snapshots/82eb0e6908390680598ca3ec1d77adfc5e1b24aa/"


def test_compute_rank():
    """测试压缩比到秩的转换"""
    r = compute_rank(4096, 4096, 0.2)
    assert r == 1638
    r = compute_rank(4096, 11008, 0.4)
    assert r == 1791
    r = compute_rank(4096, 4096, 0.999)
    assert r >= 1


def test_get_model_config():
    """测试加载模型配置"""
    config = get_model_config(MODEL_PATH)
    assert config.hidden_size == 4096
    assert config.intermediate_size == 11008
    assert config.num_hidden_layers == 32


def test_load_model():
    """测试加载完整模型"""
    model, tokenizer = load_model(MODEL_PATH)
    assert model is not None
    assert tokenizer is not None
    layers = get_linear_layers(model)
    assert len(layers) == 224  # 7 linear layers × 32 transformer layers
    names = [n for n, _ in layers]
    assert "model.layers.0.self_attn.q_proj" in names
    assert "model.layers.31.mlp.down_proj" in names
    del model
    torch.cuda.empty_cache()
