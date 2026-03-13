import pytest
import torch


def test_linear_layers_order():
    """验证线性层顺序列表"""
    from src.compress.compress_model import LINEAR_LAYERS_ORDER
    assert len(LINEAR_LAYERS_ORDER) == 7
    assert "self_attn.q_proj" in LINEAR_LAYERS_ORDER
    assert "mlp.down_proj" in LINEAR_LAYERS_ORDER


@pytest.mark.integration
def test_compress_model_whitening_only_smoke(model_path):
    """SVD-LLM(W) smoke test: 仅压缩前 2 层"""
    from src.model.loader import load_model
    from src.data.calibration import get_calibration_data
    from src.compress.compress_model import compress_model_whitening_only
    from src.model.replace import CompressedLinear

    model, tokenizer = load_model(model_path)
    calib_data = get_calibration_data(tokenizer, nsamples=4, seqlen=128)

    original_num_layers = model.config.num_hidden_layers
    model.config.num_hidden_layers = 2

    compress_model_whitening_only(model, tokenizer, calib_data, ratio=0.5)

    model.config.num_hidden_layers = original_num_layers

    assert isinstance(model.model.layers[0].self_attn.q_proj, CompressedLinear)
    assert not isinstance(model.model.layers[2].self_attn.q_proj, CompressedLinear)

    inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5)
    assert out.shape[1] > inputs["input_ids"].shape[1]

    del model
    torch.cuda.empty_cache()
