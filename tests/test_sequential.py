# tests/test_sequential.py
import pytest
import torch


def test_linear_layers_order():
    """验证线性层顺序列表"""
    from src.compress.sequential_update import LINEAR_LAYERS_ORDER
    assert len(LINEAR_LAYERS_ORDER) == 7
    assert "self_attn.q_proj" in LINEAR_LAYERS_ORDER
    assert "mlp.down_proj" in LINEAR_LAYERS_ORDER


def test_compress_model_whitening_only_smoke():
    """SVD-LLM(W) smoke test: 仅压缩前 2 层"""
    from src.model.loader import load_model
    from src.data.calibration import get_calibration_data
    from src.compress.sequential_update import compress_model_whitening_only
    from src.model.replace import CompressedLinear

    MODEL_PATH = "/home/xiyaofeng/.cache/huggingface/hub/models--jeffwan--llama-7b-hf/snapshots/82eb0e6908390680598ca3ec1d77adfc5e1b24aa/"
    model, tokenizer = load_model(MODEL_PATH)
    calib_data = get_calibration_data(tokenizer, nsamples=4, seqlen=128)

    # 仅压缩前 2 层
    original_num_layers = model.config.num_hidden_layers
    model.config.num_hidden_layers = 2

    compress_model_whitening_only(model, tokenizer, calib_data, ratio=0.5)

    model.config.num_hidden_layers = original_num_layers

    # 验证第 0 层已被替换
    q_proj = model.model.layers[0].self_attn.q_proj
    assert isinstance(q_proj, CompressedLinear)

    # 验证第 2 层未被替换
    q_proj_2 = model.model.layers[2].self_attn.q_proj
    assert isinstance(q_proj_2, torch.nn.Linear)
    assert not isinstance(q_proj_2, CompressedLinear)

    # 模型仍能推理
    inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5)
    assert out.shape[1] > inputs["input_ids"].shape[1]

    del model
    torch.cuda.empty_cache()
