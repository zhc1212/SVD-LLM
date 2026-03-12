"""端到端集成测试

仅压缩前 2 层 + 少量校准样本，验证全流程不报错。
"""
import pytest
import torch

MODEL_PATH = "/home/xiyaofeng/.cache/huggingface/hub/models--jeffwan--llama-7b-hf/snapshots/82eb0e6908390680598ca3ec1d77adfc5e1b24aa/"


def test_e2e_svd_llm_w():
    """SVD-LLM(W) 端到端: 白化压缩前2层，验证推理"""
    from src.model.loader import load_model
    from src.data.calibration import get_calibration_data
    from src.compress.sequential_update import compress_model_whitening_only
    from src.model.replace import CompressedLinear

    model, tokenizer = load_model(MODEL_PATH)
    calib_data = get_calibration_data(tokenizer, nsamples=4, seqlen=128)

    original = model.config.num_hidden_layers
    model.config.num_hidden_layers = 2
    compress_model_whitening_only(model, tokenizer, calib_data, ratio=0.5)
    model.config.num_hidden_layers = original

    # 验证压缩
    assert isinstance(model.model.layers[0].self_attn.q_proj, CompressedLinear)
    assert isinstance(model.model.layers[0].mlp.down_proj, CompressedLinear)

    # 模型应能推理
    inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10)
    assert out.shape[1] > inputs["input_ids"].shape[1]

    del model
    torch.cuda.empty_cache()


def test_e2e_svd_llm_sequential():
    """SVD-LLM 端到端: sequential update 压缩前2层"""
    from src.model.loader import load_model
    from src.data.calibration import get_calibration_data
    from src.compress.sequential_update import compress_model_sequential
    from src.model.replace import CompressedLinear

    model, tokenizer = load_model(MODEL_PATH)
    calib_data = get_calibration_data(tokenizer, nsamples=4, seqlen=128)

    original = model.config.num_hidden_layers
    model.config.num_hidden_layers = 2
    compress_model_sequential(model, tokenizer, calib_data, ratio=0.5)
    model.config.num_hidden_layers = original

    # 验证压缩
    assert isinstance(model.model.layers[0].self_attn.q_proj, CompressedLinear)
    assert isinstance(model.model.layers[1].mlp.gate_proj, CompressedLinear)
    # 第2层不应被压缩
    assert not isinstance(model.model.layers[2].self_attn.q_proj, CompressedLinear)

    # 模型应能推理
    inputs = tokenizer("The quick brown fox", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10)
    assert out.shape[1] > inputs["input_ids"].shape[1]

    del model
    torch.cuda.empty_cache()
