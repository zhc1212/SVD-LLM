import pytest
import torch
from src.model.loader import load_model
from src.data.calibration import (
    get_calibration_data,
    collect_layer_covariances,
)


@pytest.fixture(scope="module")
def model_and_tokenizer(model_path):
    model, tokenizer = load_model(model_path)
    yield model, tokenizer
    del model
    torch.cuda.empty_cache()


@pytest.mark.integration
def test_get_calibration_data(model_and_tokenizer):
    _, tokenizer = model_and_tokenizer
    samples = get_calibration_data(tokenizer, nsamples=4, seqlen=128)
    assert len(samples) == 4
    assert samples[0].shape == (1, 128)
    assert samples[0].dtype == torch.long


@pytest.mark.integration
def test_collect_layer_covariances(model_and_tokenizer):
    """测试流式协方差收集"""
    model, tokenizer = model_and_tokenizer
    samples = get_calibration_data(tokenizer, nsamples=2, seqlen=64)

    cov_dict = collect_layer_covariances(
        model, samples, layer_idx=0,
        linear_names=["self_attn.q_proj", "mlp.down_proj"],
    )

    # q_proj 输入是 4096 维
    XtX_q, N_q = cov_dict["self_attn.q_proj"]
    assert XtX_q.shape == (4096, 4096)
    assert N_q == 2 * 64

    # down_proj 输入是 11008 维
    XtX_d, N_d = cov_dict["mlp.down_proj"]
    assert XtX_d.shape == (11008, 11008)
    assert N_d == 2 * 64

    # 协方差应该对称
    C = (XtX_q / N_q).float()
    assert torch.allclose(C, C.T, atol=1e-4)
