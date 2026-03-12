import pytest
import torch
from src.model.loader import load_model
from src.data.calibration import get_calibration_data, collect_linear_input_activations

MODEL_PATH = "/home/xiyaofeng/.cache/huggingface/hub/models--jeffwan--llama-7b-hf/snapshots/82eb0e6908390680598ca3ec1d77adfc5e1b24aa/"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    model, tokenizer = load_model(MODEL_PATH)
    yield model, tokenizer
    del model
    torch.cuda.empty_cache()


def test_get_calibration_data(model_and_tokenizer):
    _, tokenizer = model_and_tokenizer
    samples = get_calibration_data(tokenizer, nsamples=4, seqlen=128)
    assert len(samples) == 4
    assert samples[0].shape == (1, 128)
    assert samples[0].dtype == torch.long


def test_collect_linear_input_activations(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    samples = get_calibration_data(tokenizer, nsamples=2, seqlen=64)
    X = collect_linear_input_activations(
        model, samples, layer_idx=0, linear_name="self_attn.q_proj"
    )
    assert X.shape == (2 * 64, 4096)
    assert X.dtype == torch.float32
    assert not torch.isnan(X).any()
