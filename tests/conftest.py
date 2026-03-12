import os
import sys
import pytest

# 确保 src 可以被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_MODEL_PATH = "/home/xiyaofeng/.cache/huggingface/hub/models--jeffwan--llama-7b-hf/snapshots/82eb0e6908390680598ca3ec1d77adfc5e1b24aa/"


def pytest_addoption(parser):
    parser.addoption(
        "--model-path",
        default=os.environ.get("SVD_LLM_MODEL_PATH", DEFAULT_MODEL_PATH),
        help="Path to the LLaMA model for integration tests",
    )


@pytest.fixture(scope="session")
def model_path(request):
    return request.config.getoption("--model-path")
