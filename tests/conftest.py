import os
import sys
import pytest

# 确保 src 可以被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def pytest_addoption(parser):
    parser.addoption(
        "--model-path",
        default=os.environ.get("SVD_LLM_MODEL_PATH"),
        help="Path to the LLaMA model for integration tests",
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run tests marked as integration",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return

    skip_integration = pytest.mark.skip(
        reason="integration tests are disabled by default; pass --run-integration to enable"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture(scope="session")
def model_path(request):
    value = request.config.getoption("--model-path")
    if not value:
        pytest.skip(
            "model path is not set; provide --model-path or SVD_LLM_MODEL_PATH for integration tests"
        )
    return value
