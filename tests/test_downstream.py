import pytest


def test_format_downstream_results():
    """测试结果格式化"""
    from src.eval.downstream import format_downstream_results

    mock_results = {
        "openbookqa": {"acc,none": 0.34, "acc_norm,none": 0.35},
        "arc_easy": {"acc,none": 0.75, "acc_norm,none": 0.72},
    }

    formatted = format_downstream_results(mock_results)
    assert formatted["openbookqa"] == 0.34
    assert formatted["arc_easy"] == 0.75
