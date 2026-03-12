import pytest
import torch


def test_perplexity_original_model():
    """测试原始模型的 WikiText-2 Perplexity 应在合理范围"""
    from src.model.loader import load_model
    from src.eval.perplexity import evaluate_perplexity

    MODEL_PATH = "/home/xiyaofeng/.cache/huggingface/hub/models--jeffwan--llama-7b-hf/snapshots/82eb0e6908390680598ca3ec1d77adfc5e1b24aa/"
    model, tokenizer = load_model(MODEL_PATH)

    ppl = evaluate_perplexity(model, tokenizer, "wikitext2")
    print(f"WikiText-2 Perplexity: {ppl:.2f}")

    # 论文报告 5.68, 允许合理偏差
    assert 4.0 < ppl < 8.0, f"Perplexity {ppl} out of expected range"

    del model
    torch.cuda.empty_cache()
