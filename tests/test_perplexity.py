import pytest
import torch


@pytest.mark.integration
def test_perplexity_original_model(model_path):
    """测试原始模型的 WikiText-2 Perplexity"""
    from src.model.loader import load_model
    from src.eval.perplexity import evaluate_perplexity

    model, tokenizer = load_model(model_path)
    ppl = evaluate_perplexity(model, tokenizer, "wikitext2")
    print(f"WikiText-2 Perplexity: {ppl:.2f}")
    assert 4.0 < ppl < 8.0, f"Perplexity {ppl} out of expected range"
    del model
    torch.cuda.empty_cache()
