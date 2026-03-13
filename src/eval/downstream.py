def evaluate_downstream(model, tokenizer, tasks=None, batch_size=8, device="cuda"):
    """使用 lm-evaluation-harness 评估下游任务

    Args:
        model: HuggingFace CausalLM
        tokenizer: tokenizer
        tasks: 任务列表
        batch_size: 评估 batch size
        device: 计算设备

    Returns:
        results: dict
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    if tasks is None:
        tasks = [
            "openbookqa",
            "arc_easy",
            "winogrande",
            "hellaswag",
            "piqa",
            "truthfulqa_mc2",
        ]

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=str(device),
    )

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        batch_size=batch_size,
    )

    return results["results"]


def format_downstream_results(results):
    """格式化下游任务结果为可读字典

    Returns:
        formatted: dict {task_name: metric_value}
    """
    formatted = {}
    for task_name, task_results in results.items():
        if "acc,none" in task_results:
            formatted[task_name] = task_results["acc,none"]
        elif "acc_norm,none" in task_results:
            formatted[task_name] = task_results["acc_norm,none"]
        elif "bleu_max,none" in task_results:
            formatted[task_name] = task_results["bleu_max,none"]
        elif "exact_match,strict-match" in task_results:
            formatted[task_name] = task_results["exact_match,strict-match"]
        elif "exact_match,flexible-extract" in task_results:
            formatted[task_name] = task_results["exact_match,flexible-extract"]

    return formatted
