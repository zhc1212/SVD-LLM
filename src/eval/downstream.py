import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LM_EVAL_TASKS_DIR = os.path.join(PROJECT_ROOT, "lm_eval_tasks")


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
    from lm_eval.tasks import TaskManager

    if tasks is None:
        tasks = [
            "openbookqa",
            "arc_easy",
            "winogrande",
            "hellaswag",
            "piqa",
            "mathqa_local",
            "truthfulqa_mc2",
        ]

    task_manager = TaskManager(include_path=LM_EVAL_TASKS_DIR)

    # Fix tokenizer for batch generation: jeffwan/llama-7b-hf has broken special
    # token IDs (all mapped to 0). Correct LLaMA values: bos=1, eos=2, unk=0.
    # Wrong eos_token_id causes CUDA indexing errors during padded batch generation.
    if tokenizer.eos_token_id == 0 and hasattr(model, "config"):
        tokenizer.bos_token_id = getattr(model.config, "bos_token_id", 1) or 1
        tokenizer.eos_token_id = getattr(model.config, "eos_token_id", 2) or 2
        tokenizer.bos_token = tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id)
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

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
        task_manager=task_manager,
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
