"""Baseline evaluation: run inference + LLM-as-judge scoring on a subset."""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from reward.evaluate import load_eval_set, generate_responses
from reward.metrics import EvalMetrics, compute_metrics, print_metrics_table
from reward.reward_fn import RewardConfig, RewardFunction
import asyncio
from tqdm.asyncio import tqdm_asyncio


async def score_responses_async(
    examples: List[dict],
    responses: List[str],
    reward_fn: RewardFunction,
) -> List[dict]:
    """Score all responses using LLM-as-judge (eval mode, k=3)."""
    tasks = [
        reward_fn.compute_async(
            question=ex["question"],
            response=resp,
            gold_answer=ex["gold_answer"],
            prompt_type=ex["prompt_type"],
            stated_belief=ex.get("stated_belief", ""),
            eval_mode=True,
        )
        for ex, resp in zip(examples, responses)
    ]
    scored = await tqdm_asyncio.gather(*tasks, desc="Judging")
    for i, s in enumerate(scored):
        s["prompt_type"] = examples[i]["prompt_type"]
    return scored


def run_baseline_eval(
    model_path: str,
    eval_set_path: str,
    output_path: str,
    subset_size: Optional[int] = None,
    device: str = "cuda",
    batch_size: int = 8,
    max_new_tokens: int = 256,
    judge_model: str = "gemini-1.5-flash",
) -> EvalMetrics:
    """Run baseline evaluation: inference + scoring on (optionally) a subset.

    Args:
        model_path: HuggingFace model checkpoint path
        eval_set_path: Path to eval set JSON
        output_path: Where to save results JSON
        subset_size: If provided, only evaluate first N examples (for quick testing)
        device: Inference device
        batch_size: Batch size for inference
        max_new_tokens: Max tokens to generate
        judge_model: Judge LLM model ID

    Returns:
        EvalMetrics object
    """
    print(f"\n[baseline_eval] Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to(device)

    print(f"[baseline_eval] Loading eval set: {eval_set_path}")
    examples = load_eval_set(eval_set_path)

    if subset_size:
        examples = examples[:subset_size]
        print(f"[baseline_eval] Using subset of {len(examples)} examples")

    counts = {
        "neutral": sum(1 for e in examples if e["prompt_type"] == "neutral"),
        "correct_belief": sum(1 for e in examples if e["prompt_type"] == "correct_belief"),
        "misconception": sum(1 for e in examples if e["prompt_type"] == "misconception"),
    }
    print(
        f"[baseline_eval] {len(examples)} examples "
        f"({counts['neutral']} neutral, {counts['correct_belief']} correct-belief, "
        f"{counts['misconception']} misconception)"
    )

    print("[baseline_eval] Running inference...")
    responses = generate_responses(
        model, tokenizer, examples,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    print(f"[baseline_eval] Scoring with LLM-as-judge ({judge_model}, k=3)...")
    rc = RewardConfig(
        mode="factual_only",
        judge_model=judge_model,
        n_samples_train=1,
        n_samples_eval=3,
    )
    reward_fn = RewardFunction(config=rc)
    scored = asyncio.run(score_responses_async(examples, responses, reward_fn))

    metrics = compute_metrics(scored)
    model_name = Path(model_path).name
    print_metrics_table(model_name, metrics)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "model": model_path,
        "subset_size": subset_size,
        "metrics": metrics.to_dict(),
        "examples": [
            {
                "question": ex["question"],
                "prompt_type": ex["prompt_type"],
                "stated_belief": ex.get("stated_belief", ""),
                "gold_answer": ex["gold_answer"],
                "response": resp,
                "r_factual": sc["r_factual"],
                "r_syc": sc["r_syc"],
                "r_contra": sc["r_contra"],
            }
            for ex, resp, sc in zip(examples, responses, scored)
        ],
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[baseline_eval] Results saved → {output_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run baseline evaluation: inference + LLM-as-judge scoring"
    )
    parser.add_argument("--model_path", required=True, help="HuggingFace checkpoint")
    parser.add_argument("--eval_set", required=True, help="Path to eval_set.json")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--subset_size", type=int, default=None,
        help="Only evaluate first N examples (for quick testing)"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--judge_model", default="gemini-1.5-flash")

    args = parser.parse_args()
    run_baseline_eval(
        model_path=args.model_path,
        eval_set_path=args.eval_set,
        output_path=args.output,
        subset_size=args.subset_size,
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        judge_model=args.judge_model,
    )
