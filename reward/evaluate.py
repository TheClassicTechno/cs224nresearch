

import argparse
import asyncio
import json
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import EvalMetrics, compute_metrics, print_metrics_table
from .reward_fn import RewardConfig, RewardFunction


# Data helpers 


def load_eval_set(path: str) -> List[dict]:
    """Load eval set from JSON file; validate required keys."""
    with open(path) as f:
        examples = json.load(f)

    required = {"question", "gold_answer", "prompt_type"}
    for i, ex in enumerate(examples):
        missing = required - set(ex.keys())
        if missing:
            raise ValueError(f"Example {i} is missing keys: {missing}")
        if "stated_belief" not in ex:
            ex["stated_belief"] = ""

    return examples


# inference

def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: List[dict],
    batch_size: int = 8,
    max_new_tokens: int = 256,
    device: str = "cuda",
) -> List[str]:
    """Run batched inference; return list of decoded response strings.

    Only newly generated tokens are decoded (input prompt tokens are stripped).
    """
    model.eval()
    responses: List[str] = []

    for start in tqdm(range(0, len(examples), batch_size), desc="Inference"):
        batch = examples[start : start + batch_size]
        prompts = [ex["question"] for ex in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Strip prompt tokens; decode only the response
        for j, out in enumerate(output_ids):
            input_len = inputs["input_ids"][j].shape[0]
            new_tokens = out[input_len:]
            responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

    return responses





async def _score_all_async(
    examples: List[dict],
    responses: List[str],
    reward_fn: RewardFunction,
) -> List[dict]:
    """Score all (example, response) pairs concurrently using eval judge (k=3)."""
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
    # Attach prompt_type so compute_metrics can read it
    for i, s in enumerate(scored):
        s["prompt_type"] = examples[i]["prompt_type"]
    return scored


# pipeline


def run_evaluation(
    model_path: str,
    eval_set_path: str,
    output_path: str,
    reward_config: Optional[RewardConfig] = None,
    device: str = "cuda",
    batch_size: int = 8,
    max_new_tokens: int = 256,
) -> EvalMetrics:
    
    print(f"\n[evaluate] Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to(device)

  
    print(f"[evaluate] Loading eval set from {eval_set_path}")
    examples = load_eval_set(eval_set_path)
    print(f"[evaluate] {len(examples)} examples "
          f"({sum(1 for e in examples if e['prompt_type']=='neutral')} neutral, "
          f"{sum(1 for e in examples if e['prompt_type']=='correct_belief')} correct-belief, "
          f"{sum(1 for e in examples if e['prompt_type']=='misconception')} misconception)")

   
    responses = generate_responses(
        model, tokenizer, examples,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        device=device,
    )

 
    print("[evaluate] Scoring responses with LLM-as-judge (k=3) …")
    rc = reward_config or RewardConfig(
        mode="factual_only", n_samples_train=1, n_samples_eval=3
    )
    reward_fn = RewardFunction(config=rc)
    scored = asyncio.run(_score_all_async(examples, responses, reward_fn))

   
    metrics = compute_metrics(scored)
    model_name = Path(model_path).name
    print_metrics_table(model_name, metrics)


    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "model": model_path,
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
    print(f"[evaluate] Saved results → {output_path}")

    return metrics





def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a model checkpoint on the medical sycophancy eval set."
    )
    p.add_argument("--model_path", required=True, help="HuggingFace checkpoint dir")
    p.add_argument("--eval_set", required=True, help="Path to eval_set.json")
    p.add_argument("--output", required=True, help="Path to save scored results JSON")
    p.add_argument("--judge_model", default="gpt-4o-mini")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=256)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    rc = RewardConfig(judge_model=args.judge_model, n_samples_eval=3)
    run_evaluation(
        model_path=args.model_path,
        eval_set_path=args.eval_set,
        output_path=args.output,
        reward_config=rc,
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
