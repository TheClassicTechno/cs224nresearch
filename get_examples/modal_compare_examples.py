"""
Sample misconception eval examples and compare SFT vs GRPO model outputs on Modal.

Run:
    modal run get_examples/modal_compare_examples.py
    modal run get_examples/modal_compare_examples.py --sample-size 7 --seed 42
"""

import json
import os
import random
from typing import Any

import modal

APP_NAME = "medquad-misconception-example-compare"
HF_DATASET = "mli5/medquad-sycophancy"
SPLIT_NAME = "eval"
PROMPT_CONDITION = "misconception"
SFT_MODEL_ID = "mli5/qwen3.5-4b-sft-medquad"
GRPO_MODEL_ID = "mli5/qwen3.5-2b-grpo-medquad-new-reward-conditioned"
OUTPUT_PATH = "/tmp/modal_compare_examples.json"
LOCAL_OUTPUT_PATH = "get_examples/modal_compare_examples_output.json"
GPU_TYPE = "L4"
DEFAULT_SAMPLE_SIZE = 7
DEFAULT_SEED = 42
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_BATCH_SIZE = 4

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "huggingface_hub",
        "tqdm",
    )
    .env({"PYTHONUNBUFFERED": "1"})
)


def _validate_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=60 * 60,
    secrets=[modal.Secret.from_dotenv()],
)
def generate_model_comparisons(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    import gc

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    _validate_positive(sample_size, "sample_size")
    _validate_positive(max_new_tokens, "max_new_tokens")

    hf_token = os.environ.get("HF_TOKEN")
    dataset = load_dataset(HF_DATASET, split="train", token=hf_token)

    examples = [
        {
            "dataset_index": idx,
            "prompt": row["new_question"],
            "gold_answer": row["answer"],
        }
        for idx, row in enumerate(dataset)
        if row.get("split") == SPLIT_NAME and row.get("prompt_condition") == PROMPT_CONDITION
    ]

    if len(examples) < sample_size:
        raise RuntimeError(
            f"Requested {sample_size} examples, but only found {len(examples)} "
            f"{PROMPT_CONDITION} examples in the {SPLIT_NAME} split."
        )

    sampled_examples = random.Random(seed).sample(examples, sample_size)

    def generate_for_model(model_id: str) -> list[str]:
        print(f"Loading tokenizer and model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token,
            trust_remote_code=True,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token,
            trust_remote_code=True,
        )
        model.eval()

        outputs_text: list[str] = []
        for start in tqdm(range(0, len(sampled_examples), DEFAULT_BATCH_SIZE), desc=model_id):
            batch = sampled_examples[start : start + DEFAULT_BATCH_SIZE]
            prompts = []
            for example in batch:
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": example["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompts.append(prompt_text)

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(model.device)

            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            for seq in generated:
                generated_ids = seq[input_len:]
                text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                outputs_text.append(text)

        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return outputs_text

    sft_outputs = generate_for_model(SFT_MODEL_ID)
    grpo_outputs = generate_for_model(GRPO_MODEL_ID)

    if len(sft_outputs) != sample_size or len(grpo_outputs) != sample_size:
        raise RuntimeError(
            "Model output counts did not match the sampled example count: "
            f"sft={len(sft_outputs)} grpo={len(grpo_outputs)} sample_size={sample_size}"
        )

    result_examples = []
    for example, sft_output, grpo_output in zip(sampled_examples, sft_outputs, grpo_outputs):
        result_examples.append(
            {
                "dataset_index": example["dataset_index"],
                "prompt": example["prompt"],
                "gold_answer": example["gold_answer"],
                "sft_model_output": sft_output,
                "grpo_model_output": grpo_output,
            }
        )

    result = {
        "dataset": HF_DATASET,
        "split": SPLIT_NAME,
        "prompt_condition": PROMPT_CONDITION,
        "sample_size": sample_size,
        "seed": seed,
        "models": {
            "sft": SFT_MODEL_ID,
            "grpo": GRPO_MODEL_ID,
        },
        "examples": result_examples,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved comparison artifact to {OUTPUT_PATH}")
    for idx, example in enumerate(result_examples, start=1):
        print(f"\n=== Example {idx} (dataset_index={example['dataset_index']}) ===")
        print(f"Prompt:\n{example['prompt']}")
        print(f"\nGold answer:\n{example['gold_answer']}")
        print(f"\nSFT model output:\n{example['sft_model_output']}")
        print(f"\nGRPO model output:\n{example['grpo_model_output']}")

    return result


@app.local_entrypoint()
def main(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> None:
    result = generate_model_comparisons.remote(
        sample_size=sample_size,
        seed=seed,
        max_new_tokens=max_new_tokens,
    )
    with open(LOCAL_OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved local artifact to {LOCAL_OUTPUT_PATH}")
    print(json.dumps(result, indent=2))
