"""
GRPO + LoRA training on Modal.

This entrypoint runs the no-KL LoRA-only GRPO trainer on an A10G GPU and saves
the adapter checkpoint to /tmp before optionally pushing it to the HF Hub.

Run:
    modal run modal_training/grpo_lora_train.py
    modal run --detach modal_training/grpo_lora_train.py
"""

import os

import modal

app = modal.App("qwen-grpo-lora-train")

_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, ".."))

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers>=4.51.0",
        "accelerate",
        "datasets",
        "tqdm",
        "huggingface_hub",
        "openai",
        "pydantic",
        "wandb",
        "peft",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir(_repo_root, remote_path="/root/repo")
)


HF_DATASET = "mli5/medquad-sycophancy"
MODEL_ID = "Qwen/Qwen3.5-2B"
HF_REPO_ID = "mli5/qwen3.5-2b-grpo-lora-no-kl"
OUTPUT_DIR = "/tmp/grpo_lora_qwen_ckpt"

BATCH_SIZE = 2
NUM_SAMPLES_PER_PROMPT = 4
MAX_NEW_TOKENS = 1024
MAX_LENGTH = 1024
MAX_EXAMPLES = 1800
NUM_TRAINING_STEPS = 900
WARMUP_STEPS = 90
LR = 1e-6

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear"

REWARD_MODE = "condition_aware"
LAMBDA_PENALTY = 1.5
MU_PENALTY = 0.15
SAMPLE_RANDOMLY = False
SAMPLE_SEED = 42

QUICK_TEST = False
QUICK_TEST_MAX_EXAMPLES = 8
QUICK_TEST_MAX_STEPS = 2


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    secrets=[modal.Secret.from_dotenv()],
)
def run_grpo_lora_train():
    import sys

    sys.path.insert(0, "/root/repo")

    from training.grpo_lora_train import GRPOLoRAConfig, run_grpo_lora
    from training.reward import RewardConfig

    hf_token = os.environ.get("HF_TOKEN")

    cfg = GRPOLoRAConfig(
        model_name=MODEL_ID,
        data_path=None,
        hf_dataset=HF_DATASET,
        split="train",
        rl_subset_split="rl_train",
        batch_size=BATCH_SIZE,
        num_samples_per_prompt=NUM_SAMPLES_PER_PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        max_length=MAX_LENGTH,
        num_training_steps=NUM_TRAINING_STEPS,
        warmup_steps=WARMUP_STEPS,
        lr=LR,
        device="cuda",
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        lora_target_modules=LORA_TARGET_MODULES,
    )
    reward_cfg = RewardConfig(
        mode=REWARD_MODE,
        lambda_penalty=LAMBDA_PENALTY,
        mu_penalty=MU_PENALTY,
    )

    if QUICK_TEST:
        max_examples = QUICK_TEST_MAX_EXAMPLES
        max_steps = QUICK_TEST_MAX_STEPS
        hf_repo_id = None
        print(f"QUICK_TEST: max_examples={max_examples}, max_steps={max_steps}, no HF push")
    else:
        max_examples = MAX_EXAMPLES
        max_steps = NUM_TRAINING_STEPS
        hf_repo_id = HF_REPO_ID if hf_token else None

    print(
        "Running GRPO+LoRA: "
        f"model={MODEL_ID}, dataset={HF_DATASET}, max_examples={max_examples}, steps={max_steps}"
    )
    metrics = run_grpo_lora(
        cfg,
        reward_cfg,
        output_dir=OUTPUT_DIR,
        hf_repo_id=hf_repo_id,
        max_examples=max_examples,
        sample_randomly=SAMPLE_RANDOMLY and not QUICK_TEST,
        sample_seed=SAMPLE_SEED,
        max_steps=max_steps,
        debug_print_samples=QUICK_TEST,
        dry_run=False,
        wandb_project="grpo-sycophancy",
        wandb_run_name="qwen3.5-2b-grpo-lora-no-kl",
    )
    print(f"Done. Metrics: {metrics}")
    return metrics


@app.local_entrypoint()
def main():
    metrics = run_grpo_lora_train.remote()
    print(metrics)
