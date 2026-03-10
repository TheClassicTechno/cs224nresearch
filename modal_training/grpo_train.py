"""
GRPO training on Modal
Run GRPO on a Modal GPU, save to /tmp, then optionally push to HF Hub.
Uses training package baked into the image via add_local_dir.

Run:
    modal run modal_training/grpo_train.py
    modal run --detach modal_training/grpo_train.py

Quick test (few examples, no HF push): set QUICK_TEST = True below.
"""

import os
import modal

app = modal.App("qwen-grpo-train")

# Bake repo into image 
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
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir(_repo_root, remote_path="/root/repo")
)


HF_DATASET = "mli5/medquad-sycophancy"
MODEL_ID = "Qwen/Qwen3.5-2B"
HF_REPO_ID = "mli5/qwen3.5-2b-grpo-medquad-reward-conditioned"
OUTPUT_DIR = "/tmp/grpo_qwen_ckpt"

BATCH_SIZE = 2
NUM_SAMPLES_PER_PROMPT = 4
NUM_TRAINING_STEPS = 500
LR = 1e-6
KL_COEFF = 0.01
REWARD_MODE = "condition_aware"
LAMBDA_PENALTY = 0.3
MU_PENALTY = 0.15

# Quick test: few examples, no HF push (flip to True to test)
QUICK_TEST = False
QUICK_TEST_MAX_EXAMPLES = 10
QUICK_TEST_MAX_STEPS = 5


@app.function(
    image=image,
    gpu="A100",
    timeout=86400,
    secrets=[modal.Secret.from_dotenv()],
)
def run_grpo_train():
    import sys
    sys.path.insert(0, "/root/repo")

    from training.grpo_train import run_grpo, GRPOConfig
    from training.reward import RewardConfig

    hf_token = os.environ.get("HF_TOKEN")

    cfg = GRPOConfig(
        model_name=MODEL_ID,
        data_path=None,
        hf_dataset=HF_DATASET,
        split="train",
        rl_subset_split="rl_train",
        batch_size=BATCH_SIZE,
        num_samples_per_prompt=NUM_SAMPLES_PER_PROMPT,
        num_training_steps=NUM_TRAINING_STEPS,
        lr=LR,
        kl_coeff=KL_COEFF,
        device="cuda",
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
        max_examples = 250
        max_steps = None
        hf_repo_id = HF_REPO_ID if hf_token else None

    print(f"Running GRPO: model={MODEL_ID}, dataset={HF_DATASET}, steps={cfg.num_training_steps if max_steps is None else max_steps}")
    metrics = run_grpo(
        cfg,
        reward_cfg,
        output_dir=OUTPUT_DIR,
        hf_repo_id=hf_repo_id,
        max_examples=max_examples,
        max_steps=max_steps,
        debug_print_samples=QUICK_TEST,
        dry_run=False,
    )
    print(f"Done. Metrics: {metrics}")
    return metrics


@app.local_entrypoint()
def main():
    metrics = run_grpo_train.remote()
    print(metrics)
