"""
Online DPO + LoRA training on Modal.

This entrypoint trains against judge-gated gold-vs-sampled preference pairs and
saves the adapter checkpoint to /tmp before optionally pushing it to the HF Hub.
"""

import os

import modal

app = modal.App("qwen-online-dpo-train")

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
HF_REPO_ID = "mli5/qwen3.5-2b-online-dpo-lora"
OUTPUT_DIR = "/tmp/online_dpo_qwen_ckpt"

BATCH_SIZE = 2
MAX_NEW_TOKENS = 256
MAX_LENGTH = 1024
NUM_TRAINING_STEPS = 900
WARMUP_STEPS = 50
LR = 1e-6
BETA = 0.1
MIN_FACTUAL_SCORE_FOR_SKIP = 3.0
MAX_EXAMPLES = 1800
SAMPLE_RANDOMLY = False
SAMPLE_SEED = 42

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear"

JUDGE_MODEL = "gpt-5-nano-2025-08-07"
WANDB_PROJECT = "online-dpo-sycophancy"
WANDB_RUN_NAME = "qwen3.5-2b-online-dpo-lora"

QUICK_TEST = False
QUICK_TEST_MAX_EXAMPLES = 8
QUICK_TEST_MAX_STEPS = 2


@app.function(
    image=image,
    gpu="A100",
    timeout=86400,
    secrets=[modal.Secret.from_dotenv()],
)
def run_online_dpo_train():
    import sys

    sys.path.insert(0, "/root/repo")

    from training.online_dpo_train import OnlineDPOConfig, run_online_dpo

    hf_token = os.environ.get("HF_TOKEN")
    os.environ["GRPO_JUDGE_MODEL"] = JUDGE_MODEL

    cfg = OnlineDPOConfig(
        model_name=MODEL_ID,
        data_path=None,
        hf_dataset=HF_DATASET,
        split="train",
        rl_subset_split="rl_train",
        batch_size=BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
        max_length=MAX_LENGTH,
        lr=LR,
        num_training_steps=NUM_TRAINING_STEPS,
        warmup_steps=WARMUP_STEPS,
        beta=BETA,
        min_factual_score_for_skip=MIN_FACTUAL_SCORE_FOR_SKIP,
        use_endorse_gate=True,
        use_overcorr_gate=True,
        device="cuda",
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        lora_target_modules=LORA_TARGET_MODULES,
    )

    if QUICK_TEST:
        max_examples = QUICK_TEST_MAX_EXAMPLES
        max_steps = QUICK_TEST_MAX_STEPS
        hf_repo_id = None
        print(f"QUICK_TEST: max_examples={max_examples}, max_steps={max_steps}, no HF push")
    else:
        max_examples = MAX_EXAMPLES
        max_steps = None
        hf_repo_id = HF_REPO_ID if hf_token else None

    print(
        "Running Online DPO: "
        f"model={MODEL_ID}, dataset={HF_DATASET}, max_examples={max_examples}, "
        f"steps={max_steps if max_steps is not None else NUM_TRAINING_STEPS}"
    )
    metrics = run_online_dpo(
        cfg,
        output_dir=OUTPUT_DIR,
        hf_repo_id=hf_repo_id,
        max_examples=max_examples,
        sample_randomly=SAMPLE_RANDOMLY and not QUICK_TEST,
        sample_seed=SAMPLE_SEED,
        max_steps=max_steps,
        debug_print_samples=QUICK_TEST,
        dry_run=False,
        wandb_project=WANDB_PROJECT,
        wandb_run_name=WANDB_RUN_NAME,
    )
    print(f"Done. Metrics: {metrics}")
    return metrics


@app.local_entrypoint()
def main():
    metrics = run_online_dpo_train.remote()
    print(metrics)
