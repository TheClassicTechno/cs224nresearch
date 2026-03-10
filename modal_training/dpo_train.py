"""
DPO training on Modal (A10G).

Offline DPO: no online scoring/rollouts at train time — much cheaper than GRPO.
Preference pairs are constructed from the dataset (gold answer = chosen, synthetic rejected).

Run:
    modal run modal_training/dpo_train.py
    modal run --detach modal_training/dpo_train.py
"""

import os
import modal

app = modal.App("qwen-dpo-train")

_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, ".."))

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers>=4.51.0",
        "accelerate",
        "peft",
        "datasets",
        "tqdm",
        "huggingface_hub",
        "wandb",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir(_repo_root, remote_path="/root/repo")
)

HF_DATASET = "mli5/medquad-sycophancy"
MODEL_ID = "technojules/qwen3.5-2b-sft-medquad"
HF_REPO_ID = "mli5/qwen3.5-2b-dpo-medquad-sycophancy"
OUTPUT_DIR = "/tmp/dpo_qwen_ckpt"

BATCH_SIZE = 4
NUM_TRAINING_STEPS = 1000
LR = 5e-5
BETA = 0.1

# Quick test: flip to True to test
QUICK_TEST = False
QUICK_TEST_MAX_EXAMPLES = 10
QUICK_TEST_MAX_STEPS = 5


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    secrets=[modal.Secret.from_dotenv()],
)
def run_dpo_train():
    import sys
    sys.path.insert(0, "/root/repo")

    from training.dpo_train import run_dpo, DPOConfig

    hf_token = os.environ.get("HF_TOKEN")

    cfg = DPOConfig(
        model_name=MODEL_ID,
        hf_dataset=HF_DATASET,
        split="train",
        dpo_subset_split="rl_train",
        batch_size=BATCH_SIZE,
        num_training_steps=NUM_TRAINING_STEPS,
        lr=LR,
        beta=BETA,
        device="cuda",
    )

    if QUICK_TEST:
        max_examples = QUICK_TEST_MAX_EXAMPLES
        max_steps = QUICK_TEST_MAX_STEPS
        hf_repo_id = None
        print(f"QUICK_TEST: max_examples={max_examples}, max_steps={max_steps}, no HF push")
    else:
        max_examples = None
        max_steps = None
        hf_repo_id = HF_REPO_ID if hf_token else None

    print(f"Running DPO: model={MODEL_ID}, dataset={HF_DATASET}, steps={NUM_TRAINING_STEPS if max_steps is None else max_steps}")
    metrics = run_dpo(
        cfg,
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
    metrics = run_dpo_train.remote()
    print(metrics)
