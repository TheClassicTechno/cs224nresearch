"""
SFT training: Qwen/Qwen2-7B on the training split of mli5/medquad-sycophancy.

Trains with LoRA (rank 8, alpha 16, target_modules='all-linear') for 2 epochs
and pushes the merged checkpoint to HF Hub.

Run:
    modal run modal_training/7bmodelSFT.py
    modal run --detach modal_training/7bmodelSFT.py   # keep running if terminal disconnects
"""

import os
import modal

app = modal.App("qwen7b-sft-train")

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
        "tiktoken",
        "einops",
        "transformers-stream-generator"
    )
    .env({"PYTHONUNBUFFERED": "1"})
)

HF_DATASET = "mli5/medquad-sycophancy"
MODEL_ID = "Qwen/Qwen2-7B"
HF_REPO_ID = "jillianchang/qwen2-7b-sft-medquad"
OUTPUT_DIR = "/tmp/sft_qwen2_7b_ckpt"

BATCH_SIZE = 2
GRAD_ACCUM = 2
EPOCHS = 2
LR = 2e-5
MAX_LENGTH = 1024
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


def preprocess(example, tokenizer, max_length):
    prompt = example["prompt"]
    response = example["response"]

    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_ids)

    enc = tokenizer(
        prompt + response,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    labels = enc["input_ids"].copy()
    for i in range(len(labels)):
        if i < prompt_len or labels[i] == tokenizer.pad_token_id:
            labels[i] = -100
    enc["labels"] = labels
    return enc


@app.function(
    image=image,
    gpu="A100",        # 40 GB VRAM — for Qwen2-7B with LoRA
    timeout=86400,     # 24h
    secrets=[modal.Secret.from_dotenv()],
)
def run_sft_train():
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        default_data_collator,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    hf_token = os.environ.get("HF_TOKEN")

    # ── 1. Load training split ───────────────────────────────────────────────
    print(f"Loading dataset: {HF_DATASET}")
    ds = load_dataset(HF_DATASET, split="train", token=hf_token)
    valid_conditions = {"neutral", "misconception", "correct_belief"}
    train_ds = ds.filter(
        lambda ex: ex.get("split") == "rl_train" and ex.get("prompt_condition") in valid_conditions
    )
    # Rename columns to prompt/response
    train_ds = train_ds.rename_columns({"new_question": "prompt", "answer": "response"})
    print(f"Training examples: {len(train_ds)}")

    # ── 2. Load tokenizer and model ──────────────────────────────────────────
    print(f"Loading tokenizer and model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )
    model.config.use_cache = False
    print("Model loaded.")

    # ── 3. LoRA ──────────────────────────────────────────────────────────────
    # "all-linear" covers all linear layers for Qwen2-7B.
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules="all-linear",
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    # ── 4. Tokenize dataset ──────────────────────────────────────────────────
    print("Tokenizing dataset...")
    tokenized = train_ds.map(
        lambda ex: preprocess(ex, tokenizer, MAX_LENGTH),
        remove_columns=train_ds.column_names,
    )

    # ── 5. Train ─────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        bf16=True,
        save_strategy="epoch",
        logging_steps=10,
        report_to=["none"],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=default_data_collator,
    )

    print("Starting SFT training...")
    train_result = trainer.train()
    print(f"Training complete. Metrics: {train_result.metrics}")

    # ── 6. Save ──────────────────────────────────────────────────────────────
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # ── 7. Push to HF Hub ────────────────────────────────────────────────────
    if hf_token:
        from huggingface_hub import HfApi, login
        print(f"Pushing to HF Hub: {HF_REPO_ID}")
        login(token=hf_token)
        api = HfApi()
        api.create_repo(repo_id=HF_REPO_ID, exist_ok=True)
        api.upload_folder(
            folder_path=OUTPUT_DIR,
            repo_id=HF_REPO_ID,
            repo_type="model",
            path_in_repo=".",
        )
        print(f"Model pushed to https://huggingface.co/{HF_REPO_ID}")
    else:
        print("[Warning] HF_TOKEN not set — skipping HF Hub push.")

    return train_result.metrics


@app.local_entrypoint()
def main():
    metrics = run_sft_train.remote()
    print(metrics)
