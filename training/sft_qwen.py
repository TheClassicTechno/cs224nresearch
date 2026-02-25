
#SFT baseline for Qwen1.5-1.8B using Hugging Face Transformers and LoRA



import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import yaml

def load_sft_data(data_path):
    #Load SFT data from JSON/JSONL file or Hugging Face dataset
    if data_path.startswith("hf:"):
    
        _, ds_name, split = data_path.split(":", 2)
        from datasets import load_dataset
        print(f"Loading Hugging Face dataset: {ds_name}, split: {split}")
        ds = load_dataset(ds_name, split=split)
        #  mli5/medquad-sycophancy uses new_question, answer
        cols = ds.column_names
        if "prompt" in cols and "response" in cols:
            filtered = ds.filter(lambda ex: ex.get("split") == "rl_train" and ex.get("prompt_condition") in {"neutral", "misconception", "correct-belief"})
            return filtered
        if "new_question" in cols and "answer" in cols:
            ds = ds.rename_columns({"new_question": "prompt", "answer": "response"})
            filtered = ds.filter(lambda ex: ex.get("split") == "rl_train" and ex.get("prompt_condition") in {"neutral", "misconception", "correct-belief"})
            return filtered
        if "question" in cols and "answer" in cols:
            ds = ds.rename_columns({"question": "prompt", "answer": "response"})
            filtered = ds.filter(lambda ex: ex.get("split") == "rl_train" and ex.get("prompt_condition") in {"neutral", "misconception", "correct-belief"})
            return filtered
        raise ValueError(
            f"Dataset {ds_name} must have 'prompt' and 'response' columns (or 'new_question'/'answer', or 'question'/'answer'). "
            f"Found: {cols}"
        )
    elif data_path.endswith(".json"):
        import json
        with open(data_path) as f:
            data = json.load(f)
        return Dataset.from_list(data)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path) as f:
            for line in f:
                data.append(json.loads(line))
        return Dataset.from_list(data)
    else:
        raise ValueError("Unsupported data format: must be .json, .jsonl, or 'hf:dataset:split'")

def preprocess(example, tokenizer, max_length=1024):
    # Concatenate prompt and response 
    text = example["prompt"] + example["response"]
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    enc["labels"] = enc["input_ids"].copy()
    return enc

def main():
    use_wandb = os.environ.get("WANDB_API_KEY") is not None

    parser = argparse.ArgumentParser(description="SFT baseline for Qwen3-1.7B")
    parser.add_argument("--config", type=str, default=None, help="YAML config file (overrides CLI args)")
    parser.add_argument("--model_name", default=None, help="Model name or path")
    parser.add_argument("--data_path", default=None, help="Path to SFT data (.json or .jsonl)")
    parser.add_argument("--output_dir", default=None, help="Where to save the fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA (requires peft)")
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # Load config from YAML 
    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    def get_cfg(key, default=None):
        v = getattr(args, key, None)
        if v is not None and v != "None":
            return v
        return config.get(key, default)

    model_name = get_cfg("model_name", "Qwen/Qwen3-1.7B")
    data_path = get_cfg("data_path")
    if not data_path:
        raise ValueError("data_path is required. Set it in your YAML config or pass --data_path (e.g. hf:dataset_name:split or path to .json/.jsonl).")
    output_dir = get_cfg("output_dir", "sft_qwen3_ckpt")
    batch_size = int(get_cfg("batch_size", 2))
    gradient_accumulation_steps = int(get_cfg("gradient_accumulation_steps", 2))
    epochs = int(get_cfg("epochs", 2))
    lr = float(get_cfg("lr", 2e-5))
    max_length = int(get_cfg("max_length", 1024))
    use_lora = get_cfg("use_lora", False)
    lora_r = int(get_cfg("lora_r", 8))
    lora_alpha = int(get_cfg("lora_alpha", 16))
    lora_dropout = float(get_cfg("lora_dropout", 0.05))
    device = get_cfg("device", "cuda")

    hf_token = os.environ.get("HF_TOKEN")
    modal_token_id = os.environ.get("MODAL_TOKEN_ID")
    modal_token_secret = os.environ.get("MODAL_TOKEN_SECRET")
    if not (modal_token_id and modal_token_secret):
        print("[Warning] MODAL_TOKEN_ID or MODAL_TOKEN_SECRET not set in environment. Modal API access may fail.")
    else:
        print(f"[Info] Modal API credentials loaded: MODAL_TOKEN_ID={modal_token_id[:6]}... (hidden)")

    if use_wandb:
        import wandb
        wandb.init(
            project="cs224n-sft",
            name=f"SFT-{model_name}",
            config={
                "model_name": model_name,
                "data_path": data_path,
                "output_dir": output_dir,
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "max_length": max_length,
                "use_lora": use_lora,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
            }
        )

    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device)

    if use_lora and (str(use_lora).lower() == "true" or use_lora is True):
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("peft is required for LoRA. Install with: pip install peft")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        print("LoRA enabled.")

    # Reduce activation memory
    model.gradient_checkpointing_enable()

    print(f"Loading data from {data_path}")
    dataset = load_sft_data(data_path)
    # Remove all original columns so only tokenized fields (input_ids, attention_mask, labels) 
    dataset = dataset.map(
        lambda ex: preprocess(ex, tokenizer, max_length),
        remove_columns=dataset.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=lr,
        bf16=True,  # Use bf16 to match Qwen3 default dtype
        save_strategy="epoch",
        logging_steps=10,
        report_to=["wandb"] if use_wandb else ["tensorboard"],
        logging_dir=os.path.join(output_dir, "logs"),
        remove_unused_columns=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("Starting supervised fine-tuning...")
    train_result = trainer.train()
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    # Log training metrics to wandb
    if use_wandb:
        wandb.log(train_result.metrics)
        wandb.finish()
    # Save as .pth file 
    try:
        import torch
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.pth"))
        print(f"Saved PyTorch .pth weights to {os.path.join(output_dir, 'pytorch_model.pth')}")
    except Exception as e:
        print(f"[Warning] Could not save .pth weights: {e}")
    print(f"Training logs saved to {os.path.join(output_dir, 'logs')}")

    # push to Hugging Face Hub 
    repo_id = get_cfg("hf_repo_id", None)
    if hf_token and repo_id:
        try:
            from huggingface_hub import HfApi, login
            print(f"Logging in to Hugging Face Hub and pushing to {repo_id}...")
            login(token=hf_token)
            api = HfApi()
            api.create_repo(repo_id=repo_id, exist_ok=True)
            api.upload_folder(
                folder_path=output_dir,
                repo_id=repo_id,
                repo_type="model",
                path_in_repo="."
            )
            print(f"Model pushed to https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"[Warning] Could not push to Hugging Face Hub: {e}")

if __name__ == "__main__":
    main()
