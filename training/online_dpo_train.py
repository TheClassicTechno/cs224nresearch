"""
Online DPO trainer using gold answers as chosen responses and live policy
samples as rejected responses when a judge-based gate says the sample is bad.
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

try:
    from .response_scorer import score_sampled_response
except ImportError:
    from response_scorer import score_sampled_response

VALID_CONDITIONS = {"neutral", "misconception", "correct_belief"}
JudgeFn = Callable[[str, str, str, str, str], Dict[str, float]]


def load_online_dpo_data(data_path: str, rl_split: str = "rl_train") -> Dataset:
    if data_path.startswith("hf:"):
        _, ds_name, split = data_path.split(":", 2)
        print(f"Loading Hugging Face dataset: {ds_name}, split: {split}")
        ds = load_dataset(ds_name, split=split)
        cols = ds.column_names

        if "prompt" in cols and "response" in cols:
            pass
        elif "new_question" in cols and "answer" in cols:
            ds = ds.rename_columns({"new_question": "prompt", "answer": "response"})
        elif "question" in cols and "answer" in cols:
            ds = ds.rename_columns({"question": "prompt", "answer": "response"})
        else:
            raise ValueError(
                "Dataset must have prompt/response or new_question/answer or "
                f"question/answer. Found: {cols}"
            )

        if "split" in ds.column_names:
            ds = ds.filter(lambda ex: ex.get("split") == rl_split)
        if "prompt_condition" in ds.column_names:
            ds = ds.filter(lambda ex: ex.get("prompt_condition") in VALID_CONDITIONS)
        else:
            raise ValueError("Dataset must contain 'prompt_condition'.")
        return ds

    if data_path.endswith(".json"):
        import json

        with open(data_path) as f:
            data = json.load(f)
        ds = Dataset.from_list(data)
    elif data_path.endswith(".jsonl"):
        import json

        data = []
        with open(data_path) as f:
            for line in f:
                data.append(json.loads(line))
        ds = Dataset.from_list(data)
    else:
        raise ValueError("data_path must be 'hf:dataset:split', .json, or .jsonl")

    cols = ds.column_names
    if "new_question" in cols and "answer" in cols and "prompt" not in cols:
        ds = ds.rename_columns({"new_question": "prompt", "answer": "response"})
    elif "question" in cols and "answer" in cols and "prompt" not in cols:
        ds = ds.rename_columns({"question": "prompt", "answer": "response"})

    if "split" in ds.column_names:
        ds = ds.filter(lambda ex: ex.get("split") == rl_split)
    if "prompt_condition" in ds.column_names:
        ds = ds.filter(lambda ex: ex.get("prompt_condition") in VALID_CONDITIONS)
    else:
        raise ValueError("Local dataset must contain 'prompt_condition'.")

    return ds


@dataclass
class OnlineDPOConfig:
    model_name: str = "Qwen/Qwen3.5-2B"
    data_path: Optional[str] = None
    hf_dataset: str = "mli5/medquad-sycophancy"
    split: str = "train"
    rl_subset_split: str = "rl_train"
    batch_size: int = 2
    max_new_tokens: int = 256
    max_length: int = 1024
    lr: float = 1e-6
    num_training_steps: int = 400
    warmup_steps: int = 40
    beta: float = 0.1
    min_factual_score_for_skip: float = 3.0
    use_endorse_gate: bool = True
    use_overcorr_gate: bool = True
    log_interval: int = 10
    device: str = "cuda"
    temperature: float = 1.0
    top_p: float = 0.95
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"


def _get_dataset(cfg: OnlineDPOConfig) -> Dataset:
    if cfg.data_path:
        return load_online_dpo_data(cfg.data_path, rl_split=cfg.rl_subset_split)
    return load_online_dpo_data(f"hf:{cfg.hf_dataset}:{cfg.split}", rl_split=cfg.rl_subset_split)


def _safe_list(batch_value: Any, fallback_len: int, default_value: str = "") -> List[Any]:
    if batch_value is None:
        return [default_value] * fallback_len
    if isinstance(batch_value, list):
        out = batch_value
    elif hasattr(batch_value, "tolist"):
        out = batch_value.tolist()
    else:
        out = [default_value] * fallback_len
    while len(out) < fallback_len:
        out.append(default_value)
    return out


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _to_int01(x: Any, default: int = 0) -> int:
    try:
        return 1 if int(x) != 0 else 0
    except Exception:
        return default


def should_create_dpo_pair(
    *,
    prompt_type: str,
    factual_score: float,
    endorse_flag: int,
    overcorr_flag: int,
    min_factual_score_for_skip: float = 3.0,
    use_endorse_gate: bool = True,
    use_overcorr_gate: bool = True,
) -> bool:
    prompt_type = (prompt_type or "").strip().lower()
    factual_failure = _to_float(factual_score, default=0.0) < float(min_factual_score_for_skip)
    endorse_failure = use_endorse_gate and prompt_type == "misconception" and _to_int01(endorse_flag) == 1
    overcorr_failure = (
        use_overcorr_gate and prompt_type == "correct_belief" and _to_int01(overcorr_flag) == 1
    )

    if prompt_type not in VALID_CONDITIONS:
        return False
    if prompt_type == "neutral":
        return factual_failure
    return factual_failure or endorse_failure or overcorr_failure


def _preview_text(text: str, max_chars: int = 400) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _format_prompt(tokenizer: AutoTokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _tokenize_completion(
    tokenizer: AutoTokenizer,
    completion_text: str,
    max_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    token_ids = tokenizer(
        completion_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    )["input_ids"]
    return torch.tensor(token_ids, dtype=torch.long, device=device)


def _completion_logprob_from_logits(
    logits: torch.Tensor,
    full_ids: torch.Tensor,
    prompt_len: int,
    response_len: int,
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
    labels = full_ids[:, 1:]
    token_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    start = prompt_len - 1
    end = start + response_len
    return token_lp[:, start:end].sum(dim=1).squeeze()


def _sequence_logprob(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    full_ids = torch.cat([prompt_ids, response_ids], dim=0).unsqueeze(0).to(device)
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    if full_ids.shape[1] < 2:
        full_ids = torch.cat([full_ids, full_ids.new_full((1, 1), pad)], dim=1)
    attention_mask = torch.ones_like(full_ids, device=device)

    with torch.set_grad_enabled(model.training):
        outputs = model(input_ids=full_ids, attention_mask=attention_mask)
    return _completion_logprob_from_logits(outputs.logits, full_ids, prompt_ids.shape[0], response_ids.shape[0])


def _dpo_loss(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    policy_margin = chosen_logps - rejected_logps
    reference_margin = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_margin - reference_margin)
    return -torch.nn.functional.logsigmoid(logits).mean(), logits


def _aggregate_condition_stats(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for cond in VALID_CONDITIONS:
        subset = [record for record in records if record["prompt_type"] == cond]
        if not subset:
            continue
        n = len(subset)
        updated = sum(1 for record in subset if record["accepted_for_dpo"])
        stats[cond] = {
            "count": float(n),
            "accepted_count": float(updated),
            "accepted_rate": updated / n,
            "mean_factual": sum(record["factual_score"] for record in subset) / n,
            "endorse_rate": sum(record["endorse_flag"] for record in subset) / n,
            "overcorr_rate": sum(record["overcorr_flag"] for record in subset) / n,
        }
    return stats


def run_online_dpo(
    cfg: OnlineDPOConfig,
    output_dir: str,
    hf_repo_id: Optional[str] = None,
    max_examples: Optional[int] = None,
    sample_randomly: bool = False,
    sample_seed: int = 42,
    max_steps: Optional[int] = None,
    debug_print_samples: bool = False,
    dry_run: bool = False,
    wandb_project: Optional[str] = "online-dpo-sycophancy",
    wandb_run_name: Optional[str] = None,
    judge_fn: Optional[JudgeFn] = None,
) -> Dict[str, Any]:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    import wandb

    use_wandb = os.environ.get("WANDB_API_KEY") is not None
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        mode="online" if use_wandb else "disabled",
        config={
            "model_name": cfg.model_name,
            "batch_size": cfg.batch_size,
            "max_new_tokens": cfg.max_new_tokens,
            "max_length": cfg.max_length,
            "lr": cfg.lr,
            "num_training_steps": cfg.num_training_steps,
            "warmup_steps": cfg.warmup_steps,
            "beta": cfg.beta,
            "min_factual_score_for_skip": cfg.min_factual_score_for_skip,
            "use_endorse_gate": cfg.use_endorse_gate,
            "use_overcorr_gate": cfg.use_overcorr_gate,
            "max_examples": max_examples,
            "sample_randomly": sample_randomly,
            "sample_seed": sample_seed,
            "max_steps": max_steps,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "lora_target_modules": cfg.lora_target_modules,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "objective": "online_dpo_gold_vs_bad_sample",
        },
    )

    print(f"Loading tokenizer and models: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    policy.to(device)
    policy.config.use_cache = False

    ref_policy = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    ref_policy.to(device)
    ref_policy.eval()
    for param in ref_policy.parameters():
        param.requires_grad_(False)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    policy = get_peft_model(policy, lora_config)
    policy.enable_input_require_grads()
    policy.gradient_checkpointing_enable()
    policy.train()
    policy.print_trainable_parameters()

    ds = _get_dataset(cfg)
    if len(ds) == 0:
        raise ValueError("No training examples found after filtering for the requested split and conditions.")
    if max_examples is not None:
        sample_size = min(max_examples, len(ds))
        if sample_randomly:
            ds = ds.shuffle(seed=sample_seed).select(range(sample_size))
        else:
            ds = ds.select(range(sample_size))
    print(f"Loaded {len(ds)} training examples after filtering.")

    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=cfg.lr)
    target_steps = max_steps if max_steps is not None else cfg.num_training_steps
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=target_steps,
    )

    global_step = 0
    batches_since_update = 0
    steps_per_epoch = max(1, math.ceil(len(ds) / cfg.batch_size))
    approx_epochs = target_steps / steps_per_epoch
    if max_examples is not None:
        sampling_desc = (
            f"random subset of {len(ds)} examples (seed={sample_seed})"
            if sample_randomly
            else f"first {len(ds)} examples"
        )
        print(f"Training on {sampling_desc}.")
    print(
        "Target optimizer steps="
        f"{target_steps} (~{approx_epochs:.2f} passes over the selected dataset, "
        f"{steps_per_epoch} steps/pass)."
    )

    pbar = tqdm(total=target_steps, desc="Online-DPO", unit="step")
    stop_reason = "completed"
    while global_step < target_steps:
        for batch in dl:
            if global_step >= target_steps:
                break

            questions = batch["prompt"] if isinstance(batch["prompt"], list) else batch["prompt"].tolist()
            prompt_types = (
                batch["prompt_condition"]
                if isinstance(batch["prompt_condition"], list)
                else batch["prompt_condition"].tolist()
            )
            gold_answers = _safe_list(batch.get("response"), len(questions), default_value="")
            stated_beliefs = _safe_list(batch.get("stated_belief"), len(questions), default_value="")

            chosen_logps: List[torch.Tensor] = []
            rejected_logps: List[torch.Tensor] = []
            ref_chosen_logps: List[torch.Tensor] = []
            ref_rejected_logps: List[torch.Tensor] = []
            judge_records: List[Dict[str, Any]] = []

            for idx, question in enumerate(questions):
                prompt_type = prompt_types[idx]
                gold_answer = gold_answers[idx]
                stated_belief = stated_beliefs[idx]

                if prompt_type not in VALID_CONDITIONS or not str(gold_answer).strip():
                    continue

                prompt_text = _format_prompt(tokenizer, question)
                prompt_ids = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=cfg.max_length,
                )["input_ids"].squeeze(0).to(device)
                prompt_attention_mask = torch.ones_like(prompt_ids, device=device).unsqueeze(0)

                policy.eval()
                with torch.no_grad():
                    generated = policy.generate(
                        prompt_ids.unsqueeze(0),
                        attention_mask=prompt_attention_mask,
                        max_new_tokens=cfg.max_new_tokens,
                        do_sample=True,
                        top_p=cfg.top_p,
                        temperature=cfg.temperature,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                policy.train()

                rejected_ids = generated[0, prompt_ids.shape[0] :]
                if rejected_ids.numel() == 0:
                    continue

                response_text = tokenizer.decode(rejected_ids, skip_special_tokens=True).strip()
                scores = score_sampled_response(
                    prompt=question,
                    response=response_text,
                    prompt_condition=prompt_type,
                    gold_answer=gold_answer,
                    stated_belief=stated_belief,
                    judge_fn=judge_fn,
                )
                factual_score = _to_float(scores.get("factual_score", 0.0), default=0.0)
                endorse_flag = _to_int01(scores.get("endorse_flag", 0), default=0)
                overcorr_flag = _to_int01(scores.get("overcorr_flag", 0), default=0)
                accepted_for_dpo = should_create_dpo_pair(
                    prompt_type=prompt_type,
                    factual_score=factual_score,
                    endorse_flag=endorse_flag,
                    overcorr_flag=overcorr_flag,
                    min_factual_score_for_skip=cfg.min_factual_score_for_skip,
                    use_endorse_gate=cfg.use_endorse_gate,
                    use_overcorr_gate=cfg.use_overcorr_gate,
                )

                judge_records.append(
                    {
                        "prompt_type": prompt_type,
                        "factual_score": factual_score,
                        "endorse_flag": endorse_flag,
                        "overcorr_flag": overcorr_flag,
                        "accepted_for_dpo": accepted_for_dpo,
                    }
                )

                if debug_print_samples and global_step < 3 and idx == 0:
                    print("\n=== Online DPO sample preview ===")
                    print(f"step={global_step + 1} prompt_type={prompt_type}")
                    print(f"prompt: {_preview_text(question)}")
                    print(f"sampled_response: {_preview_text(response_text)}")
                    print(f"gold_answer: {_preview_text(gold_answer)}")
                    print(
                        "scores: "
                        f"factual={factual_score:.2f} endorse={endorse_flag} "
                        f"overcorr={overcorr_flag} accepted={accepted_for_dpo}"
                    )
                    print("=================================\n")

                if not accepted_for_dpo:
                    continue

                remaining_tokens = cfg.max_length - prompt_ids.shape[0]
                if remaining_tokens <= 0:
                    continue
                max_completion_tokens = min(cfg.max_new_tokens, remaining_tokens)
                chosen_ids = _tokenize_completion(tokenizer, gold_answer, max_completion_tokens, device)
                rejected_ids = rejected_ids[:max_completion_tokens].to(device)
                if chosen_ids.numel() == 0 or rejected_ids.numel() == 0:
                    continue

                chosen_logps.append(_sequence_logprob(policy, tokenizer, prompt_ids, chosen_ids, device))
                rejected_logps.append(_sequence_logprob(policy, tokenizer, prompt_ids, rejected_ids, device))
                with torch.no_grad():
                    ref_chosen_logps.append(
                        _sequence_logprob(ref_policy, tokenizer, prompt_ids, chosen_ids, device)
                    )
                    ref_rejected_logps.append(
                        _sequence_logprob(ref_policy, tokenizer, prompt_ids, rejected_ids, device)
                    )

            if not chosen_logps:
                batches_since_update += 1
                if batches_since_update >= len(dl):
                    stop_reason = "no_valid_pairs"
                    print("Stopping early after a full pass without any valid DPO pairs.")
                    break
                print("Warning: no valid DPO pairs in batch; skipping optimizer step.")
                continue

            batches_since_update = 0
            chosen_logps_tensor = torch.stack(chosen_logps)
            rejected_logps_tensor = torch.stack(rejected_logps)
            ref_chosen_tensor = torch.stack(ref_chosen_logps)
            ref_rejected_tensor = torch.stack(ref_rejected_logps)

            loss, dpo_logits = _dpo_loss(
                chosen_logps_tensor,
                rejected_logps_tensor,
                ref_chosen_tensor,
                ref_rejected_tensor,
                beta=cfg.beta,
            )
            policy_margin = (chosen_logps_tensor - rejected_logps_tensor).mean().item()
            reference_margin = (ref_chosen_tensor - ref_rejected_tensor).mean().item()
            mean_dpo_logit = dpo_logits.mean().item()

            cond_stats = _aggregate_condition_stats(judge_records)
            valid_pairs = len(chosen_logps)
            skip_rate = 1.0 - (valid_pairs / max(len(judge_records), 1))

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                pairs=valid_pairs,
                skip=f"{skip_rate:.2f}",
                margin=f"{policy_margin:.3f}",
            )

            log_dict = {
                "loss": loss.item(),
                "valid_pairs": valid_pairs,
                "judged_examples": len(judge_records),
                "skip_rate": skip_rate,
                "policy_margin": policy_margin,
                "reference_margin": reference_margin,
                "mean_dpo_logit": mean_dpo_logit,
                "lr": lr_scheduler.get_last_lr()[0],
            }
            for cond in ["neutral", "misconception", "correct_belief"]:
                if cond in cond_stats:
                    stats = cond_stats[cond]
                    log_dict[f"{cond}/count"] = stats["count"]
                    log_dict[f"{cond}/accepted_count"] = stats["accepted_count"]
                    log_dict[f"{cond}/accepted_rate"] = stats["accepted_rate"]
                    log_dict[f"{cond}/mean_factual"] = stats["mean_factual"]
                    log_dict[f"{cond}/endorse_rate"] = stats["endorse_rate"]
                    log_dict[f"{cond}/overcorr_rate"] = stats["overcorr_rate"]
            wandb.log(log_dict, step=global_step + 1)

            if dry_run:
                pbar.close()
                print("Dry run complete; skipping backward/update.")
                wandb.finish()
                return {"dry_run": True, "valid_pairs": valid_pairs}

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            global_step += 1
            pbar.update(1)

        if stop_reason != "completed":
            break

    pbar.close()
    os.makedirs(output_dir, exist_ok=True)
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA adapter to {output_dir}")

    wandb.finish()

    metrics = {
        "output_dir": output_dir,
        "global_step": min(global_step, target_steps),
        "stop_reason": stop_reason,
    }
    hf_token = os.environ.get("HF_TOKEN")
    if hf_repo_id and hf_token:
        try:
            from huggingface_hub import HfApi, login

            login(token=hf_token)
            api = HfApi()
            api.create_repo(repo_id=hf_repo_id, exist_ok=True)
            api.upload_folder(
                folder_path=output_dir,
                repo_id=hf_repo_id,
                repo_type="model",
                path_in_repo=".",
            )
            print(f"Pushed to https://huggingface.co/{hf_repo_id}")
            metrics["hf_repo_id"] = hf_repo_id
        except Exception as exc:
            raise RuntimeError(f"HF Hub push failed: {exc}") from exc
    return metrics


def run_online_dpo_training() -> None:
    parser = argparse.ArgumentParser(description="Online DPO trainer with judge-gated gold-vs-sample pairs.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--hf_dataset", type=str, default="mli5/medquad-sycophancy")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--rl_subset_split", type=str, default="rl_train")
    parser.add_argument("--output_dir", type=str, default="online_dpo_qwen_ckpt")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_training_steps", type=int, default=400)
    parser.add_argument("--warmup_steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--min_factual_score_for_skip", type=float, default=3.0)
    parser.add_argument("--disable_endorse_gate", action="store_true")
    parser.add_argument("--disable_overcorr_gate", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="all-linear")
    parser.add_argument("--hf_repo_id", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--sample_randomly", action="store_true")
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--debug_print_samples", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="online-dpo-sycophancy")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--judge_model", type=str, default=None)
    args = parser.parse_args()

    if args.judge_model:
        os.environ["GRPO_JUDGE_MODEL"] = args.judge_model

    cfg = OnlineDPOConfig(
        model_name=args.model_name,
        data_path=args.data_path,
        hf_dataset=args.hf_dataset,
        split=args.split,
        rl_subset_split=args.rl_subset_split,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        lr=args.lr,
        num_training_steps=args.num_training_steps,
        warmup_steps=args.warmup_steps,
        beta=args.beta,
        min_factual_score_for_skip=args.min_factual_score_for_skip,
        use_endorse_gate=not args.disable_endorse_gate,
        use_overcorr_gate=not args.disable_overcorr_gate,
        device=args.device,
        temperature=args.temperature,
        top_p=args.top_p,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )
    metrics = run_online_dpo(
        cfg,
        output_dir=args.output_dir,
        hf_repo_id=args.hf_repo_id,
        max_examples=args.max_examples,
        sample_randomly=args.sample_randomly,
        sample_seed=args.sample_seed,
        max_steps=args.max_steps,
        debug_print_samples=args.debug_print_samples,
        dry_run=args.dry_run,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    print(metrics)


if __name__ == "__main__":
    run_online_dpo_training()
