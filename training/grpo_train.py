"""
GRPO-style trainer: for each prompt sample K responses, score each with
response_scorer, compute rewards via compute_reward, then use within-group
standardized advantages for a KL-regularized policy-gradient update.
"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from .reward import RewardConfig, compute_reward
from .response_scorer import score_sampled_response, validate_openai_key

VALID_CONDITIONS = {"neutral", "misconception", "correct_belief"}

# Save a local checkpoint every N steps; push to HF every M steps.
CHECKPOINT_EVERY_N_STEPS = 100
HF_PUSH_EVERY_N_STEPS = 250


def load_grpo_data(data_path: str, rl_split: str = "rl_train") -> Dataset:
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
                f"Dataset must have prompt/response or new_question/answer or "
                f"question/answer. Found: {cols}"
            )

        cols = ds.column_names
        if "split" in cols:
            ds = ds.filter(lambda ex: ex.get("split") == rl_split)
        if "prompt_condition" in cols:
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

    cols = ds.column_names
    if "split" in cols:
        ds = ds.filter(lambda ex: ex.get("split") == rl_split)
    if "prompt_condition" in cols:
        ds = ds.filter(lambda ex: ex.get("prompt_condition") in VALID_CONDITIONS)
    else:
        raise ValueError("Local dataset must contain 'prompt_condition'.")

    return ds


@dataclass
class GRPOConfig:
    model_name: str = "technojules/qwen3.5-2b-sft-medquad"
    data_path: Optional[str] = None
    hf_dataset: str = "mli5/medquad-sycophancy"
    split: str = "train"
    rl_subset_split: str = "rl_train"
    batch_size: int = 2
    num_samples_per_prompt: int = 4
    max_new_tokens: int = 128
    max_length: int = 1024
    lr: float = 1e-6
    num_training_steps: int = 1000
    warmup_steps: int = 100
    kl_coeff: float = 0.01
    log_interval: int = 10
    device: str = "cuda"
    # LoRA settings (applied to policy only; ref_policy stays frozen base model)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


def _get_dataset(cfg: GRPOConfig) -> Dataset:
    if cfg.data_path:
        return load_grpo_data(cfg.data_path, rl_split=cfg.rl_subset_split)
    data_path = f"hf:{cfg.hf_dataset}:{cfg.split}"
    return load_grpo_data(data_path, rl_split=cfg.rl_subset_split)


def _sequence_logprobs(
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

    with torch.set_grad_enabled(model.training):
        out = model(input_ids=full_ids)

    logits = out.logits[:, :-1]
    labels = full_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    prompt_len = prompt_ids.shape[0]
    response_lp = token_lp[:, prompt_len - 1 : prompt_len - 1 + response_ids.shape[0]]
    return response_lp.sum().squeeze()


def _safe_list(batch_value, fallback_len: int, default_value="") -> List[Any]:
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


def _aggregate_condition_stats(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for cond in VALID_CONDITIONS:
        subset = [r for r in records if r["prompt_type"] == cond]
        if not subset:
            continue
        n = len(subset)
        stats[cond] = {
            "count": float(n),
            "mean_reward": sum(r["reward"] for r in subset) / n,
            "mean_factual": sum(r["factual_score_norm"] for r in subset) / n,
            "mean_endorse": sum(r["endorse_flag"] for r in subset) / n,
            "mean_overcorr": sum(r["overcorr_flag"] for r in subset) / n,
        }
    return stats


def _save_checkpoint(
    policy,
    tokenizer,
    output_dir: str,
    step: int,
    hf_repo_id: Optional[str],
    hf_token: Optional[str],
) -> None:
    """Save LoRA adapter locally and optionally push to HF Hub."""
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    policy.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"[checkpoint] Saved to {ckpt_dir}")

    if hf_repo_id and hf_token:
        try:
            from huggingface_hub import HfApi, login
            login(token=hf_token)
            api = HfApi()
            api.create_repo(repo_id=hf_repo_id, exist_ok=True)
            api.upload_folder(
                folder_path=ckpt_dir,
                repo_id=hf_repo_id,
                repo_type="model",
                path_in_repo=f"checkpoint-{step}",
            )
            print(f"[checkpoint] Pushed checkpoint-{step} to https://huggingface.co/{hf_repo_id}")
        except Exception as e:
            print(f"[checkpoint] WARNING: HF push failed at step {step}: {e}")


def run_grpo(
    cfg: GRPOConfig,
    reward_cfg: RewardConfig,
    output_dir: str,
    hf_repo_id: Optional[str] = None,
    max_examples: Optional[int] = None,
    max_steps: Optional[int] = None,
    debug_print_samples: bool = False,
    dry_run: bool = False,
    wandb_project: Optional[str] = "grpo-sycophancy",
    wandb_run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run GRPO training from config (used by CLI and Modal). Returns final metrics dict."""

    # ── 0. Validate API keys before loading the model ────────────────────────
    validate_openai_key()

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    hf_token = os.environ.get("HF_TOKEN")
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Initialize wandb ──────────────────────────────────────────────────
    import wandb
    wandb_run = wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            "model_name": cfg.model_name,
            "batch_size": cfg.batch_size,
            "num_samples_per_prompt": cfg.num_samples_per_prompt,
            "max_new_tokens": cfg.max_new_tokens,
            "max_length": cfg.max_length,
            "lr": cfg.lr,
            "num_training_steps": cfg.num_training_steps,
            "warmup_steps": cfg.warmup_steps,
            "kl_coeff": cfg.kl_coeff,
            "reward_mode": reward_cfg.mode,
            "lambda_penalty": reward_cfg.lambda_penalty,
            "mu_penalty": reward_cfg.mu_penalty,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "max_examples": max_examples,
            "max_steps": max_steps,
        },
    )

    try:
        # ── 2. Load tokenizer ────────────────────────────────────────────────
        print(f"Loading tokenizer and model: {cfg.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ── 3. Load reference policy (frozen base model, bfloat16) ──────────
        ref_policy = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        ref_policy.eval()
        for p in ref_policy.parameters():
            p.requires_grad_(False)
        print("Reference policy loaded (frozen).")

        # ── 4. Load trainable policy with LoRA (bfloat16) ───────────────────
        # LoRA on policy only: reduces optimizer state from ~24 GB to ~100 MB.
        from peft import LoraConfig, get_peft_model, TaskType, PeftModel
        policy_base = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        # If the SFT model was saved with a LoRA adapter, we must load+merge it before
        # applying GRPO LoRA. AutoModelForCausalLM.from_pretrained may not return a
        # PeftModel subclass even when adapter_config.json is present — so we try
        # PeftModel.from_pretrained explicitly, then fall back gracefully.
        try:
            adapter_policy = PeftModel.from_pretrained(policy_base, cfg.model_name, is_trainable=False)
            policy_base = adapter_policy.merge_and_unload()
            print("SFT LoRA adapter merged into base weights for GRPO policy.")
        except Exception as e:
            print(f"No PEFT adapter to merge (or already merged): {e}. Using model as-is.")
        # Also strip any leftover peft_config attributes so get_peft_model won't warn.
        for _obj in (policy_base, policy_base.config):
            if hasattr(_obj, "peft_config"):
                try:
                    delattr(_obj, "peft_config")
                except Exception:
                    pass
        policy_base.config.use_cache = False
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules="all-linear",  # Qwen3.5-2B DeltaNet requires all-linear
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        policy = get_peft_model(policy_base, lora_config)
        policy.enable_input_require_grads()
        policy.gradient_checkpointing_enable()
        policy.train()
        policy.print_trainable_parameters()
        print("Policy loaded with LoRA adapter.")

        # ── 5. Load dataset ──────────────────────────────────────────────────
        ds = _get_dataset(cfg)
        if max_examples is not None:
            ds = ds.select(range(min(max_examples, len(ds))))
        print(f"Loaded {len(ds)} training examples after filtering.")

        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

        # Only LoRA parameters need optimizer — saves ~24 GB vs full AdamW
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, policy.parameters()), lr=cfg.lr
        )
        target_steps = max_steps if max_steps is not None else cfg.num_training_steps
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=cfg.warmup_steps,
            num_training_steps=target_steps,
        )

        global_step = 0

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

                all_logprobs: List[torch.Tensor] = []
                all_ref_logprobs: List[torch.Tensor] = []
                all_rewards: List[float] = []
                reward_records: List[Dict[str, Any]] = []
                entropies_this_batch: List[float] = []

                for i in range(len(questions)):
                    q = questions[i]
                    ptype = prompt_types[i]
                    gold = gold_answers[i]
                    stated = stated_beliefs[i]

                    if ptype not in VALID_CONDITIONS:
                        continue

                    messages = [{"role": "user", "content": q}]
                    prompt_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    prompt_enc = tokenizer(
                        prompt_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=cfg.max_length,
                    )
                    prompt_ids = prompt_enc["input_ids"].squeeze(0).to(device)

                    with torch.no_grad():
                        gen_out = policy.generate(
                            prompt_ids.unsqueeze(0).expand(cfg.num_samples_per_prompt, -1),
                            max_new_tokens=cfg.max_new_tokens,
                            do_sample=True,
                            top_p=0.95,
                            temperature=1.2,  # higher temp → more diverse samples → non-zero advantages
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        )

                    response_ids_list = [
                        gen_out[j, prompt_ids.shape[0]:]
                        for j in range(cfg.num_samples_per_prompt)
                    ]

                    group_rewards: List[float] = []
                    group_lps: List[torch.Tensor] = []
                    group_ref_lps: List[torch.Tensor] = []

                    for k in range(cfg.num_samples_per_prompt):
                        resp_ids = response_ids_list[k]
                        response_text = tokenizer.decode(resp_ids, skip_special_tokens=True).strip()

                        # score_sampled_response raises on API errors (no silent fallback)
                        scores = score_sampled_response(
                            prompt=q,
                            response=response_text,
                            prompt_condition=ptype,
                            gold_answer=gold,
                            stated_belief=stated,
                        )

                        factual_score = scores.get("factual_score", 0.0)
                        endorse_flag = _to_int01(scores.get("endorse_flag", 0), default=0)
                        overcorr_flag = _to_int01(scores.get("overcorr_flag", 0), default=0)

                        r_dict = compute_reward(
                            prompt_type=ptype,
                            factual_score=factual_score,
                            endorse_flag=endorse_flag,
                            overcorr_flag=overcorr_flag,
                            config=reward_cfg,
                        )

                        reward_value = _to_float(r_dict.get("reward", 0.0), default=0.0)
                        factual_score_norm = r_dict.get("factual_score_norm", r_dict.get("r_factual", None))
                        if factual_score_norm is None:
                            fs = _to_float(factual_score, default=0.0)
                            factual_score_norm = (fs - 1.0) / 3.0 if 1.0 <= fs <= 4.0 else fs
                        factual_score_norm = max(0.0, min(1.0, _to_float(factual_score_norm, 0.0)))

                        if debug_print_samples and global_step < 2:
                            print("\n" + "=" * 80)
                            print(f"PROMPT TYPE: {ptype}")
                            print(f"PROMPT: {q}")
                            print(f"SAMPLE {k}: {response_text}")
                            print(f"SCORES: {scores}")
                            print(f"REWARD_DICT: {r_dict}")

                        group_rewards.append(reward_value)

                        reward_records.append(
                            {
                                "prompt_type": ptype,
                                "reward": reward_value,
                                "factual_score_norm": factual_score_norm,
                                "endorse_flag": endorse_flag,
                                "overcorr_flag": overcorr_flag,
                            }
                        )

                        lp = _sequence_logprobs(policy, tokenizer, prompt_ids, resp_ids, device)
                        with torch.no_grad():
                            ref_lp = _sequence_logprobs(ref_policy, tokenizer, prompt_ids, resp_ids, device)

                        group_lps.append(lp)
                        group_ref_lps.append(ref_lp)

                    all_logprobs.extend(group_lps)
                    all_ref_logprobs.extend(group_ref_lps)
                    all_rewards.extend(group_rewards)

                    with torch.no_grad():
                        full = torch.cat([prompt_ids, response_ids_list[0]], dim=0).unsqueeze(0)
                        out = policy(input_ids=full)
                        log_p = torch.log_softmax(out.logits[:, :-1], dim=-1)
                        probs = torch.exp(log_p)
                        ent = -(probs * log_p).sum(dim=-1).mean().item()
                        entropies_this_batch.append(ent)

                if not all_rewards:
                    print("Warning: no valid rewards in batch; skipping step.")
                    continue

                logprobs = torch.stack(all_logprobs)
                ref_logprobs = torch.stack(all_ref_logprobs)
                rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32, device=device)

                rewards_reshaped = rewards_tensor.view(-1, cfg.num_samples_per_prompt)
                group_mean = rewards_reshaped.mean(dim=1, keepdim=True)
                group_std = rewards_reshaped.std(dim=1, keepdim=True, unbiased=False)
                advantages = ((rewards_reshaped - group_mean) / (group_std + 1e-8)).view(-1)

                policy_loss = -(logprobs * advantages.detach()).mean()
                kl_proxy = (logprobs - ref_logprobs.detach()).mean()
                loss = policy_loss + cfg.kl_coeff * kl_proxy

                mean_entropy = sum(entropies_this_batch) / max(len(entropies_this_batch), 1)
                mean_reward = rewards_tensor.mean().item()
                cond_stats = _aggregate_condition_stats(reward_records)

                print(
                    f"[step {global_step + 1}] "
                    f"loss={loss.item():.4f} "
                    f"policy_loss={policy_loss.item():.4f} "
                    f"kl_proxy={kl_proxy.item():.4f} "
                    f"mean_reward={mean_reward:.4f} "
                    f"entropy={mean_entropy:.4f}"
                )
                for cond in ["neutral", "misconception", "correct_belief"]:
                    if cond in cond_stats:
                        s = cond_stats[cond]
                        print(
                            f"  [{cond}] "
                            f"n={int(s['count'])} "
                            f"mean_reward={s['mean_reward']:.4f} "
                            f"mean_factual={s['mean_factual']:.4f} "
                            f"endorse_rate={s['mean_endorse']:.4f} "
                            f"overcorr_rate={s['mean_overcorr']:.4f}"
                        )

                log_dict = {
                    "loss": loss.item(),
                    "policy_loss": policy_loss.item(),
                    "kl_proxy": kl_proxy.item(),
                    "mean_reward": mean_reward,
                    "entropy": mean_entropy,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                for cond in ["neutral", "misconception", "correct_belief"]:
                    if cond in cond_stats:
                        s = cond_stats[cond]
                        log_dict[f"{cond}/mean_reward"] = s["mean_reward"]
                        log_dict[f"{cond}/mean_factual"] = s["mean_factual"]
                        log_dict[f"{cond}/endorse_rate"] = s["mean_endorse"]
                        log_dict[f"{cond}/overcorr_rate"] = s["mean_overcorr"]
                wandb.log(log_dict, step=global_step + 1)

                if dry_run:
                    print("Dry run complete; skipping backward/update.")
                    return {"dry_run": True}

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                global_step += 1

                # ── Intermediate checkpoint ──────────────────────────────────
                if global_step % CHECKPOINT_EVERY_N_STEPS == 0:
                    push_to_hf = (global_step % HF_PUSH_EVERY_N_STEPS == 0)
                    _save_checkpoint(
                        policy, tokenizer, output_dir, global_step,
                        hf_repo_id=hf_repo_id if push_to_hf else None,
                        hf_token=hf_token,
                    )

        # ── Final save: merge LoRA into weights and push full model ──────────
        print("Merging LoRA adapter into base weights...")
        merged_policy = policy.merge_and_unload()
        merged_policy.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Final merged model saved to {output_dir}")

        metrics = {"output_dir": output_dir, "global_step": global_step}

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
                print(f"Pushed final model to https://huggingface.co/{hf_repo_id}")
                metrics["hf_repo_id"] = hf_repo_id
            except Exception as e:
                raise RuntimeError(f"HF Hub push failed: {e}") from e

        return metrics

    finally:
        # Always finish wandb run — even on exception or timeout
        wandb.finish()


def run_grpo_training() -> None:
    parser = argparse.ArgumentParser(description="GRPO-style trainer.")
    parser.add_argument("--model_name", type=str, default="technojules/qwen3.5-2b-sft-medquad")
    parser.add_argument("--hf_dataset", type=str, default="mli5/medquad-sycophancy")
    parser.add_argument("--output_dir", type=str, default="grpo_qwen_ckpt")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_samples_per_prompt", type=int, default=4)
    parser.add_argument("--num_training_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--kl_coeff", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--reward_mode", type=str, default="condition_aware",
                        choices=["factual_only", "condition_aware"])
    parser.add_argument("--lambda_penalty", type=float, default=0.3)
    parser.add_argument("--mu_penalty", type=float, default=0.15)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--hf_repo_id", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--debug_print_samples", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="grpo-sycophancy")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    cfg = GRPOConfig(
        model_name=args.model_name,
        data_path=args.data_path,
        hf_dataset=args.hf_dataset,
        batch_size=args.batch_size,
        num_samples_per_prompt=args.num_samples_per_prompt,
        num_training_steps=args.num_training_steps,
        lr=args.lr,
        kl_coeff=args.kl_coeff,
        device=args.device,
    )
    reward_cfg = RewardConfig(
        mode=args.reward_mode,
        lambda_penalty=args.lambda_penalty,
        mu_penalty=args.mu_penalty,
    )
    run_grpo(
        cfg,
        reward_cfg,
        output_dir=args.output_dir,
        hf_repo_id=args.hf_repo_id,
        max_examples=args.max_examples,
        max_steps=args.max_steps,
        debug_print_samples=args.debug_print_samples,
        dry_run=args.dry_run,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    run_grpo_training()
