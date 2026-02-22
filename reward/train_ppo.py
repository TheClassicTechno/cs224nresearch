

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .baseline_eval import compare_all_models
from .evaluate import run_evaluation
from .ppo_reward_wrapper import PPORewardWrapper
from .reward_fn import ABLATION_CONFIGS, RewardConfig

logger = logging.getLogger(__name__)





def load_rl_dataset(path: str, tokenizer, max_length: int = 512) -> Dataset:
    """Load RL prompt pool JSON and tokenize.

    Returns a HuggingFace Dataset with:
      input_ids, attention_mask (tokenized question)
      + passthrough metadata: question, gold_answer, prompt_type, stated_belief.
    """
    with open(path) as f:
        records = json.load(f)

    # Tokenize in-place so TRL can form batches
    def _tok(ex: dict) -> dict:
        enc = tokenizer(
            ex["question"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            # Metadata kept as strings — passed through to reward wrapper
            "question":       ex["question"],
            "gold_answer":    ex["gold_answer"],
            "prompt_type":    ex.get("prompt_type", "neutral"),
            "stated_belief":  ex.get("stated_belief", ""),
        }

    raw = Dataset.from_list(records)
    return raw.map(_tok, remove_columns=[])


def _collate_fn(batch: List[dict]) -> List[dict]:
    """TRL expects a list of dicts (not a padded tensor batch) for PPO."""
    return batch


# Single PPO training run 


def run_ppo_training(
    sft_checkpoint: str,
    rl_prompt_pool_path: str,
    output_dir: str,
    reward_config: Optional[RewardConfig] = None,
    # Hyperparameters (all from paper proposal / standard PPO defaults)
    num_epochs: int = 1,
    batch_size: int = 8,
    mini_batch_size: int = 4,
    ppo_epochs: int = 4,
    learning_rate: float = 1.5e-5,
    kl_coef: float = 0.2,
    max_new_tokens: int = 256,
    device: str = "cuda",
) -> str:
    """Run PPO post-training from an SFT checkpoint.

   
    """

    try:
        from trl import PPOConfig, PPOTrainer
    except ImportError as e:
        raise ImportError(
            "TRL is required for PPO training.  Install with: pip install trl"
        ) from e

    rc = reward_config or RewardConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PPO Training: {rc.label()}")
    print(f"  checkpoint  : {sft_checkpoint}")
    print(f"  output      : {output_dir}")
    print(f"  lr={learning_rate}  kl_coef={kl_coef}  batch={batch_size}")
    print(f"{'='*60}")


    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for generation

    model = AutoModelForCausalLM.from_pretrained(
        sft_checkpoint, torch_dtype=torch.float16
    ).to(device)

    ref_model = AutoModelForCausalLM.from_pretrained(
        sft_checkpoint, torch_dtype=torch.float16
    ).to(device)
    ref_model.eval()

    dataset = load_rl_dataset(rl_prompt_pool_path, tokenizer)

  
    ppo_config = PPOConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        ppo_epochs=ppo_epochs,
        kl_coef=kl_coef,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        gamma=1.0,
        lam=0.95,
        log_with=None,          # Set to "wandb" to enable W&B logging
        remove_unused_columns=False,
    )

   
    reward_wrapper = PPORewardWrapper(reward_config=rc, tokenizer=tokenizer)

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=_collate_fn,
    )

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # Training loop 
    for epoch in range(num_epochs):
        print(f"\n── Epoch {epoch + 1}/{num_epochs} ──")

        for batch in ppo_trainer.dataloader:
            # Build query tensors (list of 1-D LongTensors)
            query_tensors = [
                torch.tensor(b["input_ids"], dtype=torch.long).to(device)
                for b in batch
            ]

            # Build metadata dicts for the reward wrapper
            metadata = [
                {
                    "question":      b["question"],
                    "gold_answer":   b["gold_answer"],
                    "prompt_type":   b["prompt_type"],
                    "stated_belief": b.get("stated_belief", ""),
                }
                for b in batch
            ]

            # Generate responses
            response_tensors = ppo_trainer.generate(
                query_tensors,
                return_prompt=False,
                **gen_kwargs,
            )

            # Compute rewards
            
            rewards = reward_wrapper.compute_rewards(
                query_tensors, response_tensors, metadata
            )
            _check_reward_health(rewards, rc.label())

            # 
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)


    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Save reward history for learning-curve analysis
    history_path = output_path / "reward_history.json"
    with open(history_path, "w") as f:
        json.dump(reward_wrapper.get_reward_history(), f, indent=2)

    print(f"\n[train_ppo] Checkpoint saved → {output_path}")
    return str(output_path)


def _check_reward_health(rewards: List[torch.Tensor], label: str) -> None:
    """Verification step: warn if rewards are collapsed to a single value."""
    vals = [r.item() for r in rewards]
    if len(set(round(v, 3) for v in vals)) == 1:
        logger.warning(
            "[%s] Reward collapse detected: all rewards = %.4f.  "
            "Check judge API / cache.",
            label, vals[0],
        )
        print(
            f"[WARNING] Reward collapse in {label}: all rewards = {vals[0]:.4f}.  "
            "Check judge API connectivity and cache."
        )





def run_all_ablations(
    sft_checkpoint: str,
    rl_prompt_pool_path: str,
    eval_set_path: str,
    output_dir: str = "results",
    device: str = "cuda",
    skip_training: bool = False,
    judge_model: str = "gpt-4o-mini",
    **train_kwargs,
) -> None:
    """Train all four reward ablations and evaluate each on the eval set.

   
    """
    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    results_dir = output_dir / "metrics"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # SFT baseline evaluation first
    print("\n>>> [Ablation] Step 0 — SFT Baseline Evaluation")
    from .baseline_eval import run_sft_baseline
    run_sft_baseline(
        sft_checkpoint=sft_checkpoint,
        eval_set_path=eval_set_path,
        output_dir=str(results_dir),
        device=device,
        judge_model=judge_model,
    )

    #Each PPO ablation 
    for i, rc in enumerate(ABLATION_CONFIGS, start=1):
        name = rc.label()
        ckpt_path = checkpoints_dir / name
        result_path = results_dir / f"{name}_eval.json"

        print(f"\n>>> [Ablation] Step {i}/{len(ABLATION_CONFIGS)} — {name}")

        if not skip_training:
            run_ppo_training(
                sft_checkpoint=sft_checkpoint,
                rl_prompt_pool_path=rl_prompt_pool_path,
                output_dir=str(ckpt_path),
                reward_config=rc,
                device=device,
                **train_kwargs,
            )
        else:
            print(f"  (skip_training=True; using existing checkpoint at {ckpt_path})")

        if not ckpt_path.exists():
            print(f"  [WARNING] Checkpoint not found at {ckpt_path}; skipping eval.")
            continue

        run_evaluation(
            model_path=str(ckpt_path),
            eval_set_path=eval_set_path,
            output_path=str(result_path),
            reward_config=rc,
            device=device,
        )

  
    print("\n>>> [Ablation] Final Comparison")
    compare_all_models(str(results_dir))




def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PPO training and ablation runner.")
    sub = p.add_subparsers(dest="command")


    t = sub.add_parser("train", help="Single PPO training run")
    t.add_argument("--sft_checkpoint", required=True)
    t.add_argument("--rl_prompt_pool", required=True)
    t.add_argument("--output_dir", required=True)
    t.add_argument(
        "--reward_mode", default="factual_only",
        choices=["factual_only", "truth_weighted"],
    )
    t.add_argument("--gamma", type=float, default=0.3)
    t.add_argument("--num_epochs", type=int, default=1)
    t.add_argument("--batch_size", type=int, default=8)
    t.add_argument("--mini_batch_size", type=int, default=4)
    t.add_argument("--ppo_epochs", type=int, default=4)
    t.add_argument("--learning_rate", type=float, default=1.5e-5)
    t.add_argument("--kl_coef", type=float, default=0.2)
    t.add_argument("--max_new_tokens", type=int, default=256)
    t.add_argument("--device", default="cuda")
    t.add_argument("--judge_model", default="gpt-4o-mini")


    a = sub.add_parser("ablate", help="Run all four reward ablations")
    a.add_argument("--sft_checkpoint", required=True)
    a.add_argument("--rl_prompt_pool", required=True)
    a.add_argument("--eval_set", required=True)
    a.add_argument("--output_dir", default="results")
    a.add_argument("--device", default="cuda")
    a.add_argument("--skip_training", action="store_true",
                   help="Skip training; only evaluate existing checkpoints")
    a.add_argument("--judge_model", default="gpt-4o-mini")
    a.add_argument("--num_epochs", type=int, default=1)
    a.add_argument("--batch_size", type=int, default=8)
    a.add_argument("--learning_rate", type=float, default=1.5e-5)
    a.add_argument("--kl_coef", type=float, default=0.2)

    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train":
        rc = RewardConfig(mode=args.reward_mode, gamma=args.gamma,
                          judge_model=args.judge_model)
        run_ppo_training(
            sft_checkpoint=args.sft_checkpoint,
            rl_prompt_pool_path=args.rl_prompt_pool,
            output_dir=args.output_dir,
            reward_config=rc,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            mini_batch_size=args.mini_batch_size,
            ppo_epochs=args.ppo_epochs,
            learning_rate=args.learning_rate,
            kl_coef=args.kl_coef,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )

    elif args.command == "ablate":
        run_all_ablations(
            sft_checkpoint=args.sft_checkpoint,
            rl_prompt_pool_path=args.rl_prompt_pool,
            eval_set_path=args.eval_set,
            output_dir=args.output_dir,
            device=args.device,
            skip_training=args.skip_training,
            judge_model=args.judge_model,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            kl_coef=args.kl_coef,
        )

    else:
        parser.print_help()
