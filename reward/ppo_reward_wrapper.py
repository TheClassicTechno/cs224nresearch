

import asyncio
import logging
from typing import List, Optional

import torch

from .reward_fn import RewardConfig, RewardFunction

logger = logging.getLogger(__name__)


class PPORewardWrapper:
    """Wraps RewardFunction for use inside a TRL PPO training loop.

   
    """

    def __init__(
        self,
        reward_config: Optional[RewardConfig] = None,
        tokenizer=None,
    ):
        self.config = reward_config or RewardConfig()
        self.reward_fn = RewardFunction(config=self.config)
        self.tokenizer = tokenizer

        self._step: int = 0
        self._history: List[dict] = []  # per-step stats for logging / debugging



    def compute_rewards(
        self,
        queries: List[torch.Tensor],
        responses: List[torch.Tensor],
        metadata: List[dict],
        tokenizer=None,
    ) -> List[torch.Tensor]:
        """Compute scalar rewards for one PPO step.

      
        """
        tok = tokenizer or self.tokenizer
        assert tok is not None, (
            "PPORewardWrapper requires a tokenizer for decoding. "
            "Pass it to __init__ or to compute_rewards()."
        )
        assert len(queries) == len(responses) == len(metadata), (
            f"Batch size mismatch: queries={len(queries)}, "
            f"responses={len(responses)}, metadata={len(metadata)}"
        )

       
        decoded_responses = [
            tok.decode(r, skip_special_tokens=True).strip()
            for r in responses
        ]

        # example dicts
        examples = [
            {
                "question":      meta["question"],
                "response":      resp,
                "gold_answer":   meta["gold_answer"],
                "prompt_type":   meta.get("prompt_type", "neutral"),
                "stated_belief": meta.get("stated_belief", ""),
            }
            for meta, resp in zip(metadata, decoded_responses)
        ]

        #  Compute rewards 
        scored = self.reward_fn.compute_batch(examples, eval_mode=False)

     
        self._step += 1
        self._log_step(scored, metadata)

        return [
            torch.tensor(s["reward"], dtype=torch.float32)
            for s in scored
        ]

    #  Logging 

    def _log_step(self, scored: List[dict], metadata: List[dict]) -> None:
        """Log and store per-step reward statistics."""
        rewards    = [s["reward"]    for s in scored]
        r_factuals = [s["r_factual"] for s in scored]
        r_sycs     = [s["r_syc"]     for s in scored]

        mean_r  = sum(rewards)    / len(rewards)
        mean_rf = sum(r_factuals) / len(r_factuals)
        mean_rs = sum(r_sycs)     / len(r_sycs)

        # Count prompt types for per-step breakdown
        counts = {"neutral": 0, "correct_belief": 0, "misconception": 0}
        for meta in metadata:
            pt = meta.get("prompt_type", "neutral")
            counts[pt] = counts.get(pt, 0) + 1

        step_stats = {
            "step": self._step,
            "mean_reward": mean_r,
            "mean_r_factual": mean_rf,
            "mean_r_syc": mean_rs,
            "n_neutral": counts["neutral"],
            "n_correct_belief": counts["correct_belief"],
            "n_misconception": counts["misconception"],
        }
        self._history.append(step_stats)

        logger.info(
            "[Step %4d] reward=%.4f  r_factual=%.4f  r_syc=%.4f  "
            "(neu=%d  corr=%d  misc=%d)",
            self._step, mean_r, mean_rf, mean_rs,
            counts["neutral"], counts["correct_belief"], counts["misconception"],
        )
        # Also print so it shows even without logging config
        print(
            f"[Step {self._step:4d}] "
            f"reward={mean_r:.4f}  r_factual={mean_rf:.4f}  r_syc={mean_rs:.4f}  "
            f"(neu={counts['neutral']} corr={counts['correct_belief']} "
            f"misc={counts['misconception']})"
        )

    def get_reward_history(self) -> List[dict]:
        """Return list of per-step reward stats (for plotting learning curves)."""
        return self._history

 

    @classmethod
    def from_config_dict(
        cls,
        config: dict,
        tokenizer=None,
    ) -> "PPORewardWrapper":
        """Construct from a plain dict (e.g., loaded from YAML/JSON config).

        Expected keys (all optional):
          mode, gamma, n_samples_train, n_samples_eval, judge_model, cache_dir
        """
        rc = RewardConfig(
            mode=config.get("mode", "factual_only"),
            gamma=config.get("gamma", 0.3),
            n_samples_train=config.get("n_samples_train", 1),
            n_samples_eval=config.get("n_samples_eval", 3),
            judge_model=config.get("judge_model", "gpt-4o-mini"),
            cache_dir=config.get("cache_dir", ".judge_cache"),
        )
        return cls(reward_config=rc, tokenizer=tokenizer)
