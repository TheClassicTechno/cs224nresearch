from dataclasses import dataclass
from typing import Literal, Dict


RewardMode = Literal["factual_only", "condition_aware"]


@dataclass
class RewardConfig:
    """Configuration for scalar reward used in GRPO-style training."""
    mode: RewardMode = "condition_aware"
    lambda_penalty: float = 0.3   # misconception endorsement penalty weight
    mu_penalty: float = 0.15      # correct-belief over-correction penalty weight


def normalize_factual_score(raw_score: float) -> float:
    """Map a 1–4 factuality rubric to [0, 1].

    If input is already in [0, 1], return it unchanged.
    Otherwise clamp to [1, 4] and map linearly.
    """
    if 0.0 <= raw_score <= 1.0:
        return raw_score

    clamped = max(1.0, min(4.0, raw_score))
    return (clamped - 1.0) / 3.0


def compute_reward(
    *,
    prompt_type: str,
    factual_score: float,
    endorse_flag: int = 0,
    overcorr_flag: int = 0,
    config: RewardConfig,
) -> Dict[str, float]:
    """Compute scalar reward R(x, y) and return reward components.

    Args:
        prompt_type: One of {"neutral", "misconception", "correct_belief"}.
        factual_score: Raw factuality score (1–4 rubric or already in [0, 1]).
        endorse_flag: 1 if the answer endorses a misconception.
        overcorr_flag: 1 if the answer over-corrects a correct belief.
        config: RewardConfig with reward mode and penalty weights.

    Returns:
        Dict with:
          - reward: scalar reward
          - factual_score_norm: normalized factual score in [0, 1]
          - endorse_flag: 0.0 or 1.0
          - overcorr_flag: 0.0 or 1.0
    """
    ptype = prompt_type.lower().strip()
    factual_score_norm = normalize_factual_score(float(factual_score))
    endorse = float(1 if endorse_flag else 0)
    overcorr = float(1 if overcorr_flag else 0)

    if config.mode == "factual_only":
        reward = factual_score_norm
    else:
        if ptype == "neutral":
            reward = factual_score_norm
        elif ptype == "misconception":
            reward = (
                (1.0 - config.lambda_penalty) * factual_score_norm
                - config.lambda_penalty * endorse
            )
        elif ptype == "correct_belief":
            reward = (
                (1.0 - config.mu_penalty) * factual_score_norm
                - config.mu_penalty * overcorr
            )
        else:
            # Safe fallback: treat unknown prompt types as neutral factual QA
            reward = factual_score_norm

    return {
        "reward": reward,
        "factual_score_norm": factual_score_norm,
        "endorse_flag": endorse,
        "overcorr_flag": overcorr,
    }