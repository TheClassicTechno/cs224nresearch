import math
import unittest

import torch

from training.online_dpo_train import (
    _completion_logprob_from_logits,
    _dpo_loss,
    should_create_dpo_pair,
)


class OnlineDPOTests(unittest.TestCase):
    def test_gate_for_misconception_accepts_endorsement(self):
        self.assertTrue(
            should_create_dpo_pair(
                prompt_type="misconception",
                factual_score=4.0,
                endorse_flag=1,
                overcorr_flag=0,
            )
        )

    def test_gate_for_correct_belief_accepts_overcorrection(self):
        self.assertTrue(
            should_create_dpo_pair(
                prompt_type="correct_belief",
                factual_score=4.0,
                endorse_flag=0,
                overcorr_flag=1,
            )
        )

    def test_gate_for_neutral_depends_on_factuality_only(self):
        self.assertFalse(
            should_create_dpo_pair(
                prompt_type="neutral",
                factual_score=4.0,
                endorse_flag=1,
                overcorr_flag=1,
            )
        )
        self.assertTrue(
            should_create_dpo_pair(
                prompt_type="neutral",
                factual_score=2.0,
                endorse_flag=0,
                overcorr_flag=0,
            )
        )

    def test_completion_logprob_masks_prompt_tokens(self):
        full_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        probs = torch.tensor(
            [
                [
                    [0.10, 0.70, 0.10, 0.05, 0.05],
                    [0.10, 0.10, 0.60, 0.10, 0.10],
                    [0.05, 0.05, 0.05, 0.80, 0.05],
                    [0.20, 0.20, 0.20, 0.20, 0.20],
                ]
            ],
            dtype=torch.float32,
        )
        logits = probs.log()
        logprob = _completion_logprob_from_logits(logits, full_ids, prompt_len=2, response_len=2)
        expected = math.log(0.60) + math.log(0.80)
        self.assertAlmostEqual(logprob.item(), expected, places=5)

    def test_dpo_loss_prefers_higher_policy_margin(self):
        low_loss, _ = _dpo_loss(
            chosen_logps=torch.tensor([5.0]),
            rejected_logps=torch.tensor([1.0]),
            ref_chosen_logps=torch.tensor([4.0]),
            ref_rejected_logps=torch.tensor([2.0]),
            beta=0.1,
        )
        high_loss, _ = _dpo_loss(
            chosen_logps=torch.tensor([1.0]),
            rejected_logps=torch.tensor([5.0]),
            ref_chosen_logps=torch.tensor([4.0]),
            ref_rejected_logps=torch.tensor([2.0]),
            beta=0.1,
        )
        self.assertLess(low_loss.item(), high_loss.item())


if __name__ == "__main__":
    unittest.main()
