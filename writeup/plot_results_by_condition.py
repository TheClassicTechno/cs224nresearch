"""
Plot truthfulness (accuracy) by prompt condition: Pretrained vs SFT.

Expects JSON files with "accuracy_by_condition": {"neutral": float, "correct_belief": float, "misconception": float}.
Produced by modal_training/baseline.py and modal_training/sft_eval.py (after the per-condition addition).

Usage:
    python writeup/plot_results_by_condition.py baseline.json sft.json -o fig_results_by_condition.pdf
    python writeup/plot_results_by_condition.py sft.json -o sft_only.pdf   # SFT only
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


CONDITION_LABELS = ["Neutral", "Correct belief", "Misconception"]
CONDITION_KEYS = ["neutral", "correct_belief", "misconception"]


def load_accuracies(path: str) -> dict[str, float]:
    with open(path) as f:
        data = json.load(f)
    acc = data.get("accuracy_by_condition") or data
    return {k: acc[k] for k in CONDITION_KEYS if acc.get(k) is not None}


def main():
    parser = argparse.ArgumentParser(description="Plot accuracy by condition (grouped bar)")
    parser.add_argument("baseline_json", nargs="?", help="Path to baseline eval results JSON")
    parser.add_argument("sft_json", help="Path to SFT eval results JSON")
    parser.add_argument("-o", "--output", default="results_by_condition.pdf", help="Output figure path")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    sft = load_accuracies(args.sft_json)
    baseline = load_accuracies(args.baseline_json) if args.baseline_json else None

    x = np.arange(len(CONDITION_KEYS))
    width = 0.35 if baseline else 0.5

    fig, ax = plt.subplots(figsize=(5, 4))
    if baseline:
        if set(baseline.keys()) != set(sft.keys()):
            raise ValueError("Condition keys differ between baseline and SFT results")
        ax.bar(x - width / 2, [baseline[k] for k in CONDITION_KEYS], width, label="Pretrained (zero-shot)", color="C0")
        ax.bar(x + width / 2, [sft[k] for k in CONDITION_KEYS], width, label="SFT", color="C1")
    else:
        ax.bar(x, [sft[k] for k in CONDITION_KEYS], width, label="SFT", color="C1")

    ax.set_ylabel("Factual accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_LABELS)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
