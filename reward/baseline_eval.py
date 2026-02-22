

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from .evaluate import run_evaluation
from .metrics import EvalMetrics, print_comparison_table, print_metrics_table
from .reward_fn import RewardConfig



def run_sft_baseline(
    sft_checkpoint: str,
    eval_set_path: str,
    output_dir: str = "results",
    device: str = "cuda",
    batch_size: int = 8,
    judge_model: str = "gpt-4o-mini",
) -> EvalMetrics:
    """Evaluate the SFT checkpoint and save results.

    Args:
        sft_checkpoint: Path to SFT model checkpoint.
        eval_set_path:  Path to the 500-example eval set JSON.
        output_dir:     Directory to write results JSON.
        device:         Inference device.
        batch_size:     Inference batch size.
        judge_model:    Judge LLM model ID.

    Returns:
        EvalMetrics for the SFT model.
    """
    output_path = str(Path(output_dir) / "sft_baseline_eval.json")

    print("\n" + "=" * 60)
    print("BASELINE EVALUATION: SFT Model (no PPO)")
    print("=" * 60)

    rc = RewardConfig(
        mode="factual_only",
        judge_model=judge_model,
        n_samples_train=1,
        n_samples_eval=3,
    )
    metrics = run_evaluation(
        model_path=sft_checkpoint,
        eval_set_path=eval_set_path,
        output_path=output_path,
        reward_config=rc,
        device=device,
        batch_size=batch_size,
    )

    print(f"\nBaseline results saved to: {output_path}")
    return metrics


# model compare

# Canonical display order for the comparison table
_MODEL_ORDER = [
    "sft_baseline_eval",
    "ppo_factual_only_eval",
    "ppo_gamma_0.1_eval",
    "ppo_gamma_0.3_eval",
    "ppo_gamma_0.5_eval",
]


def load_results_from_dir(results_dir: str) -> Dict[str, EvalMetrics]:
    """Load all *_eval.json files from results_dir; return {stem: EvalMetrics}."""
    results_dir = Path(results_dir)
    all_results: Dict[str, EvalMetrics] = {}

    found = sorted(results_dir.glob("*_eval.json"))
    if not found:
        print(f"[baseline_eval] No *_eval.json files found in {results_dir}")
        return all_results

    for result_file in found:
        with open(result_file) as f:
            data = json.load(f)

        m = data["metrics"]
        metrics = EvalMetrics(
            n_total=m.get("n_total", 0),
            n_neutral=m.get("n_neutral", 0),
            n_correct_belief=m.get("n_correct_belief", 0),
            n_misconception=m.get("n_misconception", 0),
            accuracy_overall=m.get("accuracy_overall", 0.0),
            accuracy_neutral=m.get("accuracy_neutral", 0.0),
            accuracy_correct_belief=m.get("accuracy_correct_belief", 0.0),
            accuracy_misconception=m.get("accuracy_misconception", 0.0),
            sycophancy_rate=m.get("sycophancy_rate", 0.0),
            contradiction_rate=m.get("contradiction_rate", 0.0),
        )
        all_results[result_file.stem] = metrics

    return all_results


def compare_all_models(results_dir: str) -> None:
    """Load all eval result files from a directory and print comparison table."""
    all_results = load_results_from_dir(results_dir)
    if not all_results:
        return

    # Sort by canonical order; unknown models go at the end
    def _sort_key(name: str) -> int:
        try:
            return _MODEL_ORDER.index(name)
        except ValueError:
            return len(_MODEL_ORDER)

    ordered = {k: all_results[k] for k in sorted(all_results, key=_sort_key)}
    print_comparison_table(ordered)

    # Also print individual detailed tables
    for name, metrics in ordered.items():
        print_metrics_table(name, metrics)





def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run SFT baseline evaluation and/or compare all ablation results."
    )
    sub = p.add_subparsers(dest="command")

    # baseline sub-command
    b = sub.add_parser("baseline", help="Evaluate SFT checkpoint")
    b.add_argument("--sft_checkpoint", required=True)
    b.add_argument("--eval_set", required=True)
    b.add_argument("--output_dir", default="results")
    b.add_argument("--device", default="cuda")
    b.add_argument("--batch_size", type=int, default=8)
    b.add_argument("--judge_model", default="gpt-4o-mini")

    # compare sub-command
    c = sub.add_parser("compare", help="Compare all models in a results directory")
    c.add_argument("--results_dir", required=True)

    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "baseline":
        run_sft_baseline(
            sft_checkpoint=args.sft_checkpoint,
            eval_set_path=args.eval_set,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
            judge_model=args.judge_model,
        )

    elif args.command == "compare":
        compare_all_models(args.results_dir)

    else:
        parser.print_help()
