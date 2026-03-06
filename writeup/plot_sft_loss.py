"""
Plot SFT training loss vs step from trainer_state.json (HF checkpoint).

Uses MODEL_ID (same as sft_eval.py). With no path, fetches trainer_state.json
from HF without downloading the full repo.

Usage:
    python writeup/plot_sft_loss.py -o sft_loss.pdf
    python writeup/plot_sft_loss.py checkpoint-900/trainer_state.json -o sft_loss.pdf
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# HF token from env (e.g. .env or HF_TOKEN=...)
HF_TOKEN = os.environ.get("HF_TOKEN")

# Same as modal_training/sft_eval.py
MODEL_ID = "technojules/qwen3-1.7b-sft-medquad"
CHECKPOINT = "checkpoint-900"


def load_trainer_state_local(path: str) -> Dict:
    path = os.path.abspath(path)
    if os.path.isdir(path):
        path = os.path.join(path, "trainer_state.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def load_trainer_state_hf(repo_id: str, checkpoint: str, token: Optional[str] = None) -> Dict:
    from huggingface_hub import hf_hub_download

    filename = f"{checkpoint}/trainer_state.json"
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model", token=token)
    with open(path, "r") as f:
        return json.load(f)


def extract_loss_steps(trainer_state: Dict) -> List[Dict]:
    """
    Returns a list of {"step": int, "loss": float, "epoch": float|None}
    """
    history = trainer_state.get("log_history", [])
    out = []
    for h in history:
        if "loss" not in h:
            continue
        if "step" not in h:
            # If step is missing, skip rather than inventing one.
            # (Inventing steps can make the plot misleading.)
            continue

        out.append(
            {
                "step": int(h["step"]),
                "loss": float(h["loss"]),
                "epoch": float(h["epoch"]) if "epoch" in h and h["epoch"] is not None else None,
            }
        )

    # Sort just in case logs are out of order
    out.sort(key=lambda x: x["step"])
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot SFT loss vs step")
    parser.add_argument("training_log", nargs="?", help="Local path to trainer_state.json or checkpoint dir; if omitted, fetches from HF (MODEL_ID)")
    parser.add_argument("-o", "--output", default="sft_loss.pdf", help="Output figure path")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--token", default=HF_TOKEN, help="HF token (default: HF_TOKEN from env)")
    args = parser.parse_args()

    if args.training_log:
        trainer_state = load_trainer_state_local(args.training_log)
    else:
        try:
            trainer_state = load_trainer_state_hf(MODEL_ID, CHECKPOINT, token=args.token)
        except Exception as e:
            err_msg = str(e).lower()
            if "401" in err_msg or "unauthorized" in err_msg or "gated" in err_msg or "authenticated" in err_msg:
                raise SystemExit(
                    f"Repo {MODEL_ID} is gated or private. Set HF_TOKEN in your env (or .env) or pass --token."
                ) from e
            raise

    points = extract_loss_steps(trainer_state)
    if not points:
        raise SystemExit("No loss entries with 'step' found in trainer_state.json.")

    steps = [p["step"] for p in points]
    losses = [p["loss"] for p in points]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(steps, losses, linewidth=1)
    ax.set_xlabel("Global step")
    ax.set_ylabel("Training loss")
    ax.set_title("SFT training loss")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()