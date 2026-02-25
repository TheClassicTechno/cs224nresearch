# cs224nresearch
Juli Huang, Michael Li, Jillian Chang
## Setup

1. Create the conda environment (first time only):
   ```bash
   conda create -n cs224nresearch python=3.11 -y
   ```

2. Activate the environment:
   ```bash
   conda activate cs224nresearch
   ```

3. Install dependencies (first time only, or when `requirements.txt` changes):
   ```bash
   pip install -r requirements.txt
   ```

4. Set up API keys in `.env` file:
   ```bash
   # Create .env file if it doesn't exist
   touch .env
   ```
   Then edit `.env`:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```


## Baseline Evaluation (Before Fine-tuning)

Run inference + LLM-as-judge scoring on a small subset to establish baseline:

```bash
python evaluation/eval.py \
  --model_path "path/to/your/sft/checkpoint" \
  --eval_set "path/to/eval_set.json" \
  --output "results/baseline_eval.json" \
  --subset_size 50 \
  --device cuda \
  --judge_model "gemini-1.5-flash"
```

This will:
1. Load your model and run inference on the eval set (or subset)
2. Score responses with LLM-as-judge (R_factual, R_syc, R_contra)
3. Compute metrics: sycophancy rate, contradiction rate, medical accuracy
4. Save results to JSON

**Quick test on 10 examples:**
```bash
python evaluation/eval.py \
  --model_path "your_model" \
  --eval_set "eval_set.json" \
  --output "results/test.json" \
  --subset_size 10
```

## Project Structure

cs224nresearch/
├── dataset/
│   ├── create_syc_dataset.py   # sycophancy conversion
│   └── dataset_utils.py        # shared loading/preprocessing
├── reward/
│   ├── judge.py                # LLM-as-judge scoring (R_factual, R_syc, R_contra)
│   ├── reward_fn.py            # Reward function with truth-weighted configs
│   ├── metrics.py              # Evaluation metrics computation
│   ├── evaluate.py             # Full evaluation pipeline
│   └── baseline_eval.py        # Baseline evaluation wrapper
├── training/
│   ├── grpo_train.py           # GRPO training loop
│   ├── ppo_train.py            # PPO training loop
│   └── reward.py               # reward model / reward functions
├── evaluation/
│   └── eval.py                 # Baseline eval: inference + scoring
├── configs/
│   ├── grpo_config.yaml
│   └── ppo_config.yaml
└── requirements.txt