# cs224nresearch

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

4. Copy `.env.example` to `.env` and fill in your API key:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env`:
   ```
   GEMINI_API_KEY=your_key_here
   ```


## Project Initial Structure

cs224nresearch/
├── dataset/
│   ├── create_syc_dataset.py   # sycophancy conversion (existing)
│   └── dataset_utils.py        # shared loading/preprocessing
├── training/
│   ├── grpo_train.py           # GRPO training loop
│   ├── ppo_train.py            # PPO training loop
│   └── reward.py               # reward model / reward functions
├── evaluation/
│   └── eval.py                 # measure sycophancy rate, response quality
├── configs/
│   ├── grpo_config.yaml
│   └── ppo_config.yaml
└── requirements.txt