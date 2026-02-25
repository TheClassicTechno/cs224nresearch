import modal

# Modal tokens are injected at runtime via secrets
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "datasets", "peft", "pyyaml", "wandb", "accelerate", "huggingface_hub")
    .add_local_dir("training", "/root/training")
    .add_local_file("sft_config.yaml", "/root/sft_config.yaml")
)

app = modal.App("cs224nresearch-sft")

@app.function(
    image=image,
    # A10G or L4 24GB: use batch_size=2 + gradient_accumulation_steps=2 
    gpu="A10G",
    secrets=[modal.Secret.from_dotenv()],
    timeout=60*60*4  # 4 hours
)
def run_sft_on_modal():
    import subprocess
    import sys
    #  see progress in Modal dashboard and logs
    result = subprocess.run(
        ["python", "/root/training/sft_qwen.py", "--config", "/root/sft_config.yaml"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    if result.returncode != 0:
        raise RuntimeError("SFT script exited with code %d" % result.returncode)
    return "SFT on Modal GPU complete!"

@app.local_entrypoint()
def main():
    run_sft_on_modal.remote()