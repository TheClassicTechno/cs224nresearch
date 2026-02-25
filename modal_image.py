import modal

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "wandb",
        # Add any other dependencies you need
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .copy_local_dir("reward", "/root/reward")
    .copy_local_dir("training", "/root/training")
    .copy_local_dir("evaluation", "/root/evaluation")
    .copy_local_file("requirements.txt", "/root/requirements.txt")
)
