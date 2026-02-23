import modal


image = modal.Image.debian_slim().pip_install("torch", "transformers")


@modal.function(image=image, gpu="A100")
def run_on_gpu():
    import torch
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name(0))

    return "GPU job complete!"


if __name__ == "__main__":
 
    run_on_gpu.remote()

