import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    # Get CUDA version
    print(f"CUDA Version: {torch.version.cuda}")

    # Get GPU name
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Name: {gpu_name}")

    # Get number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")

    # Get current GPU index
    current_gpu = torch.cuda.current_device()
    print(f"Current GPU Index: {current_gpu}")
else:
    print("CUDA is not available. Using CPU.")
