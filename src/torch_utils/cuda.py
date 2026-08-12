import torch

def obtain_cuda_device(selected_gpu=None):
    if selected_gpu is not None:
        if torch.cuda.is_available() and selected_gpu < torch.cuda.device_count():
            device = torch.device(f"cuda:{selected_gpu}")
            print(f"Using GPU {selected_gpu}")
        else:
            print(f"Selected GPU {selected_gpu} is not available, falling back to CPU.")
            device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("GPU available")
        else:
            device = torch.device("cpu")
            print("GPU not available, using CPU")
    return device
