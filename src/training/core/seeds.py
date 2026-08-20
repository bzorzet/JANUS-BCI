import random
import numpy as np
import torch

def set_seed(seed):
    """Sets the seed for model initialization while avoiding changes to DataLoader randomness."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
