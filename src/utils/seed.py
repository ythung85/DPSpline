# src/utils/seed.py
import torch
import numpy as np
import random

def set_seed(seed: int):
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
torch.cuda.manual_seed_all(seed)