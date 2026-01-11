import torch
import numpy as np

def num_para(model):
    return sum(p.numel() for p in model.parameters())
