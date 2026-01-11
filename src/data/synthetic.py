import torch
import numpy as np

def sim_data(n, dim, Type, device='cpu', seed=None):
    """
    Return X (n x dim), y (n x 1) - torch tensors on specified device.
    """
    if seed is not None:
        torch.manual_seed(seed)
    if Type == 'A':
        X = torch.rand((n, 2))
        y = torch.exp(2*torch.sin(X[:,0]*0.5*torch.pi) + 0.5*torch.cos(X[:,1]*2.5*torch.pi))
        y = y.reshape(-1,1).float()
    elif Type == 'B':
        X = torch.rand((n, dim))
        y = torch.ones((n,))
        for d in range(dim):
            a = (d+1)/2
            y = y * ((torch.abs(4*X[:,d]-2)+a)/(1+a))
        y = y.reshape(-1,1).float()
    else:
        raise ValueError("Unknown Type")
    return X.to(device), y.to(device)