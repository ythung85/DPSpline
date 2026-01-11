

import torch
import numpy as np

def diag_mat_weights(dimp, type='first'):
    """
    Return a torch Tensor D matrix:
    - 'first': (dimp-1, dimp) with [-1, 1] on diag and next diag
    - 'second': (dimp-2, dimp) second difference [-1,2,-1]
    """
    if type == 'first':
        dg = np.zeros((dimp-1, dimp))
        for i in range(dimp-1):
            dg[i,i] = -1
            dg[i,i+1] = 1
    elif type == 'second':
        dg = np.zeros((dimp-2, dimp))
        for i in range(dimp-2):
            dg[i,i] = -1
            dg[i,i+1] = 2
            dg[i,i+2] = -1
    else:
        raise ValueError("Unknown type")
    return torch.tensor(dg, dtype=torch.float32)

def spline_penalty_loss(model, lambda_vals, device):
	"""
	Penalty = sum( lambda_l * || D @ W_l ||^2 )
	"""
	penalty = 0.0
	extracted_model = model.Spline_block.model
	l_idx = 0
	
	try:
		for name, module in extracted_model.named_modules(): 
			if name == f"block_{l_idx}": 
				block = getattr(extracted_model, f'block_{l_idx}')
				W = block.block.BSL.control_p
			
				lam = lambda_vals[l_idx] if lambda_vals is not None else 0
				
				D = diag_mat_weights(W.size()[2], type='second').to(device)
				penalty += lam * torch.norm(D @ W[0].T)
				l_idx += 1
			else:
				continue
	except:
	
		for l_idx, layer in enumerate(extracted_model):
			block = getattr(extracted_model, f'block_{l_idx}')
			
			W = block.block.BSL.control_p
			
			lam = lambda_vals[l_idx] if lambda_vals is not None else 0
			
			D = diag_mat_weights(W.size()[2], type='second').to(device)
			penalty += lam * torch.norm(D @ W[0].T)
	
	return penalty