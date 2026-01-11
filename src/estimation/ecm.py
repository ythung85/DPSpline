# src/estimation/ecm.py
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

'''
def augment_with_intercept(X, weights, intercept):

    nl, batch, _ = X.shape
    _, _, nm, nk = weights.shape
    device = X.device

    X_reshaped = X.view(nl, batch, nm, nk)
    intercept_expanded = intercept.view(nl, 1, nm, 1).expand(nl, batch, nm, 1)
    
    X_aug = torch.cat([X_reshaped, intercept_expanded], dim=-1)
    
    X_new = X_aug.view(nl, batch, nm * (nk + 1))
    ones = torch.ones((nl, 1, nm, 1), device=device, dtype=weights.dtype)
    
    weights_new = torch.cat([weights, ones], dim=-1)

    return X_new, weights_new

'''

def augment_with_intercept(X_list, weights_list, intercept_list):
    
    X_new_list = []
    weights_new_list = []

    for l, (x, w, b) in enumerate(zip(X_list, weights_list, intercept_list)):

        # x shape: [batch, nm*nk]
        # w shape: [1, nm, nk]
        batch = x.size(0)
        _, nm, nk = w.size()
        device = x.device
        
        # View: [batch, nm*nk] -> [batch, nm, nk]
        x_reshaped = x.view(batch, nm, nk)
        
        # b shape: [1, nm] -> [1, nm, 1] -> expand to [batch, nm, 1]
        b_expanded = b.view(1, nm, 1).expand(batch, nm, 1)
        
        # Shape: [batch, nm, nk+1]
        x_aug = torch.cat([x_reshaped, b_expanded], dim=-1)
        
        x_final = x_aug.view(batch, nm * (nk + 1))
        
        # Shape: [1, nm, 1]
        ones = torch.ones((1, nm, 1), device=device, dtype=w.dtype)
        
        # Shape: [1, nm, nk+1]
        w_final = torch.cat([w, ones], dim=-1)
        
        X_new_list.append(x_final)
        weights_new_list.append(w_final)

    return X_new_list, weights_new_list
    
def ECM(par, initial_xi=1, initial_sigma=1, initial_lambda=1e-4, device='cpu'):
    """
    Compute per-layer lambda estimates using the original ECM logic.
    par: dict with keys 'ebasic', 'basic', 'wbasic', 'bbasic'
    Returns ls_lambda tensor of length n_block
    """
    
    
    n_block = len(par['wbasic'])
    
    #n_block, _, num_neurons, _ = par['wbasic'].size()
    ls_lambda = torch.empty(n_block, device=device)

    B_aug, WB_aug = augment_with_intercept(par['ebasic'], par['wbasic'], par['bbasic'])
    num_knots = WB_aug[0].size(-1)
    
    for l in range(n_block):

        # Initialization
        lambdab = initial_lambda
        sigma = initial_sigma
        xi = initial_xi
        
        B = B_aug[l]       # expansion matrix (dependent on implementation)
        By = par['basic'][l]      # block outputs (n_sample, n_neurons)
        WB = WB_aug[l]     # (num_knots, num_neurons) or similar
        num_neurons = WB.size()[1]
        
        DB = diag_mat_weights(WB.size()[2]).to(device)
        size = By.size()[0]
        S = DB.T @ DB
        Cov_a = (xi**2) * torch.linalg.pinv(S)
        Cov_a = Cov_a.to(device)
        Cov_e = (torch.eye(size, device=device) * sigma).to(device)

        block_y = torch.reshape(By, (-1, 1))
        # attempt to shape flatB consistent with original code:
        
        flatB = B.view(num_neurons, num_knots, size)
        
        sqr_xi = 0.0
        sqr_sig = 0.0

        for i in range(num_neurons):
            # defensive: build the per-neuron terms carefully
            A = flatB[i]
            M = A.T @ Cov_a @ A + Cov_e

            try:
                L = torch.linalg.cholesky(M)
                M_inv = torch.cholesky_inverse(L)
            except:
                M_inv = torch.linalg.pinv(M)
                
            Ncov = Cov_a - (Cov_a @ A) @ (M_inv @ A.T @ Cov_a)
            Nmu = (Cov_a @ A) @ (M_inv @ By[:, i].reshape(-1, 1))

            first_xi = S @ Ncov
            second_xi = (Nmu.T @ S @ Nmu)
            sqr_xi += torch.trace(first_xi) + second_xi

            first_sig = torch.norm(By[:, i])
            second_sig = 2 * (By[:, i] @ A.T) @ Nmu
            third_sig = torch.trace((A @ A.T) @ Ncov)
            four_sig = (Nmu.T @ A @ A.T @ Nmu)

            sqr_sig += (first_sig + second_sig + third_sig + four_sig)

            # free memory
            del A, M, M_inv, Ncov, Nmu, first_xi, second_xi, first_sig, second_sig, third_sig, four_sig

        sqr_xi = sqr_xi / num_neurons
        sqr_sig = sqr_sig / (num_neurons * size)

        # safe divide
        ls_lambda[l] = (sqr_sig / sqr_xi).item() if sqr_xi != 0 else torch.tensor(0.0, device=device)

        del Cov_a, Cov_e, flatB
        
    return ls_lambda


def compute_GCV(model, X_in, X, y):

    criterion = torch.nn.MSELoss(reduction='mean')
    XTX = torch.matmul(X.T, X)
    batch = X.size()[0]
    A = XTX

    try:
        A_inv = torch.inverse(A)
        
    except RuntimeError:
        jitter = 1e-6 * torch.eye(A.shape[0], device=A.device)
        A_inv = torch.inverse(A + jitter)
    
    M = torch.matmul(A_inv, XTX)
    dof = torch.trace(M)
    with torch.no_grad():
        DPSy = model(X_in)
        GCV = float((criterion(y, DPSy)/(1-dof/batch)**2).item()) if (dof<batch) else float(criterion(y, DPSy).item())

    return GCV
    
def ECM_layersise_update(model, par, Lambda, x, y):
    """
    Update layer-wise spline control weights using closed-form update (original logic).
    Returns updated model and GCV metric.
    """
    model.eval()
    device = x.device

    B_out = par['basic']
    B_in, B_w = augment_with_intercept(par['ebasic'], par['wbasic'], par['bbasic'])
    B_in, B_w = par['ebasic'], par['wbasic']

    n_layer = len(B_w)
    nk = B_w[0].size(-1)
    nm = [B.size()[1] for B in B_w]

    #n_layer, _, nm, nk = B_w.size()
    
    batch = x.size()[0]
    DB = diag_mat_weights(nk, 'second').to(device)

    Size = [b.size()[1] for b in B_in]

    #B_in = B_in.view(n_layer, batch, nm, nk)

    for l in range(n_layer):
        B_in[l] = B_in[l].view(batch, nm[l], nk)
        NW = torch.empty((1, nm[l], nk), device=device)
        NB = torch.empty((1, nm[l]), device=device)
        
        for i in range(nm[l]):
            B1y = B_out[l][:, i]
            BB = B_in[l][:,i,:].T
        
            # Update the weights and bias
            # Regularize with Lambda[l] / Size[l]
            mat = BB @ BB.T + Lambda[l]/ Size[l] * (DB.T @ DB)
            NW[:, i, :] = torch.linalg.solve(mat,  (BB @ B1y))
            NB[:, i] = torch.mean(B1y - (NW[:, i] @ BB))
            
        block = getattr(model.Spline_block.model, f'block_{l}')
        getattr(block.block.BSL, 'control_p').data = NW
        getattr(block.block.BSL, 'bias').data = NB

    GCV = compute_GCV(model, x, B_out[-1], y)
    return model, GCV



def ECM_update(model, max_iter, x, y, verbose = False):
	"""
	Iterate ECM: compute lambdas and update weights until convergence or max_iter.
	Returns best lambda vector and iteration count.
	"""
	BestGCV = prev = 9999
	patient = 10
	pcount = 0
	BestLambda = None
	iteration = 0
	info = {}
	num_para = 0
	for name, module in model.Spline_block.model.named_modules(): 
		if name == f"block_{num_para}": 
			num_para += 1
	Lambda_list = torch.zeros((max_iter, num_para))
	
	for i in range(max_iter):
		# forward to populate internals
		
		ECM_para = model.get_para_ecm(x)
		ECM_Lambda = ECM(ECM_para, initial_xi=1, initial_sigma=1, initial_lambda=1e-4, device=x.device)
		Lambda_list[i] = ECM_Lambda
		model, GCV = ECM_layersise_update(model, ECM_para, ECM_Lambda, x, y)
	
		if verbose:
			print('GCV:', GCV)
	
		if abs(prev - GCV) < 1e-4:
			print('GCV Converge at', i + 1, 'iteration')
			iteration = i + 1
	
			return info, opt_iter
	
		if GCV < BestGCV:
			info['Best_Lambda'] = ECM_Lambda
			info['Lambda_cand'] = Lambda_list
			info['Best_GCV'] = GCV
			info['Best_model'] = model
			opt_iter = i
			BestGCV = GCV
			pcount = 0
		else:
			pcount += 1
	
		if pcount == patient:
			print('GCV converged by patience at', i + 1, 'iteration')
			iteration = i + 1
	
			return info, opt_iter
	
		prev = GCV
		iteration = i + 1
	
		del ECM_para, ECM_Lambda
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
	
	return info, opt_iter



###
# Other Version of ECM
##

def ECM_optimized(par, initial_xi=1, initial_sigma=1, initial_lambda=1e-4, device='cpu'):
    """
    Optimized ECM using Primal Form (Woodbury/Bayesian Linear Regression style)
    Avoids N x N matrix inversion. Complexity O(N * K^2).
    """
    
    n_block = len(par['wbasic'])
    ls_lambda = torch.empty(n_block, device=device)

    # 1. 預先準備 Basis (這部分不變)
    B_aug, WB_aug = augment_with_intercept(par['ebasic'], par['wbasic'], par['bbasic'])

    for l in range(n_block):
        # Initialization
        # 注意: 這裡使用 scalar tensor 以避免維度廣播問題
        sigma_sq = torch.tensor(initial_sigma ** 2, device=device, dtype=torch.float32)
        xi_sq = torch.tensor(initial_xi ** 2, device=device, dtype=torch.float32)
        
        B = B_aug[l]       # (num_neurons, num_knots, size) ?? Check shape
        # 根據您原代碼 flatB = B.view(num_neurons, num_knots, size)
        # 所以 B 的原始維度應包含這些資訊
        By = par['basic'][l]      # (size, num_neurons)
        WB = WB_aug[l]            # (num_knots, num_neurons)

        
        num_neurons = WB.size(1)
        num_knots = WB.size(2)    # K
        size = By.size(0)         # N (Large, e.g., 200k)

        # 構建懲罰矩陣 S (K x K)
        # 這裡假設 diag_mat_weights 產生的是差分矩陣 D
        DB = diag_mat_weights(WB.size()[2]).to(device) 
        S = DB.T @ DB # (K x K)
        
        # Prior Precision Matrix (S / xi^2)
        # 我們直接計算 Precision，避免對 S 求偽逆後再求逆
        # Lambda_prior = S / xi_sq
        Lambda_prior = S / (xi_sq + 1e-8) # 加一點數值穩定

        # 為了保持與原代碼一致的 flatB
        flatB = B.T.view(num_neurons, num_knots, size) 
        
        total_sqr_xi = 0.0
        total_sqr_sig = 0.0

        for i in range(num_neurons):
            # A shape: [K, N] (knot, batch)
            A = flatB[i] 
            y = By[:, i].reshape(-1, 1) # [N, 1]

            # --- OPTIMIZATION START ---
            
            # 1. Compute Sufficient Statistics (The "Compression" step)
            # 這一步將 N 維縮減到 K 維。 O(N * K^2)
            # G = A @ A.T  --> [K, K]
            G = torch.matmul(A, A.T) 
            
            # h = A @ y    --> [K, 1]
            h = torch.matmul(A, y)

            # 2. Compute Posterior Precision & Covariance (K x K)
            # Precision = Prior_Prec + (1/sigma^2) * G
            beta = 1.0 / sigma_sq
            Posterior_Precision = Lambda_prior + beta * G
            
            # Ncov = inv(Posterior_Precision) --> [K, K]
            # 使用 cholesky_solve 或 solve 比 inv 更穩定
            # L * L.T = Posterior_Precision
            # Ncov = inv(Posterior_Precision)
            try:
                L = torch.linalg.cholesky(Posterior_Precision)
                Ncov = torch.cholesky_inverse(L)
            except RuntimeError:
                # Fallback for numerical instability
                jitter = 1e-6 * torch.eye(num_knots, device=device)
                Ncov = torch.linalg.inv(Posterior_Precision + jitter)

            # 3. Compute Posterior Mean (Nmu) --> [K, 1]
            # Nmu = Ncov @ (beta * h)
            Nmu = torch.matmul(Ncov, beta * h)

            # --- 計算 sqr_xi (參數更新項) ---
            # Formula: Trace(S * Ncov) + Nmu.T * S * Nmu
            # 這些都是 K x K 的運算，非常快
            term_trace_xi = torch.trace(torch.matmul(S, Ncov))
            term_quad_xi = torch.matmul(Nmu.T, torch.matmul(S, Nmu))
            total_sqr_xi += term_trace_xi + term_quad_xi

            # --- 計算 sqr_sig (殘差項) ---
            # 原公式展開: E[||y - A.T w||^2]
            # = y.T y - 2 y.T A.T E[w] + Tr(A A.T E[ww.T])
            # = y.T y - 2 h.T Nmu + Tr(G * (Ncov + Nmu Nmu.T))
            
            # 1. y^T y (Scalar)
            y_norm_sq = torch.sum(y ** 2)
            
            # 2. -2 h^T Nmu (Scalar)
            cross_term = -2 * torch.matmul(h.T, Nmu)
            
            # 3. Tr(G @ (Ncov + Nmu @ Nmu.T))
            # = Tr(G @ Ncov) + Tr(G @ Nmu @ Nmu.T)
            # = Tr(G @ Ncov) + Nmu.T @ G @ Nmu
            
            E_wwT = Ncov + torch.matmul(Nmu, Nmu.T)
            # 這裡可以直接用 Trace property 加速:
            # trace(G @ E_wwT)
            quad_term = torch.trace(torch.matmul(G, E_wwT))
            
            total_sqr_sig += y_norm_sq + cross_term + quad_term
            
            # --- OPTIMIZATION END ---

        # Normalize across neurons
        mean_sqr_xi = total_sqr_xi / num_neurons
        mean_sqr_sig = total_sqr_sig / (num_neurons * size) # 注意這裡要除以 N (size)

        # Update lambda
        # lambda = sigma^2 / xi^2
        if mean_sqr_xi > 1e-9:
            ls_lambda[l] = (mean_sqr_sig / mean_sqr_xi).item()
        else:
            ls_lambda[l] = 0.0

    return ls_lambda
