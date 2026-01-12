import os, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
LEVEL1 = HERE.parent

if str(LEVEL1) not in sys.path:
    sys.path.insert(0, str(LEVEL1))
    
from argparse import ArgumentParser
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch
import math
import numpy as np

from data.synthetic import sim_data
from estimation.ecm import ECM_update
from models.dps import DPS
from training.early_stopping import EarlyStopping
from training.trainer import *
from training.loss import spline_penalty_loss
from data.dataset import Dataset

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

def create_loaders(X_train, y_train, X_val, y_val, batch_size=None):
    bs = batch_size if batch_size else len(X_train)
    
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=len(X_val), shuffle=False) # Val 通常不需要 shuffle
	
    return train_loader, val_loader
	
# ---------------------------------------------------------
# 1. Improved Helper: B-Spline Basis (GPU Safe)
# ---------------------------------------------------------
def b_spline_basis(x, knots, degree):
    x = x.unsqueeze(-1)
    # knots shape: (1, 1, num_knots) to broadcast
    knots = knots.view(1, 1, -1)
    
    # Initial basis (degree 0)
    basis = ((x >= knots[..., :-1]) & (x < knots[..., 1:])).float()
    
    eps = 1e-6
    
    for d in range(1, degree + 1):
        b_prev = basis
        
        knots_left = knots[..., :-(d+1)]     # t_i
        knots_right = knots[..., d+1:]       # t_{i+d+1}
        
        # Term 1
        denom1 = knots[..., d:-1] - knots_left
        term1 = ((x - knots_left) / (denom1 + eps)) * b_prev[..., :-1]
        
        # Term 2
        denom2 = knots_right - knots[..., 1:-d]
        term2 = ((knots_right - x) / (denom2 + eps)) * b_prev[..., 1:]
        
        basis = term1 + term2
        
    return basis 

# ---------------------------------------------------------
# 2. ResNet Block for Tabular Data
# ---------------------------------------------------------
class ResNetBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(ResNetBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.linear2 = nn.Linear(hidden_dim, input_dim) # Project back to input_dim for addition
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        # Shortcut connection handling (if dims differ, projection is needed)
        self.shortcut = nn.Identity()
        if input_dim != hidden_dim:
            self.shortcut = nn.Linear(input_dim, hidden_dim) 
            # Note: The design below assumes input_dim == hidden_dim for standard ResNet blocks
            # If you want to change dimensions, the residual add needs to match dimensions.
            # Here we assume a strict ResNet block where input_dim == hidden_dim usually.

    def forward(self, x):
        # Save input for residual connection
        residual = x
        
        # First dense layer
        out = self.linear1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)
        
        # Second dense layer
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        
        # Residual connection
        out += residual
        out = F.relu(out) # ReLU after addition
        return out

# ---------------------------------------------------------
# 3. Optimized BSL Layer
# ---------------------------------------------------------
class BSL(nn.Module):
    def __init__(self, degree, num_knots, num_neurons, knots_place, bias=True):
        super(BSL, self).__init__()
        self.degree = degree
        self.num_knots = num_knots
        self.num_neurons = num_neurons
        self.knots_place = knots_place
        
        self.num_basis = num_knots - degree - 1
        self.control_p = nn.Parameter(torch.randn(1, self.num_neurons, self.num_basis))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, self.num_neurons)) # Init bias to 0 is usually safer
        else:
            self.register_parameter('bias', None)

        # Create knots and register as buffer (automatically moves with .cuda())
        self.register_buffer('knots', self._create_knots(degree, num_knots))
        self.inter = {}

    def _create_knots(self, d, k):
        # Create directly on default device, will be moved by register_buffer
        if self.knots_place == 'boundary' or True: 
            mid_knots = torch.linspace(0, 1, k - 2*d)
            left_pad = torch.zeros(d)
            right_pad = torch.ones(d)
            knots = torch.cat([left_pad, mid_knots, right_pad])
        return knots

    def forward(self, x):
        # x shape: (Batch, Neurons)
        # B-Spline input must be in [0, 1]. The previous layer (Sigmoid) guarantees this.
        
        basis_matrix = b_spline_basis(x, self.knots, self.degree)
        self.inter['basic'] = basis_matrix.reshape(x.shape[0], -1) 
        
        # (B, N, Basis) * (1, N, Basis) -> sum over Basis dim
        tout = (basis_matrix * self.control_p).sum(dim=2) 
        
        if self.bias is not None:
            tout += self.bias
            
        return tout

# ---------------------------------------------------------
# 4. BSpline Blocks & Stacks
# ---------------------------------------------------------
class BSpline_block(nn.Module):
    def __init__(self, degree, num_knots, num_neurons, knots_place, dropout=0.0, bias=True):
        super(BSpline_block, self).__init__()
        self.block = nn.Sequential(OrderedDict([
            ('bn', nn.BatchNorm1d(num_neurons)),
            ('sigmoid', nn.Sigmoid()), # Crucial: Maps input to [0, 1] for B-Splines
            ('BSL', BSL(degree, num_knots, num_neurons, knots_place, bias)),
            ('drop', nn.Dropout(dropout)),
        ]))
        
    def forward(self, x):
        return self.block(x)

class StackBS_block(nn.Module):
    def __init__(self, block, degree, num_knots, num_neurons, num_blocks, knots_place, dropout=0.0, bias=True):
        super().__init__()
        layers = OrderedDict()
        for i in range(num_blocks):
            # Input dimension for this block
            in_dim = num_neurons[i-1] if i > 0 else num_neurons[0]
            out_dim = num_neurons[i]
            
            # Dimension matching if needed between blocks
            if i > 0 and in_dim != out_dim:
                layers[f'dim_match_{i}'] = nn.Linear(in_dim, out_dim)
            
            layers[f'block_{i}'] = block(
                degree=degree, 
                num_knots=num_knots, 
                num_neurons=out_dim, 
                knots_place=knots_place,
                dropout=dropout,
                bias=bias
            )
        self.model = nn.Sequential(layers)

    def forward(self, x):
        return self.model(x)

# ---------------------------------------------------------
# 5. Main DPS Model with ResNet Backbone
# ---------------------------------------------------------
class DPS(nn.Module):
    def __init__(self, input_dim, degree, num_knots, num_neurons, num_bsl, dropout, output_dim, knots_place, bias, 
                 use_resnet=True, resnet_blocks=3, resnet_dim=256):
        """
        Modified to include ResNet Backbone.
        
        Args:
            use_resnet (bool): Whether to use ResNet blocks before Splines.
            resnet_blocks (int): Number of ResNet blocks.
            resnet_dim (int): Hidden dimension for ResNet (and subsequent Spline layers).
        """
        super(DPS, self).__init__()
        self.use_resnet = use_resnet
        
        # --- Feature Extractor (ResNet) ---
        self.first_linear = nn.Linear(input_dim, resnet_dim)
        
        if use_resnet:
            self.backbone = nn.Sequential(*[
                ResNetBlock(input_dim=resnet_dim, hidden_dim=resnet_dim, dropout=dropout)
                for _ in range(resnet_blocks)
            ])
        else:
            self.backbone = nn.Identity()

        # --- Spline Layers ---
        # If using ResNet, update num_neurons[0] to match resnet_dim
        # If user passed a list for num_neurons, we ensure the input matches backbone output
        self.spline_input_dim = resnet_dim
        
        # Adjust first layer of Spline stack to accept resnet_dim
        # We assume num_neurons is a list, e.g., [256, 128]
        if num_neurons[0] != resnet_dim:
            print(f"Warning: Adjusting first Spline layer input from {num_neurons[0]} to {resnet_dim} to match ResNet.")
            num_neurons[0] = resnet_dim

        self.Spline_block = StackBS_block(
            BSpline_block, 
            degree=degree, 
            num_knots=num_knots, 
            num_neurons=num_neurons, 
            num_blocks=num_bsl, 
            knots_place=knots_place, 
            dropout=dropout/2,
            bias=bias
        )
        
        # Final Output Layer
        self.ln2 = nn.Linear(num_neurons[-1], output_dim)
        
    def forward(self, x):
        # 1. Project to Latent Space
        x = self.first_linear(x)
        
        # 2. Deep Feature Extraction (ResNet)
        x = self.backbone(x)
        
        # 3. Interpretability & Nonlinearity (Splines)
        spout = self.Spline_block(x)
        
        # 4. Final Prediction
        output = self.ln2(spout)
        
        return output
    
    # ... (get_para_ecm method remains same as your original code) ...
    def get_para_ecm(self, x):
        ecm_para = {}
        bs_block_out = {}
        bs_spline_value = {}
        bs_spline_weight = {}
        bs_spline_bias = {}
    
        def get_activation(name):
            def hook(model, input, output):
                bs_block_out[name] = output.detach()
            return hook

        _ = self(x)
        
        handles = []
        for name, layer in self.named_modules():
            if 'block.drop' in name:
                handles.append(layer.register_forward_hook(get_activation(name)))
            elif 'block.BSL' in name:
                bs_spline_value[name] = layer.inter['basic'].detach()
                bs_spline_weight[name] = layer.control_p.detach()
                bs_spline_bias[name] = layer.bias.detach()
        
        _ = self(x)
        
        for h in handles:
            h.remove()
            
        ecm_para['basic'] = list(bs_block_out.values())
        ecm_para['ebasic'] = list(bs_spline_value.values())
        ecm_para['wbasic'] = list(bs_spline_weight.values())
        ecm_para['bbasic'] = list(bs_spline_bias.values())
        
        return ecm_para

class PenaltyScheduler:
    def __init__(self, warm_up_epochs, start_epoch=0):
        
        self.warm_up_epochs = warm_up_epochs
        self.start_epoch = start_epoch

    def get_factor(self, current_epoch):
        if current_epoch < self.start_epoch:
            return 0.0
        
        if self.warm_up_epochs == 0:
            return 1.0
            
        relative_step = current_epoch - self.start_epoch
        factor = relative_step / self.warm_up_epochs
        
        return min(max(factor, 0.0), 1.0)

def freeze_backbone(model):

    print("Freezing ResNet Backbone...")
    for param in model.backbone.parameters():
        param.requires_grad = False
		

class Trainer:
    def __init__(self, model, optimizer, criterion, device, scheduler=None, early_stopping=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.history = {'train_loss': [], 'val_loss': []}

    def train_epoch(self, train_loader, penalty_func=None, current_lambdas=None):
        
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 1. Forward Pass
            output = self.model(batch_x)
            loss = self.criterion(output, batch_y)
            
            # 2. Add Penalty 
            if penalty_func and current_lambdas is not None:
                reg_loss = penalty_func(self.model, current_lambdas, self.device)
                loss += reg_loss
            
            # 3. Backward Pass
            loss.backward()
            
            # (Optional) Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(train_loader)
        self.history['train_loss'].append(epoch_loss)
        return epoch_loss

    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                running_loss += loss.item()
                
        epoch_loss = running_loss / len(val_loader)
        self.history['val_loss'].append(epoch_loss)
        
        if self.early_stopping:
            self.early_stopping(epoch_loss, self.model)
        
        return epoch_loss

    def fit(self, train_loader, val_loader, epochs, penalty_func=None, lambda_vals=None, 
            warm_up_epochs=0, start_warm_up_epoch=0):
        
        
        # Initialize Penalty Scheduler
        pen_scheduler = PenaltyScheduler(warm_up_epochs, start_warm_up_epoch)
        
        pbar = tqdm(range(epochs), desc="Epochs")
        for epoch in pbar:
            
            # --- Warm-up Logic ---
            current_lambdas = None
            warm_up_factor = pen_scheduler.get_factor(epoch)

            if lambda_vals is not None:
                if isinstance(lambda_vals, dict):
                    current_lambdas = {k: v * warm_up_factor for k, v in lambda_vals.items()}
                elif isinstance(lambda_vals, (float, int)):
                    current_lambdas = lambda_vals * warm_up_factor
                else:
                    current_lambdas = [v * warm_up_factor for v in lambda_vals]
            
            train_loss = self.train_epoch(train_loader, penalty_func, current_lambdas)
            val_loss = self.validate(val_loader)
            
            if self.scheduler:
                self.scheduler.step()
                
            pbar.set_postfix({
                'Train': f"{train_loss:.4f}", 
                'Val': f"{val_loss:.4f}",
                'P_Factor': f"{warm_up_factor:.2f}" 
            })
            
            if self.early_stopping and self.early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
        return self.history

def evaluation(model, loader):
    criterion = nn.MSELoss(reduction='mean')
    total_loss = 0.0
    num_samples = 0
    with torch.no_grad(): # Disable gradient calculation for efficiency
        model.eval()
        for inputs, targets in loader:
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            num_samples += inputs.size(0)
            
    average_loss = total_loss / num_samples
    print(f"MSE: {average_loss:.4f}")
	
def main():

	case = 'year'
	task = 'classification' if case == 'churn' else 'regression'
	
	data_loader = Dataset(case)
	data = data_loader.get_data()
	X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
	y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
	bs = 2048
	
	print(X_train.size())
	train_loader, val_loader = create_loaders(X_train, y_train, X_val, y_val, batch_size = bs)
	test_loader, val_loader = create_loaders(X_test, y_test, X_val, y_val, batch_size = bs)
	
	
	ndim = X_train.size()[1]
	nk = 10
	hidden_config = [128, 64, 32]
	nbl = len(hidden_config)
	dp = 0.2
	Fout = 1
	device = 'cpu'
	hp = '_A'
	save_path_bs = "./best_DBS_sh_model_"+case+ hp + ".pt"
	DeepBS = DPS(input_dim = ndim, degree = 3, 
				 num_knots = nk, 
				 num_neurons = hidden_config, 
				 num_bsl = nbl,
	             #use_resnet = False,
	             #resnet_dim = 1,
				 dropout = dp, 
				 output_dim = Fout, 
				 knots_place = 'quantile', 
				 bias = True).to(device)
	
	optimizer = torch.optim.Adam(DeepBS.parameters(), lr=5e-2)
	early_stop = EarlyStopping(patience=5, verbose=False, delta=1e-4, path=save_path_bs)
	criterion = torch.nn.MSELoss(reduction='mean')
	ECM_iteration = 20
	
	
	trainer = Trainer(DeepBS, optimizer, criterion, device, early_stopping=early_stop)
	trainer.fit(train_loader, val_loader, epochs=10000)
	
	
	
	sub_sample_ECM = 2048 # It should be whole dataset
	with torch.no_grad():
		DeepBS.eval()
		DeepBS = DPS(input_dim = ndim, degree = 3, num_knots = nk, num_neurons = hidden_config, num_bsl = nbl,
	             #use_resnet = False, resnet_dim = 1, 
					 dropout = dp, output_dim = Fout, knots_place = 'quantile', bias = True).to(device)
		DeepBS.load_state_dict(torch.load(save_path_bs, weights_only=True))
		print(evaluation(DeepBS, test_loader))
		Info_ECM, iteration = ECM_update(DeepBS, ECM_iteration, X_train[:sub_sample_ECM,:], y_train[:sub_sample_ECM], True)
	
	DeepPS = DPS(input_dim = ndim, degree = 3, num_knots = nk, num_neurons = hidden_config, num_bsl = nbl,
	             #use_resnet = False, resnet_dim = 1, 
				 dropout = dp, output_dim = Fout, knots_place = 'quantile', bias = True).to(device)
	DeepPS.load_state_dict(torch.load(save_path_bs, weights_only=True))
	optimizer = torch.optim.Adam(DeepPS.parameters(), lr=1e-4)
	save_path_ps = "./best_DPS_sh_model_"+case+ hp + ".pt"
	
	early_stop = EarlyStopping(patience=20, verbose=False, delta=1e-4, path=save_path_ps)
	freeze_backbone(DeepPS)
	
	optimizer = torch.optim.Adam(
		filter(lambda p: p.requires_grad, DeepPS.parameters()), 
		lr=1e-4, # orignal good 1e-3
		weight_decay=0 
	)
	
	trainer = Trainer(DeepPS, optimizer, criterion, device, early_stopping=early_stop)
	trainer.fit(train_loader, val_loader, epochs=50, 
				penalty_func = spline_penalty_loss, 
				lambda_vals = Info_ECM['Best_Lambda'],
			   warm_up_epochs=10,  start_warm_up_epoch = 0)
	
	
	with torch.no_grad():
		DeepPS = DPS(input_dim = ndim, degree = 3, num_knots = nk, num_neurons = hidden_config, num_bsl = nbl,
	             #use_resnet = False, resnet_dim = 1, 
					 dropout = 0.1, output_dim = Fout, knots_place = 'quantile', bias = True).to(device)
		DeepPS.load_state_dict(torch.load(save_path_ps, weights_only=True))
		DeepPS.eval()
		
		print(evaluation(DeepPS, test_loader))
		
if __name__ == "__main__":
    main()