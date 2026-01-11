import os, sys
from pathlib import Path

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
import itertools

import argparse


def create_loaders(X_train, y_train, X_val, y_val, batch_size=None):
    bs = batch_size if batch_size else len(X_train)
    
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=len(X_val), shuffle=False) # Val 通常不需要 shuffle
    return train_loader, val_loader

def compute_GCV(model, X_in, X, y):
	criterion = torch.nn.MSELoss(reduction='mean')
	
	
	XTX = torch.matmul(X.T, X)
	batch = X.size()[0]
	A = XTX
	try:
		A_inv = torch.inverse(A)
		
	except RuntimeError:
		# 如果 A 接近奇異矩陣，加一點 jitter (通常 lambda > 0 就不會發生)
		jitter = 1e-6 * torch.eye(A.shape[0], device=A.device)
		A_inv = torch.inverse(A + jitter)
	
	M = torch.matmul(A_inv, XTX)
	dof = torch.trace(M)
	with torch.no_grad():
		DPSy = model(X_in)
		GCV = float((criterion(y, DPSy)/(1-dof/batch)**2).item()) if (dof<batch) else float(criterion(y, DPSy).item())
	
	return GCV


def main():
	parser = argparse.ArgumentParser(description="Train a model")
	parser.add_argument('--startidx', type = int)
	args = parser.parse_args()
	
	ntrain = 200
	nval = 200
	ntest = 1000
	Dtype = 'A'
	ndim = 2
	learning_rate = 1e-1
	ndf = 20
	nk = 7
	nbl = 1   
	Fout = 1
	data = {}
	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	criterion = torch.nn.MSELoss(reduction='mean')
	
	## Storing parameter ##
	result = {}
	Lambdalist = {}
	GCV_1 = np.zeros((ndf))
	GCV_2 = np.zeros((ndf))
	MSPE_1 = np.zeros((ndf))
	MSPE_2 = np.zeros((ndf))
	
	Iterlist = np.zeros((ndf, 1))
	for d in range(args.startidx, args.startidx + ndf):
	
		####################
		# Data Preparation #
		####################
	
		torch.manual_seed(d)
		X_train, y_train = sim_data(ntrain, ndim, Dtype)
		X_val, y_val = sim_data(nval, ndim, Dtype)
		X_test, y_test = sim_data(ntest, ndim, Dtype)
		epstrain = torch.normal(0, torch.var(y_train)*0.01, size=y_train.size())
		epsval = torch.normal(0, torch.var(y_val)*0.01, size=y_val.size())
		epstest = torch.normal(0, torch.var(y_test)*0.01, size=y_test.size())
	
		y_train, y_test, y_val = y_train + epstrain, y_test + epstest, y_val + epsval
		data[str(d+1)] = {'TrainX': X_train, 'Trainy': y_train, 'TestX': X_test, 'Testy': y_test, 'ValX': X_val, 'Valy': y_val}
	
	
		train_loader, val_loader = create_loaders(X_train, y_train, X_val, y_val)
		
		_config = {0: [10], 1: [15], 2: [20], 3: [10, 10], 4: [15, 15], 5: [20, 20], 6: [10, 10, 10], 7: [15, 15, 15], 8: [20, 20, 20], 9: [10, 10, 10, 10], 10: [15, 15, 15, 15], 11: [20, 20, 20, 20]}

		_result = torch.zeros(len(_config))
		_result_mspe = torch.zeros(len(_config))
		
		device = "cpu"
		criterion = torch.nn.MSELoss(reduction='mean')
		
		ndim = X_train.size()[1]
		Fout = 1
		
		for key, value in _config.items():
		
		
			DeepBS = DPS(input_dim = ndim, degree = 3, num_knots = nk, num_neurons = value, num_bsl = len(value), dropout = 0.0, output_dim = Fout, knots_place = 'quantile', bias = True).to(device)
			
			optimizer = torch.optim.Adam(DeepBS.parameters(), lr=learning_rate)
			save_path_bs = f"./logs/best_DBS_model_{key}_{d}.pt"
			early_stop = EarlyStopping(patience=30, verbose=False, delta=1e-4, path=save_path_bs)
			
			trainer = Trainer(DeepBS, optimizer, criterion, device, early_stopping=early_stop)
			trainer.fit(train_loader, val_loader, epochs=10000)
			
			with torch.no_grad():
				DeepBS.load_state_dict(torch.load(save_path_bs, weights_only=True))
				DeepBS.eval()
				# Hook
				activations = {}
				def get_last_layer_hook(module, input, output):
					activations['last_layer'] = output.detach()
				handle = DeepBS.Spline_block.register_forward_hook(get_last_layer_hook)
				output = DeepBS(X_train)
				last_neurons = activations['last_layer']
				handle.remove()
				
				# GCV Calculation
				_result[key] = compute_GCV(DeepBS, X_train, last_neurons, y_train)
				_result_mspe[key] = criterion(DeepBS(X_test), y_test)
				
		
		min_idx = torch.argmin(_result).item()
		min_mspe = _result_mspe[min_idx]
		min_archi = _config[min_idx]
		neuron_options = [min_archi[0] - 5, min_archi[0], min_archi[0] + 5]
		
		opt_nbl = len(min_archi)
		combinations = list(itertools.product(neuron_options, repeat=opt_nbl))
		dps_model_config = {i : list(combo) for i, combo in enumerate(combinations)}
		
		
		_result_gcv = torch.zeros(len(dps_model_config))
		_result_gcv_mspe = torch.zeros(len(dps_model_config))
		
		
		for i, hidden_config in dps_model_config.items():
	
			DeepBS = DPS(input_dim = ndim, degree = 3, num_knots = nk, num_neurons = hidden_config, num_bsl = len(hidden_config), dropout = 0.0, output_dim = Fout, knots_place = 'quantile', bias = True).to(device)
			optimizer = torch.optim.Adam(DeepBS.parameters(), lr=learning_rate)
			save_path_bs = f"./logs/best_DBS_model_gcv_{d}.pt"
			early_stop = EarlyStopping(patience=50, verbose=False, delta=1e-3, path=save_path_bs)
			trainer = Trainer(DeepBS, optimizer, criterion, device, early_stopping=early_stop)
			trainer.fit(train_loader, val_loader, epochs=1000)
			
			with torch.no_grad():
				DeepBS.load_state_dict(torch.load(save_path_bs, weights_only=True))
				DeepBS.eval()
				# Hook
				activations = {}
				def get_last_layer_hook(module, input, output):
					activations['last_layer'] = output.detach()
				handle = DeepBS.Spline_block.register_forward_hook(get_last_layer_hook)
				output = DeepBS(X_train)
				last_neurons = activations['last_layer']
				handle.remove()
				
				# GCV Calculation
				_result_gcv[i] = compute_GCV(DeepBS, X_train, last_neurons, y_train)
				_result_gcv_mspe[i] = criterion(DeepBS(X_test), y_test)
				
		min_gcv_idx = torch.argmin(_result_gcv).item()
		min_gcv_mspe = _result_gcv_mspe[min_gcv_idx]
		min_gcv_archi = dps_model_config[min_gcv_idx]
		
		
		GCV_1[d-args.startidx] = torch.min(_result)
		GCV_2[d-args.startidx] = torch.min(_result_gcv)
		MSPE_1[d-args.startidx] = min_mspe
		MSPE_2[d-args.startidx] = min_gcv_mspe
		print('min_archi: ', min_archi)
		
		print(MSPE_1)
		print(MSPE_2)
		
	print('results')
	result['results1_mspe'] = _result_mspe
	result['results2_mspe'] = _result_gcv_mspe
	result['MSPE1'] = MSPE_1
	result['MSPE2'] = MSPE_2
    print(f"Proposed Method | MSPE: {np.mean(MSPE_1):.4f} | STD: {np.std(MSPE_1):.4f}")
    print(f"Sensitivity Method | MSPE: {np.mean(MSPE_2):.4f} | STD: {np.std(MSPE_2):.4f}")

    np.save('./logs/GCV.npy', result) 

if __name__ == "__main__":
    main()