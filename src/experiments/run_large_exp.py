import os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
LEVEL1 = HERE.parent

if str(LEVEL1) not in sys.path:
    sys.path.insert(0, str(LEVEL1))
    
import argparse
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch
import math
import numpy as np
from sklearn.metrics import roc_curve, auc

import pandas as pd
import zipfile

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

def create_loaders(X_train, y_train, X_val, y_val, batch_size=None):
    bs = batch_size if batch_size else len(X_train)
    
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=len(X_val), shuffle=False) # Val 通常不需要 shuffle
    return train_loader, val_loader



def main():

	parser = argparse.ArgumentParser(description="Train a model")
	parser.add_argument('--case', type = str, help = 'dataset', choices=['ca', 'bike', 'churn', 'year'])
	parser.add_argument('--hidden_config', default = "", type=str, help='Path to the dataset')
	parser.add_argument('--Fin', type = int, default=2)
	parser.add_argument('--Fout', type = int, default=1)
	
	## DBS Model Setting
	parser.add_argument('--nk', type = int, default=15)
	parser.add_argument('--knot_place', type = str, default='quantile')
	parser.add_argument('--dropout', type = float, default=1e-1)
	parser.add_argument('--hc', type=int, nargs ="*", default=[30, 30], help='hidden configuration for DPS')
	parser.add_argument('--nl', type = int, default=2)
	parser.add_argument('--nepochs', type = int, default=10000)
	parser.add_argument('--lr', type = float, default=1e-2)
	parser.add_argument('--hp', type = str, default = 'A')
	parser.add_argument('--bs', type = int, default = 2048)
	
	## ECM setting
	parser.add_argument('--ECM_Iter', type = int, default=20)
	## Fine-tune Setting
	parser.add_argument('--fine_tune_nepochs', type = int, default=2000)
	parser.add_argument('--fine_tune_lr', type = float, default=5e-4)
	args = parser.parse_args()
	
	## Data Preprocessing 
	print(f"Case: {args.case}")
	data_loader = Dataset(args.case)
	data = data_loader.get_data()
	
	X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
	y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
	task = data['task']
	
	train_loader, val_loader = create_loaders(X_train, y_train, X_val, y_val, batch_size = args.bs)
	learning_rate = args.lr
	nbl = len(args.hc)
	_, ndim = X_train.size()
	Fout = 1
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	device = "cpu"
	
	if task == 'regression':
		criterion = torch.nn.MSELoss(reduction='mean')
	else:
		criterion = nn.BCEWithLogitsLoss()
		
	DeepBS = DPS(input_dim = ndim, degree = 3, 
				 num_knots = args.nk, 
				 num_neurons = args.hc, 
				 num_bsl = nbl, 
				 dropout = args.dropout, 
				 output_dim = Fout, 
				 knots_place = 'quantile', 
				 bias = True).to(device)
	
	optimizer = torch.optim.Adam(DeepBS.parameters(), lr=args.lr)
	
	save_path_bs = f"./best_model/best_DBS_model_"+args.case+"_"+args.hp+".pt"
	early_stop = EarlyStopping(patience=50, verbose=False, delta=1e-3, path=save_path_bs)
	
	trainer = Trainer(DeepBS, optimizer, criterion, device, early_stopping=early_stop)
	trainer.fit(train_loader, val_loader, epochs=args.nepochs)
	
	print('Model_config: ', args.hc)
	with torch.no_grad():
		DeepBS = DPS(input_dim = ndim, degree = 3, num_knots = args.nk, num_neurons = args.hc, num_bsl = nbl, dropout = args.dropout, output_dim = Fout, knots_place = 'quantile', bias = True).to(device)
		DeepBS.load_state_dict(torch.load(save_path_bs, weights_only=True))
		DeepBS.eval()
		
		Info_ECM, iteration = ECM_update(DeepBS, args.ECM_Iter, X_train[:1024,:], y_train[:1024])
		
		print(task)
		if task == 'classification':
			pred_train_probs = torch.sigmoid(DeepBS(X_train))
			pred_test_probs = torch.sigmoid(DeepBS(X_test))
			
			fpr, tpr, thresholds = roc_curve(y_train, pred_train_probs.detach().numpy())
			print(f"(DBS) Training AUC: {auc(fpr, tpr):.4f}")
			fpr, tpr, thresholds = roc_curve(y_test, pred_test_probs.detach().numpy())
			print(f"(DBS) Testing AUC: {auc(fpr, tpr):.4f}")
			
		else:
			print(f"(DBS) Training MSE: {criterion(DeepBS(X_train), y_train):.4f}")
			print(f"(DBS) Testing MSE: {criterion(DeepBS(X_test), y_test):.4f}")
			
	
	'''
	DPS Training
	
	'''
	DeepPS = DPS(input_dim = ndim, degree = 3, num_knots = args.nk, num_neurons = args.hc, num_bsl = nbl, dropout = 0.1, output_dim = Fout, knots_place = 'quantile', bias = True).to(device)
	DeepPS.load_state_dict(torch.load(save_path_bs, weights_only=True))
	optimizer = torch.optim.Adam(DeepPS.parameters(), lr=args.fine_tune_lr)
	
	save_path_ps = f"./best_model/best_DPS_model_"+args.case+"_"+args.hp+".pt"
	early_stop = EarlyStopping(patience=50, verbose=False, delta=1e-3, path=save_path_ps)
	
	
	trainer = Trainer(DeepPS, optimizer, criterion, device, early_stopping=early_stop)
	trainer.fit(train_loader, val_loader, epochs=args.fine_tune_nepochs, penalty_func = spline_penalty_loss)
	
	
	'''
	Model Evaluation
	
	'''
	
	with torch.no_grad():
		DeepPS = DPS(input_dim = ndim, degree = 3, num_knots = args.nk, num_neurons = args.hc, num_bsl = nbl, dropout = args.dropout, output_dim = Fout, knots_place = 'quantile', bias = True).to(device)
		DeepPS.load_state_dict(torch.load(save_path_ps, weights_only=True))
		DeepPS.eval()
		
		if task == 'classification':
			pred_train_probs = torch.sigmoid(DeepPS(X_train))
			pred_test_probs = torch.sigmoid(DeepPS(X_test))
			
			fpr, tpr, thresholds = roc_curve(y_train, pred_train_probs.detach().numpy())
			print(f"(DPS) Training AUC: {auc(fpr, tpr):.4f}")
			fpr, tpr, thresholds = roc_curve(y_test, pred_test_probs.detach().numpy())
			print(f"(DPS) Testing AUC: {auc(fpr, tpr):.4f}")
			
		else:
			print(f"(DPS) Training MSE: {criterion(DeepPS(X_train), y_train):.4f}")
			print(f"(DPS) Testing MSE: {criterion(DeepPS(X_test), y_test):.4f}")

if __name__ == "__main__":
    main()