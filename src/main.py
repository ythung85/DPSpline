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

parser = ArgumentParser()
## Dataset Setting
parser.add_argument('--trainsize', type = int, default= 200)
parser.add_argument('--valsize', type = int, default=200)
parser.add_argument('--testsize', type = int, default=1000)
parser.add_argument('--rep', type = int, default=3)
parser.add_argument('--data', type = str, default='A')
parser.add_argument('--Fin', type = int, default=2)
parser.add_argument('--Fout', type = int, default=1)

## DBS Model Setting
parser.add_argument('--nk', type = int, default=15)
parser.add_argument('--knot_place', type = str, default='quantile')
parser.add_argument('--hc', nargs ="*", default=[30, 30], type=int, help='hidden configuration for DPS')
parser.add_argument('--nl', type = int, default=2)
parser.add_argument('--nepochs', type = int, default=10000)
parser.add_argument('--lr', type = float, default=1e-2)

## ECM setting
parser.add_argument('--ECM_Iter', type = int, default=20)
## Fine-tune Setting
parser.add_argument('--fine_tune_nepochs', type = int, default=2000)
parser.add_argument('--fine_tune_lr', type = float, default=1e-3)


args = parser.parse_args()

def create_loaders(X_train, y_train, X_val, y_val, batch_size=None):
    bs = batch_size if batch_size else len(X_train)
    
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=len(X_val), shuffle=False) # Val 通常不需要 shuffle
    return train_loader, val_loader


if __name__ == "__main__":

    ntrain = args.trainsize
    nval = args.valsize
    ntest = args.testsize
    Dtype = args.data
    ndim = args.Fin
    learning_rate = args.lr
    ndf = args.rep
    hidden_config = args.hc
    nbl = len(hidden_config)
    nk = args.nk
    kp = args.knot_place
    #nbl = args.nl    
    Fout = args.Fout
    data = {}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    criterion = torch.nn.MSELoss(reduction='mean')

    ## Storing parameter ##
    result = {}
    Lambdalist = {}
    Bres = np.zeros((ndf, 1))
    Pres = np.zeros((ndf, 1))
    Iterlist = np.zeros((ndf, 1))
    for d in range(ndf):

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

        print(hidden_config)
        DeepBS = DPS(input_dim = ndim, degree = 3, num_knots = nk, num_neurons = hidden_config, num_bsl = nbl, dropout = 0.0, output_dim = Fout, knots_place = kp, bias = True).to(device)
        optimizer = torch.optim.Adam(DeepBS.parameters(), lr=learning_rate)
        save_path_bs = f"best_DBS_model_d{d+1}.pt"
        early_stop = EarlyStopping(patience=50, verbose=False, delta=1e-3, path=save_path_bs)


        trainer = Trainer(DeepBS, optimizer, criterion, device, early_stopping=early_stop)
        trainer.fit(train_loader, val_loader, epochs=args.nepochs)



        # Load best model for ECM
        with torch.no_grad():
            DeepBS = DPS(input_dim = ndim, degree = 3, num_knots = nk, num_neurons = hidden_config, num_bsl = nbl, dropout = 0.0, output_dim = Fout, knots_place = kp, bias = True).to(device)
            DeepBS.load_state_dict(torch.load(save_path_bs, weights_only=True))
            DeepBS.eval()


            score_bs = criterion(y_test, DeepBS(X_test)).item()
            Bres[d] = score_bs

            Info_ECM, iteration = ECM_update(DeepBS, args.ECM_Iter, X_train, y_train)

            Lambdalist[str(d+1)] = Info_ECM['Best_Lambda']

            del DeepBS     
        torch.cuda.empty_cache()

        


        ###########
        #   DPS   #
        ###########


        DeepPS = DPS(input_dim = ndim, degree = 3, num_knots = nk, num_neurons = hidden_config, num_bsl = nbl, dropout = 0.0, output_dim = Fout, knots_place = kp, bias = True).to(device)
        DeepPS.load_state_dict(torch.load(save_path_bs, weights_only=True))

        optimizer = torch.optim.Adam(DeepPS.parameters(), lr=args.fine_tune_lr)

        save_path_ps = f"best_DPS_model_d{d+1}.pt"
        early_stop = EarlyStopping(patience=50, verbose=False, delta=1e-4, path=save_path_ps)

        trainer = Trainer(DeepPS, optimizer, criterion, device, early_stopping=early_stop)
        trainer.fit(train_loader, val_loader, epochs=args.fine_tune_nepochs,
        	penalty_func = spline_penalty_loss,
        	lambda_vals = Lambdalist[str(d+1)])

        with torch.no_grad():
            DeepPS = DPS(input_dim = ndim, degree = 3, num_knots = nk, num_neurons = hidden_config, num_bsl = nbl, dropout = 0.0, output_dim = Fout, knots_place = kp, bias = True).to(device)
            DeepPS.load_state_dict(torch.load(save_path_ps, weights_only=True))
            DeepPS.eval()

            score_ps = criterion(y_test, DeepPS(X_test)).item()
            Pres[d] = score_ps

            del DeepPS     
        torch.cuda.empty_cache()

    result['DeepBS'] = Bres	
    result['DeepPS'] = Pres


    print(np.mean(result['DeepBS']), np.std(result['DeepBS']))
    print(np.mean(result['DeepPS']), np.std(result['DeepPS']))
    
