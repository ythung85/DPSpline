'''
Command for runing this DPS.py - May 8th 
python3 DPS-Double-Descent.py --data A --nk 10 --nm 10 --rep 1 --nbl 1

'''

from argparse import ArgumentParser
from torch import nn
import torch
import math
import numpy as np
from sklearn.preprocessing import SplineTransformer
from collections import OrderedDict


parser = ArgumentParser()
# Basic Setting
parser.add_argument('--trainsize', default = 400, type = int, help = 'training data size')
parser.add_argument('--testsize', default = 400, type = int, help = 'testing data size')
parser.add_argument('--data', type = str, help = 'simulated data type')
parser.add_argument('--Fin', default = 2, type = int, help = 'input dimension')
parser.add_argument('--Fout', default = 1, type = int, help = 'output dimension')

# Neural Architecture
parser.add_argument('--nk', default = 10, type = int, help = 'number of knot')
parser.add_argument('--nbl', default = 1, type = int)
parser.add_argument('--nm', default = 50, type = int, help = 'number of neuron')
parser.add_argument('--nl', default = 1, type= int, help = 'number of spline-layer')
parser.add_argument('--dp', default = 0.0, type = float, help = 'dropout percentage')

# Training Setting
parser.add_argument('--nepochs', default = 10000, type = int, help = 'total number of training epochs')
parser.add_argument('--rep', default = 1, type = int, help = 'Number of different initialization used to train the model')
parser.add_argument('--lr', default = 1e-1, type = float, help = 'initial learning rate')
parser.add_argument('--fine_tune_epoch', default = 10000, type = int, help = 'total number of fine tuning epochs')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################################################################
#                       Data preprocessing                       #
##################################################################
def sim_data(n, dim, Type):
	if Type == 'A':
		X = torch.rand((n,2))
		y = torch.exp(2*torch.sin(X[:,0]*0.5*torch.pi)+ 0.5*torch.cos(X[:,1]*2.5*torch.pi))
		y = y.reshape(-1,1)
		y = y.float()
		
	elif Type == 'B':
		X = torch.rand((n, dim))
		y = 1
		for d in range(dim):
			a = (d+1)/2
			y *= ((torch.abs(4*X[:,d]-2)+a)/(1+a))
		y = y.reshape(-1,1)
		y = y.float()
	else:
		pass
	
	return X, y

def diag_mat_weights(dimp, type = 'first'):
	if type == 'first':
		dg = np.zeros((dimp-1, dimp))
		for i in range(dimp-1):
			dg[i,i] = -1
			dg[i,i+1]= 1
	elif type == 'second':
		dg = np.zeros((dimp-2, dimp))
		for i in range(dimp-2):
			dg[i,i] = -1
			dg[i,i+1]= 2
			dg[i,i+2]= -1
	else:
		pass
	return torch.Tensor(dg)

def num_para(model):
	tp = 0
	for param in model.parameters():
		tp += param.numel()
	return tp
    
##################################################################
#                          DPS Modeling                          #
##################################################################

class BSL(nn.Module):
    def __init__(self, degree, num_knots, num_neurons, bias = True):
        super(BSL, self).__init__()
        self.degree = degree
        self.num_knots = num_knots
        self.num_neurons = num_neurons
        self.control_p = nn.Parameter(torch.randn(self.num_knots, self.num_neurons))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(self.num_neurons))
        else:
            self.register_parameter('bias', None)
            
        self.inter = {}
    
    def basis_function(self, x, i, k, t):
    
        # Base case: degree 0 spline
        if k == 0:
            return ((t[i] <= x) & (x < t[i + 1])).float()
    
        # Recursive case
        denom1 = t[i + k] - t[i]
        denom2 = t[i + k + 1] - t[i + 1]
    
        term1 = 0
        if denom1 != 0:
            term1 = (x - t[i]) / denom1 * self.basis_function(x, i, k - 1, t)
    
        term2 = 0
        if denom2 != 0:
            term2 = (t[i + k + 1] - x) / denom2 * self.basis_function(x, i + 1, k - 1, t)
    
        return term1 + term2

    def knots_distribution(self, dg, nk):

        knots = torch.cat([torch.linspace(-0.002, -0.001, steps=dg),            # Add repeated values at the start for clamping
            torch.linspace(0, 1, nk-2*dg-2),  # Uniform knot spacing in the middle
            torch.linspace(1.001, 1.002, steps=dg)           # Add repeated values at the end for clamping
            ]).view(-1,1)
     
        return knots
    
    def basis_function2(self, x, spl):
        basis_output = spl.fit_transform(x.cpu().numpy())
        return basis_output
            
    def forward(self, x):
        batch_size, num_features = x.size()
        device = x.device
        
        # Create knot vector and apply B-spline basis functions for each feature
    
        basises = []
        knots = self.knots_distribution(self.degree, self.num_knots)
        #knots = knots.to(device)
        spl = SplineTransformer(n_knots=self.num_knots, degree=self.degree, knots = knots)

        
        for feature in range(num_features):
            # Calculate B-spline basis functions for this feature
            
            basis = self.basis_function2(x[:, feature].reshape(-1,1), spl)
            basis = torch.Tensor(basis).to(device)
            basises.append(basis)
        
        self.inter['basic'] = torch.reshape(torch.stack(basises, dim = 1), (batch_size, self.num_knots * self.num_neurons)).T
        basises = torch.stack(basises)
        tout = basises.permute(1,2,0) * self.control_p
        tout = tout.sum(dim =1)
        if self.bias is not None:
            tout += self.bias        
            
        return tout


class NormLayer(nn.Module):
    def __init__(self):
        super(NormLayer, self).__init__()

    def forward(self, x):
        min_val = torch.min(x, axis = 1).values.reshape(-1,1)
        max_val = torch.max(x, axis = 1).values.reshape(-1,1)

        x = (x - min_val)/(max_val - min_val)  # Rescale to [0, 1]
        return x.detach()
        
class BSpline_block(nn.Module):
    def __init__(self, degree, num_knots, num_neurons, dropout = 0.0, bias = True):
        super(BSpline_block, self).__init__()

        self.block = nn.Sequential(OrderedDict([
            ('norm', NormLayer()),
            ('BSL', BSL(degree = degree, num_knots = num_knots, num_neurons = num_neurons, bias = bias)),
            ('drop', nn.Dropout(dropout)),
        ]))
        
    def forward(self, x):
        return self.block(x)
        
class StackBS_block(nn.Module):
    def __init__(self, block, degree, num_knots, num_neurons, num_blocks, dropout = 0.0, bias = True):
        super().__init__()
        self.model = nn.ModuleDict({
            f'block_{i}': block(degree = degree, num_knots = num_knots, num_neurons = num_neurons)
            for i in range(num_blocks)
        })

    def forward(self, x):
        for name, block in self.model.items():
            x = block(x)
        return x


class DPS(nn.Module):
    def __init__(self, input_dim, degree, num_knots, num_neurons, num_bsl, dropout, output_dim, bias):
        super(DPS, self).__init__()
        self.num_neurons = num_neurons
        self.num_knots = num_knots
        self.ln1 = nn.Linear(input_dim, num_neurons)
        self.Spline_block = StackBS_block(BSpline_block, degree = degree, num_knots = num_knots, num_neurons = num_neurons, num_blocks = num_bsl, dropout = dropout)
        self.ln2 = nn.Linear(num_neurons, output_dim)
        
    def forward(self, x):
        
        x = self.ln1(x)
        spout = self.Spline_block(x)
        output = self.ln2(spout)
        
        return output

    def get_para_ecm(self, x):

        '''
        ecm_para: A dictionary that collects the parameter we need to the following ECM algorithm.
        ecm_para.basic: Store the output of each B-Spline block; Dimension = [n_sample, n_neurons]
        ecm_para.ebasic Store the weight matrix of each B-Spline expansion; Dimension = [n_knots * n_neurons, n_sample]

        '''
        ecm_para = {}
        bs_block_out = {}
        bs_spline_weight = {}
        bs_spline_value = {}
        bs_spline_bias = {}

        _ = self(x)
        
        def get_activation(name):
            def hook(model, input, output):
                bs_block_out[name] = output.detach()
            return hook

        handles = []
        for name, layer in self.named_modules():
            if 'block.drop' in name:
                handles.append(layer.register_forward_hook(get_activation(name)))
            elif 'block.BSL' in name:
                bs_spline_value[name] = layer.inter['basic'].detach()
                bs_spline_weight[name] = layer.control_p.detach()
                bs_spline_bias[name] = layer.bias.detach()
        # Run forward pass (triggers hooks)
        _ = self(x)

        # Clean up hooks
        for h in handles:
            h.remove()
            
        ecm_para['basic'] = torch.stack(list(bs_block_out.values()), dim=0)
        ecm_para['ebasic'] = torch.stack(list(bs_spline_value.values()), dim=0)
        ecm_para['wbasic'] = torch.stack(list(bs_spline_weight.values()), dim=0)
        ecm_para['bbasic'] = torch.stack(list(bs_spline_bias.values()), dim=0)
        del bs_block_out, bs_spline_weight, bs_spline_value, bs_spline_bias, _
        torch.cuda.empty_cache()
        
        return ecm_para

    def fit(self, x):
        return 0

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss  # because we want to minimize val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
def train_one_epoch(model, x, y, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    return loss

                
def validate(model, x, y, criterion, device):
    model.eval()
    total_loss = 0.0
    x, y = x.to(device), y.to(device)
    outputs = model(x)
    loss = criterion(outputs, y)

    return loss
    
##################################################################
#                          ECM Iteration                         #
##################################################################

def ECM(par, initial_xi = 1, initial_sigma = 1, initial_lambda = 1e-4):
    lambdab = initial_lambda
    sigma = initial_sigma
    xi = initial_xi
    
    n_block, num_knots, num_neurons = par['wbasic'].size()
    ls_lambda = torch.empty(n_block)
    
    for l in range(n_block):
        B = par['ebasic'][l]
        By = par['basic'][l]
        WB = par['wbasic'][l]
        
        DB = diag_mat_weights(WB.size()[0]).to(device)
        size = B.size()[1]
        S = DB.T @ DB
        Cov_a = (xi**2)* torch.linalg.pinv(S)
        Cov_a.to(device)
        Cov_e = (torch.eye(size*num_neurons)* sigma).to(device)
        
        block_y = torch.reshape(By, (-1,1))
        flatB = B.view(num_neurons, num_knots, size)
            
        sqr_xi= 0
        sqr_sig = 0

        for i in range(num_neurons):
            Ncov = (Cov_a -(Cov_a @ flatB[i]) @ (torch.linalg.pinv(flatB[i].T @ Cov_a @ flatB[i] + Cov_e[size*i:size*(i+1),size*i:size*(i+1)]) @ flatB[i].T @ Cov_a))
            Nmu = (Cov_a @ flatB[i]) @ (torch.linalg.pinv(flatB[i].T @ Cov_a @ flatB[i] + Cov_e[size*i:size*(i+1),size*i:size*(i+1)])) @ By[:,i].reshape(-1,1)
            
            first_xi = S @ Ncov
            second_xi = (Nmu.T @ S @ Nmu)
            sqr_xi += torch.trace(first_xi) + second_xi
                
            first_sig = torch.norm(By[:,i])
            second_sig = 2 * (By[:,i] @ flatB[i].T) @ Nmu 
            third_sig = torch.trace((flatB[i] @ flatB[i].T) @ Ncov)
            four_sig = (Nmu.T @ flatB[i] @ flatB[i].T @ Nmu)
            
            sqr_sig += (first_sig + second_sig + third_sig + four_sig)
            
            del first_xi, second_xi, first_sig, second_sig, third_sig, four_sig
            torch.cuda.empty_cache()
            
        sqr_xi /= num_neurons
        sqr_sig /= (num_neurons*size)

        ls_lambda[l] = (sqr_sig/sqr_xi).item()
        
        del Cov_a, Cov_e, flatB
        torch.cuda.empty_cache()
        
    return ls_lambda
    
def ECM_layersise_update(model, par, Lambda, x, y):

    model.eval()
    device = x.device
    
    B_out, B_in, B_w, B_b = par['basic'], par['ebasic'], par['wbasic'], par['bbasic']
    n_layer, nk, nm = B_w.size()
    DB = diag_mat_weights(B_w[0].size()[0], 'second').to(device)

    Project_matrix = (torch.linalg.pinv(B_in[-1].T @ B_in[-1]) @ B_in[-1].T @ B_in[-1])
    Size = [b.size()[1] for b in B_in]

    B_in = B_in.view(n_layer, nm, nk, Size[0])

    for l in range(n_layer):    
        NW = torch.empty((nk, nm)).to(device)
        NB = torch.empty((nm)).to(device)
        
        for i in range(nm):
            B1y = B_out[l][:,i] - B_b[l][i]
            BB = B_in[l][i].T
    
            # Update the weights and bias
            NW[:, i] = (torch.inverse(BB.T @ BB + (Lambda[l]/Size[l]) * (DB.T @ DB)) @ BB.T @ B1y)
            NB[i] = torch.mean(B_out[l][:,i] - (NW[:,i] @ BB.T))
                
        # update the weight
        block = getattr(model.Spline_block.model, f'block_{l}')
        getattr(block.block.BSL, 'control_p').data = NW
        getattr(block.block.BSL, 'bias').data = NB

        del NW, NB, B1y, BB, block
        torch.cuda.empty_cache()
        
    with torch.no_grad():
        DPSy = model(x)
        Update_Train_Loss = np.round(criterion(y, DPSy.detach()).item(), 5)
        GCV = np.round((torch.norm(y - DPSy)/(Size[-1]-torch.trace(Project_matrix))).item(), 5)
    
    return model, GCV

def ECM_update(model, max_iter, x, y):
    BestGCV = prev = 9999
    patient = 10
    pcount = 0
    for i in range(max_iter):
        _ = model(X_train)
        ECM_para = model.get_para_ecm(x)
        ECM_Lambda = ECM(ECM_para, initial_xi = 1, initial_sigma = 1, initial_lambda = 1e-4)

        model, GCV = ECM_layersise_update(model, ECM_para, ECM_Lambda, x, y)

        if np.abs(prev - GCV) < 5e-5:
            print('GCV Converge at ',i+1,' iteration')
            break
            
        if GCV < BestGCV:
            BestLambda = ECM_Lambda
            BestGCV = GCV
            pcount = 0
        else:
            pcount += 1

        if pcount == patient:
            print('GCV Converge at ',i,' iteration')
            break

        prev = GCV

        del ECM_para, ECM_Lambda
    
    return BestLambda


if __name__ == "__main__":

    ntrain = args.trainsize
    ntest = args.testsize
    Dtype = args.data
    ndim = args.Fin
    learning_rate = args.lr
    ndf = args.rep
    nl = args.nbl
    nm = args.nm
    nk = args.nk    
    Fout = args.Fout
    nepoch = args.nepochs
    fine_tune_epoch = args.fine_tune_epoch
    data = {}

    knot_list = [10, 20, 30, 40, 50, 60, 70, 80]

    # Prepare the data #

    for d in range(ndf):
        torch.manual_seed(d)
        X_train, y_train = sim_data(ntrain, ndim, Dtype)
        X_test, y_test = sim_data(ntest, ndim, Dtype)
        X_val, y_val = sim_data(100, ndim, Dtype)
        epstrain = torch.normal(0, 0.1, size=y_train.size())
        epstest = torch.normal(0, 0.1, size=y_test.size())
        epsval = torch.normal(0, 0.1, size=y_val.size())

        y_train, y_test = y_train + epstrain, y_test + epstest; y_val = y_val + epsval
        data[str(d+1)] = {'TrainX': X_train, 'Trainy': y_train, 'TestX': X_test, 'Testy': y_test, 'ValX': X_val, 'Valy': y_val}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.MSELoss(reduction='mean')


    Bres = np.zeros((len(knot_list)))
    Pres = np.zeros((len(knot_list))) 

    result = {}
    for i in range(len(knot_list)):
        '''
        for d in range(ndf):
            X_train = data[str(d+1)]['TrainX']; X_test = data[str(d+1)]['TestX']; X_val = data[str(d+1)]['ValX'] 
            y_train = data[str(d+1)]['Trainy']; y_test = data[str(d+1)]['Testy']; y_val = data[str(d+1)]['Valy']
        
            DeepBS = DPS(input_dim = ndim, degree = 3, num_knots = knot_list[i], num_neurons = nm, num_bsl = nl, dropout = 0.0, output_dim = Fout, bias = True).to(device)
            print('Knot Number: ', knot_list[i])
            optimizer = torch.optim.Adam(DeepBS.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            best_model_path = "best_DBS_model_k"+str(knot_list[i])+".pt"
            early_stopping = EarlyStopping(patience=30, verbose=False, delta=1e-4, path= best_model_path)
        
            for epoch in range(nepoch):
                optimizer.zero_grad()
                DeepBS.train()
                
                output = DeepBS(X_train)
                loss = criterion(output, y_train)
                val_loss = validate(DeepBS, X_val, y_val, criterion, device)
            
                if epoch % 10 == 0:
                    print(f"Epoch {epoch+1:02d} | Train Loss: {loss:.4f} | " f"Val Loss: {val_loss:.4f}")
                
                early_stopping(val_loss, DeepBS)
                if early_stopping.early_stop:
                    print("Early stopping triggered. Restoring best model...")        
                    break
                loss.backward()
                optimizer.step()

            del DeepBS, output, loss, val_loss
            torch.cuda.empty_cache()

        '''
        print('Loading best DBS model for DPS training and ECM where k = ', knot_list[i])
        model = DPS(input_dim = ndim, degree = 3, num_knots = knot_list[i], num_neurons = nm, num_bsl = nl, dropout = 0.0, output_dim = Fout, bias = True).to(device)
        model.load_state_dict(torch.load('./best_DBS_model_k'+str(knot_list[i])+'.pt', weights_only = True))
        model.eval()
        with torch.no_grad():
            Bres[i] = criterion(model(X_test), y_test)
            BestLambda = ECM_update(model, 20, X_train, y_train)
            
        del model
        torch.cuda.empty_cache()

        print('Start runing fast-tuning ...')

        for d in range(ndf):
            DeepPS = DPS(input_dim = ndim, degree = 3, num_knots = knot_list[i], num_neurons = nm, num_bsl = nl, dropout = 0.0, output_dim = Fout, bias = True).to(device)
            DeepPS.load_state_dict(torch.load('./best_DBS_model_k'+str(knot_list[i])+'.pt', weights_only = True))
            ft_learning_rate = 1e-2
            optimizer = torch.optim.Adam(DeepPS.parameters(), lr= ft_learning_rate)
            best_model_path = "best_DPS_model_k"+str(knot_list[i])+".pt"
            early_stopping = EarlyStopping(patience=50, verbose=False, delta=1e-5, path= best_model_path)
            n = X_train.size()[0]
            BestLambda[0] = 0.6
            for epoch in range(1, fine_tune_epoch):
        
                # Forward pass: Compute predicted y by passing x to the modelsp
                output = DeepPS(X_train)
                loss = criterion(output, y_train)
                
                for l in range(nl):
                    block = getattr(DeepPS.Spline_block.model, f'block_{l}')
                    W = getattr(block.block.BSL, 'control_p')
                    loss += BestLambda[l]/X_train.size()[0] * torch.norm(diag_mat_weights(W.size()[0]).to(device) @ W)

                val_loss = validate(DeepPS, X_val, y_val, criterion, device)
            
                if epoch % 10 == 0:
                    print(f"Epoch {epoch+1:02d} | Train Loss: {loss:.4f} | " f"Val Loss: {val_loss:.4f}")
                
                early_stopping(val_loss, DeepPS)
                if early_stopping.early_stop:
                    print("Early stopping triggered. Restoring best model...")        
                    break
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        model = DPS(input_dim = ndim, degree = 3, num_knots = knot_list[i], num_neurons = nm, num_bsl = nl, dropout = 0.0, output_dim = Fout, bias = True).to(device)
        model.load_state_dict(torch.load('./best_DPS_model_k'+str(knot_list[i])+'.pt', weights_only = True))
        model.eval()
        with torch.no_grad():
            Pres[i] = criterion(model(X_test), y_test)
            
        del model
        torch.cuda.empty_cache()
        
    result['DBS'] = Bres
    result['DPS'] = Pres
    
    
    print('Result for B/P: \n')
    print('Knots Candidate: ', knot_list)
    print('| MBS | Performance: ', result['DBS'])
    print('| DPS | Performance: ', result['DPS'])
    print('End')
    
    #python3 DPS.py --data A --nk 15 --nm 50 --rep 1













