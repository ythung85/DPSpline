from argparse import ArgumentParser
from sklearn.preprocessing import SplineTransformer
from collections import OrderedDict
from torch import nn
import torch
import numpy as np
import pickle


parser = ArgumentParser()

# Basic Setting
parser.add_argument('--trainsize', default = 400, type = int, help = 'training data size')
parser.add_argument('--testsize', default = 400, type = int, help = 'testing data size')
parser.add_argument('--data', type = str, help = 'simulated data type')
parser.add_argument('--Fin', default = 2, type = int, help = 'input dimension')
parser.add_argument('--Fout', default = 1, type = int, help = 'output dimension')

# Neural Architecture
parser.add_argument('--nk', default = 10, type = int, help = 'number of knot')
parser.add_argument('--nm', default = 50, type = int, help = 'number of neuron')
parser.add_argument('--nl', default = 1, type= int, help = 'number of spline-layer')
parser.add_argument('--dp', default = 0.0, type = float, help = 'dropout percentage')

# Training Setting
parser.add_argument('--nepoch', default = 1000, type = int, help = 'total number of training epochs')
parser.add_argument('--rep', default = 1, type = int, help = 'Number of different initialization used to train the model')
parser.add_argument('--lr', default = 1e-3, type = float, help = 'initial learning rate')
parser.add_argument('--fine_tune_epoch', default = 1000, type = int, help = 'total number of fine tuning epochs')

args = parser.parse_args()

def sim_data(n, dim, Type):
    if Type == 'A':
        X = torch.rand((n,dim))
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

def norm(x):
    return (x-torch.min(x))/(torch.max(x)-torch.min(x))



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

        sqr_xi /= num_neurons
        sqr_sig /= (num_neurons*size)

        ls_lambda[l] = (sqr_sig/sqr_xi).item()
        
        del Cov_a, Cov_e, flatB
    
    return ls_lambda
    
def ECM_layersise_update(model, par, Lambda, x, y):
    criterion = torch.nn.MSELoss(reduction='mean')
    model.eval()
    '''
    with torch.no_grad():
        DSy = model(x)
        print('Training Error: ', np.round(criterion(y, DSy.detach()).item(), 5))
    '''

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

    
    with torch.no_grad():
        DPSy = model(x)
        #Update_Train_Loss = np.round(criterion(y, DPSy.detach()).item(), 5)
        GCV = np.round((torch.norm(y - DPSy)/(Size[-1]-torch.trace(Project_matrix))).item(), 5)
    
    return model, GCV

def ECM_update(model, max_iter, x, y):
    BestGCV = 9999
    patient = 5
    pcount = 0
    for _ in range(max_iter):
        _ = model(x)
        ECM_para = model.get_para_ecm(x)
        ECM_Lambda = ECM(ECM_para, initial_xi = 1, initial_sigma = 1, initial_lambda = 1e-4)

        model, GCV = ECM_layersise_update(model, ECM_para, ECM_Lambda, x, y)

        if GCV < BestGCV:
            BestLambda = ECM_Lambda
            BestGCV = GCV
            pcount = 0
        else:
            pcount += 1

        if pcount == patient:
            break

        del ECM_para, ECM_Lambda
    
    return BestLambda

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
        basis_output = spl.fit_transform(x.cpu().detach().numpy())
        return basis_output
            
    def forward(self, x):
        batch_size, num_features = x.size()
        device = x.device
        
        # Create knot vector and apply B-spline basis functions for each feature

        '''
        knots = torch.cat([
                        torch.zeros(self.degree),               # Add repeated values at the start for clamping
                        torch.linspace(0, 1, self.num_knots - self.degree + 1),  # Uniform knot spacing in the middle
                        torch.ones(self.degree)                 # Add repeated values at the end for clamping
                    ]).to(device)

        # Apply B-spline basis functions for each feature

        basises = []
        
        
        for feature in range(num_features):
            # Calculate B-spline basis functions for this feature
            basis = torch.stack([self.basis_function(x[:, feature], i, self.degree, knots) 
                                 for i in range(self.num_knots)], dim=-1)
            basises.append(basis)
            
        '''
    
        basises = []
        knots = self.knots_distribution(self.degree, self.num_knots)
        #knots = knots.to(device)
        spl = SplineTransformer(n_knots=self.num_knots, degree=self.degree, knots = knots)

        
        for feature in range(num_features):
            # Calculate B-spline basis functions for this feature
            
            basis = self.basis_function2(x[:, feature].reshape(-1,1), spl)
            basis = torch.Tensor(basis).to(device)
            basises.append(basis)
        
        
        if num_features == 1:
            tout = basises[0] @ self.control_p
            self.inter['basic'] = basises[0].T
        else:
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
            f'block_{i}': block(degree = degree, num_knots = num_knots, num_neurons = num_neurons, dropout = dropout, bias = bias)
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
        #self.nm1 = NormLayer() 
        #self.sp1 = BSL(degree = degree, num_knots = num_knots, num_neurons = num_neurons, bias = True)
        self.Spline_block = StackBS_block(BSpline_block, degree = degree, num_knots = num_knots, num_neurons = num_neurons, num_blocks = num_bsl, dropout = dropout)
        self.ln2 = nn.Linear(num_neurons, output_dim)
        #self.inter = {}
        
    def forward(self, x):
        
        x = self.ln1(x)
        #x = self.nm1(x)
        # # # # # # # # # # # # # #
        #          SPLINE         #
        # # # # # # # # # # # # # #
        
        spout = self.Spline_block(x)

        '''  
        ln1out = self.nm1(ln1out)
        device = ln1out.device
        batch_size, _ = x.size()
        
        # # # # # # # # # # # # # #
        #          SPLINE         #
        # # # # # # # # # # # # # #
        
        sp1out = self.sp1(ln1out)
        bslist = self.sp1.inter['basic']
        
        self.inter['ebasic'] = bslist
        self.inter['basic'] = sp1out
        '''
        
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
        del bs_block_out, bs_spline_weight, bs_spline_value, bs_spline_bias
        
        return ecm_para

    def fit(self, x):
        return 0


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.MSELoss(reduction='mean') 
    ntrain = args.trainsize
    ntest = args.testsize
    Dtype = args.data
    ndim = args.Fin
    learning_rate = args.lr
    nepoch = args.nepoch
    fine_tune_epoch = args.fine_tune_epoch

    ndf = args.rep
    nm = args.nm
    nk = args.nk 
    nl = args.nl   
    Fout = args.Fout
    dp = args.dp
    data = {}

    for d in range(ndf):
        torch.manual_seed(d)

        X_train, y_train = sim_data(ntrain, ndim, Dtype)
        X_train, y_train = X_train.to(device), y_train.to(device) 
        X_val, y_val = sim_data(100, ndim, Dtype)
        X_val, y_val = X_val.to(device), y_val.to(device)
        X_test, y_test = sim_data(ntest, ndim, Dtype)
        X_test, y_test = X_test.to(device), y_test.to(device) 

        epstrain = torch.normal(0, torch.var(y_train)*0.01, size=y_train.size()).to(device)
        epstest = torch.normal(0, torch.var(y_test)*0.01, size=y_test.size()).to(device)
        epsval = torch.normal(0, torch.var(y_val)*0.01, size=y_val.size()).to(device)

        y_train, y_test, y_val = y_train + epstrain, y_test + epstest, y_val + epsval
        data[str(d+1)] = {'TrainX': X_train, 'Trainy': y_train, 'TestX': X_test, 'Testy': y_test, 'ValX': X_val, 'Valy': y_val}


    result = {}
    Lambdalist = {}
    Bres = np.zeros((ndf))
    Pres = np.zeros((ndf))

    for d in range(ndf):
        print('dataset: ', str(d+1))
        X_train = data[str(d+1)]['TrainX']; X_test = data[str(d+1)]['TestX']
        y_train = data[str(d+1)]['Trainy']; y_test = data[str(d+1)]['Testy']

        
        conv = False

        while not conv:
            DeepBS = DPS(input_dim = ndim, degree = 3, num_knots = nk, num_neurons = nm, num_bsl = nl, dropout = dp, output_dim = Fout, bias = True).to(device)
            optimizer = torch.optim.Adam(DeepBS.parameters(), lr=learning_rate)
            bloss_list = []; tor = 1e-5; lr_tor = 1e-6
            patientc = 30; patientr = 10; tpat = 0; bloss = 9999
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

            for t in range(nepoch):
                # Forward pass: Compute predicted y by passing x to the modelsp
                pyb_af = DeepBS(X_train)
                loss = criterion(y_train, pyb_af); bloss_list.append(loss.item())
                scheduler.step()

                if (t > 0) and ((bloss_list[t-1]-bloss_list[t])<tor):        
                    if tpat < patientc:
                        tpat += 1
                        pass
                    else:
                        if t < patientc + 1:
                            conv = False
                        else:
                            conv = True
                        print('Convergence!')
                        break
                    
                else:
                    if loss < bloss:
                        print('Current loss: ', loss.item(), ' | , previous best loss: ', bloss, ' | saving best model ...')
                        torch.save(DeepBS.state_dict(), './EXA'+str(X_train.size()[0])+'h'+str(nm)+'k'+str(nk)+'data'+str(d+1))
                        bloss = loss.item()
                        tpat = 0
                    tpat += 1
            
                if tpat == patientc:
                    if t < patientc + 1:
                        conv = False
                    else:
                        conv = True
                        print('Convergence!')
                    break

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if conv:
                break
    
        ## ECM -> Find optimal Lambda
        with torch.no_grad():
            model = DPS(input_dim = ndim, degree = 3, num_knots = nk, num_neurons = nm, num_bsl = nl, dropout = dp, output_dim = Fout, bias = True).to(device)
            model.load_state_dict(torch.load( './EXA'+str(X_train.size()[0])+'h'+str(nm)+'k'+str(nk)+'data'+str(d+1), weights_only = True))
            BMSPE = criterion(y_test, model(X_test).detach()).item()
            print(BMSPE)
            Bres[d] = BMSPE
            BestLambda = ECM_update(model, 10, X_train, y_train)
            Lambdalist[str(d+1)] = BestLambda

    result['DeepBS'] = Bres

    ## DPS fine-tuning
    print('Start runing fast-tuning ...')


    for d in range(ndf):
        print('Dataset '+str(d+1))
        DeepPS = DPS(input_dim = ndim, degree = 3, num_knots = nk, num_neurons = nm, num_bsl = nl, dropout = dp, output_dim = Fout, bias = True).to(device)
        DeepPS.load_state_dict(torch.load( './EXA'+str(X_train.size()[0])+'h'+str(nm)+'k'+str(nk)+'data'+str(d+1), weights_only = True))
        optimizer = torch.optim.Adam(DeepPS.parameters(), lr= 1e-3)
        n = X_train.size()[0]
        best_model_path = 'Best_DPS_d'+str(d+1)+'.pt'
        early_stopping = EarlyStopping(patience=30, verbose=False, delta=1e-3, path= best_model_path)

        ## Access to the Weight matrix for spline
        DPS_Wstack = []
        for l in range(nl):    
            block = getattr(model.Spline_block.model, f'block_{l}')
            DPS_Wstack.append(getattr(block.block.BSL, 'control_p').data)

        for t in range(1, fine_tune_epoch):
                                                
            # Forward pass: Compute predicted y by passing x to the modelsp
            pyb_af = DeepPS(X_train)
            loss = criterion(y_train, pyb_af)
            
            for l in range(nl):
                loss += (BestLambda[l]/n * torch.norm(diag_mat_weights(DPS_Wstack[l].size()[0]).to(device) @ DPS_Wstack[l]))
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            early_stopping(loss, DeepPS)

            if early_stopping.early_stop:
                print("Early stopping triggered. Restoring best model...")        
                break
                
        with torch.no_grad():
            DeepPS = DPS(input_dim = ndim, degree = 3, num_knots = nk, num_neurons = nm, num_bsl = nl, dropout = dp, output_dim = Fout, bias = True).to(device)
            DeepPS.load_state_dict(torch.load(best_model_path, weights_only = True))

            PMSPE = criterion(y_test, DeepPS(X_test).detach()).item()
            Pres[d] = PMSPE

    result['DeepPS'] = Pres
	
    with open('simulation_output.pkl', 'wb') as file:
        pickle.dump(result, file)


    print('Result for B/P: \n')
    print('Number of Dataset: ', ndf)
    print('| DeepBS | Means: ', result['DeepBS'].mean(),' | Std: ',result['DeepBS'].std())
    print('| DeepPS | Means: ', result['DeepPS'].mean(),' | Std: ',result['DeepPS'].std())