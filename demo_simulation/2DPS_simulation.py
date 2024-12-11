from argparse import ArgumentParser
from torch import nn
import torch
import math
import numpy as np


parser = ArgumentParser()
parser.add_argument('--trainsize', type = int)
parser.add_argument('--testsize', type = int)
parser.add_argument('--data', type = str)
parser.add_argument('--Fin', type = int)
parser.add_argument('--Fout', type = int)
parser.add_argument('--nk', type = int)
parser.add_argument('--nm', type = int)
parser.add_argument('--rep', type = int)
parser.add_argument('--lr', type = float)

args = parser.parse_args()

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
	
def norm(x):
	return (x-torch.min(x))/(torch.max(x)-torch.min(x))

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


class PRODBSplineLayerMultiFeature(nn.Module):
	def __init__(self, input_dim, degree, num_knots, output_dim, num_neurons, bias = True):
		super(PRODBSplineLayerMultiFeature, self).__init__()
		self.degree = degree
		self.num_knots = num_knots
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.num_neurons = num_neurons
		
		if input_dim == 2:
			self.control_p = nn.Parameter(torch.randn(self.num_knots**2, self.output_dim))
		else:
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
	
	def forward(self, x):
		batch_size, num_features = x.size()
		device = x.device
		
		# Create knot vector
		# knots = torch.linspace(0, 1, self.num_knots + self.degree + 1).to(device)
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

class MPSv3(nn.Module):
    def __init__(self, input_dim, degree, num_knots, num_neurons, output_dim, bias):
        super(MPSv3, self).__init__()
        self.num_neurons = num_neurons
        self.num_knots = num_knots
        self.ln1 = nn.Linear(input_dim, num_neurons)
        self.nm1 = NormLayer() 
        self.sp1 = PRODBSplineLayerMultiFeature(input_dim = 1, degree = degree, num_knots = num_knots, num_neurons = num_neurons, output_dim= output_dim, bias = True)
        self.nm2 = NormLayer() 
        self.sp2 = PRODBSplineLayerMultiFeature(input_dim = 1, degree = degree, num_knots = num_knots, num_neurons = num_neurons, output_dim= output_dim, bias = True)
        self.ln2 = nn.Linear(num_neurons, output_dim)
        self.relu = nn.ReLU()
        self.inter = {}
        
    def forward(self, x):
        ln1out = self.ln1(x)
        ln1out = self.nm1(ln1out)
        
        device = ln1out.device
        batch_size, _ = x.size()
        
        # # # # # # # # # # # # # #
        #         SPLINE 1        #
        # # # # # # # # # # # # # #
        
        sp1out = self.sp1(ln1out)
        bslist = self.sp1.inter['basic']
        
        self.inter['ebasic'] = bslist
        self.inter['basic'] = sp1out

        # # # # # # # # # # # # # #
        #         SPLINE 2        #
        # # # # # # # # # # # # # #
        
        sp2out = self.sp2(ln1out)
        bslist2 = self.sp2.inter['basic']
        
        self.inter['ebasic2'] = bslist2
        self.inter['basic2'] = sp2out

        ln2out = self.ln2(sp2out)
        ln2out = self.relu(ln2out)
        
        return ln2out

def num_para(model):
	tp = 0
	for param in model.parameters():
		tp += param.numel()
	return tp


def ECM(model, num_neurons, num_knots, initial_xi = 1, initial_sigma = 1, initial_lambda = 1e-4, L = None):
    lambdab = initial_lambda
    sigma = initial_sigma
    xi = initial_xi

    if L == 1:
        B = model.inter['ebasic']
        By = model.inter['basic']
        WB = model.sp1.control_p
    else:
        B = model.inter['ebasic2']
        By = model.inter['basic2']
        WB = model.sp2.control_p
        
    DB = diag_mat_weights(WB.size()[0]).to(device)
    size = B.size()[1]
    S = DB.T @ DB
    Cov_a = (xi**2)* torch.linalg.pinv(S)
    Cov_e = torch.eye(size*num_neurons)* sigma
    
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
    
    sqr_xi /= num_neurons
    sqr_sig /= (num_neurons*size)
    
    Lambda = sqr_sig/sqr_xi
    
    return Lambda.item()



if __name__ == "__main__":

	ntrain = args.trainsize
	ntest = args.testsize
	Dtype = args.data
	ndim = args.Fin
	learning_rate = args.lr
	ndf = args.rep
	nm = args.nm
	nk = args.nk    
	Fout = args.Fout
	data = {}

	for d in range(ndf):
		torch.manual_seed(d)
		X_train, y_train = sim_data(ntrain, ndim, Dtype)
		X_test, y_test = sim_data(ntest, ndim, Dtype)
		epstrain = torch.normal(0, 0.1, size=y_train.size())
		epstest = torch.normal(0, 0.1, size=y_test.size())

		y_train, y_test = y_train + epstrain, y_test + epstest
		data[str(d+1)] = {'TrainX': X_train, 'Trainy': y_train, 'TestX': X_test, 'Testy': y_test}


	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	criterion = torch.nn.MSELoss(reduction='mean')

	result = {}
	Lambdalist = {}
	Bres = np.zeros((ndf, 1))
	Pres = np.zeros((ndf, 1))

	for d in range(ndf):
		print('dataset: ', str(d+1))
		X_train = data[str(d+1)]['TrainX']; X_test = data[str(d+1)]['TestX']
		y_train = data[str(d+1)]['Trainy']; y_test = data[str(d+1)]['Testy']

		
		conv = False

		while not conv:
			MBS = MPSv3(input_dim = ndim, degree = 3, num_knots = nk, num_neurons = nm, output_dim = Fout, bias = True).to(device)
			learning_r = 1e-1
			#criterion = torch.nn.MSELoss(reduction='mean')
			optimizer = torch.optim.Adam(MBS.parameters(), lr=learning_r)
			Iteration = 10000; bloss_list = []; tor = 1e-5; lr_tor = 1e-6
			patientc = 10; patientr = 5; tpat = 0; bloss = 9999

			for t in range(Iteration):
				# Forward pass: Compute predicted y by passing x to the modelsp
				pyb_af = MBS(X_train)
				loss = criterion(y_train, pyb_af); bloss_list.append(loss.item())
				
				if (t > 0) and ((bloss_list[t-1]-bloss_list[t])<tor):        
					if (tpat % patientr) == 0:
						learning_r *= 0.2 
						tpat += 1
						#print('Learning rate reduce to ', learning_r)
						optimizer = torch.optim.Adam(MBS.parameters(), lr=learning_r)
						if learning_r <= lr_tor:
							if t < patientc + 1:
								conv = False
							else:
								conv = True
							print('Convergence!')
							break
					elif tpat < patientc:
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
						#print('Current loss: ', loss.item(), ' | , previous best loss: ', bloss, ' | saving best model ...')
						torch.save(MBS.state_dict(), './EXA'+str(X_train.size()[0])+'h'+str(nm)+'k'+str(nk)+'data'+str(d+1))
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
		
		with torch.no_grad():
		    print('------------------------------------------')
		    print('Before adding penalty ... ')
		    eval_model = MPSv3(input_dim = ndim, degree = 3, num_knots = nk, num_neurons = nm, output_dim = Fout, bias = True).to(device)
		    eval_model.load_state_dict(torch.load('./EXA'+str(X_train.size()[0])+'h'+str(nm)+'k'+str(nk)+'data'+str(d+1), weights_only = True))
		    print('Training Error: ', np.round(criterion(y_train, eval_model(X_train).detach()).item(), 5), ' | Testing Error: ', np.round(criterion(y_test, eval_model(X_test).detach()).item(), 5))
		    print('------------------------------------------')
		    print('After adding penalty ... ')

		    
		    WB = eval_model.sp1.control_p
		    DB = diag_mat_weights(WB.size()[0], 'second').to(device)
		    BestGCV = 9999
		    
		    for i in range(10):
		        MPSy = eval_model(X_train)
		        LambdaB1 = ECM(model = eval_model, num_neurons = nm, num_knots = nk, L = 1)
		        LambdaB2 = ECM(model = eval_model, num_neurons = nm, num_knots = nk, L = 2)
		        
		        B1 = eval_model.inter['ebasic']
		        B2 = eval_model.inter['ebasic2']
		        P2 = (torch.linalg.pinv(B2.T @ B2) @ B2.T @ B2)
		        
		        By1 = eval_model.inter['basic']
		        By2 = eval_model.inter['basic2']
		        
		        size1 = B1.size()[1]
		        size2 = B2.size()[1]

		        B1 = B1.view(nm, nk, size1)
		        B2 = B2.view(nm, nk, size2)

		        NW1 = torch.empty((nk, nm))
		        NW2 = torch.empty((nk, nm))
		        NB1 = torch.empty((nm))
		        NB2 = torch.empty((nm))
		        for i in range(nm):
		            B1y = By1[:,i] - eval_model.sp1.bias.data[i]
		            B2y = By2[:,i] - eval_model.sp2.bias.data[i]

		            BB1 = B1[i].T
		            BB2 = B2[i].T
		            PB1 = (torch.linalg.pinv(BB1.T @ BB1) @ BB1.T @ BB1)
		            PB2 = (torch.linalg.pinv(BB2.T @ BB2) @ BB2.T @ BB2)

		            # Update the weights and bias
		            NW1[:, i] = (torch.inverse(BB1.T @ BB1 + (LambdaB1/size1) * (DB.T @ DB)) @ BB1.T @ B1y)
		            NW2[:, i] = (torch.inverse(BB2.T @ BB2 + (LambdaB2/size2) * (DB.T @ DB)) @ BB2.T @ B2y)
		            NB1[i] = torch.mean(By1[:,i] - (NW1[:,i] @ BB1.T))
		            NB2[i] = torch.mean(By2[:,i] - (NW2[:,i] @ BB2.T))
		            
		        # update the weight
		        getattr(eval_model.sp1, 'control_p').data = NW1
		        getattr(eval_model.sp2, 'control_p').data = NW2
		        getattr(eval_model.sp1, 'bias').data = NB1
		        getattr(eval_model.sp2, 'bias').data = NB2
		        

		        MPSy = eval_model(X_train)
		        trainloss = np.round(criterion(y_train, MPSy.detach()).item(), 5)
		        GCV = np.round((torch.norm(y_train - MPSy)/(size2-torch.trace(P2))).item(), 5)
		        
		        if GCV < BestGCV:
		            BestLambdaB1, BestLambdaB2 = LambdaB1, LambdaB2
		            BestGCV = GCV
		            
		        MPSy = eval_model(X_test)
		        print('Lambda: ', np.round(LambdaB1, 5),' and ', np.round(LambdaB2, 5),'| Training Loss: ', trainloss,'| GCV: ', GCV,' | Testing Error: ', np.round(criterion(y_test, MPSy.detach()).item(), 5))
		        Lambdalist[str(d+1)] = [BestLambdaB1, BestLambdaB2]


		with torch.no_grad():
			eval_model = MPSv3(input_dim = ndim, degree = 3, num_knots = nk, num_neurons = nm, output_dim = Fout, bias = True).to(device)
			eval_model.load_state_dict(torch.load('./EXA'+str(X_train.size()[0])+'h'+str(nm)+'k'+str(nk)+'data'+str(d+1), weights_only = True))
			BMSPE = criterion(y_test, eval_model(X_test).detach()).item()
			print(BMSPE)
			Bres[d, 0] = BMSPE

	result['MBS'] = Bres


	print('Start runing fast-tuning ...')

	Fast_tun_epoch = 1001
	for d in range(ndf):
		print('Dataset '+str(d+1))
		eval_model = MPSv3(input_dim = ndim, degree = 3, num_knots = nk, num_neurons = nm, output_dim = Fout, bias = True).to(device)
		eval_model.load_state_dict(torch.load( './EXA'+str(X_train.size()[0])+'h'+str(nm)+'k'+str(nk)+'data'+str(d+1), weights_only = True))
		optimizer = torch.optim.Adam(eval_model.parameters(), lr= learning_rate)
		n = X_train.size()[0]
		
		LambdaB1, LambdaB2 = Lambdalist[str(d+1)][0], Lambdalist[str(d+1)][1]
		
		for t in range(1, Fast_tun_epoch):
											   
			# Forward pass: Compute predicted y by passing x to the modelsp
			pyb_af = eval_model(X_train)
			WB1 = eval_model.sp1.control_p.data; WB2 = eval_model.sp2.control_p.data
			DB1 = diag_mat_weights(WB1.size()[0]).to(device); DB2 = diag_mat_weights(WB2.size()[0]).to(device)


			loss = criterion(y_train, pyb_af) + (LambdaB1/n) * torch.norm(DB1 @ WB1) + (LambdaB2/n) * torch.norm(DB2 @ WB2)
		
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		

		with torch.no_grad():
			PMSPE = criterion(y_test, eval_model(X_test).detach()).item()
			Pres[d, 0] = PMSPE

	result['DPS'] = Pres

	np.save('repsim.npy', result, allow_pickle = True)

	print('Result for B/P: \n')
	print('Number of Dataset: ', ndf)
	print('| MBS | Means: ', result['MBS'].mean(),' | Std: ',result['MBS'].std())
	print('| DPS | Means: ', result['DPS'].mean(),' | Std: ',result['DPS'].std())















