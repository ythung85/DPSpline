from sklearn.model_selection import train_test_split
from sklearn.preprocessing import SplineTransformer
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from torch import nn
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from collections import Counter
import glob
import cv2
import os


def proc_brain(imgdir, w, h):

    WIDTH, HEIGHT = w, h
    
    x = []
    for i in range(len(imgdir)):
        # Read and resize image
        full_size_image = cv2.imread(imgdir[i])
        x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC)/255.0) 

    return x

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
        
        knots = torch.cat([torch.linspace(0, 1, nk-2)          # Add repeated values at the end for clamping
            ]).view(-1,1)

        return knots
    
    def basis_function2(self, x, spl):
        basis_output = spl.fit_transform(x.cpu().numpy())
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
            f'block_{i}': block(degree = degree, num_knots = num_knots, num_neurons = num_neurons)
            for i in range(num_blocks)
        })

    def forward(self, x):
        for name, block in self.model.items():
            x = block(x)
        return x

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    



if __name__ == "__main__":
	train1 = glob.glob('/Users/a080528/Downloads/chest_xray/train/PNEUMONIA/*.jpeg')
	train2 = glob.glob('/Users/a080528/Downloads/chest_xray/train/NORMAL/*.jpeg')

	test1 = glob.glob('/Users/a080528/Downloads/chest_xray/test/PNEUMONIA/*.jpeg')
	test2 = glob.glob('/Users/a080528/Downloads/chest_xray/test/NORMAL/*.jpeg')

	train = [train1, train2]; test = [test1, test2]

	trainx = []
	for f in train:
	    x = proc_brain(f, 224, 224)
	    trainx.append(x)

	testx = []
	for f in test:
	    x = proc_brain(f, 224, 224)
	    testx.append(x)

	Xtraind = np.concatenate((np.array(trainx[0]), np.array(trainx[1])))
	y_train = np.array([1]*len(trainx[0]))
	y_train = np.concatenate((y_train, [0]*len(trainx[1])))
	Xtestd = np.concatenate((np.array(testx[0]), np.array(testx[1])))
	y_test = np.array([1]*len(testx[0]))
	y_test = np.concatenate((y_test, [0]*len(testx[1])))


	trainsize = 1000; testsize = 300

	np.random.seed(42)
	trainid = np.random.choice(len(Xtraind), trainsize)
	testid = np.random.choice(len(Xtestd), testsize)

	print(f"(Training) Number of glioma image: {Counter(y_train[trainid])[0]} |  Number of non-tumor image: {Counter(y_train[trainid])[1]} ")
	print(f" (Testing) Number of glioma image: {Counter(y_test[testid])[0]} |  Number of non-tumor image: {Counter(y_test[testid])[1]} ")

	X_train = torch.Tensor(Xtraind[trainid]).permute(0, 3, 1, 2); y_train = torch.Tensor(y_train[trainid]).type(torch.LongTensor)
	X_test = torch.Tensor(Xtestd[testid]).permute(0, 3, 1, 2); y_test = torch.Tensor(y_test[testid]).type(torch.LongTensor)


	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	"""
	Model setting:

	`device`: running the program with cpu or gpu
	`tmc`: the classifier that equip with DNN-S 
	`nm` : number of neuron in DNN-S
	`nk` : number of knot in DNN-S
	`patientc` : (early-stop crierion) If the model didn't improve in n epoch then stop.
	`patientr` : If the model didn't improve in n epoch then decrease learning rate with specific factor.

	"""
	import torchvision

	model = torchvision.models.resnet50(pretrained=True).to(device)
    
	for param in model.parameters():
	    param.requires_grad = False   
	  
	'''
	model.fc = nn.Sequential(
	               nn.Linear(2048, nm),
	                #StackBS_block(BSpline_block, degree = dg, num_knots = nk, num_neurons = nm, num_blocks = nl, dropout = 0.0),
	                nn.Linear(nm, 10)).to(device)
	'''
	dg = 3; nm = 50; nk = 10; doutput = 2; nl = 1; Iteration = 10000

	num_ftrs = model.fc.in_features
	model.fc = torch.nn.Linear(num_ftrs, 1024)
	model.fc = nn.Sequential(
	    torch.nn.Dropout(0.5),
	    torch.nn.Linear(num_ftrs, 1024),
	    torch.nn.Dropout(0.2),
	    torch.nn.Linear(1024, 512),
	    torch.nn.Dropout(0.2),
	    torch.nn.Linear(512, 256),
	    torch.nn.Dropout(0.2),
	    torch.nn.Linear(256, nm),
	    torch.nn.Dropout(0.2),
	    StackBS_block(BSpline_block, degree = dg, num_knots = nk, num_neurons = nm, num_blocks = nl, dropout = 0.0),
	    torch.nn.Linear(nm, doutput),
	).to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

	dataset = TensorDataset(X_train, y_train)
	traindataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
	dataset = TensorDataset(X_test, y_test)
	testdataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

	# Iterate through the dataloader
	for batch_data, batch_labels in dataloader:
	    model.train()
	    for batch_idx, (data, target) in enumerate(traindataloader):
	        data, target = data.to(device), target.to(device)
	        optimizer.zero_grad()
	        output = model(data)
	        loss = criterion(output, target)
	        loss.backward()
	        optimizer.step()
	        if batch_idx % 20 == 0:
	            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
	                Iteration, batch_idx * len(data), len(train_loader.dataset),
	                100. * batch_idx / len(train_loader), loss.item()))