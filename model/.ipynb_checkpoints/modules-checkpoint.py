from argparse import ArgumentParser
from torch import nn
import torch
import math
import numpy as np


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