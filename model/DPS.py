from argparse import ArgumentParser
from torch import nn
import torch
import math
import numpy as np
from .modules import PRODBSplineLayerMultiFeature, NormLayer

class MPSv3(nn.Module):
	def __init__(self, input_dim, degree, num_knots, num_neurons, output_dim, bias):
		super(MPSv3, self).__init__()
		self.num_neurons = num_neurons
		self.num_knots = num_knots
		self.ln1 = nn.Linear(input_dim, num_neurons)
		self.nm1 = NormLayer() 
		self.sp1 = PRODBSplineLayerMultiFeature(input_dim = 1, degree = degree, num_knots = num_knots, num_neurons = num_neurons, output_dim= output_dim, bias = True)
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

		ln2out = self.ln2(sp1out)
		ln2out = self.relu(ln2out)
		
		return ln2out