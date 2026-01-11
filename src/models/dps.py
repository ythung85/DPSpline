import torch
import torch.nn as nn
from collections import OrderedDict

# ---------------------------------------------------------
# Helper Function: PyTorch-native B-Spline Basis Calculation
# ---------------------------------------------------------
def b_spline_basis(x, knots, degree):

    x = x.unsqueeze(-1)
    knots = knots.view(1, 1, -1)
    
    basis = ((x >= knots[..., :-1]) & (x < knots[..., 1:])).float()
    
    for d in range(1, degree + 1):

        b_prev = basis
        
        knots_left = knots[..., :-(d+1)]     # t_i
        knots_right = knots[..., d+1:]       # t_{i+d+1}
        
        eps = 1e-6
        
        # Term 1 calculation
        denom1 = knots[..., d:-1] - knots_left
        term1 = ((x - knots_left) / (denom1 + eps)) * b_prev[..., :-1]
        
        # Term 2 calculation
        denom2 = knots_right - knots[..., 1:-d]
        term2 = ((knots_right - x) / (denom2 + eps)) * b_prev[..., 1:]
        
        basis = term1 + term2
        
    return basis # Shape: (B, N, Num_Knots - Degree - 1)

# ---------------------------------------------------------
# Optimized BSL Layer
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
            self.bias = nn.Parameter(torch.randn(1, self.num_neurons))
        else:
            self.register_parameter('bias', None)

        self.register_buffer('knots', self._create_knots(degree, num_knots))
        
        self.inter = {}

    def _create_knots(self, d, k):

        device = torch.device('cpu') # 
        if self.knots_place == 'boundary' or True: 
            mid_knots = torch.linspace(0, 1, k - 2*d).to(device)
            
            left_pad = torch.zeros(d).to(device)
            right_pad = torch.ones(d).to(device)
            knots = torch.cat([left_pad, mid_knots, right_pad])
            
            
        return knots

    def forward(self, x):

        basis_matrix = b_spline_basis(x, self.knots, self.degree)
        
        
        self.inter['basic'] = basis_matrix.reshape(x.shape[0], -1) 
        
        
        tout = (basis_matrix * self.control_p).sum(dim=2) # Result: (B, N)
        
        if self.bias is not None:
            tout += self.bias
            
        return tout

class BSpline_block(nn.Module):
    def __init__(self, degree, num_knots, num_neurons, knots_place, dropout=0.0, bias=True):
        super(BSpline_block, self).__init__()
        
        self.block = nn.Sequential(OrderedDict([
            ('bn', nn.BatchNorm1d(num_neurons)),
            ('sigmoid', nn.Sigmoid()),
            ('BSL', BSL(degree=degree, 
                        num_knots=num_knots, 
                        num_neurons=num_neurons, 
                        knots_place=knots_place, 
                        bias=bias)),

            ('drop', nn.Dropout(dropout)),
        ]))
        
    def forward(self, x):
        return self.block(x)

class StackBS_block(nn.Module):
    def __init__(self, block, degree, num_knots, num_neurons, num_blocks, knots_place, dropout = 0.0, bias = True):
        super().__init__()

        layers = OrderedDict()
        for i in range(num_blocks):
            current_neurons = num_neurons[i]
            
            if i > 0 and num_neurons[i] != num_neurons[i-1]:
                layers[f'dim_match_{i}'] = nn.Linear(num_neurons[i-1], num_neurons[i])
            

            layers[f'block_{i}'] = block(
                degree=degree, 
                num_knots=num_knots, 
                num_neurons=current_neurons, 
                knots_place=knots_place,
                dropout=dropout,
                bias=bias
            )
            
        self.model = nn.Sequential(layers)
            

    def forward(self, x):

        return self.model(x)
    
class DPS(nn.Module):
	def __init__(self, input_dim, degree, num_knots, num_neurons, num_bsl, dropout, output_dim, knots_place, bias):
		super(DPS, self).__init__()
		self.num_neurons = num_neurons
		self.num_knots = num_knots
		self.knots_place = knots_place
		self.ln1 = nn.Linear(input_dim, num_neurons[0])
		self.Spline_block = StackBS_block(
			BSpline_block, 
			degree = degree, 
			num_knots = num_knots, 
			num_neurons = num_neurons, 
			num_blocks = num_bsl, 
			knots_place = knots_place, 
			dropout = dropout,
			bias = bias
		)
		self.ln2 = nn.Linear(num_neurons[-1], output_dim)
		
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
				
		# Run forward pass (triggers hooks)
		_ = self(x)
	
		# Clean up hooks
		for h in handles:
			h.remove()
			
		ecm_para['basic'] = list(bs_block_out.values())
		ecm_para['ebasic'] = list(bs_spline_value.values())
		ecm_para['wbasic'] = list(bs_spline_weight.values())
		ecm_para['bbasic'] = list(bs_spline_bias.values())
		del bs_block_out, bs_spline_weight, bs_spline_value, bs_spline_bias
		
		return ecm_para

	@staticmethod
	def get_predictions(x, threshold=0.5):
		"""
		Converts raw model outputs (logits) into binary class labels.
		"""
		# Apply sigmoid to get probabilities between 0 and 1
		probs = torch.sigmoid(x)
		# Apply threshold: 1 if prob >= 0.5, else 0
		return (probs >= threshold).float()
	
	def fit(self, x):
		return 0
