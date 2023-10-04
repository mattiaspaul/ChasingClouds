import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchinfo
from torch.utils.checkpoint import checkpoint
import time
#import matplotlib.pyplot as plt

from torch.autograd import Function
from torch.autograd.functional import jacobian
device = 'cpu'

class DiVRoC(Function):
    @staticmethod
    def forward(ctx, input, grid, shape):
        device = input.device
        dtype = input.dtype
        
        output = -jacobian(lambda x: (F.grid_sample(x, grid) - input).pow(2).mul(0.5).sum(), torch.zeros(shape, 
dtype=dtype, device=device))
        
        ctx.save_for_backward(input, grid, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grid, output = ctx.saved_tensors
        
        B, C = input.shape[:2]
        input_dims = input.shape[2:]
        output_dims = grad_output.shape[2:]
    
        y = jacobian(lambda x: F.grid_sample(grad_output.unsqueeze(2).view(B*C, 1, *output_dims), x).mean(), 
grid.unsqueeze(1).repeat(1, C, *([1]*(len(input_dims)+1))).view(B*C, *input_dims, len(input_dims))).view(B, C, 
*input_dims, len(input_dims))
        
        grad_grid = (input.numel()*input.unsqueeze(-1)*y).sum(1)
        
        grad_input = F.grid_sample(grad_output, grid)
        
        return grad_input, grad_grid, None


