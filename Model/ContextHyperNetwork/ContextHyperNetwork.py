import torch 
import torch.nn as nn 
from torch import Tensor

#from ContextSpectralLayer import ContextSpectralLayer

class ContextSpectralLayer(nn.Module):
    def __init__(self, context_dim, input_dim, out_dim, rank=None):
        super().__init__()
        
        # Dimension spectrale interne
        if rank is None:
            rank = min(context_dim, input_dim)
        
        self.rank = rank
        
        # Bases spectrales partagées
        self.Qc = nn.Parameter(torch.randn(context_dim, rank) )
        self.Qx = nn.Parameter(torch.randn(input_dim, rank))
        
        # Spectres par neurone
        self.Lambda = nn.Parameter(torch.randn(out_dim, rank))

    def forward(self, x, c):
        """
        x : (batch, input_dim)
        c : (batch, context_dim)
        """

        # Projection spectrale
        c_tilde = c @ self.Qc          # (batch, rank)
        x_tilde = x @ self.Qx          # (batch, rank)

        # Interaction mode par mode
        z = c_tilde * x_tilde          # (batch, rank)

        # Mélange multi-neurones
        y = z @ self.Lambda.T          # (batch, out_dim)

        return y
    
class ContextHyperNetwork(nn.Module):
    def __init__(self,layers):
        super().__init__()
        
        self.layers = nn.ModuleList([
            ContextSpectralLayer(layers[0],layers[i],layers[i+1]) for i in range(len(layers)-1)
        ])

    def forward(self, x):
        c = x 
        for layer in self.layers:
            x = layer(x, c)
        return x 