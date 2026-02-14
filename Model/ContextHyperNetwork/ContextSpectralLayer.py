import torch
import torch.nn as nn

# --- 1. LA RMSNORM (Stabilité sans décentrage) ---
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.gamma

# --- 2. LA COUCHE HYPER-SPECTRALE AVEC RMS ---
class ContextSpectralLayer(nn.Module):
    def __init__(self, context_dim, input_dim,out_dim):
        super().__init__()
        self.input_dim = input_dim
        
        # Générateur de poids (W = c*A + b)
        self.weight_generator =  nn.Parameter(torch.randn(context_dim, input_dim))
        
        # Mélange spectral (Lambda)
        self.lambda_mix = nn.Linear(input_dim, out_dim)
        
        # RMSNorm pour calmer le jeu après le carré
        self.norm = RMSNorm(input_dim)

    def forward(self, x, c):
        batch_size = c.shape[0]
        
        # A. x calcule W
        W = c @ self.weight_generator
        
        # B. Projection sur l'état précédent (W @ x_prev)
        output = torch.sum(W * x, dim=1, keepdim=True)
        
        # C. Activation Spectrale (Le Carré)
        energy = output ** 2
        
        # D. Lambda + Biais + RMSNorm
        # On normalise juste avant de sortir pour la couche suivante
        x_next_raw = self.lambda_mix(energy)
        return self.norm(x_next_raw)