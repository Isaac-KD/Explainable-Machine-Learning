import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


class RecurentSpectralLayer(nn.Module):
    """
    Couche spectrale quadratique :  out = W_ew @ (W_bc @ x)²
    Architecture :
        h   = x @ W_bc.T + b_bc          (base_change, projection vers l'espace propre)
        out = (h ⊙ h) @ W_ew.T + b_ew   (pondération par les valeurs propres)

    Par rapport à nn.Linear :
        ✓ Appels F.linear directs → moins d'indirection Python
        ✓ Paramètres exposés directement → le compilateur JIT peut les fusionner
        ✓ Initialisation orthogonale de W_bc → stabilité dès l'epoch 0
    """

    __slots__ = ()  # empêche la création d'attributs dynamiques accidentels

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()

        # Matrice de changement de base : in_dim → out_dim
        self.W_bc  = nn.Parameter(torch.empty(out_dim, in_dim))
        self.b_bc  = nn.Parameter(torch.zeros(out_dim))

        # Poids spectraux (valeurs propres) : out_dim → out_dim
        self.W_ew  = nn.Parameter(torch.empty(out_dim, out_dim))
        self.b_ew  = nn.Parameter(torch.zeros(out_dim))

        self._init_parameters()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _init_parameters(self) -> None:
        # W_bc orthogonale → préserve les normes, évite explosion/disparition
        nn.init.orthogonal_(self.W_bc)

        # W_ew petite valeur uniforme → sortie proche de 0 au départ
        bound = 1.0 / math.sqrt(self.W_ew.size(1))
        nn.init.uniform_(self.W_ew, -bound, bound)

    # ------------------------------------------------------------------
    # Forward — optimisé pour torch.compile et AMP
    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        # h : (B, out_dim)
        h = F.linear(x, self.W_bc, self.b_bc)
        # sortie quadratique : (B, out_dim)
        return F.linear(h * h, self.W_ew, self.b_ew)

    # ------------------------------------------------------------------
    # Utilitaire
    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        in_dim  = self.W_bc.size(1)
        out_dim = self.W_bc.size(0)
        return f"in_dim={in_dim}, out_dim={out_dim}"


class RecurentBilineaireSpectralLayer(nn.Module):
    """
    Couche spectrale bilinéaire (bases gauche/droite distinctes) :
        out = W_ew @ (W_r @ x  ⊙  W_l @ x)

    Optimisation clé : les deux projections (right + left) sont fusionnées
    en un seul appel F.linear via la concaténation des poids → une seule
    opération GEMM au lieu de deux, divisé par 2 le nombre de lancements CUDA.

    Forme des poids fusionnés :
        W_proj : (2*out_dim, in_dim)   [W_r stacked W_l]
        b_proj : (2*out_dim,)
    """

    __slots__ = ()

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.out_dim = out_dim

        # Projection fusionnée right + left → une seule GEMM
        self.W_proj = nn.Parameter(torch.empty(2 * out_dim, in_dim))
        self.b_proj = nn.Parameter(torch.zeros(2 * out_dim))

        # Poids spectraux
        self.W_ew   = nn.Parameter(torch.empty(out_dim, out_dim))
        self.b_ew   = nn.Parameter(torch.zeros(out_dim))

        self._init_parameters()

    def _init_parameters(self) -> None:
        # Initialise chaque moitié indépendamment pour simuler deux matrices orthogonales
        nn.init.orthogonal_(self.W_proj[:self.out_dim])
        nn.init.orthogonal_(self.W_proj[self.out_dim:])

        bound = 1.0 / math.sqrt(self.W_ew.size(1))
        nn.init.uniform_(self.W_ew, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        # Une seule GEMM pour les deux projections
        h = F.linear(x, self.W_proj, self.b_proj)          # (B, 2*out_dim)
        h_r, h_l = h.chunk(2, dim=-1)                       # chacun (B, out_dim)
        return F.linear(h_r * h_l, self.W_ew, self.b_ew)   # (B, out_dim)

    def extra_repr(self) -> str:
        in_dim = self.W_proj.size(1)
        return f"in_dim={in_dim}, out_dim={self.out_dim}"
