"""
feynman_benchmark_config.py — Comparaison MLP × KAN × RAN sur le dataset Feynman
==================================================================================

Usage :
    python benchmark.py feynman_benchmark_config.py
    python benchmark.py feynman_benchmark_config.py --output results/feynman/ --verbose
    python benchmark.py feynman_benchmark_config.py --no-plots

Ce fichier définit :
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  3 familles de modèles :                                                │
  │   • MLP  — Multi-Layer Perceptron (baseline classique, GELU)            │
  │   • KAN  — Kolmogorov–Arnold Network (B-splines cubiques apprenables)   │
  │   • RAN  — Recurrent Attention Network (couches spectrales quadratiques)│
  ├─────────────────────────────────────────────────────────────────────────┤
  │  22 tâches Feynman couvrant :                                           │
  │   easy (8) · medium (8) · hard (5) · very_hard (1)                     │
  │  Formules physiques des Feynman Lectures I, II, III +                   │
  │  thermodynamique et relativité restreinte                               │
  └─────────────────────────────────────────────────────────────────────────┘

Références :
  Udrescu & Tegmark (2020). AI Feynman. Science Advances 6(16), eaay2631.
  Liu et al. (2024). KAN: Kolmogorov–Arnold Networks. arXiv:2404.19756.

Architecture KAN :
  KANLinear — B-splines cubiques (ordre 3, grille G=5 noeuds) sur chaque
  connexion i->j. Recurrence de Cox-De Boor. Residu SiLU (base_weight).
  Pas de dependance externe : implementation auto-contenue ci-dessous.
"""

import torch
import torch.nn as nn
import math

# -- Import des taches Feynman -------------------------------------------------
from feynman_tasks import FEYNMAN_TASKS


# ==============================================================================
#  IMPLEMENTATION KAN (auto-contenue, sans dependance externe)
# ==============================================================================

class KANLinear(nn.Module):
    """
    Couche KAN elementaire.

    Pour chaque connexion (i -> j) :
        y_j = sum_i [ w_b[j,i]*SiLU(x_i)  +  sum_k c[j,i,k]*B_k(x_i) ]

    ou B_k sont des B-splines cubiques sur une grille uniforme [-1, 1].

    Args :
      in_features   : dimension d'entree
      out_features  : dimension de sortie
      grid_size     : nombre d'intervalles de la grille (defaut 5)
      spline_order  : degre des B-splines (defaut 3 = cubique)
      scale_noise   : amplitude du bruit d'initialisation des splines
      scale_base    : gain Xavier pour la partie residu (SiLU)
      grid_range    : bornes de la grille [l, r] (defaut -1 a 1)
    """

    def __init__(
        self,
        in_features : int,
        out_features: int,
        grid_size   : int   = 5,
        spline_order: int   = 3,
        scale_noise : float = 0.1,
        scale_base  : float = 1.0,
        grid_range  : tuple = (-1.0, 1.0),
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order

        h    = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1,
        )
        self.register_buffer("grid", grid)

        n_basis = grid_size + spline_order
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, n_basis) * scale_noise
        )
        self.spline_scaler = nn.Parameter(torch.ones(out_features, in_features))
        self.base_weight   = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        self.base_weight.data *= scale_base

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evalue les B-splines en x via recurrence de Cox-De Boor.
        x     : (batch, in)
        retour: (batch, in, n_basis)
        """
        x = x.clamp(self.grid[self.spline_order], self.grid[-self.spline_order - 1])
        x = x.unsqueeze(-1)
        grid = self.grid

        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        for k in range(1, self.spline_order + 1):
            dl = grid[k:-1]   - grid[:-(k + 1)]
            dr = grid[k + 1:] - grid[1:-k]
            left  = (x - grid[:-(k + 1)]) / (dl + 1e-8) * bases[..., :-1]
            right = (grid[k + 1:] - x)    / (dr + 1e-8) * bases[..., 1:]
            bases = left + right
        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out   = nn.functional.linear(nn.functional.silu(x), self.base_weight)
        B          = self.b_splines(x)
        w          = self.spline_weight * self.spline_scaler.unsqueeze(-1)
        spline_out = torch.einsum("bik,oik->bo", B, w)
        return base_out + spline_out


class KAN(nn.Module):
    """
    Kolmogorov-Arnold Network : empilement de couches KANLinear.

    Args :
      layers_dim   : [in, h1, ..., out]
      grid_size    : taille de grille B-spline
      spline_order : ordre des B-splines (3 recommande)
      dropout      : taux de dropout inter-couches
    """

    def __init__(self, layers_dim: list, grid_size: int = 5,
                 spline_order: int = 3, dropout: float = 0.0):
        super().__init__()
        assert len(layers_dim) >= 2
        self.dropout = nn.Dropout(dropout)
        self.layers  = nn.ModuleList([
            KANLinear(layers_dim[i], layers_dim[i + 1],
                      grid_size=grid_size, spline_order=spline_order)
            for i in range(len(layers_dim) - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.dropout(x)
            x = layer(x)
        return x


# ==============================================================================
#  ADAPTATEUR ENTREE/SORTIE
#  Permet d'utiliser des backbones a dims fixes sur toutes les taches Feynman.
# ==============================================================================

class AdaptedModel(nn.Module):
    """
    Encapsule un backbone (MLP / KAN / RAN) avec :
      * projecteur d'entree  : task_in  -> backbone_in
      * projecteur de sortie : backbone_out -> task_out
    """
    def __init__(self, backbone, task_in, task_out, backbone_in, backbone_out):
        super().__init__()
        self.proj_in  = nn.Linear(task_in, backbone_in)   if task_in  != backbone_in  else nn.Identity()
        self.backbone = backbone
        self.proj_out = nn.Linear(backbone_out, task_out) if backbone_out != task_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_out(self.backbone(self.proj_in(x)))


# ==============================================================================
#  DIMENSIONS MAXIMALES (pour les adaptateurs)
# ==============================================================================

_MAX_IN  = max(t["input_dim"]  for t in FEYNMAN_TASKS)   # = 5
_MAX_OUT = max(t["output_dim"] for t in FEYNMAN_TASKS)   # = 1


# ==============================================================================
#  FACTORIES DE MODELES
# ==============================================================================

# -- MLP -----------------------------------------------------------------------

def _mlp_s():
    return AdaptedModel(
        nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 32)),
        _MAX_IN, _MAX_OUT, 32, 32)

def _mlp_m():
    return AdaptedModel(
        nn.Sequential(
            nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 64),
        ), _MAX_IN, _MAX_OUT, 64, 64)

def _mlp_l():
    return AdaptedModel(
        nn.Sequential(
            nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 128),
        ), _MAX_IN, _MAX_OUT, 128, 128)


# -- KAN -----------------------------------------------------------------------

def _kan_s():
    return AdaptedModel(KAN([32, 32, 32], grid_size=5, spline_order=3),
                        _MAX_IN, _MAX_OUT, 32, 32)

def _kan_m():
    return AdaptedModel(KAN([64, 64, 64], grid_size=5, spline_order=3),
                        _MAX_IN, _MAX_OUT, 64, 64)

def _kan_l():
    return AdaptedModel(KAN([128, 64, 64, 128], grid_size=7, spline_order=3),
                        _MAX_IN, _MAX_OUT, 128, 128)


# -- RAN -----------------------------------------------------------------------

def _ran_s():
    from Model.RecurentHyperNetwork.RecurentAttentionNetwork import RecurentAttentionNetwork
    return AdaptedModel(
        RecurentAttentionNetwork([32, 32, 32], use_layernorm=True, dropout=0.0),
        _MAX_IN, _MAX_OUT, 32, 32)

def _ran_m():
    from Model.RecurentHyperNetwork.RecurentAttentionNetwork import RecurentAttentionNetwork
    return AdaptedModel(
        RecurentAttentionNetwork([64, 64, 64], use_layernorm=True, dropout=0.0),
        _MAX_IN, _MAX_OUT, 64, 64)

def _ran_l():
    from Model.RecurentHyperNetwork.RecurentAttentionNetwork import RecurentAttentionNetwork
    return AdaptedModel(
        RecurentAttentionNetwork([128, 64, 64, 128], use_layernorm=True, dropout=0.0),
        _MAX_IN, _MAX_OUT, 128, 128)


# ==============================================================================
#  1. MODELES A TESTER
# ==============================================================================

MODELS = [

    # ──────────────────────── MLP ─────────────────────────────────────────
    {
        "name"         : "MLP_small   [32x2]",
        "model_class"  : None,
        "model_kwargs" : {},
        "model_factory": _mlp_s,
        # GELU, 2 couches cachees, ~3k params hors adaptateurs
        # Baseline rapide, bon pour taches lineaires.
    },
    {
        "name"         : "MLP_medium  [64x3]",
        "model_class"  : None,
        "model_kwargs" : {},
        "model_factory": _mlp_m,
        # GELU, 3 couches cachees, ~16k params
    },
    {
        "name"         : "MLP_large   [128x4]",
        "model_class"  : None,
        "model_kwargs" : {},
        "model_factory": _mlp_l,
        # GELU, 4 couches cachees, ~82k params
    },

    # ──────────────────────── KAN ─────────────────────────────────────────
    {
        "name"         : "KAN_small   [32x2]  g=5",
        "model_class"  : None,
        "model_kwargs" : {},
        "model_factory": _kan_s,
        # B-splines cubiques, grille 5 noeuds, ~11k params.
        # Avantage attendu sur fonctions transcendantales (exp, sin, 1/r^2).
        # Ref : Liu et al. (2024), arXiv:2404.19756
    },
    {
        "name"         : "KAN_medium  [64x2]  g=5",
        "model_class"  : None,
        "model_kwargs" : {},
        "model_factory": _kan_m,
        # ~41k params
    },
    {
        "name"         : "KAN_large   [128->64->64->128]  g=7",
        "model_class"  : None,
        "model_kwargs" : {},
        "model_factory": _kan_l,
        # Grille etendue (7 noeuds) pour taches hard/very_hard.
        # ~420k params (dont splines). Convergence plus lente.
    },

    # ──────────────────────── RAN ─────────────────────────────────────────
    {
        "name"         : "RAN_small   [32x2]",
        "model_class"  : None,
        "model_kwargs" : {},
        "model_factory": _ran_s,
        # RecurentSpectralLayer : h=W_base(x), out=W_eigen(h*h)
        # Non-linearite quadratique naturelle -> fort sur polynomes deg-2.
    },
    {
        "name"         : "RAN_medium  [64x2]",
        "model_class"  : None,
        "model_kwargs" : {},
        "model_factory": _ran_m,
    },
    {
        "name"         : "RAN_large   [128->64->64->128]",
        "model_class"  : None,
        "model_kwargs" : {},
        "model_factory": _ran_l,
    },
]


# ==============================================================================
#  2. TACHES — 22 equations Feynman (depuis feynman_tasks.py)
# ==============================================================================

TASKS = FEYNMAN_TASKS


# ==============================================================================
#  3. PARAMETRES D'ENTRAINEMENT
# ==============================================================================

TRAIN_CONFIG = {
    # -- Dataset ---------------------------------------------------------------
    "n_samples"   : 5000,      # 5k/tache : resultats robustes, ~2h CPU
                               # Reduire a 1000 pour un test rapide (~10 min)
                               # Passer a 10000 pour reproduire les scores publies

    # -- Optimisation ----------------------------------------------------------
    "epochs"      : 500,       # Augmenter a 2000 pour KAN_large sur taches hard
    "batch_size"  : 128,
    "optimizer"   : "adamw",   # AdamW : regularise les poids spline KAN
    "lr"          : 1e-3,
    "weight_decay": 1e-4,
    "scheduler"   : "cosine",  # CosineAnnealingLR : ameliore convergence KAN

    # -- Stabilite -------------------------------------------------------------
    "patience"    : 0,         # desactive (toutes les epochs tournent)
    "device"      : "cpu",     # changer en "cuda" ou "mps" si disponible

    # -- Reproductibilite ------------------------------------------------------
    "seed"        : 42,
}


# ==============================================================================
#  4. CROSS-VALIDATION
# ==============================================================================

CROSS_VAL = {
    "n_folds"  : 5,
    "shuffle"  : True,
    "val_ratio": 0.15,
}


# ==============================================================================
#  RESULTATS ATTENDUS (ordre de grandeur)
# ==============================================================================
#
#  Taches "easy"  (Gaussienne, Carnot, Pendule, Centre de masse) :
#    MLP ≈ KAN ≈ RAN  -- tous convergent sous 100 epochs, R^2 > 0.99
#
#  Taches "medium"  (Lorentz, Doppler, Snell, Kepler) :
#    KAN  > MLP ≈ RAN  -- splines capturent sqrt(1-beta^2) et log efficacement
#    RAN  > MLP         -- structure quadratique aide sur I.15.1 (x'=gamma*(x-vt))
#
#  Taches "hard"  (Planck, Bose-Einstein, Maxwell, Debye-Hückel, Larmor) :
#    KAN >> MLP >> RAN  -- exp(-x) est la force naturelle des B-splines
#    Note : sorties en log10 -> tache rendue plus douce
#
#  Tache "very_hard"  (Rabi III.9.52, 5 vars, oscillations) :
#    Tous les modeles peinent -- R^2 < 0.5 attendu sans curriculum learning
#    KAN_large + grille fine (g>=10) recommande pour ce cas precis
#
#  Efficacite parametrique :
#    KAN >> MLP  (Liu et al. 2024 : 100x moins de params pour meme MSE)
#    RAN ~= MLP  en general, RAN > MLP sur taches polynomiales (deg-2)
# ==============================================================================
