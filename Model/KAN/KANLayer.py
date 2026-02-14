"""
KANLayer.py — Couche KAN (Kolmogorov-Arnold Network) vectorisée
================================================================

Formulation :
    Pour chaque paire (entrée i, sortie j) :
        phi_ij(x_i) = w_b * SiLU(x_i)  +  w_s * spline(x_i)
    Sortie_j = sum_i phi_ij(x_i)

Optimisations :
  - Grille B-spline entièrement vectorisée (batch, in_dim, n_bases) en une passe
  - Aucune boucle Python sur les nœuds
  - Mise à jour adaptative de grille (update_grid) disponible pendant l'entraînement
  - Initialisation contrôlée pour éviter l'explosion des activations

Référence : Liu et al., "KAN: Kolmogorov-Arnold Networks", arXiv:2404.19756
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────────────────────
#  Noyau B-spline vectorisé
# ─────────────────────────────────────────────────────────────

def b_splines(x: torch.Tensor, grid: torch.Tensor, k: int) -> torch.Tensor:
    """
    Base B-spline d'ordre k — entièrement vectorisée, sans boucle sur les nœuds.

    Args:
        x    : (batch, in_dim)
        grid : (in_dim, n_grid + 2k + 1)  — nœuds étendus par feature
        k    : ordre du spline (3 = cubique)

    Returns:
        bases : (batch, in_dim, n_grid + k)
    """
    # (batch, in_dim, 1)
    x = x.unsqueeze(-1)
    # (1, in_dim, G)
    g = grid.unsqueeze(0)

    # Bases d'ordre 0 : indicatrices [g_i, g_{i+1})
    bases = ((x >= g[:, :, :-1]) & (x < g[:, :, 1:])).float()

    # Récurrence de Cox-de Boor
    for p in range(1, k + 1):
        # Termes gauche et droit
        gl = g[:, :, :-(p + 1)]
        gr = g[:, :,  p: -1   ]
        dl = (gr - gl).clamp(min=1e-8)

        gl2 = g[:, :, 1:  -p  ]
        gr2 = g[:, :, p+1:    ]
        dr  = (gr2 - gl2).clamp(min=1e-8)

        left  = (x - gl)  / dl * bases[:, :, :-1]
        right = (gr2 - x) / dr * bases[:, :,  1:]
        bases = left + right

    return bases.contiguous()   # (batch, in_dim, n_grid + k)


# ─────────────────────────────────────────────────────────────
#  KANLayer
# ─────────────────────────────────────────────────────────────

class KANLayer(nn.Module):
    """
    Couche KAN vectorisée.

    Paramètres :
      spline_weights : (out_dim, in_dim, n_bases)
      scale_base     : (out_dim, in_dim)   — pondération activation résiduelle
      scale_spline   : (out_dim, in_dim)   — pondération spline

    Args:
        in_dim      : dimension d'entrée
        out_dim     : dimension de sortie
        n_grid      : nombre d'intervalles de grille  (résolution du spline)
        k           : ordre du B-spline  (3 = cubique)
        grid_range  : étendue initiale de la grille  [-r, r]
        grid_eps    : marge pour la mise à jour adaptative de grille
    """

    def __init__(
        self,
        in_dim     : int,
        out_dim    : int,
        n_grid     : int   = 5,
        k          : int   = 3,
        grid_range : float = 1.0,
        grid_eps   : float = 0.02,
    ):
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.n_grid  = n_grid
        self.k       = k
        self.n_bases = n_grid + k
        self.grid_eps = grid_eps

        self.base_activation = nn.SiLU()

        # ── Grille initiale régulière ──────────────────────────────
        # Points intérieurs : n_grid + 1 valeurs dans [-r, r]
        pts  = torch.linspace(-grid_range, grid_range, n_grid + 1)
        step = pts[1] - pts[0]
        ext_l = pts[0]  - step * torch.arange(k, 0, -1)
        ext_r = pts[-1] + step * torch.arange(1, k + 1)
        full  = torch.cat([ext_l, pts, ext_r])          # (n_grid + 2k + 1,)
        grid  = full.unsqueeze(0).expand(in_dim, -1).contiguous()
        self.register_buffer("grid", grid)               # pas apprenable

        # ── Paramètres apprenables ─────────────────────────────────
        self.spline_weights = nn.Parameter(torch.empty(out_dim, in_dim, self.n_bases))
        self.scale_base     = nn.Parameter(torch.empty(out_dim, in_dim))
        self.scale_spline   = nn.Parameter(torch.empty(out_dim, in_dim))

        self._init_parameters()

    # ── Initialisation ─────────────────────────────────────────────

    def _init_parameters(self) -> None:
        # scale_base ~ N(0, 1/sqrt(in)) pour variance unité en sortie
        nn.init.normal_(self.scale_base, std=1.0 / self.in_dim ** 0.5)
        nn.init.ones_(self.scale_spline)
        # Spline weights : Xavier sur la vue aplatie
        w = self.spline_weights.view(self.out_dim, -1)
        nn.init.xavier_uniform_(w)

    # ── Mise à jour adaptative de grille ───────────────────────────

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor) -> None:
        """
        Recalcule la grille sur la distribution réelle des activations.
        Appeler toutes les N epochs, pas à chaque pas.

        Args:
            x : (batch, in_dim) — batch représentatif
        """
        batch = x.size(0)
        x_sorted, _ = x.sort(dim=0)

        q   = torch.linspace(self.grid_eps, 1 - self.grid_eps,
                             self.n_grid + 1, device=x.device)
        idx = (q * (batch - 1)).long().clamp(0, batch - 1)
        adaptive = x_sorted[idx].T                        # (in_dim, n_grid+1)

        step = (adaptive[:, -1:] - adaptive[:, :1]) / self.n_grid
        uniform = adaptive[:, :1] + step * torch.arange(
            self.n_grid + 1, device=x.device)

        new_pts  = self.grid_eps * uniform + (1 - self.grid_eps) * adaptive
        step_new = new_pts[:, 1:2] - new_pts[:, :1]
        ext_l    = new_pts[:, :1]  - step_new * torch.arange(
            self.k, 0, -1, device=x.device)
        ext_r    = new_pts[:, -1:] + step_new * torch.arange(
            1, self.k + 1, device=x.device)
        new_grid = torch.cat([ext_l, new_pts, ext_r], dim=1)
        self.grid.copy_(new_grid)

    # ── Forward ────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, in_dim)
        Returns:
            y : (batch, out_dim)
        """
        # 1. Branche résiduelle SiLU
        # F.linear : (batch, in) × (out, in).T → (batch, out)
        base_out = F.linear(self.base_activation(x), self.scale_base)

        # 2. Branche spline
        # bases : (batch, in_dim, n_bases)
        bases = b_splines(x, self.grid, self.k)

        # Fusion poids spline + échelle : (out_dim, in_dim, n_bases)
        w = self.spline_weights * self.scale_spline.unsqueeze(-1)

        # Contraction tensorielle → (batch, out_dim)
        spline_out = torch.einsum("bin,oin->bo", bases, w)

        return base_out + spline_out
