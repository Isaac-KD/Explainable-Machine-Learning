"""
KANNetwork.py — Réseau KAN complet, drop-in replacement de RecurentAttentionNetwork
=====================================================================================

Usage identique à RecurentAttentionNetwork :
    net = KANNetwork([8, 32, 32, 8], use_layernorm=True, dropout=0.1)
    y   = net(x)

Pour comparer dans experiment_config.py :
    from KANNetwork import KANNetwork
    {"name": "KAN_medium", "model_class": KANNetwork,
     "model_kwargs": {"layers_dim": [8,32,32,8], "n_grid": 5, "k": 3}}
"""

import torch
import torch.nn as nn
from typing import List
from .KANLayer import KANLayer


class KANNetwork(nn.Module):
    """
    Réseau KAN multi-couches avec LayerNorm et Dropout optionnels.

    L'interface est volontairement identique à RecurentAttentionNetwork pour
    faciliter la comparaison dans benchmark.py / experiment_config.py.

    Args:
        layers_dim   : liste [in_dim, hidden..., out_dim]
        n_grid       : résolution de la grille B-spline par couche
        k            : ordre du spline (3 = cubique)
        grid_range   : étendue initiale [-r, r] de la grille
        use_layernorm: normalisation entre couches (recommandé)
        dropout      : taux de dropout
        update_grid_every : toutes les N epochs, mettre à jour la grille
                            (0 = jamais). Non géré ici, exposé comme attribut
                            pour un callback d'entraînement personnalisé.
    """

    def __init__(
        self,
        layers_dim        : List[int],
        n_grid            : int   = 5,
        k                 : int   = 3,
        grid_range        : float = 1.0,
        use_layernorm     : bool  = True,
        dropout           : float = 0.0,
        update_grid_every : int   = 0,
    ):
        super().__init__()
        assert len(layers_dim) >= 2, "Il faut au moins une dim d'entrée et de sortie"

        self.layers_dim        = layers_dim
        self.use_layernorm     = use_layernorm
        self.update_grid_every = update_grid_every
        self.dropout           = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        for i in range(len(layers_dim) - 1):
            in_d, out_d = layers_dim[i], layers_dim[i + 1]
            self.layers.append(
                KANLayer(in_d, out_d, n_grid=n_grid, k=k, grid_range=grid_range)
            )
            if use_layernorm:
                self.layers.append(nn.LayerNorm(out_d))

    # ── Mise à jour globale de toutes les grilles ──────────────────

    @torch.no_grad()
    def update_grids(self, x: torch.Tensor) -> None:
        """
        Propage x à travers le réseau et met à jour chaque grille KAN.
        À appeler depuis une callback d'entraînement, ex. toutes les 100 epochs.

        Args:
            x : (batch, layers_dim[0]) — batch représentatif (ex. batch de val)
        """
        for layer in self.layers:
            if isinstance(layer, KANLayer):
                layer.update_grid(x)
            x = layer(x)   # propager pour la couche suivante

    # ── Forward ────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.dropout(x)
            x = layer(x)
        return x
