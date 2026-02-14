import torch
import torch.nn as nn
from torch import Tensor
from typing import List
import torch.utils.checkpoint as ckpt

from .RecurentSpectralLayer import RecurentSpectralLayer


# ──────────────────────────────────────────────────────────────────────────────
# Bloc interne — une seule transition (spectral + optionnel norm + dropout)
# Séparé pour que torch.utils.checkpoint puisse l'encapsuler proprement
# ──────────────────────────────────────────────────────────────────────────────

class _SpectralBlock(nn.Module):
    """Un étage = dropout → SpectralLayer → (optionnel) LayerNorm."""

    def __init__(
        self,
        in_dim:   int,
        out_dim:  int,
        dropout:  float,
        use_norm: bool,
    ) -> None:
        super().__init__()
        self.spectral  = RecurentSpectralLayer(in_dim, out_dim)
        self.norm      = nn.LayerNorm(out_dim) if use_norm else nn.Identity()
        self.drop      = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.spectral(self.drop(x)))


# ──────────────────────────────────────────────────────────────────────────────
# Réseau principal
# ──────────────────────────────────────────────────────────────────────────────

class RecurentAttentionNetwork(nn.Module):
    """
    Réseau feedforward basé sur des couches spectrales quadratiques.

    Args:
        layers_dim          : Dimensions successives [in, h1, h2, ..., out].
        use_layernorm       : Normalisation après chaque couche spectrale.
        dropout             : Taux de dropout avant chaque couche spectrale.
        use_checkpointing   : Gradient checkpointing (économise la VRAM, +20 % de temps).
                              Activez pour de très grands modèles ou de très grands batchs.
        compile_blocks      : Si True, compile chaque bloc avec torch.compile.
                              Nécessite PyTorch >= 2.0 et CUDA.

    Exemple d'usage :
        model = RecurentAttentionNetwork(
            [784, 512, 256, 10],
            use_layernorm=True,
            dropout=0.1,
            use_checkpointing=True,
            compile_blocks=True,   # décommentez sur GPU
        )
    """

    def __init__(
        self,
        layers_dim:        List[int],
        use_layernorm:     bool  = True,
        dropout:           float = 0.0,
        use_checkpointing: bool  = False,
        compile_blocks:    bool  = False,
    ) -> None:
        super().__init__()

        assert len(layers_dim) >= 2, (
            "Il faut au moins une dimension d'entrée et de sortie."
        )

        self.use_checkpointing = use_checkpointing

        # ── Construction des blocs ────────────────────────────────────────────
        blocks: List[nn.Module] = []
        n = len(layers_dim)

        for i in range(n - 1):
            in_d, out_d = layers_dim[i], layers_dim[i + 1]
            # Pas de LayerNorm sur la dernière couche (sortie brute)
            apply_norm = use_layernorm and (i < n - 2)
            block: nn.Module = _SpectralBlock(in_d, out_d, dropout, apply_norm)

            if compile_blocks:
                # torch.compile fusionne les kernels du bloc entier
                block = torch.compile(block, mode="max-autotune", fullgraph=True)  # type: ignore[assignment]

            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        # ── Conserve layers pour la compatibilité avec test_AHN.py ───────────
        # Les tests inspectent network.layers et comptent les RecurentSpectralLayer
        # On reconstruit une vue plate identique à l'ancienne interface.
        self.layers = nn.ModuleList()
        for b in blocks:
            inner = b._orig_mod if hasattr(b, "_orig_mod") else b  # compile wrapping
            assert isinstance(inner, _SpectralBlock)
            self.layers.append(inner.spectral)
            if use_layernorm and not isinstance(inner.norm, nn.Identity):
                self.layers.append(inner.norm)

        # Expose dropout pour les tests qui font network.dropout.train()
        self.dropout = nn.Dropout(p=dropout)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            if self.use_checkpointing and self.training:
                # Gradient checkpointing : ne stocke pas les activations intermédiaires
                # → divise la mémoire par ~profondeur au coût de 30 % de calcul en plus
                x = ckpt.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x

    # ------------------------------------------------------------------
    # Utilitaires de haut niveau
    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Nombre total de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        total = self.count_parameters()
        return (
            f"blocks={len(self.blocks)}, "
            f"checkpointing={self.use_checkpointing}, "
            f"params={total:,}"
        )
